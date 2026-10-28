//! The prompt: a card holding a multi-line editor, with the session's settings in its top edge
//! and a hint line or slash-command menu below it, drawn inline under the transcript and redrawn
//! in place on every keystroke.

use std::io::Write;
use std::mem;

use anyhow::Result;
use crossterm::cursor::{MoveToColumn, MoveUp};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::queue;
use crossterm::style::{Color, Stylize};
use crossterm::terminal::{Clear, ClearType};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::frame::{self, INDENT, fit, truncate};
use crate::{ACCENT, FOCUS, USER};

/// A slash command: its name, its argument (empty if none) and what it does.
pub type Command = (&'static str, &'static str, &'static str);

const PLACEHOLDER: &str = "Ask anything, or type / for commands";
const KEYS: &[(&str, &str)] = &[
    ("enter", "send"),
    ("\\ enter", "new line"),
    ("/", "commands"),
    ("↑", "history"),
    ("ctrl-c ×2", "quit"),
];
const MENU_KEYS: &[(&str, &str)] = &[
    ("↑↓", "choose"),
    ("tab", "complete"),
    ("enter", "run"),
    ("esc", "cancel"),
];
/// "› " before the first line of text.
const PROMPT_W: usize = 2;
const KEY: Color = Color::AnsiValue(250);

pub enum Input {
    Line(String),
    Exit,
}

enum Action {
    Edit,
    Submit,
    Exit,
}

pub struct Editor {
    commands: &'static [Command],
    /// Shown in the rule above the text.
    status: String,
    /// The highlighted entry of the command menu.
    selected: usize,
    history: Vec<String>,
    buf: Vec<char>,
    cursor: usize,
    /// Position in `history` while browsing it, and the unsent text to come back to.
    browsing: Option<usize>,
    draft: Vec<char>,
    exit_armed: bool,
    /// Rows from the top of the drawn box down to the terminal cursor.
    cursor_row: usize,
}

impl Editor {
    pub fn new(commands: &'static [Command]) -> Self {
        Self {
            commands,
            status: String::new(),
            selected: 0,
            history: Vec::new(),
            buf: Vec::new(),
            cursor: 0,
            browsing: None,
            draft: Vec::new(),
            exit_armed: false,
            cursor_row: 0,
        }
    }

    /// Draws the prompt at the cursor, with `status` in its rule, and edits until the user sends
    /// a line or quits; the prompt is erased before returning.
    pub fn read(&mut self, out: &mut impl Write, status: &str) -> Result<Input> {
        self.status = status.to_string();
        self.selected = 0;
        self.buf.clear();
        self.cursor = 0;
        self.browsing = None;
        self.exit_armed = false;
        self.cursor_row = 0;
        loop {
            self.draw(out)?;
            let before = self.buf.clone();
            let action = match event::read()? {
                Event::Key(k) if k.kind != KeyEventKind::Release => self.key(k),
                Event::Paste(s) => {
                    self.insert(&s.replace("\r\n", "\n").replace('\r', "\n"));
                    Action::Edit
                }
                _ => Action::Edit,
            };
            if self.buf != before {
                self.selected = 0;
            }
            match action {
                Action::Edit => {}
                Action::Submit => {
                    self.erase(out)?;
                    let line: String = mem::take(&mut self.buf).into_iter().collect();
                    let line = line.trim().to_string();
                    if self.history.last() != Some(&line) {
                        self.history.push(line.clone());
                    }
                    return Ok(Input::Line(line));
                }
                Action::Exit => {
                    self.erase(out)?;
                    return Ok(Input::Exit);
                }
            }
        }
    }

    fn key(&mut self, k: KeyEvent) -> Action {
        let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
        let alt = k.modifiers.contains(KeyModifiers::ALT);
        let shift = k.modifiers.contains(KeyModifiers::SHIFT);
        let ctrl_c = ctrl && k.code == KeyCode::Char('c');
        if !ctrl_c {
            self.exit_armed = false;
        }
        match k.code {
            KeyCode::Char('c') if ctrl => {
                if !self.buf.is_empty() {
                    self.set(Vec::new());
                } else if mem::replace(&mut self.exit_armed, true) {
                    return Action::Exit;
                }
            }
            KeyCode::Char('d') if ctrl => {
                if self.buf.is_empty() {
                    return Action::Exit;
                }
                if self.cursor < self.buf.len() {
                    self.buf.remove(self.cursor);
                }
            }
            KeyCode::Char('a') if ctrl => self.cursor = self.line_start(self.cursor),
            KeyCode::Char('e') if ctrl => self.cursor = self.line_end(self.cursor),
            KeyCode::Char('u') if ctrl => {
                let start = self.line_start(self.cursor);
                self.buf.drain(start..self.cursor);
                self.cursor = start;
            }
            KeyCode::Char('k') if ctrl => {
                let end = self.line_end(self.cursor);
                self.buf.drain(self.cursor..end);
            }
            KeyCode::Char('w') if ctrl => self.delete_word(),
            // Ctrl-J is a raw line feed: a newline in every terminal.
            KeyCode::Char('j') if ctrl => self.insert("\n"),
            KeyCode::Char('b') if alt => self.cursor = self.word_left(),
            KeyCode::Char('f') if alt => self.cursor = self.word_right(),
            KeyCode::Char(c) if !ctrl => self.insert(&c.to_string()),
            KeyCode::Enter if shift || alt => self.insert("\n"),
            KeyCode::Enter => return self.enter(),
            KeyCode::Tab => self.complete(),
            KeyCode::Backspace if alt || ctrl => self.delete_word(),
            KeyCode::Backspace => {
                if self.cursor > 0 {
                    self.cursor -= 1;
                    self.buf.remove(self.cursor);
                }
            }
            KeyCode::Delete => {
                if self.cursor < self.buf.len() {
                    self.buf.remove(self.cursor);
                }
            }
            KeyCode::Left if alt || ctrl => self.cursor = self.word_left(),
            KeyCode::Right if alt || ctrl => self.cursor = self.word_right(),
            KeyCode::Left => self.cursor = self.cursor.saturating_sub(1),
            KeyCode::Right => self.cursor = (self.cursor + 1).min(self.buf.len()),
            KeyCode::Home => self.cursor = self.line_start(self.cursor),
            KeyCode::End => self.cursor = self.line_end(self.cursor),
            KeyCode::Up => self.up(),
            KeyCode::Down => self.down(),
            KeyCode::Esc => self.set(Vec::new()),
            _ => {}
        }
        Action::Edit
    }

    fn enter(&mut self) -> Action {
        // A trailing backslash continues onto a new line, as in a shell.
        if self.cursor > 0 && self.buf[self.cursor - 1] == '\\' {
            self.buf[self.cursor - 1] = '\n';
            return Action::Edit;
        }
        let text = self.text();
        if text.trim().is_empty() {
            return Action::Edit;
        }
        if !self.commands.iter().any(|(c, _, _)| *c == text)
            && let Some((c, _, _)) = self.menu().get(self.selected)
        {
            self.set(c.chars().collect());
        }
        Action::Submit
    }

    fn complete(&mut self) {
        if let Some((c, _, _)) = self.menu().get(self.selected) {
            self.set(format!("{c} ").chars().collect());
        }
    }

    /// The commands matching a partly typed one; empty unless the text is a command prefix.
    fn menu(&self) -> Vec<&'static Command> {
        let text = self.text();
        if !is_command_prefix(&text) {
            return Vec::new();
        }
        self.commands
            .iter()
            .filter(|(c, _, _)| c.starts_with(&text))
            .collect()
    }

    /// Moves the menu highlight, wrapping around; false if no menu is open.
    fn step_menu(&mut self, forward: bool) -> bool {
        let n = self.menu().len();
        if n == 0 || self.browsing.is_some() {
            return false;
        }
        self.selected = if forward {
            (self.selected + 1) % n
        } else {
            (self.selected + n - 1) % n
        };
        true
    }

    fn insert(&mut self, s: &str) {
        for c in s.chars() {
            let expanded = match c {
                '\t' => "    ".chars().collect(),
                '\n' => vec!['\n'],
                c if c.is_control() => continue,
                c => vec![c],
            };
            for c in expanded {
                self.buf.insert(self.cursor, c);
                self.cursor += 1;
            }
        }
    }

    fn set(&mut self, buf: Vec<char>) {
        self.buf = buf;
        self.cursor = self.buf.len();
    }

    fn text(&self) -> String {
        self.buf.iter().collect()
    }

    fn line_start(&self, i: usize) -> usize {
        self.buf[..i]
            .iter()
            .rposition(|&c| c == '\n')
            .map_or(0, |p| p + 1)
    }

    fn line_end(&self, i: usize) -> usize {
        self.buf[i..]
            .iter()
            .position(|&c| c == '\n')
            .map_or(self.buf.len(), |p| i + p)
    }

    /// Up moves through the command menu when it is open, within a multi-line message, and
    /// through history from its first line.
    fn up(&mut self) {
        if self.step_menu(false) {
            return;
        }
        let start = self.line_start(self.cursor);
        if start == 0 {
            let i = match self.browsing {
                None if !self.history.is_empty() => {
                    self.draft = self.buf.clone();
                    self.history.len() - 1
                }
                Some(i) if i > 0 => i - 1,
                _ => return,
            };
            self.browsing = Some(i);
            let entry = self.history[i].chars().collect();
            self.set(entry);
            return;
        }
        let col = self.cursor - start;
        let prev = self.line_start(start - 1);
        self.cursor = prev + col.min(start - 1 - prev);
    }

    fn down(&mut self) {
        if self.step_menu(true) {
            return;
        }
        let end = self.line_end(self.cursor);
        if end == self.buf.len() {
            let Some(i) = self.browsing else { return };
            let entry = if i + 1 < self.history.len() {
                self.browsing = Some(i + 1);
                self.history[i + 1].chars().collect()
            } else {
                self.browsing = None;
                mem::take(&mut self.draft)
            };
            self.set(entry);
            return;
        }
        let col = self.cursor - self.line_start(self.cursor);
        let next = end + 1;
        self.cursor = next + col.min(self.line_end(next) - next);
    }

    fn word_left(&self) -> usize {
        let mut i = self.cursor;
        while i > 0 && self.buf[i - 1].is_whitespace() {
            i -= 1;
        }
        while i > 0 && !self.buf[i - 1].is_whitespace() {
            i -= 1;
        }
        i
    }

    fn word_right(&self) -> usize {
        let mut i = self.cursor;
        while i < self.buf.len() && self.buf[i].is_whitespace() {
            i += 1;
        }
        while i < self.buf.len() && !self.buf[i].is_whitespace() {
            i += 1;
        }
        i
    }

    fn delete_word(&mut self) {
        let start = self.word_left();
        self.buf.drain(start..self.cursor);
        self.cursor = start;
    }

    fn draw(&mut self, out: &mut impl Write) -> Result<()> {
        let width = frame::text_width();
        let text_w = width - PROMPT_W;
        let (rows, (row, col)) = layout(&self.buf, self.cursor, text_w);
        let parts: Vec<&str> = self.status.split(" · ").collect();
        let status = fit(&parts, width.saturating_sub(8));
        let mut lines = vec![frame::edge(
            ['┌', '┐'],
            ("you", USER),
            (&status, Color::DarkGrey),
            FOCUS,
            width,
        )];
        for (i, r) in rows.iter().enumerate() {
            let prefix = if i == 0 {
                format!("{} ", "›".with(USER).bold())
            } else {
                " ".repeat(PROMPT_W)
            };
            let (content, w) = if self.buf.is_empty() {
                let p = truncate(PLACEHOLDER, text_w);
                let w = p.width();
                (p.dark_grey().to_string(), w)
            } else {
                (r.clone(), r.width())
            };
            lines.push(frame::row(
                &format!("{prefix}{content}"),
                PROMPT_W + w,
                FOCUS,
                width,
            ));
        }
        lines.push(frame::edge(
            ['└', '┘'],
            ("", USER),
            ("", USER),
            FOCUS,
            width,
        ));
        // Under the card, as wide as it.
        lines.extend(self.hints(width + INDENT + 2));

        self.erase(out)?;
        write!(out, "{}", lines.join("\r\n"))?;
        let target = 1 + row;
        let up = lines.len() - 1 - target;
        if up > 0 {
            queue!(out, MoveUp(up as u16))?;
        }
        queue!(out, MoveToColumn((INDENT + PROMPT_W + col) as u16))?;
        self.cursor_row = target;
        out.flush()?;
        Ok(())
    }

    fn hints(&self, width: usize) -> Vec<String> {
        let menu = self.menu();
        if !menu.is_empty() {
            let label = |(c, a, _): &Command| {
                if a.is_empty() {
                    c.to_string()
                } else {
                    format!("{c} {a}")
                }
            };
            let label_w = menu.iter().map(|c| label(c).width()).max().unwrap_or(0);
            let mut lines: Vec<String> = menu
                .iter()
                .enumerate()
                .map(|(i, &cmd)| {
                    let (c, a, d) = cmd;
                    let pad = " ".repeat(label_w - label(cmd).width() + 2);
                    let d = truncate(d, width.saturating_sub(label_w + 6));
                    let a = if a.is_empty() {
                        String::new()
                    } else {
                        format!(" {a}")
                    };
                    if i == self.selected {
                        format!(
                            " {} {}{}{pad}{d}",
                            "▸".with(ACCENT),
                            c.with(ACCENT).bold(),
                            a.with(ACCENT)
                        )
                    } else {
                        format!("   {c}{}{pad}{}", a.dark_grey(), d.dark_grey())
                    }
                })
                .collect();
            lines.push(keys(MENU_KEYS, width));
            return lines;
        }
        let text = self.text();
        if let Some((name, rest)) = text.split_once(char::is_whitespace)
            && rest.trim().is_empty()
            && let Some((c, a, d)) = self.commands.iter().find(|(c, _, _)| *c == name)
            && !a.is_empty()
        {
            let d = truncate(d, width.saturating_sub(c.width() + a.width() + 8));
            return vec![format!(
                "   {} {}  {}",
                c.with(ACCENT),
                a.with(ACCENT),
                d.dark_grey()
            )];
        }
        if self.exit_armed {
            return vec![format!("   {}", "press ctrl-c again to quit".yellow())];
        }
        vec![keys(KEYS, width)]
    }

    /// Moves to the top of the drawn prompt and clears it.
    fn erase(&mut self, out: &mut impl Write) -> Result<()> {
        if self.cursor_row > 0 {
            queue!(out, MoveUp(self.cursor_row as u16))?;
        }
        queue!(out, MoveToColumn(0), Clear(ClearType::FromCursorDown))?;
        self.cursor_row = 0;
        out.flush()?;
        Ok(())
    }
}

fn is_command_prefix(text: &str) -> bool {
    text.starts_with('/') && !text.contains(char::is_whitespace)
}

/// Key-and-action pairs on one line, dropping those that do not fit.
fn keys(pairs: &[(&str, &str)], width: usize) -> String {
    let mut line = "  ".to_string();
    let mut w = 2;
    for (i, (k, d)) in pairs.iter().enumerate() {
        let sep = if i == 0 { " " } else { " · " };
        let add = sep.width() + k.width() + 1 + d.width();
        if w + add >= width {
            break;
        }
        line.push_str(&format!(
            "{}{} {}",
            sep.dark_grey(),
            k.with(KEY),
            d.dark_grey()
        ));
        w += add;
    }
    line
}

/// Wraps the buffer into rows at most `width` columns wide; returns them with the cursor's
/// (row, column).
fn layout(buf: &[char], cursor: usize, width: usize) -> (Vec<String>, (usize, usize)) {
    let mut rows = vec![String::new()];
    let mut col = 0;
    let mut at = (0, 0);
    for (i, &ch) in buf.iter().enumerate() {
        if ch == '\n' {
            if i == cursor {
                at = (rows.len() - 1, col);
            }
            rows.push(String::new());
            col = 0;
            continue;
        }
        let w = ch.width().unwrap_or(0);
        if col + w > width {
            rows.push(String::new());
            col = 0;
        }
        if i == cursor {
            at = (rows.len() - 1, col);
        }
        rows.last_mut().expect("rows start non-empty").push(ch);
        col += w;
    }
    if cursor >= buf.len() {
        if col >= width {
            rows.push(String::new());
            col = 0;
        }
        at = (rows.len() - 1, col);
    }
    (rows, at)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chars(s: &str) -> Vec<char> {
        s.chars().collect()
    }

    #[test]
    fn wraps_rows_and_places_the_cursor() {
        let (rows, at) = layout(&chars("abcdefg"), 7, 4);
        assert_eq!(rows, ["abcd", "efg"]);
        assert_eq!(at, (1, 3));
        // A full last row puts the cursor at the start of a new one.
        let (rows, at) = layout(&chars("abcd"), 4, 4);
        assert_eq!(rows, ["abcd", ""]);
        assert_eq!(at, (1, 0));
        let (rows, at) = layout(&chars("ab\ncd"), 2, 4);
        assert_eq!(rows, ["ab", "cd"]);
        assert_eq!(at, (0, 2));
        assert_eq!(layout(&chars("ab\ncd"), 3, 4).1, (1, 0));
    }

    #[test]
    fn moves_between_lines_and_history() {
        let mut e = Editor::new(&[]);
        e.history = vec!["first".into(), "second".into()];
        e.set(chars("one\ntwo"));
        e.up();
        assert_eq!(e.cursor, 3);
        e.up();
        assert_eq!(e.text(), "second");
        e.up();
        assert_eq!(e.text(), "first");
        e.down();
        e.down();
        assert_eq!(e.text(), "one\ntwo");
    }

    #[test]
    fn completes_commands_on_enter() {
        let mut e = Editor::new(&[("/clear", "", ""), ("/help", "", "")]);
        e.set(chars("/cl"));
        assert!(matches!(e.enter(), Action::Submit));
        assert_eq!(e.text(), "/clear");
        // Arrows move through the menu, wrapping around.
        e.set(chars("/"));
        e.down();
        assert!(matches!(e.enter(), Action::Submit));
        assert_eq!(e.text(), "/help");
        e.set(chars("/"));
        e.selected = 0;
        e.up();
        e.complete();
        assert_eq!(e.text(), "/help ");
        e.set(chars("line\\"));
        assert!(matches!(e.enter(), Action::Edit));
        assert_eq!(e.text(), "line\n");
    }
}
