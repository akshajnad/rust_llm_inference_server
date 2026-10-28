//! Formats a streamed reply for the terminal as it arrives, inside its card: word wrapping
//! under a hanging indent, each finished line closed with the card's right border, plus the
//! Markdown a small model actually writes (headings, bullets, **bold**, `code` and fenced
//! blocks).

use std::mem;

use crossterm::style::{Attribute, Color, ContentStyle};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::ACCENT;
use crate::frame::{self, BORDER, INDENT};

const CODE: Color = Color::AnsiValue(179);

pub struct Renderer {
    /// The column text stops at; the card's right border follows it.
    width: usize,
    col: usize,
    /// The card's left border, starting every line after the first (the caller draws that one),
    /// and its right border, ending every line.
    left: String,
    right: String,
    /// Where wrapped lines continue: under the text of a list item, else at `INDENT`.
    hang: usize,
    out: String,
    /// The word being received; printed once whitespace ends it, so it can wrap whole.
    word: String,
    spaces: usize,
    /// Line breaks seen but not printed yet, so trailing ones never print.
    newlines: usize,
    started: bool,
    /// At the start of a source line, where block syntax (`#`, ```` ``` ````, `-`) counts.
    line_start: bool,
    /// The current line of a fenced code block, while inside one.
    fence: Option<String>,
    /// The opening fence's info string (`python`) is still arriving.
    fence_header: bool,
    heading: bool,
    bold: bool,
    code: bool,
}

impl Renderer {
    /// A renderer for a card whose text is `text_width` columns wide.
    pub fn new(text_width: usize) -> Self {
        Self {
            width: INDENT + text_width,
            col: INDENT,
            left: frame::left(BORDER),
            right: frame::right(BORDER),
            hang: INDENT,
            out: String::new(),
            word: String::new(),
            spaces: 0,
            newlines: 0,
            started: false,
            line_start: true,
            fence: None,
            fence_header: false,
            heading: false,
            bold: false,
            code: false,
        }
    }

    /// Terminal column of the cursor after everything taken so far.
    pub fn col(&self) -> usize {
        self.col
    }

    /// Output produced since the last call.
    pub fn take(&mut self) -> String {
        mem::take(&mut self.out)
    }

    pub fn push(&mut self, text: &str) {
        for ch in text.chars() {
            if let Some(line) = &mut self.fence {
                if ch == '\n' {
                    self.code_line();
                } else {
                    line.push(ch);
                }
                continue;
            }
            match ch {
                '\n' => {
                    self.flush_word();
                    if self.fence.is_some() {
                        // The line that opened a fence ended.
                        self.fence_header = false;
                        continue;
                    }
                    self.spaces = 0;
                    self.newlines += 1;
                    self.line_start = true;
                    self.heading = false;
                    self.bold = false;
                    self.code = false;
                }
                ' ' => {
                    self.flush_word();
                    self.spaces += 1;
                }
                '\t' => {
                    self.flush_word();
                    self.spaces += 4;
                }
                '\r' => {}
                _ => self.word.push(ch),
            }
        }
    }

    /// Prints whatever is still buffered and closes the last line; returns the remaining
    /// output.
    pub fn finish(&mut self) -> String {
        if self.fence.is_some() {
            if self.fence.as_ref().is_some_and(|l| !l.is_empty()) {
                self.code_line();
            }
            self.fence = None;
        } else {
            self.flush_word();
        }
        if self.started {
            self.close_line();
        }
        self.take()
    }

    fn code_line(&mut self) {
        let line = self.fence.replace(String::new()).unwrap_or_default();
        if mem::take(&mut self.fence_header) {
            return;
        }
        if line.trim_start().starts_with("```") {
            self.fence = None;
            self.line_start = true;
            return;
        }
        self.emit_newlines();
        // A bar sets code apart from prose.
        self.emit("▎ ", style(Some(BORDER), false));
        self.emit(&line.replace('\t', "    "), style(Some(CODE), false));
        self.newlines = 1;
    }

    fn flush_word(&mut self) {
        if self.word.is_empty() {
            return;
        }
        let mut word = mem::take(&mut self.word);
        let mut list_item = false;
        if mem::take(&mut self.line_start) {
            if word.starts_with("```") {
                self.fence = Some(String::new());
                self.fence_header = true;
                self.spaces = 0;
                return;
            }
            if word.len() <= 6 && word.chars().all(|c| c == '#') {
                self.heading = true;
                self.spaces = 0;
                return;
            }
            if word == "*" || word == "-" || word == "+" || word == "•" {
                word = "•".to_string();
                list_item = true;
            }
            list_item |= word
                .strip_suffix('.')
                .is_some_and(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()));
        }
        let segments = self.inline(&word);
        let width: usize = segments.iter().map(|(s, _)| s.width()).sum();
        self.emit_newlines();
        if self.heading && self.col == INDENT {
            // The space after the dropped `#` marker.
            self.spaces = 0;
        }
        if self.col > self.hang && self.col + self.spaces + width > self.width {
            self.break_line();
        } else {
            self.out.push_str(&" ".repeat(self.spaces));
            self.col += self.spaces;
        }
        self.spaces = 0;
        for (s, st) in segments {
            self.emit(&s, st);
        }
        if list_item {
            self.hang = (self.col + 1).min(INDENT + (self.width - INDENT) / 2);
        }
    }

    /// Splits a word into styled runs, consuming `**` and backtick markers.
    fn inline(&mut self, word: &str) -> Vec<(String, ContentStyle)> {
        let mut runs: Vec<(String, (bool, bool))> = Vec::new();
        let mut chars = word.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '`' {
                self.code = !self.code;
                continue;
            }
            if c == '*' && !self.code && chars.peek() == Some(&'*') {
                chars.next();
                self.bold = !self.bold;
                continue;
            }
            let key = (self.bold || self.heading, self.code);
            match runs.last_mut() {
                Some((s, k)) if *k == key => s.push(c),
                _ => runs.push((c.to_string(), key)),
            }
        }
        runs.into_iter()
            .map(|(s, (bold, code))| {
                let fg = if code {
                    Some(CODE)
                } else {
                    self.heading.then_some(ACCENT)
                };
                (s, style(fg, bold))
            })
            .collect()
    }

    /// Prints text, hard-wrapping anything wider than a line.
    fn emit(&mut self, text: &str, st: ContentStyle) {
        let mut chunk = String::new();
        for ch in text.chars() {
            let w = ch.width().unwrap_or(0);
            if self.col + w > self.width && self.col > self.hang {
                self.out
                    .push_str(&st.apply(mem::take(&mut chunk)).to_string());
                self.break_line();
            }
            chunk.push(ch);
            self.col += w;
        }
        if !chunk.is_empty() {
            self.out.push_str(&st.apply(chunk).to_string());
        }
    }

    fn emit_newlines(&mut self) {
        if !self.started {
            // Leading blank lines and spaces never print.
            self.started = true;
            self.newlines = 0;
            self.spaces = 0;
            return;
        }
        if self.newlines > 0 {
            // At most one blank line between paragraphs.
            for _ in 0..self.newlines.min(2) {
                self.close_line();
                self.out.push_str("\r\n");
                self.out.push_str(&self.left);
                self.col = INDENT;
            }
            self.hang = INDENT;
            self.newlines = 0;
        }
    }

    fn break_line(&mut self) {
        self.close_line();
        self.out.push_str("\r\n");
        self.out.push_str(&self.left);
        self.out.push_str(&" ".repeat(self.hang - INDENT));
        self.col = self.hang;
    }

    /// Pads the current line out to the right border and draws it.
    fn close_line(&mut self) {
        self.out
            .push_str(&" ".repeat(self.width.saturating_sub(self.col)));
        self.out.push_str(&self.right);
    }
}

fn style(fg: Option<Color>, bold: bool) -> ContentStyle {
    let mut s = ContentStyle::new();
    s.foreground_color = fg;
    if bold {
        s.attributes.set(Attribute::Bold);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rendered lines without escape codes.
    fn raw(input: &[&str], text_width: usize) -> Vec<String> {
        let mut r = Renderer::new(text_width);
        let mut out = String::new();
        for piece in input {
            r.push(piece);
            out.push_str(&r.take());
        }
        out.push_str(&r.finish());
        let mut text = String::new();
        let mut chars = out.chars();
        while let Some(c) = chars.next() {
            if c == '\x1b' {
                chars.by_ref().find(|&c| c == 'm');
            } else if c != '\r' {
                text.push(c);
            }
        }
        text.split('\n').map(str::to_string).collect()
    }

    /// Rendered lines without the card's borders.
    fn plain(input: &[&str], text_width: usize) -> Vec<String> {
        raw(input, text_width)
            .iter()
            .map(|l| {
                let l = l.strip_prefix(" │ ").unwrap_or(l);
                l.trim_end_matches('│').trim_end().to_string()
            })
            .collect()
    }

    #[test]
    fn closes_every_line_at_the_border() {
        let lines = raw(&["one two three four five\n\nsix"], 10);
        assert_eq!(lines.len(), 5);
        // The caller draws the first line's left border.
        assert_eq!(lines[0], "one two    │");
        for l in &lines[1..] {
            assert!(l.starts_with(" │ ") && l.ends_with(" │"), "{l:?}");
            assert_eq!(l.chars().count(), INDENT + 10 + 2, "{l:?}");
        }
    }

    #[test]
    fn wraps_words_inside_the_card() {
        let lines = plain(
            &["The sky", " is blue because", " of Rayleigh scattering."],
            19,
        );
        assert_eq!(
            lines,
            ["The sky is blue", "because of Rayleigh", "scattering."]
        );
    }

    #[test]
    fn hangs_list_items_under_their_text() {
        let lines = plain(
            &["Steps:\n- one two three four\n12. five six seven\nend"],
            15,
        );
        assert_eq!(
            lines,
            [
                "Steps:",
                "• one two three",
                "  four",
                "12. five six",
                "    seven",
                "end"
            ]
        );
    }

    #[test]
    fn styles_markdown() {
        let lines = plain(
            &[
                "\n## Why\n\n### How\nIt is **really** `blue`.\n* one\n",
                "```rust\nlet x = 1;\n```\nDone\n\n\n",
            ],
            80,
        );
        assert_eq!(
            lines,
            [
                "Why",
                "",
                "How",
                "It is really blue.",
                "• one",
                "▎ let x = 1;",
                "Done"
            ]
        );
    }
}
