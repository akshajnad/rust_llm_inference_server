//! The cards the transcript and the prompt are drawn in: boxes with a label set into the top edge
//! and an optional status at the right end of an edge, their text capped in width so long lines
//! stay readable on wide terminals.

use crossterm::style::{Color, Stylize};
use crossterm::terminal;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

/// Columns before a card's text: a margin, the border and a space (` │ `).
pub const INDENT: usize = 3;
/// The widest a card's text gets, however wide the terminal.
const MAX_TEXT: usize = 96;
pub const BORDER: Color = Color::AnsiValue(240);

/// Width of a card's text in the current terminal.
pub fn text_width() -> usize {
    let term = terminal::size().map_or(80, |(w, _)| w as usize);
    // The margin, both borders, their padding, and an empty last column.
    term.saturating_sub(INDENT + 3).clamp(16, MAX_TEXT)
}

/// A horizontal edge for a card `width` text columns wide, between two `corners`, with `label`
/// near its left end and `status` near its right; either may be empty.
pub fn edge(
    corners: [char; 2],
    label: (&str, Color),
    status: (&str, Color),
    border: Color,
    width: usize,
) -> String {
    // The text and the space either side of it.
    let span = width + 2;
    let mut left = String::new();
    let mut used = 0;
    let label_text = truncate(label.0, span.saturating_sub(4));
    if !label_text.is_empty() {
        left = format!(
            "{}{}{}",
            "─ ".with(border),
            label_text.as_str().with(label.1).bold(),
            " ".with(border)
        );
        used += label_text.width() + 3;
    }
    let mut right = String::new();
    if !status.0.is_empty() && used + status.0.width() + 4 <= span {
        right = format!("{}{}", status.0.with(status.1), " ─".with(border));
        used += status.0.width() + 2;
        left.push_str(&"─".repeat(span - used - 1).with(border).to_string());
        left.push(' ');
    } else {
        left.push_str(&"─".repeat(span - used).with(border).to_string());
    }
    format!(
        " {}{left}{right}{}",
        corners[0].with(border),
        corners[1].with(border)
    )
}

/// The border and padding before a card's text.
pub fn left(border: Color) -> String {
    format!(" {} ", "│".with(border))
}

/// The padding and border after a card's text.
pub fn right(border: Color) -> String {
    format!(" {}", "│".with(border))
}

/// One line of a card: `text`, `text_w` columns wide (it may carry styling), padded out to the
/// right border.
pub fn row(text: &str, text_w: usize, border: Color, width: usize) -> String {
    format!(
        "{}{text}{}{}",
        left(border),
        " ".repeat(width.saturating_sub(text_w)),
        right(border)
    )
}

/// Word-wraps `text` to `width` columns, keeping its line breaks and indentation; a word wider
/// than a line is split.
pub fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    for para in text.replace('\t', "    ").split('\n') {
        let mut line: Vec<char> = Vec::new();
        let mut w = 0;
        // Spaces in the indentation are not places to break.
        let indent = para.chars().take_while(|&c| c == ' ').count();
        for ch in para.chars() {
            let cw = ch.width().unwrap_or(0);
            if w + cw > width {
                match line.iter().rposition(|&c| c == ' ') {
                    Some(sp) if sp >= indent && sp > 0 => {
                        let rest = line.split_off(sp + 1);
                        lines.push(line.iter().collect::<String>().trim_end().to_string());
                        line = rest;
                    }
                    _ => lines.push(line.drain(..).collect()),
                }
                w = line.iter().map(|c| c.width().unwrap_or(0)).sum();
                if ch == ' ' && line.is_empty() {
                    continue;
                }
            }
            line.push(ch);
            w += cw;
        }
        lines.push(line.into_iter().collect());
    }
    lines
}

/// Joins `parts` with ` · `, leaving off trailing ones that would not fit in `width` columns.
pub fn fit(parts: &[impl AsRef<str>], width: usize) -> String {
    let mut line = String::new();
    for (i, p) in parts.iter().enumerate() {
        let next = if i == 0 {
            p.as_ref().to_string()
        } else {
            format!("{line} · {}", p.as_ref())
        };
        if next.width() > width {
            break;
        }
        line = next;
    }
    line
}

pub fn truncate(s: &str, width: usize) -> String {
    let mut w = 0;
    s.chars()
        .take_while(|c| {
            w += c.width().unwrap_or(0);
            w <= width
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(s: &str) -> String {
        let mut out = String::new();
        let mut chars = s.chars();
        while let Some(c) = chars.next() {
            if c == '\x1b' {
                chars.by_ref().find(|&c| c == 'm');
            } else {
                out.push(c);
            }
        }
        out
    }

    #[test]
    fn edges_and_rows_line_up() {
        let c = Color::Reset;
        let top = plain(&edge(['┌', '┐'], ("you", c), ("temp 0.6", c), c, 20));
        assert_eq!(top, " ┌─ you ───── temp 0.6 ─┐");
        let bottom = plain(&edge(['└', '┘'], ("", c), ("", c), c, 20));
        assert_eq!(bottom, " └──────────────────────┘");
        assert_eq!(plain(&row("hi", 2, c, 20)), " │ hi                   │");
        // A status that does not fit is left off.
        let top = plain(&edge(['┌', '┐'], ("you", c), ("a long status", c), c, 12));
        assert_eq!(top, " ┌─ you ────────┐");
    }

    #[test]
    fn wraps_at_spaces() {
        assert_eq!(
            wrap("the sky is blue\n  because", 9),
            ["the sky", "is blue", "  because"]
        );
        assert_eq!(wrap("  abcdefgh", 6), ["  abcd", "efgh"]);
        assert_eq!(wrap("abcdefghij", 4), ["abcd", "efgh", "ij"]);
    }

    #[test]
    fn fits_whole_parts() {
        assert_eq!(
            fit(&["model", "temp 0.6", "new chat"], 16),
            "model · temp 0.6"
        );
        assert_eq!(fit(&["model", "temp 0.6"], 3), "");
    }
}
