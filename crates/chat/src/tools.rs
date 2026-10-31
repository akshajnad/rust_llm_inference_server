//! `rlis code`'s tools: reading, creating and editing files (the project's, the directory it
//! was started in, and with the user's approval any others) and running shell commands there.
//! The model asks for them in plain text, in the formats small coding models write reliably
//! (Qwen2.5-Coder 3B ignores most of Qwen's JSON tool-call format):
//!
//! - `READ path` lines read files (or list a directory);
//! - `RUN command` lines run commands (a lone `RUN` and then a fenced block runs a script);
//! - the file's path alone on a line and then a fenced `<<<<<<< SEARCH` / `=======` /
//!   `>>>>>>> REPLACE` block edits it (aider's format, which Qwen2.5-Coder was tuned on);
//! - the path and then a fenced block of anything else writes the whole file. Models also
//!   write `READ path` and then a fenced block, echoing their own reads: that creates the file
//!   if it doesn't exist, and reads it (ignoring the block, which is made up) if it does.
//!
//! Small models often show new code without its path line. A block still creates a new file
//! when the line before it names one (`### HTML (index.html)`, ``Create `snake.py`:``) or its
//! first line is a comment naming one (`# snake.py`); other code blocks are counted as unsaved,
//! so the agent can ask the model to save them.
//!
//! `Reply` reads a reply as it streams in: it hides these requests from the transcript (their
//! results get cards of their own) and says when a request is complete and the model has gone
//! on to imagine its result, so generation can stop there. Reads run at once; writes and edits
//! become a `Change` for the user to approve before anything touches the disk.

use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail, ensure};

/// Characters of a file returned by one read; the model asks for later lines with `READ path:N`.
const MAX_READ: usize = 12000;
/// The project's instructions for agents, the first found of these, go in the system prompt.
const NOTES_FILES: &[&str] = &["RLIS.md", "AGENTS.md", "CLAUDE.md"];
const MAX_NOTES: usize = 4000;
/// Lines of code in a block shown without a file before it counts as unsaved code.
const MIN_UNSAVED_LINES: usize = 5;
const MAX_LIST: usize = 300;
/// Files listed in the system prompt.
const MAX_PROMPT_FILES: usize = 200;
/// Directories skipped when listing a directory that is not in a git repository.
const SKIP_DIRS: &[&str] = &["target", "node_modules", "__pycache__", "venv"];

/// File names without an extension (besides `*file`: `Makefile`, `Dockerfile`, ...) that a
/// path line may name.
const EXTENSIONLESS: &[&str] = &[
    "LICENSE",
    "README",
    "CHANGELOG",
    "NOTICE",
    "AUTHORS",
    "CODEOWNERS",
    "VERSION",
    "BUILD",
    "WORKSPACE",
];

const SEARCH: &str = "<<<<<<< SEARCH";
const DIVIDER: &str = "=======";
const REPLACE: &str = ">>>>>>> REPLACE";

/// What the model asked for.
#[derive(Debug, PartialEq)]
pub enum Request {
    /// A file from line `from` (1-based), or a directory's listing.
    Read { path: String, from: usize },
    /// Each search text replaced with its replacement, in order. Empty search texts create the
    /// file.
    Edit {
        path: String,
        pairs: Vec<(String, String)>,
    },
    /// The whole file.
    Write { path: String, content: String },
    /// `READ path` and a fenced block: a new file with this content, or a read of an existing
    /// one.
    ReadOrCreate { path: String, content: String },
    /// A shell command (or script) to run in the project's directory.
    Run { command: String },
    /// A request that couldn't be read, with what was wrong, for the model to try again.
    Invalid(String),
}

impl Request {
    /// The request in a few words, for its card's title.
    pub fn title(&self) -> String {
        match self {
            Request::Read { path, from: 1 } => format!("read {path}"),
            Request::Read { path, from } => format!("read {path}:{from}"),
            Request::Edit { path, .. } => format!("edit {path}"),
            Request::Write { path, .. } | Request::ReadOrCreate { path, .. } => {
                format!("write {path}")
            }
            Request::Run { command } => {
                let first = command.lines().next().unwrap_or_default();
                match command.lines().nth(1) {
                    Some(_) => format!("run {first} …"),
                    None => format!("run {first}"),
                }
            }
            Request::Invalid(_) => "request".into(),
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Read,
    Change,
    Run,
}

/// A fenced block (or bare SEARCH/REPLACE block) after a path line, while it arrives.
struct Block {
    path: String,
    /// The opening backticks; `None` for a SEARCH/REPLACE block written without a fence.
    fence: Option<String>,
    /// Fences opened inside it, as in a Markdown file with code blocks.
    depth: usize,
    lines: Vec<String>,
    /// Opened by a `READ` line rather than a path alone.
    after_read: bool,
    /// Opened by a lone `RUN` line: a script.
    run: bool,
}

/// A fenced block shown as text, with no path line before it.
struct Anonymous {
    fence: String,
    lang: String,
    depth: usize,
    lines: Vec<String>,
}

/// One reply, read as it streams in.
#[derive(Default)]
pub struct Reply {
    /// The reply as kept: everything before the cut.
    text: String,
    /// Complete lines waiting for the next ones to decide whether they start a request.
    held: Vec<String>,
    /// The line being received, and how much of it has been shown.
    line: String,
    shown: usize,
    block: Option<Block>,
    anonymous: Option<Anonymous>,
    /// A new file named by the last line shown, for a fence right after it.
    named: Option<String>,
    /// Code blocks shown without a file to save them in.
    unsaved: usize,
    /// The project's directory, to tell new files from existing ones; names are only inferred
    /// for new files.
    root: Option<PathBuf>,
    requests: Vec<Request>,
    last: Option<Kind>,
    cut: bool,
}

impl Reply {
    /// For a reply from the agent working in `root`.
    pub fn new(root: &Path) -> Self {
        Self {
            root: Some(root.to_path_buf()),
            ..Self::default()
        }
    }

    /// The requests in a whole reply.
    #[cfg(test)]
    pub fn parse(text: &str) -> Vec<Request> {
        Reply::read(text, None).requests
    }

    /// A whole reply from the agent working in `root`.
    pub fn read(text: &str, root: Option<&Path>) -> Reply {
        let mut r = Reply {
            root: root.map(Path::to_path_buf),
            ..Reply::default()
        };
        r.push(text);
        r.finish(false);
        r
    }

    pub fn into_requests(self) -> Vec<Request> {
        self.requests
    }

    /// Code blocks the model showed without saving them to a file.
    pub fn unsaved_code(&self) -> usize {
        self.unsaved
    }

    /// The part of the reply that can be shown now, given the next piece of it.
    pub fn push(&mut self, piece: &str) -> String {
        let mut out = String::new();
        for c in piece.chars() {
            if self.cut {
                break;
            }
            if c == '\n' {
                let line = std::mem::take(&mut self.line);
                let shown = std::mem::take(&mut self.shown);
                self.end_line(line, shown, &mut out);
            } else {
                self.line.push(c);
            }
        }
        // Stream the line being received unless it may start a request (inside a code block
        // shown as text, unless it may close it).
        let streams = match &self.anonymous {
            Some(_) => !self.line.trim_start().starts_with('`'),
            None => {
                self.held.is_empty()
                    && self.last.is_none()
                    && (self.shown > 0 || !may_be_special(&self.line))
            }
        };
        if !self.cut && self.block.is_none() && streams {
            out += &self.line[self.shown..];
            self.shown = self.line.len();
        }
        out
    }

    /// The rest of what can be shown, once the reply has ended; `truncated` when it hit the
    /// length limit, which leaves an open block incomplete.
    pub fn finish(&mut self, truncated: bool) -> String {
        let mut out = String::new();
        if !self.cut && !self.line.is_empty() {
            let line = std::mem::take(&mut self.line);
            let shown = std::mem::take(&mut self.shown);
            self.end_line(line, shown, &mut out);
        }
        if let Some(a) = self.anonymous.take() {
            self.end_anonymous(a);
        }
        // A block the reply ended in without closing is whole unless the length limit cut it.
        if let Some(b) = self.block.take() {
            match truncated {
                false => self.end_block(b),
                true => self.requests.push(Request::Invalid(format!(
                    "the block for {} was cut off at the length limit; nothing was changed",
                    b.path
                ))),
            }
        }
        // Lines held to see what follows them: nothing does.
        if !self.cut && self.block.is_none() {
            self.decide(&mut out, true);
        }
        out
    }

    /// The reply up to the cut (all of it if there was none).
    pub fn text(&self) -> &str {
        &self.text
    }

    /// A request is complete and the model went on past it: stop generating.
    pub fn cut(&self) -> bool {
        self.cut
    }

    /// The file being written, while its block arrives.
    pub fn writing(&self) -> Option<&str> {
        self.block.as_ref().map(|b| b.path.as_str())
    }

    fn end_line(&mut self, line: String, shown: usize, out: &mut String) {
        if self.block.is_some() {
            self.text += &line;
            self.text.push('\n');
            self.feed(line);
            return;
        }
        if self.anonymous.is_some() {
            *out += &line[shown..];
            out.push('\n');
            self.text += &line;
            self.text.push('\n');
            self.feed_anonymous(line);
            return;
        }
        if shown > 0 {
            self.named = self.new_file_in(&line);
            *out += &line[shown..];
            out.push('\n');
            self.text += &line;
            self.text.push('\n');
            return;
        }
        // Small models act out the prompt's examples: their own lines labeled `You:`.
        let line = match line.strip_prefix("You:") {
            Some(rest) => rest.trim_start().to_string(),
            None => line,
        };
        self.held.push(line);
        self.decide(out, false);
    }

    /// Shows, hides or keeps holding the held lines; at the `end` of the reply, nothing is
    /// held any longer.
    fn decide(&mut self, out: &mut String, end: bool) {
        while let Some(first) = self.held.first() {
            // A user's turn the model made up: before any request, an echo of the request;
            // after one, the model imagining its result.
            if first.starts_with("User:") {
                if self.last.is_some() {
                    return self.stop();
                }
                self.keep(1);
                continue;
            }
            if let Some(command) = run_line(first) {
                // Commands may follow anything (edit, then test); anything after them waits
                // for their output.
                self.requests.push(Request::Run { command });
                self.last = Some(Kind::Run);
                self.keep(1);
                continue;
            }
            if first.trim() == "RUN" {
                match self.held.get(1).and_then(|l| fence_open(l)) {
                    None if self.held.len() < 2 && !end => return,
                    None => {}
                    Some(fence) => {
                        self.open_block(String::new(), Some(fence), false, 2);
                        if let Some(b) = &mut self.block {
                            b.run = true;
                        }
                        return;
                    }
                }
            }
            if let Some(read) = read_line(first) {
                let second = self.held.get(1);
                if second.is_none() && !end {
                    return;
                }
                if let (Request::Read { path, .. }, Some(fence)) =
                    (&read, second.and_then(|l| fence_open(l)))
                {
                    if matches!(self.last, Some(Kind::Read | Kind::Run)) {
                        return self.stop();
                    }
                    let path = path.clone();
                    return self.open_block(path, Some(fence), true, 2);
                }
                if matches!(self.last, Some(Kind::Change | Kind::Run)) {
                    return self.stop();
                }
                self.requests.push(read);
                self.last = Some(Kind::Read);
                self.keep(1);
                continue;
            }
            if first.trim().is_empty() {
                if self.last.is_none() {
                    out.push('\n');
                }
                self.keep(1);
                continue;
            }
            if let Some(path) = path_line(first) {
                match self.held.get(1) {
                    None if !end => return,
                    Some(second) if fence_open(second).is_some() || second.trim() == SEARCH => {
                        if matches!(self.last, Some(Kind::Read | Kind::Run)) {
                            return self.stop();
                        }
                        let fence = fence_open(second);
                        let opening = if fence.is_some() { 2 } else { 1 };
                        return self.open_block(path, fence, false, opening);
                    }
                    _ => {}
                }
            }
            // Ordinary text; after a request, the model imagining what came of it.
            if self.last.is_some() {
                return self.stop();
            }
            if let Some(fence) = fence_open(first) {
                let lang = fence_lang(first);
                if let Some(path) = self.named.take().filter(|_| !is_shell(&lang)) {
                    return self.open_block(path, Some(fence), false, 1);
                }
                *out += first;
                out.push('\n');
                self.keep(1);
                self.anonymous = Some(Anonymous {
                    fence,
                    lang,
                    depth: 0,
                    lines: Vec::new(),
                });
                // What else is held is the block's, up to its end.
                while self.anonymous.is_some() && !self.held.is_empty() {
                    let line = self.held.remove(0);
                    *out += &line;
                    out.push('\n');
                    self.text += &line;
                    self.text.push('\n');
                    self.feed_anonymous(line);
                }
                continue;
            }
            if !first.trim().is_empty() {
                self.named = self.new_file_in(first);
            }
            *out += first;
            out.push('\n');
            self.keep(1);
        }
    }

    /// A new file that `line` names, if any.
    fn new_file_in(&self, line: &str) -> Option<String> {
        let name = named_file(line)?;
        self.is_new(&name).then_some(name)
    }

    /// Whether `path` doesn't exist yet (always, with no project to look in).
    fn is_new(&self, path: &str) -> bool {
        let Some(root) = &self.root else {
            return true;
        };
        let p = Path::new(path);
        let abs = match p.is_absolute() {
            true => p.to_path_buf(),
            false => root.join(p),
        };
        !abs.exists()
    }

    /// Adds a line to the code block shown as text, closing it at its fence.
    fn feed_anonymous(&mut self, line: String) {
        let a = self.anonymous.as_mut().expect("a block is open");
        let t = line.trim();
        if t == a.fence && a.depth == 0 {
            let a = self.anonymous.take().expect("a block is open");
            return self.end_anonymous(a);
        }
        if t == a.fence {
            a.depth -= 1;
        } else if t.starts_with(a.fence.as_str()) && t.len() > a.fence.len() {
            a.depth += 1;
        }
        a.lines.push(line);
    }

    /// A code block shown as text creates the file its first line names (`# snake.py`), if
    /// that is a new file; otherwise code in it is counted as unsaved.
    fn end_anonymous(&mut self, a: Anonymous) {
        if is_shell(&a.lang) {
            return;
        }
        let name = a.lines.first().and_then(|l| comment_name(l));
        if let Some(path) = name.filter(|n| self.is_new(n)) {
            let mut content = a.lines.join("\n");
            content.push('\n');
            self.requests.push(Request::Write { path, content });
            self.last = Some(Kind::Change);
            return;
        }
        let code = a.lines.iter().filter(|l| !l.trim().is_empty()).count();
        if code >= MIN_UNSAVED_LINES
            && !matches!(
                a.lang.as_str(),
                "text" | "txt" | "plaintext" | "output" | "diff"
            )
        {
            self.unsaved += 1;
        }
    }

    /// Opens a block at the held lines: the first `opening` of them (the path or `READ` line
    /// and the fence, or just the fence after a line naming the file) open it; a bare SEARCH
    /// line is its first.
    fn open_block(
        &mut self,
        path: String,
        fence: Option<String>,
        after_read: bool,
        opening: usize,
    ) {
        self.keep(opening);
        self.block = Some(Block {
            path,
            fence,
            depth: 0,
            lines: Vec::new(),
            after_read,
            run: false,
        });
        for line in std::mem::take(&mut self.held) {
            self.text += &line;
            self.text.push('\n');
            self.feed(line);
        }
    }

    /// Moves the first `n` held lines into the kept text.
    fn keep(&mut self, n: usize) {
        for line in self.held.drain(..n) {
            self.text += &line;
            self.text.push('\n');
        }
    }

    fn stop(&mut self) {
        self.cut = true;
        self.held.clear();
        self.line.clear();
    }

    /// Adds a line to the open block, closing it at its closing fence (or, for a bare
    /// SEARCH/REPLACE block, at the REPLACE marker).
    fn feed(&mut self, line: String) {
        let b = self.block.as_mut().expect("a block is open");
        let t = line.trim();
        let closed = match &b.fence {
            None => t == REPLACE,
            Some(f) if t == f && b.depth == 0 => true,
            Some(f) if t == f => {
                b.depth -= 1;
                false
            }
            Some(f) => {
                if t.starts_with(f.as_str()) && t.len() > f.len() {
                    b.depth += 1;
                }
                false
            }
        };
        if closed && b.fence.is_some() {
            let b = self.block.take().expect("a block is open");
            return self.end_block(b);
        }
        b.lines.push(line);
        if closed {
            let b = self.block.take().expect("a block is open");
            self.end_block(b);
        }
    }

    fn end_block(&mut self, b: Block) {
        if b.run {
            let command = b.lines.join("\n").trim().to_string();
            self.requests.push(match command.is_empty() {
                true => Request::Invalid("the RUN block is empty".into()),
                false => Request::Run { command },
            });
            self.last = Some(Kind::Run);
            return;
        }
        let is_edit = b.lines.first().is_some_and(|l| l.trim() == SEARCH);
        let request = match is_edit {
            true => match search_replace(&b.lines) {
                Ok(pairs) => Request::Edit {
                    path: b.path,
                    pairs,
                },
                Err(e) => Request::Invalid(format!("the edit to {}: {e}", b.path)),
            },
            false => {
                let mut content = b.lines.join("\n");
                if !content.is_empty() {
                    content.push('\n');
                }
                match b.after_read {
                    true => Request::ReadOrCreate {
                        path: b.path,
                        content,
                    },
                    false => Request::Write {
                        path: b.path,
                        content,
                    },
                }
            }
        };
        self.requests.push(request);
        self.last = Some(Kind::Change);
    }
}

/// The SEARCH/REPLACE pairs in a block's lines.
fn search_replace(lines: &[String]) -> Result<Vec<(String, String)>, String> {
    let mut pairs = Vec::new();
    let mut i = 0;
    let join = |ls: &[String]| ls.iter().map(|l| format!("{l}\n")).collect::<String>();
    while i < lines.len() {
        if lines[i].trim().is_empty() {
            i += 1;
            continue;
        }
        if lines[i].trim() != SEARCH {
            return Err(format!("expected {SEARCH}, found {:?}", lines[i]));
        }
        let start = i + 1;
        let div = (start..lines.len())
            .find(|&j| lines[j].trim() == DIVIDER)
            .ok_or(format!("no {DIVIDER} line"))?;
        let end = (div + 1..lines.len())
            .find(|&j| lines[j].trim() == REPLACE)
            .ok_or(format!("no {REPLACE} line"))?;
        pairs.push((join(&lines[start..div]), join(&lines[div + 1..end])));
        i = end + 1;
    }
    match pairs.is_empty() {
        true => Err("no SEARCH/REPLACE pairs".into()),
        false => Ok(pairs),
    }
}

/// Whether a line still arriving could turn out to start a request, so it shouldn't be
/// shown yet: a word with no space after it yet (a path), `READ `, `RUN `, a fence or a marker.
fn may_be_special(line: &str) -> bool {
    let t = line.trim_start();
    !t.contains(char::is_whitespace)
        || t.starts_with("READ ")
        || t.starts_with("RUN ")
        || t.starts_with("User:")
        || t.starts_with("You:")
        || t.starts_with("```")
        || t.starts_with("<<<<<<<")
}

/// `RUN command`, maybe with the command in backticks.
fn run_line(line: &str) -> Option<String> {
    let rest = line.trim().strip_prefix("RUN ")?.trim();
    let rest = match rest.strip_prefix('`').and_then(|r| r.strip_suffix('`')) {
        Some(inner) if !inner.contains('`') => inner.trim(),
        _ => rest,
    };
    (!rest.is_empty()).then(|| rest.to_string())
}

/// The language named after a fence's backticks, lowercased.
fn fence_lang(line: &str) -> String {
    line.trim()
        .trim_start_matches('`')
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_lowercase()
}

/// A fenced block of shell commands or terminal output, not a file's content.
fn is_shell(lang: &str) -> bool {
    matches!(
        lang,
        "bash" | "sh" | "shell" | "zsh" | "console" | "terminal" | "shell-session" | "cmd"
    )
}

/// A file a line of prose names: in backticks, parentheses or bold, or as a word that looks
/// like a file name (`### HTML (index.html)`, ``Create `snake.py`:``, `Save it as main.go`).
fn named_file(line: &str) -> Option<String> {
    if let Some(p) = path_line(line) {
        return is_file_name(&p, true).then_some(p);
    }
    let t = line.trim().trim_start_matches('#').trim();
    let mut quoted = Vec::new();
    for (open, close) in [('`', '`'), ('(', ')'), ('*', '*'), ('"', '"')] {
        let mut rest = t;
        while let Some(i) = rest.find(open) {
            let after = rest[i + 1..].trim_start_matches(open);
            let Some(j) = after.find(close) else { break };
            quoted.push(after[..j].trim().to_string());
            rest = after[j + 1..].trim_start_matches(close);
        }
    }
    if let Some(q) = quoted.into_iter().rev().find(|q| is_file_name(q, true)) {
        return Some(q);
    }
    t.split_whitespace()
        .rev()
        .map(|w| w.trim_matches(|c: char| ":,;!?.\"'`*()[]".contains(c)))
        .find(|w| is_file_name(w, false))
        .map(String::from)
}

/// A file named by a comment that is a code block's first line: `# snake.py`, `// main.go`,
/// `<!-- index.html -->`, `/* File: style.css */`.
fn comment_name(line: &str) -> Option<String> {
    let mut t = line.trim();
    for p in ["<!--", "/*", "//", "--", "#", ";", "%"] {
        if let Some(rest) = t.strip_prefix(p) {
            t = rest;
            break;
        }
    }
    if t.len() == line.trim().len() {
        return None;
    }
    let t = t.trim_end_matches("-->").trim_end_matches("*/").trim();
    let lower = t.to_lowercase();
    let t = ["file:", "filename:", "file name:", "path:"]
        .iter()
        .find_map(|p| lower.starts_with(p).then(|| t[p.len()..].trim()))
        .unwrap_or(t);
    is_file_name(t, true).then(|| t.to_string())
}

/// Whether `s` looks like a file's name or path: one word with an extension. Bare words in
/// prose (not `quoted`) must also not be capitalized names like `Node.js`.
fn is_file_name(s: &str, quoted: bool) -> bool {
    let Some((stem, ext)) = s.rsplit_once('.') else {
        return false;
    };
    let name = stem.rsplit('/').next().unwrap_or(stem);
    !s.is_empty()
        && s.len() < 200
        && !s.contains(char::is_whitespace)
        && !s.contains("://")
        && !s.starts_with(['#', '<', '>', '=', '-', '+', '@'])
        && !name.is_empty()
        && (1..=10).contains(&ext.len())
        && ext.chars().all(|c| c.is_ascii_alphanumeric())
        && ext.chars().any(|c| c.is_ascii_alphabetic())
        && !matches!(s, "e.g" | "i.e" | "etc")
        && (quoted || s.contains('/') || !name.starts_with(|c: char| c.is_uppercase()))
}

/// `READ path` or `READ path:N`.
fn read_line(line: &str) -> Option<Request> {
    let rest = line.trim().strip_prefix("READ ")?.trim();
    let rest = rest.trim_matches('`');
    if rest.is_empty() || rest.contains(char::is_whitespace) {
        return None;
    }
    let (path, from) = match rest.rsplit_once(':') {
        Some((p, n)) if n.parse::<usize>().is_ok() => (p, n.parse().unwrap_or(1)),
        _ => (rest, 1),
    };
    Some(Request::Read {
        path: path.to_string(),
        from: from.max(1),
    })
}

/// A line that is just a file path, as written before a block: maybe in backticks or bold,
/// maybe with a colon after it. A bare word counts only as a file name that usually has no
/// extension (`Makefile`, `LICENSE`), so a line like `Example:` before a code block isn't one.
fn path_line(line: &str) -> Option<String> {
    let t = line
        .trim()
        .trim_end_matches(':')
        .trim_matches(|c| c == '`' || c == '*');
    let t = t.trim_end_matches(':');
    let bare_name = t.ends_with("file") || EXTENSIONLESS.contains(&t);
    let plausible = !t.is_empty()
        && t.len() < 200
        && !t.contains(char::is_whitespace)
        && (t.contains('.') || t.contains('/') || bare_name)
        && !t.starts_with(['#', '<', '>', '=', '-', '+'])
        && !t.ends_with('.');
    plausible.then(|| t.to_string())
}

/// The backticks of a line that opens a fence.
fn fence_open(line: &str) -> Option<String> {
    let t = line.trim();
    let ticks = t.chars().take_while(|&c| c == '`').count();
    (ticks >= 3).then(|| "`".repeat(ticks))
}

/// What the agent did, or wants to do.
pub enum Action {
    /// A read: the result for the model and a summary for the user.
    Done { output: String, summary: String },
    /// A write or edit, not made yet.
    Change(Change),
    /// A write or edit that leaves the file as it is (models echo files back whole).
    Unchanged { output: String },
    /// A command to run, and why it needs the user's approval, if it does.
    Run {
        command: String,
        risk: Option<&'static str>,
    },
}

pub struct Change {
    /// As the model named it, relative to the project root.
    pub path: String,
    abs: PathBuf,
    /// The file's text now; `None` when the change creates it.
    pub old: Option<String>,
    pub new: String,
    /// Outside the project: always asks, even when the user approved all changes.
    pub outside: bool,
}

/// The directory the tools work in. Paths outside it work too, but changes there always need
/// the user's approval.
pub struct Workspace {
    root: PathBuf,
    /// Files the model has read or written in this conversation. It may change only these (and
    /// new files): small models otherwise edit files from memory, or replace them with a guess.
    seen: HashSet<PathBuf>,
}

impl Workspace {
    pub fn new(root: PathBuf) -> Self {
        Self {
            root,
            seen: HashSet::new(),
        }
    }

    /// For a new conversation, which has read nothing yet.
    pub fn forget(&mut self) {
        self.seen.clear();
    }

    /// Writes an approved change; returns the result for the model.
    pub fn apply(&mut self, change: &Change) -> Result<String> {
        if let Some(parent) = change.abs.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&change.abs, &change.new).with_context(|| format!("writing {}", change.path))?;
        self.seen.insert(change.abs.clone());
        Ok(match change.old {
            None => format!("Created {}.", change.path),
            Some(_) => format!("Applied the change to {}.", change.path),
        })
    }

    /// Refuses a change to an existing file the model hasn't read.
    fn ensure_seen(&self, abs: &Path, path: &str) -> Result<()> {
        ensure!(
            !abs.exists() || self.seen.contains(abs),
            "you have not read {path}; READ it first, then change it"
        );
        Ok(())
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The instructions, with worked examples (which small models need to follow them), the
    /// project's files and its notes for agents; `extra` is the user's own system prompt.
    pub fn system_prompt(&self, extra: Option<&str>) -> String {
        let files = self.files(&self.root);
        let mut listing = files
            .iter()
            .take(MAX_PROMPT_FILES)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n");
        if files.len() > MAX_PROMPT_FILES {
            listing += &format!(
                "\n[... and {} more; READ a directory to list its files]",
                files.len() - MAX_PROMPT_FILES
            );
        }
        if files.is_empty() {
            listing = "(none yet)".into();
        }
        let mut s = format!(
            "You are rlis, a coding agent. You work directly in the user's project at {root}: \
             you can read, create and edit its files and run commands in its directory. When \
             the user asks for code, put it in files; never only show code in your reply.

How to act (each request goes on lines of its own; results come back in the next message):

READ path/to/file      read a file (READ . lists the project; READ file:200 reads from line 200)
RUN command            run a shell command in the project's directory and see its output

When the user asks you to run something (tests, a build, a script), RUN it yourself instead of \
telling them the command. To delete, move or copy files, RUN rm, mv or cp; the user approves \
those first. Never say you did something you have not done with READ, RUN or a file block.

To create or replace a whole file, write its path alone on a line, then its content in a \
fenced block:

hello.py
```python
print(\"hello\")
```

To change part of an existing file (READ it first), write its path alone on a line, then a \
fenced SEARCH/REPLACE block: lines copied exactly from the file, indentation included, and the \
lines to put in their place:

hello.py
```python
{SEARCH}
print(\"hello\")
{DIVIDER}
print(\"hello, world\")
{REPLACE}
```

Paths are relative to the project; a path outside it (/abs/path or ~/path) needs the user's \
approval to change. Don't RUN programs that wait for input or open windows (games, servers): \
tell the user how to start them instead. Change only what the user asks for, and nothing when \
they only ask a question. When the task is done, reply with a one-sentence summary.

Example:
User: write a script that prints the squares of 1 to 5
You: squares.py
```python
for i in range(1, 6):
    print(i * i)
```
RUN python3 squares.py
User: Created squares.py.
Output of `python3 squares.py` (exit 0):
```
1
4
9
16
25
```
You: I created squares.py, which prints the squares of 1 to 5.

Example:
User: greet() should say hello, not hi.
You: READ app/greet.py
User: Contents of app/greet.py:
```
def greet(name):
    return \"hi \" + name
```
You: app/greet.py
```python
{SEARCH}
    return \"hi \" + name
{DIVIDER}
    return \"hello \" + name
{REPLACE}
```
User: Applied the change to app/greet.py.
You: greet() now says hello instead of hi.

Example:
User: What does app/greet.py do?
You: READ app/greet.py
User: Contents of app/greet.py:
```
def greet(name):
    return \"hello \" + name
```
You: It defines greet(name), which returns \"hello \" followed by the name.

Files in the project:
{listing}",
            root = self.root.display()
        );
        if let Some(notes) = self.notes() {
            s += &format!("\n\nProject notes:\n{notes}");
        }
        if let Some(extra) = extra {
            s += &format!("\n\n{extra}");
        }
        s
    }

    /// The first of the project's instruction files, shortened to `MAX_NOTES`.
    fn notes(&self) -> Option<String> {
        let text = NOTES_FILES
            .iter()
            .find_map(|f| fs::read_to_string(self.root.join(f)).ok())?;
        let text = text.trim();
        if text.is_empty() {
            return None;
        }
        Some(match text.len() > MAX_NOTES {
            true => {
                let end = (0..=MAX_NOTES)
                    .rev()
                    .find(|&i| text.is_char_boundary(i))
                    .unwrap_or(0);
                format!("{}\n[...]", &text[..end])
            }
            false => text.to_string(),
        })
    }

    /// Runs a read, or prepares a change or command; errors are the model's to fix.
    pub fn run(&mut self, request: &Request) -> Result<Action> {
        match request {
            Request::Read { path, from } => {
                let abs = self.resolve(path)?;
                ensure!(abs.exists(), "{path} does not exist");
                let outside = self.outside(&abs);
                if abs.is_dir() {
                    let files = match outside {
                        true => list_dir(&abs),
                        false => self.files(&abs),
                    };
                    let mut output = files
                        .iter()
                        .take(MAX_LIST)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join("\n");
                    if files.len() > MAX_LIST {
                        output += &format!("\n[... and {} more]", files.len() - MAX_LIST);
                    }
                    let mut summary = format!("{} files", files.len());
                    if outside {
                        summary += " · outside the project";
                    }
                    let output = format!("Files in {path}:\n```\n{output}\n```");
                    return Ok(Action::Done { output, summary });
                }
                let text = self.read(path)?;
                let (body, mut summary) = excerpt_file(&text, path, *from)?;
                if outside {
                    summary += " · outside the project";
                }
                self.seen.insert(abs);
                let output = format!("Contents of {path}:\n```\n{body}```");
                Ok(Action::Done { output, summary })
            }
            Request::ReadOrCreate { path, content } => {
                let abs = self.resolve(path)?;
                if !abs.exists() {
                    let path = path.clone();
                    let content = content.clone();
                    return self.run(&Request::Write { path, content });
                }
                // The model echoing a file it just wrote, or making up one it wants to read.
                if self.seen.contains(&abs) && self.read(path)? == recipe_tabs(path, content) {
                    return Ok(unchanged(path));
                }
                let path = path.clone();
                self.run(&Request::Read { path, from: 1 })
            }
            Request::Write { path, content } => {
                let abs = self.resolve(path)?;
                ensure!(!abs.is_dir(), "{path} is a directory");
                self.ensure_seen(&abs, path)?;
                let old = abs.exists().then(|| self.read(path)).transpose()?;
                let new = recipe_tabs(path, content);
                if old.as_ref() == Some(&new) {
                    return Ok(unchanged(path));
                }
                let outside = self.outside(&abs);
                Ok(Action::Change(Change {
                    path: path.clone(),
                    abs,
                    old,
                    new,
                    outside,
                }))
            }
            Request::Edit { path, pairs } => {
                let abs = self.resolve(path)?;
                if !abs.exists() {
                    ensure!(
                        pairs.iter().all(|(s, _)| s.trim().is_empty()),
                        "{path} does not exist; to create it, write its path and then its \
                         whole content in a fenced block"
                    );
                    let new: String = pairs.iter().map(|(_, r)| r.as_str()).collect();
                    let new = recipe_tabs(path, &new);
                    let outside = self.outside(&abs);
                    return Ok(Action::Change(Change {
                        path: path.clone(),
                        abs,
                        old: None,
                        new,
                        outside,
                    }));
                }
                self.ensure_seen(&abs, path)?;
                let old = self.read(path)?;
                let mut text = old.clone();
                for (search, replace) in pairs {
                    ensure!(
                        !search.trim().is_empty(),
                        "the SEARCH part for {path} is empty, but the file exists; copy the \
                         lines to replace into it"
                    );
                    text = match text.matches(search.as_str()).count() {
                        1 => text.replacen(search.as_str(), replace, 1),
                        0 => replace_lines(&text, search, replace).with_context(|| {
                            format!(
                                "the SEARCH part does not match {path}; READ it again and copy \
                                 the lines to replace exactly"
                            )
                        })?,
                        n => bail!(
                            "the SEARCH part appears {n} times in {path}; include more of the \
                             lines around it so it matches once"
                        ),
                    };
                }
                let text = recipe_tabs(path, &text);
                if text == old {
                    return Ok(unchanged(path));
                }
                let outside = self.outside(&abs);
                Ok(Action::Change(Change {
                    path: path.clone(),
                    abs,
                    old: Some(old),
                    new: text,
                    outside,
                }))
            }
            Request::Run { command } => Ok(Action::Run {
                command: command.clone(),
                risk: needs_approval(command),
            }),
            Request::Invalid(why) => bail!("{why}"),
        }
    }

    fn read(&self, path: &str) -> Result<String> {
        let abs = self.resolve(path)?;
        ensure!(abs.exists(), "{path} does not exist");
        let bytes = fs::read(&abs).with_context(|| format!("reading {path}"))?;
        String::from_utf8(bytes)
            .ok()
            .with_context(|| format!("{path} is not a text file"))
    }

    /// `path` as an absolute path, without touching the disk: relative to the project, or
    /// absolute, or under the home directory (`~/...`).
    fn resolve(&self, path: &str) -> Result<PathBuf> {
        let t = path.trim();
        let p = match t.strip_prefix('~') {
            Some(rest) if rest.is_empty() || rest.starts_with('/') => {
                let home = env::var_os("HOME").context("HOME is not set, so ~ means nothing")?;
                PathBuf::from(home).join(rest.trim_start_matches('/'))
            }
            _ => PathBuf::from(t),
        };
        let mut out = match p.is_absolute() {
            true => PathBuf::from("/"),
            false => self.root.clone(),
        };
        for c in p.components() {
            match c {
                Component::Normal(n) => out.push(n),
                Component::RootDir | Component::CurDir => {}
                Component::ParentDir => {
                    out.pop();
                }
                Component::Prefix(_) => bail!("{path} is not a path here"),
            }
        }
        Ok(out)
    }

    /// Whether `abs` is outside the project, symlinks resolved (`/tmp` is `/private/tmp`).
    pub fn outside(&self, abs: &Path) -> bool {
        !real(abs).starts_with(real(&self.root))
    }

    /// Files under `dir`, relative to the root: what git doesn't ignore in a repository, else
    /// everything but hidden files and build directories.
    fn files(&self, dir: &Path) -> Vec<String> {
        let rel = dir.strip_prefix(&self.root).unwrap_or(Path::new(""));
        let spec = match rel.as_os_str().is_empty() {
            true => ".".to_string(),
            false => rel.display().to_string(),
        };
        let git = Command::new("git")
            .args([
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "--",
            ])
            .arg(&spec)
            .current_dir(&self.root)
            .stdin(Stdio::null())
            .stderr(Stdio::null())
            .output();
        if let Ok(out) = git
            && out.status.success()
        {
            return String::from_utf8_lossy(&out.stdout)
                .lines()
                // Tracked files deleted from the working tree.
                .filter(|f| self.root.join(f).exists())
                .map(String::from)
                .collect();
        }
        let mut files = Vec::new();
        walk(&self.root, dir, &mut files);
        files.sort();
        files
    }
}

/// A makefile's indented lines start with a tab, as make requires; models indent recipes with
/// spaces. Other files are left as they are.
fn recipe_tabs(path: &str, text: &str) -> String {
    let name = Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    if !matches!(name.as_str(), "Makefile" | "makefile" | "GNUmakefile") && !name.ends_with(".mk") {
        return text.to_string();
    }
    text.split_inclusive('\n')
        .map(|l| match l.strip_prefix(' ') {
            Some(_) if !l.trim().is_empty() => format!("\t{}", l.trim_start_matches(' ')),
            _ => l.to_string(),
        })
        .collect()
}

/// The note restating the user's request after each round of results: small models lose it
/// behind the file contents they read, and make up work instead (edits nobody asked for).
pub fn restate(request: &str) -> String {
    format!(
        "(The user's request: {} Do what it says next, or if it is done, reply with a short \
         summary. Write each file you create or rewrite as its path alone on a line, then its \
         content in a fenced block; RUN a command to check your work.)",
        request.trim()
    )
}

/// Reminders for a reply that asked for nothing: small models show new code instead of saving
/// it, or refuse ordinary tasks, now and then, and one reminder of what they can do usually
/// sets them right. Each is sent once a turn.
#[derive(Default)]
pub struct Nudges {
    code: bool,
    refusal: bool,
    claim: bool,
    /// Something was written this turn, so code shown afterwards needn't be saved.
    saved: bool,
    /// Anything was asked for this turn, so a summary of it is no made-up claim.
    acted: bool,
}

impl Nudges {
    /// Notes the requests of a reply.
    pub fn saw(&mut self, requests: &[Request]) {
        self.acted |= !requests.is_empty();
        self.saved |= requests.iter().any(|r| {
            matches!(
                r,
                Request::Write { .. } | Request::Edit { .. } | Request::ReadOrCreate { .. }
            )
        });
    }

    /// For a reply with no requests, with `unsaved` code blocks, to the user's `request`: a
    /// note for the user and the message for the model, if it needs a reminder.
    pub fn next(
        &mut self,
        reply: &str,
        unsaved: usize,
        request: &str,
    ) -> Option<(&'static str, String)> {
        let (note, message) = if is_refusal(reply) && !self.refusal {
            self.refusal = true;
            (
                "The model declined; reminding it that it can do this",
                "You can do this: you are a coding agent with access to the files in this \
                 project and a shell. Do what the user asked: write files as each file's path \
                 alone on a line, then its whole content in a fenced block, and RUN commands \
                 with a RUN line.",
            )
        } else if !self.acted && !self.claim && !is_question(request) && claims_action(reply) {
            self.claim = true;
            (
                "The model said it did something it didn't; asking it to do it",
                "You have not done that: nothing has changed yet. Do it now with RUN lines or \
                 file blocks; the user approves changes and risky commands.",
            )
        } else if unsaved > 0 && !self.saved && !self.code && !is_question(request) {
            self.code = true;
            (
                "The model showed code without saving it; asking it to write the files",
                "You showed code but didn't save it, so no file was changed. Write each file as \
                 its path alone on a line, then its whole content in a fenced block.",
            )
        } else {
            return None;
        };
        Some((
            note,
            format!("{message}\n\n(The user's request: {})", request.trim()),
        ))
    }
}

/// Whether the user's message asks a question rather than for work (`what does main.py do?`,
/// but not `can you make a snake game?`): code in the answer to one needn't be saved.
pub fn is_question(message: &str) -> bool {
    let t = message.trim();
    let first = t
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_lowercase();
    let asking = [
        "what", "why", "how", "explain", "describe", "where", "which", "who", "when", "is", "are",
        "does", "do", "did", "should", "tell",
    ];
    let requesting = ["can", "could", "would", "will", "please"];
    asking.contains(&first.as_str()) || (t.ends_with('?') && !requesting.contains(&first.as_str()))
}

/// Whether a reply is a short refusal ("I'm sorry, but I can't assist with that"), which small
/// models give for ordinary tasks now and then.
pub fn is_refusal(reply: &str) -> bool {
    let t = reply.to_lowercase().replace('’', "'");
    t.len() < 400
        && [
            "i'm sorry",
            "i am sorry",
            "i can't",
            "i cannot",
            "i'm unable",
            "i am unable",
            "unable to",
        ]
        .iter()
        .any(|p| t.contains(p))
        && [
            "assist", "help", "fulfill", "comply", "do that", "create", "write", "run", "execute",
            "delete",
        ]
        .iter()
        .any(|p| t.contains(p))
}

/// Whether a short reply says something was done ("Deleted old.py.", "old.py deleted", "I've
/// created it"), which small models say instead of doing it, sometimes acting out both sides of
/// the prompt's examples ("User: delete old.py\nYou: Applied the deletion to old.py."). Only
/// meaningful for a turn in which nothing was done.
pub fn claims_action(reply: &str) -> bool {
    let lower = reply.trim().to_lowercase();
    let t = lower.rsplit("you:").next().unwrap_or_default();
    let done = [
        "deleted",
        "removed",
        "created",
        "renamed",
        "moved",
        "copied",
        "ran",
        "updated",
        "added",
        "fixed",
        "changed",
        "installed",
        "applied",
        "done",
        "i've",
        "i have",
    ];
    lower.len() < 300
        && t.split(|c: char| !c.is_alphanumeric() && c != '\'')
            .any(|w| done.contains(&w))
}

/// `p` with symlinks resolved as far as it exists.
fn real(p: &Path) -> PathBuf {
    for a in p.ancestors() {
        if let Ok(r) = a.canonicalize() {
            return r.join(p.strip_prefix(a).unwrap_or(Path::new("")));
        }
    }
    p.to_path_buf()
}

/// A directory's entries (directories with a slash after them), for a directory outside the
/// project, which could be all of the home directory and is not walked.
fn list_dir(dir: &Path) -> Vec<String> {
    let mut out: Vec<String> = fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .map(|e| {
                    let name = e.file_name().to_string_lossy().into_owned();
                    match e.path().is_dir() {
                        true => format!("{name}/"),
                        false => name,
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    out.sort();
    out
}

/// Why a shell command needs the user's approval, if it does: commands that delete, move or
/// write files (other than through the agent's own changes, which are approved separately),
/// change git history, install software or run with more privileges. Builds, tests and
/// read-only commands run at once.
pub fn needs_approval(command: &str) -> Option<&'static str> {
    if writes_through_redirect(command) {
        return Some("writes to a file");
    }
    let mut after_pipe = false;
    for (segment, sep) in segments(command) {
        let words: Vec<&str> = segment.split_whitespace().collect();
        if let Some(why) = risky(&words, after_pipe) {
            return Some(why);
        }
        after_pipe = sep == "|";
    }
    None
}

/// The commands in a command line with the separator after each: `;`, `&&`, `||`, `|`, `&`
/// or a newline (quotes respected, roughly).
fn segments(command: &str) -> Vec<(String, &'static str)> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quote = None;
    let mut chars = command.chars().peekable();
    while let Some(c) = chars.next() {
        if let Some(q) = quote {
            if c == q {
                quote = None;
            }
            cur.push(c);
            continue;
        }
        let sep = match c {
            '\'' | '"' => {
                quote = Some(c);
                cur.push(c);
                continue;
            }
            ';' | '\n' => ";",
            '&' if chars.peek() == Some(&'&') => {
                chars.next();
                "&&"
            }
            '|' if chars.peek() == Some(&'|') => {
                chars.next();
                "||"
            }
            '|' => "|",
            // `2>&1` and `>&2` are redirects, not separators.
            '&' if cur.ends_with('>') || chars.peek() == Some(&'>') => {
                cur.push(c);
                continue;
            }
            '&' => "&",
            _ => {
                cur.push(c);
                continue;
            }
        };
        out.push((std::mem::take(&mut cur), sep));
    }
    out.push((cur, ""));
    out
}

/// Whether a command line redirects output into a file (not `/dev/null` or another
/// descriptor).
fn writes_through_redirect(command: &str) -> bool {
    let mut quote = None;
    let b: Vec<char> = command.chars().collect();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        i += 1;
        match quote {
            Some(q) if c == q => quote = None,
            Some(_) => {}
            None if c == '\'' || c == '"' => quote = Some(c),
            None if c == '>' => {
                while i < b.len() && matches!(b[i], '>' | '|') {
                    i += 1;
                }
                if i < b.len() && b[i] == '&' {
                    continue;
                }
                while i < b.len() && b[i] == ' ' {
                    i += 1;
                }
                let target: String = b[i..]
                    .iter()
                    .take_while(|c| !c.is_whitespace() && !matches!(c, ';' | '&' | '|'))
                    .collect();
                if target != "/dev/null" {
                    return true;
                }
            }
            None => {}
        }
    }
    false
}

/// Why one command (its words) needs approval; `after_pipe` when its input is piped in.
fn risky(words: &[&str], after_pipe: bool) -> Option<&'static str> {
    // Skip `VAR=value` assignments and wrappers that run the rest.
    let mut w = words;
    while let Some(first) = w.first() {
        let wrapper = matches!(
            *first,
            "(" | "{" | "then" | "do" | "else" | "exec" | "command" | "nohup" | "time" | "env"
        );
        let assignment = first.contains('=') && !first.starts_with(['=', '-']);
        if !wrapper && !assignment {
            break;
        }
        w = &w[1..];
    }
    let first = w.first()?;
    let prog = first
        .trim_start_matches('(')
        .rsplit('/')
        .next()
        .unwrap_or(first);
    let args = &w[1..];
    let has = |a: &str| args.contains(&a);
    let sub = args
        .iter()
        .find(|a| !a.starts_with('-'))
        .copied()
        .unwrap_or("");
    match prog {
        "rm" | "rmdir" | "unlink" | "shred" | "srm" => Some("deletes files"),
        "mv" | "cp" | "ln" | "rsync" | "scp" | "dd" | "install" | "ditto" => {
            Some("moves or overwrites files")
        }
        "touch" | "tee" | "truncate" | "patch" | "unzip" => Some("writes files"),
        "chmod" | "chown" | "chgrp" | "chflags" | "xattr" => Some("changes file permissions"),
        "sudo" | "su" | "doas" => Some("runs as another user"),
        "kill" | "pkill" | "killall" | "launchctl" | "shutdown" | "reboot" | "halt" => {
            Some("stops processes or changes the system")
        }
        "osascript" | "crontab" | "mkfs" | "diskutil" | "fdisk" => Some("changes the system"),
        "defaults" if matches!(sub, "write" | "delete" | "import") => Some("changes settings"),
        "sed" | "perl" | "ruby"
            if args
                .iter()
                .any(|a| a.starts_with("-i") || *a == "--in-place") =>
        {
            Some("edits files in place")
        }
        "find" if has("-delete") || has("-exec") || has("-execdir") || has("-ok") => {
            Some("deletes or changes the files it finds")
        }
        "xargs" => {
            let rest: Vec<&str> = args
                .iter()
                .skip_while(|a| a.starts_with('-'))
                .copied()
                .collect();
            risky(&rest, false)
        }
        "git" => {
            // Skip git's own options (`-C dir`, `-c key=value`).
            let mut a = args;
            while let Some(x) = a.first() {
                match *x {
                    "-C" | "-c" | "--git-dir" | "--work-tree" => a = a.get(2..).unwrap_or(&[]),
                    x if x.starts_with('-') => a = &a[1..],
                    _ => break,
                }
            }
            let (sub, rest) = match a.split_first() {
                Some((s, r)) => (*s, r),
                None => return None,
            };
            let risky_sub = match sub {
                "commit" | "push" | "reset" | "checkout" | "switch" | "restore" | "clean"
                | "rebase" | "merge" | "rm" | "mv" | "apply" | "am" | "cherry-pick" | "revert"
                | "pull" | "init" | "clone" | "worktree" | "filter-branch" | "gc" | "prune" => true,
                "stash" => !matches!(rest.first(), Some(&"list" | &"show")),
                "branch" => rest.iter().any(|r| {
                    matches!(*r, "-d" | "-D" | "--delete" | "-m" | "-M" | "--move" | "-f")
                }),
                "tag" => rest
                    .iter()
                    .any(|r| !r.starts_with('-') || matches!(*r, "-d" | "-a")),
                "remote" => matches!(rest.first(), Some(&"add" | &"remove" | &"rm" | &"set-url")),
                "config" => !rest.iter().any(|r| matches!(*r, "--get" | "--list" | "-l")),
                _ => false,
            };
            risky_sub.then_some("changes the git repository")
        }
        "npm" | "pnpm" | "yarn" | "bun" | "npx"
            if matches!(
                sub,
                "install"
                    | "i"
                    | "add"
                    | "remove"
                    | "rm"
                    | "uninstall"
                    | "un"
                    | "ci"
                    | "update"
                    | "upgrade"
                    | "link"
                    | "publish"
            ) =>
        {
            Some("installs or removes packages")
        }
        "pip" | "pip3" | "poetry" | "pipx" | "conda"
            if matches!(sub, "install" | "uninstall" | "add" | "remove" | "update") =>
        {
            Some("installs or removes packages")
        }
        "uv" => {
            let rest: Vec<&str> = args.iter().copied().filter(|a| *a != "pip").collect();
            let sub = rest
                .iter()
                .find(|a| !a.starts_with('-'))
                .copied()
                .unwrap_or("");
            matches!(sub, "install" | "uninstall" | "add" | "remove" | "sync")
                .then_some("installs or removes packages")
        }
        p if p.starts_with("python") && args.windows(2).any(|a| a == ["-m", "pip"]) => args
            .iter()
            .any(|a| matches!(*a, "install" | "uninstall"))
            .then_some("installs or removes packages"),
        "cargo" if matches!(sub, "install" | "uninstall" | "add" | "remove" | "publish") => {
            Some("installs or removes packages")
        }
        "brew" | "port" | "apt" | "apt-get" | "yum" | "dnf" | "pacman" | "gem" | "go"
            if matches!(
                sub,
                "install"
                    | "uninstall"
                    | "remove"
                    | "reinstall"
                    | "upgrade"
                    | "get"
                    | "link"
                    | "unlink"
                    | "purge"
            ) =>
        {
            Some("installs or removes packages")
        }
        "curl"
            if args.iter().any(|a| {
                matches!(*a, "-o" | "-O" | "--output" | "--remote-name")
                    || (a.starts_with('-') && !a.starts_with("--") && a.contains(['o', 'O']))
            }) =>
        {
            Some("downloads to a file")
        }
        "wget"
            if !args
                .iter()
                .any(|a| matches!(*a, "-O-" | "-qO-") || a.ends_with("O-"))
                && !args.windows(2).any(|a| a == ["-O", "-"]) =>
        {
            Some("downloads to a file")
        }
        "sh" | "bash" | "zsh" | "fish" | "python" | "python3" | "node" | "ruby" | "perl"
            if after_pipe && args.iter().all(|a| a.starts_with('-')) =>
        {
            Some("runs a script piped into it")
        }
        _ => None,
    }
}

fn unchanged(path: &str) -> Action {
    Action::Unchanged {
        output: format!("{path} already has that content; nothing to change."),
    }
}

fn walk(root: &Path, dir: &Path, files: &mut Vec<String>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') || files.len() > 10 * MAX_LIST {
            continue;
        }
        let path = e.path();
        if path.is_dir() {
            if !SKIP_DIRS.contains(&name.as_str()) {
                walk(root, &path, files);
            }
        } else if let Ok(rel) = path.strip_prefix(root) {
            files.push(rel.display().to_string());
        }
    }
}

/// The file from `start` (1-based) on, as much as fits in `MAX_READ`, with a note on what was
/// left out; and a summary for the user.
fn excerpt_file(text: &str, path: &str, start: usize) -> Result<(String, String)> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return Ok(("(empty file)\n".into(), "empty".into()));
    }
    ensure!(
        start <= lines.len(),
        "{path} has only {} lines",
        lines.len()
    );
    let mut out = String::new();
    let mut end = start - 1;
    for l in &lines[start - 1..] {
        if end >= start && out.len() + l.len() > MAX_READ {
            break;
        }
        out += l;
        out.push('\n');
        end += 1;
    }
    let summary = match (start, end) {
        (1, e) if e == lines.len() => format!("{e} lines"),
        (s, e) => format!("lines {s}-{e} of {}", lines.len()),
    };
    if end < lines.len() {
        out += &format!(
            "[... lines {}-{} not shown; READ {path}:{} reads on]\n",
            end + 1,
            lines.len(),
            end + 1
        );
    }
    Ok((out, summary))
}

/// Replaces the one run of whole lines in `text` that equals `old` line by line, ignoring
/// indentation and trailing spaces, with `new`. Small models often copy a line's text right but
/// its indentation wrong; `new` is shifted by the same amount so it lines up with the file.
fn replace_lines(text: &str, old: &str, new: &str) -> Option<String> {
    let lines: Vec<&str> = text.lines().collect();
    let mut want: Vec<&str> = old.lines().collect();
    while want.first().is_some_and(|l| l.trim().is_empty()) {
        want.remove(0);
    }
    while want.last().is_some_and(|l| l.trim().is_empty()) {
        want.pop();
    }
    if want.is_empty() || want.len() > lines.len() {
        return None;
    }
    let hits: Vec<usize> = (0..=lines.len() - want.len())
        .filter(|&i| {
            want.iter()
                .enumerate()
                .all(|(k, w)| lines[i + k].trim() == w.trim())
        })
        .collect();
    let [at] = hits[..] else { return None };

    let indent = |l: &str| l[..l.len() - l.trim_start().len()].to_string();
    let k = want.iter().position(|l| !l.trim().is_empty())?;
    let (theirs, ours) = (indent(want[k]), indent(lines[at + k]));
    let new_lines: Vec<String> = new
        .lines()
        .map(
            |l| match (ours.strip_suffix(&theirs), theirs.strip_suffix(&ours)) {
                _ if l.trim().is_empty() => l.to_string(),
                (Some(extra), _) => format!("{extra}{l}"),
                (None, Some(extra)) => l.strip_prefix(extra).unwrap_or(l).to_string(),
                (None, None) => l.to_string(),
            },
        )
        .collect();
    let mut out: Vec<String> = lines[..at].iter().map(|l| l.to_string()).collect();
    out.extend(new_lines);
    out.extend(lines[at + want.len()..].iter().map(|l| l.to_string()));
    let mut s = out.join("\n");
    if text.ends_with('\n') {
        s.push('\n');
    }
    Some(s)
}

/// A line diff: every line of both texts tagged ' ' (both), '-' (only `old`) or '+' (only
/// `new`).
pub fn diff(old: &str, new: &str) -> Vec<(char, String)> {
    let a: Vec<&str> = old.lines().collect();
    let b: Vec<&str> = new.lines().collect();
    let pre = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
    let suf = a[pre..]
        .iter()
        .rev()
        .zip(b[pre..].iter().rev())
        .take_while(|(x, y)| x == y)
        .count();
    let (ma, mb) = (&a[pre..a.len() - suf], &b[pre..b.len() - suf]);
    let mut out: Vec<(char, String)> = a[..pre].iter().map(|l| (' ', l.to_string())).collect();
    if ma.len() * mb.len() > 4_000_000 {
        out.extend(ma.iter().map(|l| ('-', l.to_string())));
        out.extend(mb.iter().map(|l| ('+', l.to_string())));
    } else {
        // lcs[i][j]: the longest common subsequence of ma[i..] and mb[j..].
        let w = mb.len() + 1;
        let mut lcs = vec![0u32; (ma.len() + 1) * w];
        for i in (0..ma.len()).rev() {
            for j in (0..mb.len()).rev() {
                lcs[i * w + j] = match ma[i] == mb[j] {
                    true => lcs[(i + 1) * w + j + 1] + 1,
                    false => lcs[(i + 1) * w + j].max(lcs[i * w + j + 1]),
                };
            }
        }
        let (mut i, mut j) = (0, 0);
        while i < ma.len() || j < mb.len() {
            if i < ma.len() && j < mb.len() && ma[i] == mb[j] {
                out.push((' ', ma[i].to_string()));
                (i, j) = (i + 1, j + 1);
            } else if j == mb.len() || (i < ma.len() && lcs[(i + 1) * w + j] >= lcs[i * w + j + 1])
            {
                out.push(('-', ma[i].to_string()));
                i += 1;
            } else {
                out.push(('+', mb[j].to_string()));
                j += 1;
            }
        }
    }
    out.extend(a[a.len() - suf..].iter().map(|l| (' ', l.to_string())));
    out
}

/// Which lines of `diff` to show: the changes with `context` unchanged lines around each; `None`
/// stands for a run of unchanged lines left out.
pub fn excerpt(diff: &[(char, String)], context: usize) -> Vec<Option<usize>> {
    let near = |i: usize| {
        let lo = i.saturating_sub(context);
        let hi = (i + context + 1).min(diff.len());
        diff[lo..hi].iter().any(|(t, _)| *t != ' ')
    };
    let mut out = Vec::new();
    for i in 0..diff.len() {
        if near(i) {
            let gap = match out.last() {
                Some(last) => *last != Some(i - 1),
                None => i > 0,
            };
            if gap {
                out.push(None);
            }
            out.push(Some(i));
        }
    }
    if out
        .last()
        .is_some_and(|l| *l != Some(diff.len().wrapping_sub(1)))
    {
        out.push(None);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Streams `text` in pieces of `n` characters; returns what was shown and the reply.
    fn stream(text: &str, n: usize) -> (String, Reply) {
        let mut r = Reply::default();
        let chars: Vec<char> = text.chars().collect();
        let mut shown = String::new();
        for piece in chars.chunks(n) {
            shown += &r.push(&piece.iter().collect::<String>());
        }
        shown += &r.finish(false);
        (shown, r)
    }

    #[test]
    fn reads_and_cuts_after_them() {
        let text = "Let me look.\nREAD stats.py\nREAD `src/`\nstats.py says:\ndef median(xs):\n";
        for n in [1, 3, 100] {
            let (shown, r) = stream(text, n);
            assert_eq!(shown, "Let me look.\n");
            assert!(r.cut());
            assert_eq!(r.text(), "Let me look.\nREAD stats.py\nREAD `src/`\n");
            assert_eq!(
                Reply::parse(r.text()),
                [
                    Request::Read {
                        path: "stats.py".into(),
                        from: 1
                    },
                    Request::Read {
                        path: "src/".into(),
                        from: 1
                    }
                ]
            );
        }
        // A READ line followed by a fenced block: a file to create (or, if it exists, to read).
        let text = "READ Makefile\n```make\ntest:\n    python3 -m unittest\n```\n\nREAD NOTES.md\n```\n- notes\n```\nDone.\n";
        for n in [1, 5, 1000] {
            let (shown, r) = stream(text, n);
            assert_eq!(shown, "");
            assert!(r.cut());
            assert_eq!(
                Reply::parse(r.text()),
                [
                    Request::ReadOrCreate {
                        path: "Makefile".into(),
                        content: "test:\n    python3 -m unittest\n".into()
                    },
                    Request::ReadOrCreate {
                        path: "NOTES.md".into(),
                        content: "- notes\n".into()
                    }
                ]
            );
        }
        // Reads, then made-up contents for one of them: the reads alone.
        let (_, r) = stream("READ a.py\nREAD b.py\n```python\nx = 1\n```\n", 3);
        assert_eq!(r.text(), "READ a.py\n");
        assert_eq!(
            Reply::parse("READ a.rs:120"),
            [Request::Read {
                path: "a.rs".into(),
                from: 120
            }]
        );
    }

    #[test]
    fn edits_and_writes() {
        let text = "Fixing it.\n\nstats.py\n```python\n<<<<<<< SEARCH\n    return s[n // 2]\n=======\n    return s[n // 2] / 2\n>>>>>>> REPLACE\n```\n\n**README.md**:\n````markdown\n# stats\n```python\nmedian([1])\n```\n````\nApplied the change to stats.py.\n";
        for n in [1, 4, 1000] {
            let (shown, r) = stream(text, n);
            assert_eq!(shown, "Fixing it.\n\n");
            assert!(r.cut());
            assert!(r.text().ends_with("````\n"));
            assert_eq!(
                Reply::parse(r.text()),
                [
                    Request::Edit {
                        path: "stats.py".into(),
                        pairs: vec![(
                            "    return s[n // 2]\n".into(),
                            "    return s[n // 2] / 2\n".into()
                        )]
                    },
                    Request::Write {
                        path: "README.md".into(),
                        content: "# stats\n```python\nmedian([1])\n```\n".into()
                    }
                ]
            );
        }
    }

    #[test]
    fn plain_replies_stream_through() {
        let text = "The median is `s[n // 2]`.\nDone.\n\n```python\nx = 1\n```\nmain.rs is fine.";
        let (shown, r) = stream(text, 2);
        assert_eq!(shown, format!("{text}\n"));
        assert!(!r.cut() && Reply::parse(r.text()).is_empty());
        // A word before a code block is no file name unless it's one without an extension.
        assert!(Reply::parse("Example:\n```python\nx = 1\n```\n").is_empty());
        // Words stream as they arrive; only a line's first word waits.
        let mut r = Reply::default();
        assert_eq!(r.push("The median"), "The median");
        assert_eq!(r.push(" is\nREAD"), " is\n");
    }

    #[test]
    fn unfinished_blocks_are_invalid() {
        let mut r = Reply::default();
        r.push("a.py\n```python\nx = 1\n");
        r.finish(true);
        assert!(matches!(&r.requests[..], [Request::Invalid(_)]));
        let bad = Reply::parse("a.py\n```\n<<<<<<< SEARCH\nx\n```\n");
        assert!(matches!(&bad[..], [Request::Invalid(_)]));
    }

    #[test]
    fn makefile_recipes_get_tabs() {
        let make = "test:\n    python3 -m unittest\n\tmake lint\n\nX = 1\n";
        assert_eq!(
            recipe_tabs("sub/Makefile", make),
            "test:\n\tpython3 -m unittest\n\tmake lint\n\nX = 1\n"
        );
        assert_eq!(recipe_tabs("rules.mk", "a:\n  b\n"), "a:\n\tb\n");
        assert_eq!(recipe_tabs("main.py", "if x:\n    y\n"), "if x:\n    y\n");
    }

    #[test]
    fn diffs_lines() {
        let d = diff("a\nb\nc\nd\n", "a\nB\nc\nd\ne\n");
        let tags: String = d.iter().map(|(t, _)| *t).collect();
        assert_eq!(tags, " -+  +");
        let shown = excerpt(
            &diff("1\n2\n3\n4\n5\n6\n7\n8\n", "1\n2\n3\n4\nfive\n6\n7\n8\n"),
            1,
        );
        assert_eq!(shown, [None, Some(3), Some(4), Some(5), Some(6), None]);
    }

    #[test]
    fn replaces_lines_ignoring_indentation() {
        let file = "def f(x):\n    if x:\n        return 1\n    return 2\n";
        // The model dropped one level of indentation from both old and new text.
        let got = replace_lines(file, "if x:\n    return 1\n", "if x > 0:\n    return 1\n");
        assert_eq!(
            got.as_deref(),
            Some("def f(x):\n    if x > 0:\n        return 1\n    return 2\n")
        );
        assert_eq!(replace_lines(file, "return 3", "return 4"), None);
    }

    #[test]
    fn paths_reach_outside_the_project() {
        let ws = Workspace::new(PathBuf::from("/p"));
        assert_eq!(ws.resolve("src/../a.py").unwrap(), Path::new("/p/a.py"));
        assert_eq!(ws.resolve("/p/a.py").unwrap(), Path::new("/p/a.py"));
        assert_eq!(ws.resolve("../x").unwrap(), Path::new("/x"));
        assert_eq!(ws.resolve("/etc/hosts").unwrap(), Path::new("/etc/hosts"));
        let home = PathBuf::from(env::var("HOME").unwrap());
        assert_eq!(ws.resolve("~/notes.txt").unwrap(), home.join("notes.txt"));
        assert_eq!(ws.resolve("~").unwrap(), home);
        assert!(!ws.outside(Path::new("/p/src/a.py")));
        assert!(ws.outside(Path::new("/x")) && ws.outside(Path::new("/etc/hosts")));
    }

    #[test]
    fn changes_outside_are_marked() -> Result<()> {
        let base = std::env::temp_dir().join(format!("rlis-code-out-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(base.join("proj"))?;
        fs::write(base.join("elsewhere.txt"), "old\n")?;
        let mut ws = Workspace::new(base.join("proj"));
        let outside = base.join("new.txt").display().to_string();
        for (path, out) in [
            ("inside.txt", false),
            (outside.as_str(), true),
            ("../made.txt", true),
        ] {
            let write = Request::Write {
                path: path.into(),
                content: "x\n".into(),
            };
            let Action::Change(change) = ws.run(&write)? else {
                panic!("a new file")
            };
            assert_eq!(change.outside, out, "{path}");
        }
        // Outside files are read like any other, and need a read before a change.
        let edit = Request::Edit {
            path: "../elsewhere.txt".into(),
            pairs: vec![("old\n".into(), "new\n".into())],
        };
        assert!(ws.run(&edit).is_err());
        let Action::Done { summary, .. } = ws.run(&Request::Read {
            path: "../elsewhere.txt".into(),
            from: 1,
        })?
        else {
            panic!("a read")
        };
        assert!(summary.contains("outside the project"), "{summary}");
        assert!(matches!(
            ws.run(&edit)?,
            Action::Change(Change { outside: true, .. })
        ));
        // A directory outside is listed, not walked.
        let Action::Done { output, .. } = ws.run(&Request::Read {
            path: "..".into(),
            from: 1,
        })?
        else {
            panic!("a listing")
        };
        assert!(
            output.contains("proj/") && output.contains("elsewhere.txt"),
            "{output}"
        );
        let _ = fs::remove_dir_all(&base);
        Ok(())
    }

    #[test]
    fn runs_commands_and_cuts_after_them() {
        let text =
            "Let me test it.\nRUN python3 -m unittest -q\nRUN `ls src`\nOK, all 3 tests pass.\n";
        for n in [1, 3, 100] {
            let (shown, r) = stream(text, n);
            assert_eq!(shown, "Let me test it.\n");
            assert!(r.cut());
            assert_eq!(
                Reply::parse(r.text()),
                [
                    Request::Run {
                        command: "python3 -m unittest -q".into()
                    },
                    Request::Run {
                        command: "ls src".into()
                    }
                ]
            );
        }
        // A change and then a command to check it; a read after a command waits for it.
        let reqs = Reply::parse("a.py\n```python\nx = 1\n```\nRUN python3 a.py\nREAD a.py\n");
        assert_eq!(reqs.len(), 2, "{reqs:?}");
        assert!(matches!(&reqs[1], Request::Run { command } if command == "python3 a.py"));
        // A script after a lone RUN line.
        let reqs = Reply::parse("RUN\n```bash\ncd src\nmake test\n```\n");
        assert_eq!(
            reqs,
            [Request::Run {
                command: "cd src\nmake test".into()
            }]
        );
        // `RUN` inside a code block shown as text (a Dockerfile) is no command.
        let reqs =
            Reply::parse("Like this:\n```dockerfile\nFROM python:3\nRUN pip install x\n```\n");
        assert!(reqs.is_empty(), "{reqs:?}");
    }

    #[test]
    fn reads_replies_that_act_out_the_examples() {
        let text = "User: delete old.py\nYou: RUN rm old.py\nUser: Deleted old.py.\nYou: Done.\n";
        for n in [1, 4, 100] {
            let (shown, r) = stream(text, n);
            assert_eq!(shown, "");
            assert!(r.cut());
            assert_eq!(
                Reply::parse(r.text()),
                [Request::Run {
                    command: "rm old.py".into()
                }]
            );
        }
        let (shown, _) = stream("You: It returns the sum.\n", 3);
        assert_eq!(shown, "It returns the sum.\n");
    }

    #[test]
    fn names_files_from_the_line_before_a_block() {
        let html = "<!DOCTYPE html>\n<html></html>\n";
        for line in [
            "### HTML (index.html)",
            "Create `index.html`:",
            "**index.html**",
            "Save this as index.html:",
            "File: index.html",
        ] {
            let text = format!("Here is the app.\n\n{line}\n```html\n{html}```\nThen open it.\n");
            for n in [1, 7, 1000] {
                let (shown, r) = stream(&text, n);
                assert!(shown.starts_with("Here is the app.\n"), "{line}: {shown:?}");
                assert!(!shown.contains("DOCTYPE"), "{line}: {shown:?}");
                assert!(r.cut(), "{line}");
                assert_eq!(
                    Reply::parse(r.text()),
                    [Request::Write {
                        path: "index.html".into(),
                        content: html.into()
                    }],
                    "{line}"
                );
            }
        }
        // Names that are no files, and shell blocks, write nothing.
        for text in [
            "A server in Node.js:\n```javascript\nconst a = 1;\n```\n",
            "Run it with `python3 fib.py`:\n```bash\npython3 fib.py\n```\n",
            "For example (e.g. this):\n```python\nx = 1\n```\n",
        ] {
            assert!(Reply::parse(text).is_empty(), "{text}");
        }
    }

    #[test]
    fn names_files_from_a_first_line_comment() {
        let text = "```python\n# snake.py\nimport curses\n```\n";
        assert_eq!(
            Reply::parse(text),
            [Request::Write {
                path: "snake.py".into(),
                content: "# snake.py\nimport curses\n".into()
            }]
        );
        let reqs = Reply::parse("```html\n<!-- index.html -->\n<p>hi</p>\n```\n");
        assert!(matches!(&reqs[..], [Request::Write { path, .. }] if path == "index.html"));
        let reqs = Reply::parse("```c\n#include <stdio.h>\nint main() {}\n```\n");
        assert!(reqs.is_empty());
    }

    #[test]
    fn counts_unsaved_code() {
        let code = "```python\ndef fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n```\n";
        let r = Reply::read(code, None);
        assert_eq!((r.unsaved_code(), r.requests.len()), (1, 0));
        // Short snippets, commands and output are not code to save.
        for text in [
            "```python\nx = 1\n```\n",
            "```bash\na\nb\nc\nd\ne\n```\n",
            "```text\na\nb\nc\nd\ne\n```\n",
        ] {
            assert_eq!(Reply::read(text, None).unsaved_code(), 0, "{text}");
        }
        // A name in the line before is only taken for a new file.
        let dir = std::env::temp_dir().join(format!("rlis-code-named-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("old.py"), "x = 1\n").unwrap();
        let text = format!("In `old.py`:\n{code}");
        let r = Reply::read(&text, Some(&dir));
        assert_eq!((r.unsaved_code(), r.requests.len()), (1, 0));
        let r = Reply::read(&text.replace("old.py", "new.py"), Some(&dir));
        assert_eq!(r.requests.len(), 1);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn risky_commands_need_approval() {
        for safe in [
            "ls -la",
            "python3 snake.py",
            "cargo test 2>&1 | tail -20",
            "git status && git diff",
            "grep -rn foo src > /dev/null",
            "echo \"a > b\"",
            "npm run build",
            "git log --oneline -5",
            "FOO=1 make test",
            "curl -s https://example.com",
            "mkdir -p build",
            "find . -name '*.py'",
            "git -C sub stash list",
        ] {
            assert_eq!(needs_approval(safe), None, "{safe}");
        }
        for risky in [
            "rm -rf build",
            "ls && rm a.txt",
            "echo hi > notes.txt",
            "cat a >> b",
            "sed -i '' 's/a/b/' x.py",
            "git commit -m wip",
            "git -C sub push",
            "git checkout main",
            "pip install requests",
            "python3 -m pip install requests",
            "uv pip install x",
            "npm install",
            "brew install jq",
            "cargo install ripgrep",
            "curl -fsSL https://x.sh | sh",
            "curl -o out.bin https://x",
            "wget https://x/file.zip",
            "sudo ls",
            "find . -name '*.pyc' -delete",
            "ls | xargs rm",
            "mv a b",
            "chmod +x run.sh",
            "kill 123",
        ] {
            assert!(needs_approval(risky).is_some(), "{risky}");
        }
    }

    #[test]
    fn tells_questions_and_refusals() {
        for q in [
            "what does main.py do?",
            "How do I run it",
            "explain fib",
            "is this right?",
        ] {
            assert!(is_question(q), "{q}");
        }
        for t in [
            "make me a snake game",
            "can you add a test?",
            "fix the bug in a.py",
        ] {
            assert!(!is_question(t), "{t}");
        }
        assert!(is_refusal(
            "I'm sorry, but I can't assist with that request."
        ));
        assert!(is_refusal("I’m sorry, I cannot help with that."));
        assert!(is_refusal(
            "I'm sorry, but I can't run Python scripts directly. Please run the command \
             `python3 fibonacci.py` in your terminal."
        ));
        assert!(!is_refusal("I created snake.py."));
        assert!(claims_action("Deleted old.py."));
        assert!(claims_action("I've created the file for you."));
        assert!(claims_action("User: delete old.py\nYou: old.py deleted."));
        assert!(claims_action(
            "User: delete old.py\nYou: Applied the deletion to old.py."
        ));
        assert!(!claims_action("It prints the squares of 1 to 5."));
        assert!(!is_refusal(&format!(
            "I can't find it. {}",
            "x".repeat(500)
        )));
    }

    #[test]
    fn changes_need_a_read_first() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("rlis-code-test-{}", std::process::id()));
        fs::create_dir_all(&dir)?;
        fs::write(dir.join("a.py"), "x = 1\n")?;
        let mut ws = Workspace::new(dir.clone());
        let edit = Request::Edit {
            path: "a.py".into(),
            pairs: vec![("x = 1\n".into(), "x = 2\n".into())],
        };
        let refused = ws
            .run(&edit)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(refused.contains("READ it first"), "{refused}");
        let read = Request::Read {
            path: "a.py".into(),
            from: 1,
        };
        assert!(matches!(ws.run(&read)?, Action::Done { .. }));
        let Action::Change(change) = ws.run(&edit)? else {
            panic!("an edit after the read")
        };
        assert_eq!(change.new, "x = 2\n");
        // New files need no read, and what the model wrote counts as read.
        let write = Request::Write {
            path: "sub/b.py".into(),
            content: "y = 1\n".into(),
        };
        let Action::Change(change) = ws.run(&write)? else {
            panic!("a new file")
        };
        ws.apply(&change)?;
        assert_eq!(fs::read_to_string(dir.join("sub/b.py"))?, "y = 1\n");
        assert!(matches!(ws.run(&write)?, Action::Unchanged { .. }));
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    /// The directory `rlis code` runs in, read and written at its top level, in a git
    /// repository and outside one.
    #[test]
    fn reads_and_writes_its_own_directory() -> Result<()> {
        for git in [false, true] {
            let dir =
                std::env::temp_dir().join(format!("rlis-code-dir-{}-{git}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(dir.join("src"))?;
            fs::write(dir.join("main.py"), "print(1)\n")?;
            fs::write(dir.join("src/lib.py"), "x = 1\n")?;
            if git {
                let ok = Command::new("git")
                    .args(["init", "-q"])
                    .current_dir(&dir)
                    .status()?
                    .success();
                assert!(ok);
            }
            let mut ws = Workspace::new(dir.clone());
            for root in [".", "./", "", &dir.display().to_string()] {
                let Action::Done { output, .. } = ws.run(&Request::Read {
                    path: root.into(),
                    from: 1,
                })?
                else {
                    panic!("a listing")
                };
                assert!(
                    output.contains("main.py") && output.contains("src/lib.py"),
                    "{git} {root:?}: {output}"
                );
            }
            assert!(ws.system_prompt(None).contains("src/lib.py"));

            // Files at the top level, with and without an extension, from the model's reply.
            let reply =
                "notes.md\n```markdown\n# Notes\n```\n\nMakefile\n```make\nall:\n\techo hi\n```\n";
            let requests = Reply::parse(reply);
            assert_eq!(requests.len(), 2, "{requests:?}");
            for r in &requests {
                let Action::Change(change) = ws.run(r)? else {
                    panic!("a new file: {r:?}")
                };
                ws.apply(&change)?;
            }
            assert_eq!(fs::read_to_string(dir.join("notes.md"))?, "# Notes\n");
            // `READ` and a block: reads a file that exists, creates one that doesn't.
            let echo = Request::ReadOrCreate {
                path: "main.py".into(),
                content: "made up\n".into(),
            };
            assert!(matches!(ws.run(&echo)?, Action::Done { .. }));
            let echo = Request::ReadOrCreate {
                path: "main.py".into(),
                content: "print(1)\n".into(),
            };
            assert!(matches!(ws.run(&echo)?, Action::Unchanged { .. }));
            let create = Request::ReadOrCreate {
                path: "lib/new.py".into(),
                content: "y = 2\n".into(),
            };
            let Action::Change(change) = ws.run(&create)? else {
                panic!("a new file")
            };
            ws.apply(&change)?;
            assert_eq!(fs::read_to_string(dir.join("lib/new.py"))?, "y = 2\n");
            assert_eq!(
                fs::read_to_string(dir.join("Makefile"))?,
                "all:\n\techo hi\n"
            );
            let _ = fs::remove_dir_all(&dir);
        }
        Ok(())
    }

    #[test]
    fn reads_long_files_in_parts() -> Result<()> {
        let text: String = (0..3000).map(|i| format!("line {i}\n")).collect();
        let (out, summary) = excerpt_file(&text, "big.txt", 1)?;
        assert!(out.len() < MAX_READ + 200 && out.contains("READ big.txt:"));
        assert!(summary.starts_with("lines 1-"));
        let (_, summary) = excerpt_file(&text, "big.txt", 2990)?;
        assert_eq!(summary, "lines 2990-3000 of 3000");
        assert!(excerpt_file(&text, "big.txt", 4000).is_err());
        Ok(())
    }
}
