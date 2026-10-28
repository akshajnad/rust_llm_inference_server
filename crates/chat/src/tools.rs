//! `rlis code`'s tools: reading, creating and editing the files under the directory it was
//! started in. The model asks for them in plain text, in the formats small coding models write
//! reliably (Qwen2.5-Coder 3B ignores most of Qwen's JSON tool-call format):
//!
//! - `READ path` lines read files (or list a directory);
//! - the file's path alone on a line and then a fenced `<<<<<<< SEARCH` / `=======` /
//!   `>>>>>>> REPLACE` block edits it (aider's format, which Qwen2.5-Coder was tuned on);
//! - the path and then a fenced block of anything else writes the whole file.
//!
//! `Reply` reads a reply as it streams in: it hides these requests from the transcript (their
//! results get cards of their own) and says when a request is complete and the model has gone
//! on to imagine its result, so generation can stop there. Reads run at once; writes and edits
//! become a `Change` for the user to approve before anything touches the disk.

use std::collections::HashSet;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail, ensure};

/// Characters of a file returned by one read; the model asks for later lines with `READ path:N`.
const MAX_READ: usize = 12000;
const MAX_LIST: usize = 300;
/// Files listed in the system prompt.
const MAX_PROMPT_FILES: usize = 200;
/// Directories skipped when listing a directory that is not in a git repository.
const SKIP_DIRS: &[&str] = &["target", "node_modules", "__pycache__", "venv"];

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
            Request::Write { path, .. } => format!("write {path}"),
            Request::Invalid(_) => "request".into(),
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Read,
    Change,
}

/// A fenced block (or bare SEARCH/REPLACE block) after a path line, while it arrives.
struct Block {
    path: String,
    /// The opening backticks; `None` for a SEARCH/REPLACE block written without a fence.
    fence: Option<String>,
    /// Fences opened inside it, as in a Markdown file with code blocks.
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
    requests: Vec<Request>,
    last: Option<Kind>,
    cut: bool,
}

impl Reply {
    /// The requests in a whole reply.
    pub fn parse(text: &str) -> Vec<Request> {
        let mut r = Reply::default();
        r.push(text);
        r.finish(false);
        r.requests
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
        // Stream the line being received unless it may start a request.
        if !self.cut
            && self.block.is_none()
            && self.held.is_empty()
            && self.last.is_none()
            && (self.shown > 0 || !may_be_special(&self.line))
        {
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
        for line in std::mem::take(&mut self.held) {
            if self.last.is_none() {
                out += &line;
                out.push('\n');
                self.text += &line;
                self.text.push('\n');
            }
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
        if shown > 0 {
            *out += &line[shown..];
            out.push('\n');
            self.text += &line;
            self.text.push('\n');
            return;
        }
        self.held.push(line);
        self.decide(out);
    }

    /// Shows, hides or keeps holding the held lines.
    fn decide(&mut self, out: &mut String) {
        while let Some(first) = self.held.first() {
            if let Some(read) = read_line(first) {
                if self.last == Some(Kind::Change) {
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
                let Some(second) = self.held.get(1) else {
                    return;
                };
                let fence = fence_open(second);
                if fence.is_some() || second.trim() == SEARCH {
                    if self.last == Some(Kind::Read) {
                        return self.stop();
                    }
                    // The path and the fence open the block; a bare SEARCH line is its first.
                    self.keep(if fence.is_some() { 2 } else { 1 });
                    self.block = Some(Block {
                        path,
                        fence,
                        depth: 0,
                        lines: Vec::new(),
                    });
                    for line in std::mem::take(&mut self.held) {
                        self.text += &line;
                        self.text.push('\n');
                        self.feed(line);
                    }
                    return;
                }
            }
            // Ordinary text; after a request, the model imagining what came of it.
            if self.last.is_some() {
                return self.stop();
            }
            *out += first;
            out.push('\n');
            self.keep(1);
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
                Request::Write {
                    path: b.path,
                    content,
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
/// shown yet: a word with no space after it yet (a path), `READ `, a fence or a marker.
fn may_be_special(line: &str) -> bool {
    let t = line.trim_start();
    !t.contains(char::is_whitespace)
        || t.starts_with("READ ")
        || t.starts_with("```")
        || t.starts_with("<<<<<<<")
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
/// maybe with a colon after it.
fn path_line(line: &str) -> Option<String> {
    let t = line
        .trim()
        .trim_end_matches(':')
        .trim_matches(|c| c == '`' || c == '*');
    let t = t.trim_end_matches(':');
    let plausible = !t.is_empty()
        && t.len() < 200
        && !t.contains(char::is_whitespace)
        && (t.contains('.') || t.contains('/'))
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
}

pub struct Change {
    /// As the model named it, relative to the project root.
    pub path: String,
    abs: PathBuf,
    /// The file's text now; `None` when the change creates it.
    pub old: Option<String>,
    pub new: String,
}

/// The directory the tools work in; no path may leave it.
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

    /// The instructions, with a worked example (which small models need to follow them) and the
    /// project's files; `extra` is the user's own system prompt.
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
            "You are rlis, a coding agent working in the project at {root}. You read the \
             project's files and change them to do what the user asks.

To read files, reply with one READ line per file and nothing else:
READ path/to/file.py
Their contents come back in the next message. A long file comes in parts: READ \
path/to/file.py:200 reads from line 200. READ a directory to list its files.

To change part of a file, write the file's path alone on a line, then a fenced \
SEARCH/REPLACE block: lines copied exactly from the file, indentation included, and the lines \
to put in their place:

path/to/file.py
```python
{SEARCH}
    return x
{DIVIDER}
    return x + 1
{REPLACE}
```

To create a file, write its path alone on a line, then its whole content in a fenced block.

Read a file before you change it. You are told whether each change was applied. When the task \
is done, reply with a short summary of what you changed.

An example of a whole task:

User: greet() should say hello, not hi. Add a test for it.
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

tests/test_greet.py
```python
from app.greet import greet


def test_greet():
    assert greet(\"Ann\") == \"hello Ann\"
```
User: Applied the change to app/greet.py.
Created tests/test_greet.py.
You: greet() now says hello instead of hi, and tests/test_greet.py tests it.

Files in the project:
{listing}",
            root = self.root.display()
        );
        if let Some(extra) = extra {
            s += &format!("\n\n{extra}");
        }
        s
    }

    /// Runs a read, or prepares a change; errors are the model's to fix.
    pub fn run(&mut self, request: &Request) -> Result<Action> {
        match request {
            Request::Read { path, from } => {
                let abs = self.resolve(path)?;
                ensure!(abs.exists(), "{path} does not exist");
                if abs.is_dir() {
                    let files = self.files(&abs);
                    let mut output = files
                        .iter()
                        .take(MAX_LIST)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join("\n");
                    if files.len() > MAX_LIST {
                        output += &format!("\n[... and {} more]", files.len() - MAX_LIST);
                    }
                    let summary = format!("{} files", files.len());
                    let output = format!("Files in {path}:\n```\n{output}\n```");
                    return Ok(Action::Done { output, summary });
                }
                let text = self.read(path)?;
                let (body, summary) = excerpt_file(&text, path, *from)?;
                self.seen.insert(abs);
                let output = format!("Contents of {path}:\n```\n{body}```");
                Ok(Action::Done { output, summary })
            }
            Request::Write { path, content } => {
                let abs = self.resolve(path)?;
                ensure!(!abs.is_dir(), "{path} is a directory");
                self.ensure_seen(&abs, path)?;
                let old = abs.exists().then(|| self.read(path)).transpose()?;
                if old.as_ref() == Some(content) {
                    return Ok(unchanged(path));
                }
                Ok(Action::Change(Change {
                    path: path.clone(),
                    abs,
                    old,
                    new: content.clone(),
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
                    return Ok(Action::Change(Change {
                        path: path.clone(),
                        abs,
                        old: None,
                        new,
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
                if text == old {
                    return Ok(unchanged(path));
                }
                Ok(Action::Change(Change {
                    path: path.clone(),
                    abs,
                    old: Some(old),
                    new: text,
                }))
            }
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

    /// `path` inside the project, without touching the disk; `..` may not climb out of it.
    fn resolve(&self, path: &str) -> Result<PathBuf> {
        let p = Path::new(path.trim());
        let rel = p.strip_prefix(&self.root).unwrap_or(p);
        let mut out = self.root.clone();
        for c in rel.components() {
            match c {
                Component::Normal(n) => out.push(n),
                Component::CurDir => {}
                Component::ParentDir if out != self.root => {
                    out.pop();
                }
                _ => bail!("{path} is outside the project"),
            }
        }
        Ok(out)
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
        let text = "Let me look.\nREAD stats.py\nREAD `src/`\n```python\ndef median(xs):\n";
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
    fn paths_stay_in_the_project() {
        let ws = Workspace::new(PathBuf::from("/p"));
        assert_eq!(ws.resolve("src/../a.py").unwrap(), Path::new("/p/a.py"));
        assert_eq!(ws.resolve("/p/a.py").unwrap(), Path::new("/p/a.py"));
        assert!(ws.resolve("../x").is_err());
        assert!(ws.resolve("/etc/passwd").is_err());
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
