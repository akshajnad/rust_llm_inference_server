//! `rlis fix`: N attempts at a patch from one request. The task, the failing test's output and
//! the code it points at go to the server once with `n` completions, which rlis prefills once
//! and forks. Each candidate diff is applied in its own git worktree and the test command runs
//! there, a few at a time; the smallest passing diff wins.
//!
//! The user's working tree is only read (`git stash create` snapshots its tracked files
//! without touching it) unless `--apply` is given. Worktrees live under
//! `<git dir>/rlis-fix/<run>/` and are removed on exit, including on errors.

use std::collections::BTreeSet;
use std::fs;
use std::io::Read;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::Parser;
use crossterm::style::Stylize;
use serde_json::{Value, json};

use crate::{ACCENT, ensure_server, http, model_name};

/// What `rlis fix` starts when no server is running: the coding model, with room for a prompt
/// of a few files plus the reply.
const QWEN_SERVE_ARGS: &[&str] = &[
    "--model",
    "models/Qwen2.5-Coder-1.5B-Instruct-Q8_0.gguf",
    "--tokenizer",
    "models/qwen2.5-coder-1.5b-instruct",
    "--served-model-name",
    "qwen2.5-coder-1.5b-instruct",
    "--max-seq-len",
    "8192",
];

const SYSTEM: &str = "You fix bugs in code repositories. Reply with one unified diff in a \
```diff block: paths relative to the repository root with a/ and b/ prefixes, and a few \
unchanged context lines around each change so it applies. Change only what the fix needs. \
Never edit the tests.";

/// Characters of test output, file contents and grep hits put in the prompt.
const MAX_OUTPUT: usize = 4000;
const MAX_FILES: usize = 16000;
const MAX_GREP_LINES: usize = 40;

#[derive(Parser)]
#[command(
    name = "rlis fix",
    about = "N parallel attempts at a patch that makes a test pass"
)]
pub struct FixArgs {
    /// Git repository to fix.
    #[arg(long, default_value = ".")]
    repo: PathBuf,
    /// Shell command that exits 0 when the bug is fixed; run from the repository root.
    #[arg(long)]
    test: String,
    /// What is wrong, in words.
    #[arg(long)]
    task: String,
    /// Attempts, all from one request (`n`).
    #[arg(long, default_value_t = 8)]
    n: usize,
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,
    /// Seed of attempt 0; attempt i uses seed + i.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value_t = 1024)]
    max_tokens: usize,
    /// Test commands running at once.
    #[arg(long, default_value_t = 4)]
    jobs: usize,
    /// Seconds before a test run is killed and counted as a failure.
    #[arg(long, default_value_t = 60)]
    timeout: u64,
    /// Files (repository-relative) a candidate may not change, e.g. the test file.
    #[arg(long)]
    protect: Vec<String>,
    /// Apply the winning patch to the working tree.
    #[arg(long)]
    apply: bool,
    /// Also write the full result (every attempt's diff and outcome) as JSON.
    #[arg(long)]
    json: Option<PathBuf>,
    /// Server to use; a local one with Qwen2.5-Coder is started if nothing answers there.
    #[arg(long, default_value = "http://127.0.0.1:8000")]
    url: String,
    /// Exit instead of starting a server when none is running.
    #[arg(long)]
    no_start: bool,
    /// Arguments for `rlis serve` when starting one (default: the Qwen2.5-Coder model).
    #[arg(last = true)]
    serve_args: Vec<String>,
}

/// One candidate's fate.
#[derive(Default)]
struct Attempt {
    finish_reason: String,
    /// The diff as the model wrote it (empty if it wrote none).
    raw_diff: String,
    applied: bool,
    /// Applied only with `patch --fuzz`.
    fuzzy: bool,
    /// Sections for protected files (the tests) were dropped from the diff.
    dropped_protected: bool,
    /// Why it didn't apply or wasn't tested.
    rejected: Option<String>,
    /// `git diff` of the worktree after applying: the patch as it actually landed.
    diff: String,
    lines_changed: usize,
    passed: bool,
    exit: Option<i32>,
    test_secs: f64,
}

pub fn run(args: FixArgs) -> Result<()> {
    let total = Instant::now();
    anyhow::ensure!(
        args.n >= 1 && args.jobs >= 1,
        "--n and --jobs must be at least 1"
    );
    let repo = git(&args.repo, &["rev-parse", "--show-toplevel"])
        .with_context(|| format!("{} is not a git repository", args.repo.display()))?;
    let repo = PathBuf::from(repo.trim());
    let git_dir = PathBuf::from(git(&repo, &["rev-parse", "--absolute-git-dir"])?.trim());
    let serve_args = match args.serve_args.is_empty() {
        true => QWEN_SERVE_ARGS.iter().map(|s| s.to_string()).collect(),
        false => args.serve_args.clone(),
    };
    let _server = ensure_server(&args.url, args.no_start, &serve_args)?;
    let model = model_name(&args.url).unwrap_or_else(|| "unknown model".into());

    // The working tree's tracked state as a commit, or HEAD when it is clean.
    let stash = git(&repo, &["stash", "create"])?;
    let base = match stash.trim() {
        "" => git(&repo, &["rev-parse", "HEAD"])?.trim().to_string(),
        s => s.to_string(),
    };
    let run_id = format!("{}-{}", std::process::id(), unix_millis());
    let trees = Worktrees::new(&repo, git_dir.join("rlis-fix").join(run_id));

    // Context: the failing test, the files it names, and grep hits for the task's identifiers.
    let t = Instant::now();
    let base_dir = trees.add(&base, "base")?;
    let baseline = run_test(&base_dir, &args.test, args.timeout)?;
    if baseline.passed {
        println!("the test already passes at the current tree; nothing to fix");
        return Ok(());
    }
    let prompt = build_prompt(&base_dir, &args, &baseline.output)?;
    let context_secs = t.elapsed().as_secs_f64();

    let t = Instant::now();
    println!(
        "{} {} attempts from {model} ({} prompt characters)",
        "rlis fix".with(ACCENT),
        args.n,
        prompt.len()
    );
    let body = json!({
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "n": args.n,
        "temperature": args.temperature,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
    });
    let resp = http::request(
        &args.url,
        "POST",
        "/v1/chat/completions",
        Some(&serde_json::to_vec(&body)?),
    )?;
    let status = resp.status;
    let text = resp.text()?;
    if status != 200 {
        bail!("the server answered {status}: {}", text.trim());
    }
    let reply: Value = serde_json::from_str(&text)?;
    let generate_secs = t.elapsed().as_secs_f64();
    let prompt_tokens = reply["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
    let completion_tokens = reply["usage"]["completion_tokens"].as_u64().unwrap_or(0);

    let mut attempts: Vec<Attempt> = (0..args.n).map(|_| Attempt::default()).collect();
    for choice in reply["choices"].as_array().context("no choices")? {
        let i = choice["index"].as_u64().context("choice without index")? as usize;
        let a = attempts.get_mut(i).context("choice index out of range")?;
        a.finish_reason = choice["finish_reason"].as_str().unwrap_or("").into();
        a.raw_diff = extract_diff(choice["message"]["content"].as_str().unwrap_or(""));
    }

    // Apply each candidate in its own worktree, then test the ones that applied.
    let t = Instant::now();
    let mut dirs = Vec::with_capacity(args.n);
    for (i, a) in attempts.iter_mut().enumerate() {
        let dir = trees.add(&base, &i.to_string())?;
        apply_candidate(&dir, a, &args.protect)?;
        dirs.push(dir);
    }
    let attempts = Mutex::new(attempts);
    let next = AtomicUsize::new(0);
    thread::scope(|s| -> Result<()> {
        let workers: Vec<_> = (0..args.jobs.min(args.n))
            .map(|_| {
                s.spawn(|| -> Result<()> {
                    loop {
                        let i = next.fetch_add(1, Ordering::SeqCst);
                        if i >= args.n {
                            return Ok(());
                        }
                        if !attempts.lock().unwrap()[i].applied {
                            continue;
                        }
                        let r = run_test(&dirs[i], &args.test, args.timeout)?;
                        let a = &mut attempts.lock().unwrap()[i];
                        (a.passed, a.exit, a.test_secs) = (r.passed, r.exit, r.secs);
                    }
                })
            })
            .collect();
        workers
            .into_iter()
            .try_for_each(|w| w.join().expect("test worker panicked"))
    })?;
    let attempts = attempts.into_inner().unwrap();
    let tests_secs = t.elapsed().as_secs_f64();

    // Smallest passing diff wins; ties go to the lowest index.
    let winner = attempts
        .iter()
        .enumerate()
        .filter(|(_, a)| a.passed)
        .min_by_key(|(i, a)| (a.lines_changed, *i))
        .map(|(i, _)| i);
    let applied = attempts.iter().filter(|a| a.applied).count();
    let passed = attempts.iter().filter(|a| a.passed).count();
    if args.apply
        && let Some(w) = winner
    {
        git_apply(&repo, &attempts[w].diff, &[])
            .context("applying the winning patch to the working tree")?;
    }
    let total_secs = total.elapsed().as_secs_f64();

    for (i, a) in attempts.iter().enumerate() {
        let outcome = match (&a.rejected, a.passed) {
            (Some(why), _) => why.clone(),
            (None, true) => format!("passed ({:.1} s)", a.test_secs),
            (None, false) => match a.exit {
                Some(code) => format!("failed (exit {code}, {:.1} s)", a.test_secs),
                None => format!("failed (killed after {} s)", args.timeout),
            },
        };
        let fuzz = if a.fuzzy { ", with fuzz" } else { "" };
        let dropped = if a.dropped_protected {
            ", test edits dropped"
        } else {
            ""
        };
        let size = match a.applied {
            true => format!("{} lines{fuzz}{dropped}", a.lines_changed),
            false => String::new(),
        };
        println!("  #{i}  {outcome:<32} {}", size.dark_grey());
    }
    println!(
        "{} attempts · {applied} applied · {passed} passed · {}",
        args.n,
        match winner {
            Some(w) => format!("winner #{w}").with(ACCENT).to_string(),
            None => "no fix".red().to_string(),
        }
    );
    println!(
        "{}",
        format!(
            "context {context_secs:.1} s · generate {generate_secs:.1} s ({prompt_tokens} prompt + \
             {completion_tokens} output tokens) · tests {tests_secs:.1} s · total {total_secs:.1} s"
        )
        .dark_grey()
    );
    if let Some(w) = winner {
        println!("\n{}", attempts[w].diff.trim_end());
        if args.apply {
            println!("\napplied to {}", repo.display());
        }
    }

    if let Some(path) = &args.json {
        let out = json!({
            "repo": repo,
            "task": args.task,
            "test": args.test,
            "model": model,
            "n": args.n,
            "temperature": args.temperature,
            "seed": args.seed,
            "attempts": attempts.iter().map(|a| json!({
                "finish_reason": a.finish_reason,
                "raw_diff": a.raw_diff,
                "applied": a.applied,
                "fuzzy": a.fuzzy,
                "dropped_protected": a.dropped_protected,
                "rejected": a.rejected,
                "diff": a.diff,
                "lines_changed": a.lines_changed,
                "passed": a.passed,
                "exit": a.exit,
                "test_secs": a.test_secs,
            })).collect::<Vec<_>>(),
            "applied": applied,
            "passed": passed,
            "winner": winner,
            "prompt_chars": prompt.len(),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "secs": {
                "context": context_secs,
                "generate": generate_secs,
                "tests": tests_secs,
                "total": total_secs,
            },
        });
        fs::write(path, serde_json::to_string_pretty(&out)? + "\n")
            .with_context(|| format!("writing {}", path.display()))?;
    }
    Ok(())
}

/// Worktrees of one run under `root`; all removed on drop, also when the run fails.
struct Worktrees {
    repo: PathBuf,
    root: PathBuf,
}

impl Worktrees {
    /// Also removes worktrees left by runs that died without cleaning up (killed, or
    /// interrupted with ctrl-c): run directories are named `<pid>-<millis>`.
    fn new(repo: &Path, root: PathBuf) -> Self {
        if let Ok(entries) = root
            .parent()
            .map_or(Ok(None), |p| fs::read_dir(p).map(Some))
            && let Some(entries) = entries
        {
            for e in entries.flatten() {
                let name = e.file_name().to_string_lossy().into_owned();
                let pid = name.split('-').next().unwrap_or_default();
                let alive = Command::new("kill")
                    .args(["-0", pid])
                    .stderr(Stdio::null())
                    .status()
                    .is_ok_and(|s| s.success());
                if !alive {
                    let _ = fs::remove_dir_all(e.path());
                }
            }
            let _ = git(repo, &["worktree", "prune"]);
        }
        Self {
            repo: repo.to_path_buf(),
            root,
        }
    }

    fn add(&self, commit: &str, name: &str) -> Result<PathBuf> {
        let dir = self.root.join(name);
        fs::create_dir_all(&self.root)?;
        let path = dir.to_str().context("non-UTF-8 path")?;
        git(&self.repo, &["worktree", "add", "--detach", path, commit])?;
        Ok(dir)
    }
}

impl Drop for Worktrees {
    fn drop(&mut self) {
        // `git worktree remove` needs git 2.17; deleting the directories and pruning works
        // everywhere.
        let _ = fs::remove_dir_all(&self.root);
        if let Some(parent) = self.root.parent() {
            let _ = fs::remove_dir(parent);
        }
        let _ = git(&self.repo, &["worktree", "prune"]);
    }
}

struct TestRun {
    passed: bool,
    exit: Option<i32>,
    output: String,
    secs: f64,
}

/// Runs `cmd` with `sh -c` in `dir`, killing its whole process group after `timeout` seconds.
fn run_test(dir: &Path, cmd: &str, timeout: u64) -> Result<TestRun> {
    let start = Instant::now();
    let out_path = dir.join(".rlis-fix-test-output");
    let out = fs::File::create(&out_path)?;
    let mut child = Command::new("sh")
        .args(["-c", cmd])
        .current_dir(dir)
        .stdin(Stdio::null())
        .stdout(out.try_clone()?)
        .stderr(out)
        .process_group(0)
        .spawn()
        .with_context(|| format!("running {cmd:?}"))?;
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break Some(status);
        }
        if start.elapsed() > Duration::from_secs(timeout) {
            let _ = Command::new("kill")
                .args(["-KILL", &format!("-{}", child.id())])
                .status();
            let _ = child.wait();
            break None;
        }
        thread::sleep(Duration::from_millis(20));
    };
    let mut output = String::new();
    fs::File::open(&out_path)?.read_to_string(&mut output)?;
    fs::remove_file(&out_path)?;
    let exit = status.and_then(|s| s.code());
    Ok(TestRun {
        passed: exit == Some(0),
        exit,
        output,
        secs: start.elapsed().as_secs_f64(),
    })
}

/// The task, the test's output, the tracked files the output or task names, and grep hits
/// for identifiers in the task.
fn build_prompt(dir: &Path, args: &FixArgs, output: &str) -> Result<String> {
    // Paths in the output are absolute (inside the base worktree) or repository-relative.
    let prefix = format!("{}/", dir.display());
    let output = tail(&output.replace(&prefix, ""), MAX_OUTPUT);
    let tracked: BTreeSet<String> = git(dir, &["ls-files"])?.lines().map(String::from).collect();
    let mut files = Vec::new();
    for word in output
        .split(|c: char| c.is_whitespace() || "\"'(),:".contains(c))
        .chain(
            args.task
                .split(|c: char| c.is_whitespace() || "\"'`(),:".contains(c)),
        )
    {
        if tracked.contains(word) && !files.contains(&word.to_string()) {
            files.push(word.to_string());
        }
    }
    let mut prompt = format!(
        "Task: {}\n\nThe test command `{}` fails:\n```\n{}\n```\n",
        args.task.trim(),
        args.test,
        output.trim_end()
    );
    let mut budget = MAX_FILES;
    for f in &files {
        let text = fs::read_to_string(dir.join(f)).unwrap_or_default();
        if text.len() > budget {
            continue;
        }
        budget -= text.len();
        prompt += &format!("\n{f}:\n```\n{}\n```\n", text.trim_end());
    }
    let idents = identifiers(&args.task);
    if !idents.is_empty() {
        let mut cmd = vec!["grep", "-n", "-w", "-I"];
        for id in &idents {
            cmd.extend(["-e", id.as_str()]);
        }
        // Exit status 1 just means no matches.
        let hits = git(dir, &cmd).unwrap_or_default();
        let hits: Vec<&str> = hits.lines().take(MAX_GREP_LINES).collect();
        if !hits.is_empty() {
            prompt += &format!(
                "\nWhere {} appear (git grep):\n```\n{}\n```\n",
                idents.join(", "),
                hits.join("\n")
            );
        }
    }
    prompt += "\nReply with the diff that fixes the bug.";
    Ok(prompt)
}

/// Code-like words in the task: anything in backticks, plus snake_case and CamelCase words.
fn identifiers(task: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let quoted = task.split('`').skip(1).step_by(2);
    let words = task.split(|c: char| !(c.is_alphanumeric() || c == '_'));
    for w in quoted.chain(words) {
        let w = w.trim_end_matches("()");
        let code_like = w.contains('_')
            || w.chars().skip(1).any(char::is_uppercase)
            || task.contains(&format!("`{w}`"))
            || task.contains(&format!("`{w}()`"));
        let ident = !w.is_empty()
            && w.chars().all(|c| c.is_alphanumeric() || c == '_')
            && !w.starts_with(|c: char| c.is_ascii_digit());
        if ident && code_like && !out.iter().any(|o| o == w) {
            out.push(w.to_string());
        }
    }
    out
}

/// The diff in a reply: every ```diff (or ```patch) block, else everything from the first
/// `--- ` line.
fn extract_diff(reply: &str) -> String {
    let mut blocks = Vec::new();
    let mut rest = reply;
    while let Some(i) = rest.find("```") {
        let after = &rest[i + 3..];
        let (lang, body) = after.split_once('\n').unwrap_or((after, ""));
        let Some(end) = body.find("```") else { break };
        if matches!(lang.trim(), "diff" | "patch")
            || body.starts_with("--- ")
            || body.starts_with("diff --git")
        {
            blocks.push(&body[..end]);
        }
        rest = &body[end + 3..];
    }
    let diff = match blocks.is_empty() {
        false => blocks.concat(),
        true => match reply
            .find("\n--- ")
            .map(|i| i + 1)
            .or(reply.starts_with("--- ").then_some(0))
        {
            Some(i) => reply[i..].to_string(),
            None => String::new(),
        },
    };
    match diff.ends_with('\n') || diff.is_empty() {
        true => diff,
        false => diff + "\n",
    }
}

/// Paths a diff touches, from its `---`/`+++` headers, with any a/ or b/ prefix removed.
fn touched(diff: &str) -> BTreeSet<String> {
    diff.lines()
        .filter_map(|l| l.strip_prefix("+++ ").or_else(|| l.strip_prefix("--- ")))
        .map(|p| p.split('\t').next().unwrap_or(p).trim())
        .filter(|p| *p != "/dev/null")
        .map(|p| {
            p.strip_prefix("a/")
                .or_else(|| p.strip_prefix("b/"))
                .unwrap_or(p)
                .to_string()
        })
        .collect()
}

/// `diff` without the file sections that touch any of `files`. Sections start at `diff --git`
/// lines, or at `--- ` headers followed by `+++ ` when the diff has no `diff --git` lines.
fn without_files(diff: &str, files: &[String]) -> String {
    let lines: Vec<&str> = diff.split_inclusive('\n').collect();
    let git_style = lines.iter().any(|l| l.starts_with("diff --git "));
    let starts: Vec<usize> = (0..lines.len())
        .filter(|&i| match git_style {
            true => lines[i].starts_with("diff --git "),
            false => {
                lines[i].starts_with("--- ")
                    && lines.get(i + 1).is_some_and(|n| n.starts_with("+++ "))
            }
        })
        .collect();
    let mut out: String = lines[..starts.first().copied().unwrap_or(lines.len())].concat();
    for (k, &start) in starts.iter().enumerate() {
        let section = lines[start..starts.get(k + 1).copied().unwrap_or(lines.len())].concat();
        if !touched(&section).iter().any(|p| files.contains(p)) {
            out += &section;
        }
    }
    out
}

/// Applies the candidate in `dir` (git apply, else `patch` with fuzz) and records the result.
fn apply_candidate(dir: &Path, a: &mut Attempt, protect: &[String]) -> Result<()> {
    if a.raw_diff.trim().is_empty() {
        a.rejected = Some(format!("no diff (finish: {})", a.finish_reason));
        return Ok(());
    }
    // Edits to protected files (the tests) are dropped; the rest of the patch still counts.
    let diff = without_files(&a.raw_diff, protect);
    a.dropped_protected = diff != a.raw_diff;
    if diff.trim().is_empty() {
        a.rejected = Some("only edits protected files".into());
        return Ok(());
    }
    let strip = match diff.lines().any(|l| l.starts_with("+++ b/")) {
        true => "-p1",
        false => "-p0",
    };
    if git_apply(dir, &diff, &[strip]).is_err() {
        let patch_file = dir.join(".rlis-fix.patch");
        fs::write(&patch_file, &diff)?;
        let file = patch_file.to_str().context("non-UTF-8 path")?;
        let patch = |check: bool| -> Result<bool> {
            let mut cmd = Command::new("patch");
            cmd.args([strip, "-F", "3", "-N", "-t", "-s", "-i", file])
                .current_dir(dir)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null());
            if check {
                cmd.arg("-C");
            }
            Ok(cmd.status()?.success())
        };
        let ok = patch(true)? && patch(false)?;
        fs::remove_file(&patch_file)?;
        if !ok {
            a.rejected = Some("did not apply".into());
            return Ok(());
        }
        a.fuzzy = true;
    }
    a.diff = git(dir, &["diff"])?;
    if a.diff.is_empty() {
        a.rejected = Some("applied, but changed nothing".into());
        return Ok(());
    }
    if let Some(p) = touched(&a.diff).iter().find(|p| protect.contains(p)) {
        a.rejected = Some(format!("touches {p}"));
        return Ok(());
    }
    a.applied = true;
    a.lines_changed = a
        .diff
        .lines()
        .filter(|l| {
            (l.starts_with('+') && !l.starts_with("+++"))
                || (l.starts_with('-') && !l.starts_with("---"))
        })
        .count();
    Ok(())
}

fn git_apply(dir: &Path, diff: &str, extra: &[&str]) -> Result<()> {
    let mut child = Command::new("git")
        .args(["apply", "--recount", "--whitespace=nowarn"])
        .args(extra)
        .current_dir(dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()?;
    std::io::Write::write_all(child.stdin.as_mut().expect("piped"), diff.as_bytes())?;
    let out = child.wait_with_output()?;
    anyhow::ensure!(
        out.status.success(),
        "git apply: {}",
        String::from_utf8_lossy(&out.stderr).trim()
    );
    Ok(())
}

/// Runs git in `dir` and returns its stdout; an error carries stderr.
fn git(dir: &Path, args: &[&str]) -> Result<String> {
    let out = Command::new("git")
        .args(args)
        .current_dir(dir)
        .stdin(Stdio::null())
        .output()
        .context("running git")?;
    anyhow::ensure!(
        out.status.success(),
        "git {}: {}",
        args.join(" "),
        String::from_utf8_lossy(&out.stderr).trim()
    );
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// The last `max` bytes of `s` (on a line boundary when possible).
fn tail(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut start = s.len() - max;
    while !s.is_char_boundary(start) {
        start += 1;
    }
    let cut = &s[start..];
    let cut = cut.find('\n').map_or(cut, |i| &cut[i + 1..]);
    format!("[...]\n{cut}")
}

fn unix_millis() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_fenced_diffs() {
        let reply = "Here:\n```diff\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n```\nDone.";
        assert_eq!(
            extract_diff(reply),
            "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n"
        );
        let bare = "Fix:\n--- x.py\n+++ x.py\n@@ -1 +1 @@\n-a\n+b";
        assert_eq!(
            extract_diff(bare),
            "--- x.py\n+++ x.py\n@@ -1 +1 @@\n-a\n+b\n"
        );
        assert_eq!(extract_diff("```python\nprint(1)\n```"), "");
    }

    #[test]
    fn drops_protected_sections() {
        let src = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n";
        let test = "diff --git a/t.py b/t.py\n--- a/t.py\n+++ b/t.py\n@@ -1 +1 @@\n-c\n+d\n";
        let protect = ["t.py".to_string()];
        assert_eq!(without_files(&format!("{src}{test}"), &protect), src);
        assert_eq!(without_files(&format!("{test}{src}"), &protect), src);
        // Plain `---`/`+++` headers, with a removed line that itself starts with dashes.
        let plain = "--- x.py\n+++ x.py\n@@ -1,2 +1 @@\n--- old\n-a\n--- t.py\n+++ t.py\n@@ -1 +1 @@\n-c\n+d\n";
        assert_eq!(
            without_files(plain, &protect),
            "--- x.py\n+++ x.py\n@@ -1,2 +1 @@\n--- old\n-a\n"
        );
    }

    #[test]
    fn touched_strips_prefixes() {
        let d = "--- a/src/x.py\t2024\n+++ b/src/x.py\n--- /dev/null\n+++ b/new.py\n";
        assert_eq!(
            touched(d).into_iter().collect::<Vec<_>>(),
            ["new.py", "src/x.py"]
        );
    }

    #[test]
    fn identifiers_are_code_like_words() {
        assert_eq!(
            identifiers("`median()` is wrong for even lists; parse_date and HttpClient too"),
            ["median", "parse_date", "HttpClient"]
        );
    }
}
