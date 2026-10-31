//! How well a model drives `rlis code`: small coding tasks, each in a fresh directory, run
//! through the agent's own parser, tools, nudges and commands (changes approved, commands run;
//! ones that need approval are declined), then checked on disk. Not run by `cargo test`:
//!
//! ```sh
//! RLIS_PROBE_URL=http://127.0.0.1:8000 cargo test --release -p chat probe -- --ignored --nocapture
//! ```
//!
//! It reports each task's outcome, decode speed, and the server's peak memory footprint.

use std::fs;
use std::net::Shutdown;
use std::path::Path;
use std::process::Command;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use serde_json::{Value, json};

use crate::api::{self, Update};
use crate::models::Model;
use crate::tools::{self, Action, Nudges, Reply, Workspace};
use crate::{http, shell};

const MAX_STEPS: usize = 8;
const MAX_TOKENS: usize = 3000;

struct Task {
    name: &'static str,
    prompt: &'static str,
    /// Files the project starts with.
    setup: &'static [(&'static str, &'static str)],
    /// Whether the task got done, from the directory afterwards and the commands that ran.
    check: fn(&Path, &[String]) -> bool,
}

/// Files under `dir` with extension `ext`, and their text.
fn with_ext(dir: &Path, ext: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for e in walk(dir) {
        if e.ends_with(&format!(".{ext}")) {
            out.push((
                e.clone(),
                fs::read_to_string(dir.join(&e)).unwrap_or_default(),
            ));
        }
    }
    out
}

fn walk(dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if let Ok(rel) = p.strip_prefix(dir) {
                out.push(rel.display().to_string());
            }
        }
    }
    out
}

fn passes(dir: &Path, cmd: &str) -> bool {
    shell::run(dir, cmd, Duration::from_secs(30), |_| false).is_ok_and(|r| r.passed())
}

const STATS: &str = "def median(xs):\n    s = sorted(xs)\n    n = len(s)\n    return s[n // 2]\n";
const TEST_STATS: &str = "import unittest\nfrom stats import median\n\n\nclass T(unittest.TestCase):\n    def test_odd(self):\n        self.assertEqual(median([3, 1, 2]), 2)\n\n    def test_even(self):\n        self.assertEqual(median([4, 1, 3, 2]), 2.5)\n\n\nif __name__ == \"__main__\":\n    unittest.main()\n";
const UTILS: &str = "def add(a, b):\n    return a + b\n";

fn tasks() -> Vec<Task> {
    vec![
        Task {
            name: "fibonacci script",
            prompt: "write a python script that prints the first 20 fibonacci numbers",
            setup: &[],
            check: |d, _| {
                with_ext(d, "py").iter().any(|(f, _)| {
                    let out = Command::new("python3").arg(f).current_dir(d).output();
                    out.is_ok_and(|o| String::from_utf8_lossy(&o.stdout).contains("4181"))
                })
            },
        },
        Task {
            name: "snake game",
            prompt: "make me a snake game",
            setup: &[],
            check: |d, _| {
                ["py", "html", "js"]
                    .iter()
                    .flat_map(|e| with_ext(d, e))
                    .any(|(_, t)| t.lines().count() >= 30)
            },
        },
        Task {
            name: "todo app (html+js)",
            prompt: "code a todo list app in html and javascript",
            setup: &[],
            check: |d, _| !with_ext(d, "html").is_empty(),
        },
        Task {
            name: "rust word count",
            prompt: "create a rust program that counts words in a file",
            setup: &[],
            check: |d, _| with_ext(d, "rs").iter().any(|(_, t)| t.contains("fn main")),
        },
        Task {
            name: "go hello world",
            prompt: "write hello world in go",
            setup: &[],
            check: |d, _| with_ext(d, "go").iter().any(|(_, t)| t.contains("ello")),
        },
        Task {
            name: "add to existing file",
            prompt: "add a function to utils.py that reverses a string",
            setup: &[("utils.py", UTILS)],
            check: |d, _| {
                let t = fs::read_to_string(d.join("utils.py")).unwrap_or_default();
                t.contains("def add(a, b)") && (t.contains("[::-1]") || t.contains("reversed("))
            },
        },
        Task {
            name: "run the tests",
            prompt: "run the tests",
            setup: &[
                ("stats.py", "def mean(xs):\n    return sum(xs) / len(xs)\n"),
                (
                    "test_stats.py",
                    "import unittest\nfrom stats import mean\n\n\nclass T(unittest.TestCase):\n    def test_mean(self):\n        self.assertEqual(mean([1, 2, 3]), 2)\n",
                ),
            ],
            check: |_, ran| {
                ran.iter()
                    .any(|c| c.contains("unittest") || c.contains("pytest"))
            },
        },
        Task {
            name: "fix a failing test",
            prompt: "the median function is wrong for even-length lists; fix it",
            setup: &[("stats.py", STATS), ("test_stats.py", TEST_STATS)],
            check: |d, _| passes(d, "python3 -m unittest -q"),
        },
        Task {
            name: "delete a file",
            prompt: "delete old.py",
            setup: &[("old.py", "x = 1\n"), ("main.py", "print(1)\n")],
            check: |_, ran| ran.iter().any(|c| c.contains("rm") && c.contains("old.py")),
        },
        Task {
            name: "question (no changes)",
            prompt: "what does utils.py do?",
            setup: &[("utils.py", UTILS)],
            check: |d, _| {
                with_ext(d, "py").len() == 1
                    && fs::read_to_string(d.join("utils.py")).unwrap_or_default() == UTILS
            },
        },
    ]
}

/// One streamed reply, stopped at the parser's cut like the chat does: its text, tokens and
/// decode speed.
fn generate(
    url: &str,
    root: &Path,
    messages: &[Value],
    native: bool,
) -> Result<(String, u64, Option<f64>)> {
    let max_tokens = std::env::var("RLIS_PROBE_MAX_TOKENS")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(MAX_TOKENS);
    let body = json!({
        "messages": messages,
        "stream": true,
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": max_tokens,
    });
    let body = serde_json::to_vec(&body)?;
    let (tx, rx) = mpsc::channel();
    let url = url.to_string();
    thread::spawn(move || api::stream(&url, &body, native, &tx));
    let mut reply = Reply::new(root);
    let (mut pieces, mut first, mut last, mut finish) = (0u64, None, Instant::now(), None);
    let mut socket = None;
    for update in rx {
        match update {
            Update::Connected(s) => socket = Some(s),
            Update::Thinking => {
                pieces += 1;
                first.get_or_insert_with(Instant::now);
                last = Instant::now();
            }
            Update::Text(t) if !reply.cut() => {
                pieces += 1;
                first.get_or_insert_with(Instant::now);
                last = Instant::now();
                reply.push(&t);
                if reply.cut()
                    && let Some(s) = &socket
                {
                    let _ = s.shutdown(Shutdown::Both);
                }
            }
            Update::Finish(f) => finish = Some(f),
            Update::Rejected { message, .. } => anyhow::bail!("{message}"),
            Update::End(_) => break,
            _ => {}
        }
    }
    reply.finish(finish.as_deref() == Some("length"));
    let secs = first.map(|f| last.duration_since(f).as_secs_f64());
    let rate = secs
        .filter(|&s| s > 0.0 && pieces > 1)
        .map(|s| (pieces - 1) as f64 / s);
    Ok((reply.text().to_string(), pieces, rate))
}

/// Runs one task like `rlis code --yes` would (declining commands that need approval);
/// returns the commands it ran, its tokens and decode rates.
fn attempt(
    url: &str,
    dir: &Path,
    prompt: &str,
    log: &mut String,
) -> Result<(Vec<String>, u64, Vec<f64>)> {
    let mut ws = Workspace::new(dir.to_path_buf());
    let native = crate::model_name(url)
        .and_then(|m| Model::from_served(&m))
        .is_some_and(Model::native);
    let system = ws.system_prompt(None);
    let system = if native {
        crate::native::system(&system)
    } else {
        system
    };
    let mut messages = vec![
        json!({"role": "system", "content": system}),
        json!({"role": "user", "content": prompt}),
    ];
    let mut nudges = Nudges::default();
    let (mut ran, mut tokens, mut rates) = (Vec::new(), 0, Vec::new());
    for _ in 0..MAX_STEPS {
        let (text, n, rate) = generate(url, dir, &messages, native)?;
        tokens += n;
        rates.extend(rate);
        *log += &format!("--- reply\n{text}\n");
        // The chat stops a model that says again what it said before.
        if messages
            .iter()
            .any(|m| m["role"] == "assistant" && m["content"] == text.as_str())
        {
            *log += "--- stopped: repeated reply\n";
            break;
        }
        let reply = Reply::read(&text, Some(dir));
        let unsaved = reply.unsaved_code();
        let requests = reply.into_requests();
        messages.push(json!({"role": "assistant", "content": text}));
        if requests.is_empty() {
            match nudges.next(&text, unsaved, prompt) {
                Some((note, message)) => {
                    *log += &format!("--- nudge: {note}\n");
                    messages.push(json!({"role": "user", "content": message}));
                    continue;
                }
                None => break,
            }
        }
        nudges.saw(&requests);
        let mut results = Vec::new();
        for r in &requests {
            let result = match ws.run(r) {
                Err(e) => format!("Error: {e:#}"),
                Ok(Action::Done { output, .. } | Action::Unchanged { output }) => output,
                Ok(Action::Change(c)) => ws.apply(&c).unwrap_or_else(|e| format!("Error: {e:#}")),
                Ok(Action::Run {
                    command,
                    risk: Some(_),
                }) => {
                    ran.push(command.clone());
                    format!("The user declined to run `{command}`; it did not run.")
                }
                Ok(Action::Run {
                    command,
                    risk: None,
                }) => {
                    ran.push(command.clone());
                    let run = shell::run(dir, &command, Duration::from_secs(20), |_| false)?;
                    match run.exit {
                        Some(code) => format!(
                            "Output of `{command}` (exit {code}):\n```\n{}\n```",
                            run.output.trim_end()
                        ),
                        None => format!(
                            "`{command}` was killed after 20s; it may wait for input or never end."
                        ),
                    }
                }
            };
            *log += &format!(
                "--- {}\n{}\n",
                r.title(),
                result.lines().next().unwrap_or_default()
            );
            results.push(result);
        }
        results.push(tools::restate(prompt));
        messages.push(json!({"role": "user", "content": results.join("\n\n")}));
    }
    Ok((ran, tokens, rates))
}

fn peak_footprint(url: &str) -> Option<f64> {
    let text = http::request(url, "GET", "/metrics", None)
        .ok()?
        .text()
        .ok()?;
    let line = text
        .lines()
        .find(|l| l.starts_with("rlis_process_memory_footprint_peak_bytes"))?;
    let bytes: f64 = line.split_whitespace().last()?.parse().ok()?;
    Some(bytes / 1e9)
}

#[test]
#[ignore = "needs a running server: RLIS_PROBE_URL"]
fn probe() -> Result<()> {
    let url = std::env::var("RLIS_PROBE_URL").expect("set RLIS_PROBE_URL");
    let only = std::env::var("RLIS_PROBE_ONLY").ok();
    let model: Value =
        serde_json::from_str(&http::request(&url, "GET", "/v1/models", None)?.text()?)?;
    let model = model["data"][0]["id"].as_str().unwrap_or("?").to_string();
    let base = std::env::temp_dir().join(format!("rlis-probe-{}", std::process::id()));
    let (mut passed, mut total_tokens, mut all_rates, mut total) = (0, 0, Vec::new(), 0);
    let start = Instant::now();
    for (i, task) in tasks().iter().enumerate() {
        if only.as_deref().is_some_and(|o| !task.name.contains(o)) {
            continue;
        }
        total += 1;
        let dir = base.join(i.to_string());
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir)?;
        for (f, text) in task.setup {
            fs::write(dir.join(f), text)?;
        }
        let mut log = String::new();
        let t = Instant::now();
        let (ok, tokens, rates) = match attempt(&url, &dir, task.prompt, &mut log) {
            Ok((ran, tokens, rates)) => ((task.check)(&dir, &ran), tokens, rates),
            Err(e) => {
                log += &format!("--- error: {e:#}\n");
                (false, 0, Vec::new())
            }
        };
        passed += ok as usize;
        total_tokens += tokens;
        all_rates.extend(rates);
        println!(
            "{} {:<24} {:>5} tokens {:>6.1}s",
            if ok { "PASS" } else { "FAIL" },
            task.name,
            tokens,
            t.elapsed().as_secs_f64()
        );
        if !ok || std::env::var("RLIS_PROBE_LOG").is_ok() {
            for l in log.lines().take(60) {
                println!("      {l}");
            }
        }
    }
    all_rates.sort_by(f64::total_cmp);
    let median = all_rates.get(all_rates.len() / 2).copied().unwrap_or(0.0);
    println!(
        "{model}: {passed}/{total} passed · {total_tokens} tokens in {:.0}s · decode median {median:.1} tok/s · peak footprint {}",
        start.elapsed().as_secs_f64(),
        peak_footprint(&url).map_or("?".into(), |g| format!("{g:.2} GB"))
    );
    let _ = fs::remove_dir_all(&base);
    Ok(())
}
