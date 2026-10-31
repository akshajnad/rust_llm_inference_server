//! Shell commands for `rlis code` and `rlis fix`: `sh -c` in a directory, stdout and stderr
//! merged and streamed as they arrive, in a process group of its own so a timeout or the user
//! stopping it kills everything it started.

use std::io::Read;
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

/// Output kept from one command; the middle of anything longer is dropped.
const MAX_KEEP: usize = 1 << 20;
/// How long to wait for the rest of the output once the command has exited.
const DRAIN: Duration = Duration::from_millis(200);

pub struct Run {
    /// `None` when it was killed (timed out or stopped).
    pub exit: Option<i32>,
    pub output: String,
    pub secs: f64,
    pub timed_out: bool,
    pub stopped: bool,
}

impl Run {
    pub fn passed(&self) -> bool {
        self.exit == Some(0)
    }
}

/// Runs `cmd` in `dir`. `tick` is called every few milliseconds with the output that arrived
/// since the last call (often none); returning true kills the command.
pub fn run(
    dir: &Path,
    cmd: &str,
    timeout: Duration,
    mut tick: impl FnMut(&str) -> bool,
) -> Result<Run> {
    let start = Instant::now();
    // `exec 2>&1` sends the script's stderr to the same pipe, in order.
    let mut child = Command::new("sh")
        .arg("-c")
        .arg(format!("exec 2>&1\n{cmd}"))
        .current_dir(dir)
        .env("TERM", "dumb")
        .env("NO_COLOR", "1")
        .env("PAGER", "cat")
        .env("GIT_PAGER", "cat")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .process_group(0)
        .spawn()
        .with_context(|| format!("running {cmd:?}"))?;
    let mut pipe = child.stdout.take().expect("stdout is piped");
    let (tx, rx) = mpsc::channel::<Vec<u8>>();
    thread::spawn(move || {
        let mut buf = [0u8; 8192];
        while let Ok(n) = pipe.read(&mut buf) {
            if n == 0 || tx.send(buf[..n].to_vec()).is_err() {
                break;
            }
        }
    });

    let mut out = Output::default();
    let (mut timed_out, mut stopped) = (false, false);
    let status = loop {
        let fresh: String = rx.try_iter().map(|bytes| out.push(&bytes)).collect();
        let stop = tick(&fresh);
        if let Some(status) = child.try_wait()? {
            break Some(status);
        }
        timed_out = start.elapsed() > timeout;
        if timed_out || stop {
            stopped = !timed_out;
            kill_group(&mut child);
            break None;
        }
        thread::sleep(Duration::from_millis(20));
    };
    // What the command wrote just before it ended; anything it left running in the background
    // (which could hold the pipe open forever) goes with its group.
    let deadline = Instant::now() + DRAIN;
    while let Ok(bytes) = rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
        tick(&out.push(&bytes));
    }
    if status.is_some() {
        kill_group(&mut child);
    }
    Ok(Run {
        exit: status.and_then(|s| s.code()),
        output: out.finish(),
        secs: start.elapsed().as_secs_f64(),
        timed_out,
        stopped,
    })
}

fn kill_group(child: &mut Child) {
    let _ = Command::new("kill")
        .args(["-KILL", &format!("-{}", child.id())])
        .stderr(Stdio::null())
        .status();
    let _ = child.wait();
}

/// The output so far, decoded as it arrives (a UTF-8 character may be split across reads).
#[derive(Default)]
struct Output {
    text: String,
    pending: Vec<u8>,
    dropped: usize,
}

impl Output {
    /// Adds `bytes`; returns the text they complete.
    fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let valid = match std::str::from_utf8(&self.pending) {
            Ok(_) => self.pending.len(),
            // An incomplete character at the end waits for the next read; anything else
            // invalid is replaced.
            Err(e) if e.error_len().is_none() => e.valid_up_to(),
            Err(_) => self.pending.len(),
        };
        let done: Vec<u8> = self.pending.drain(..valid).collect();
        let s = String::from_utf8_lossy(&done).into_owned();
        self.text += &s;
        if self.text.len() > MAX_KEEP {
            let head = floor(&self.text, MAX_KEEP / 4);
            let cut = floor(&self.text, head + self.text.len() - MAX_KEEP / 2);
            self.dropped += cut - head;
            self.text.replace_range(head..cut, "");
        }
        s
    }

    fn finish(mut self) -> String {
        if !self.pending.is_empty() {
            let rest = std::mem::take(&mut self.pending);
            self.text += &String::from_utf8_lossy(&rest);
        }
        if self.dropped > 0 {
            let head = floor(&self.text, MAX_KEEP / 4);
            let note = format!("\n[... {} bytes of output dropped ...]\n", self.dropped);
            self.text.insert_str(head, &note);
        }
        self.text
    }
}

/// The largest char boundary at or below `i`.
fn floor(s: &str, i: usize) -> usize {
    (0..=i.min(s.len()))
        .rev()
        .find(|&i| s.is_char_boundary(i))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sh(cmd: &str, timeout: u64) -> Result<Run> {
        run(
            &std::env::temp_dir(),
            cmd,
            Duration::from_secs(timeout),
            |_| false,
        )
    }

    #[test]
    fn captures_output_and_exit() -> Result<()> {
        let r = sh("echo out; echo err >&2; exit 3", 10)?;
        assert_eq!(r.exit, Some(3));
        assert_eq!(r.output, "out\nerr\n");
        assert!(!r.passed() && !r.timed_out && !r.stopped);
        let mut streamed = String::new();
        let r = run(
            &std::env::temp_dir(),
            "printf 'a\\nb\\n'",
            Duration::from_secs(10),
            |s| {
                streamed += s;
                false
            },
        )?;
        assert!(r.passed());
        assert_eq!(streamed, "a\nb\n");
        Ok(())
    }

    #[test]
    fn kills_on_timeout_and_stop() -> Result<()> {
        let r = sh("echo started; sleep 30", 1)?;
        assert!(r.timed_out && r.exit.is_none() && r.secs < 10.0);
        assert_eq!(r.output, "started\n");
        let start = Instant::now();
        let r = run(
            &std::env::temp_dir(),
            "sleep 30",
            Duration::from_secs(60),
            |_| start.elapsed() > Duration::from_millis(300),
        )?;
        assert!(r.stopped && !r.timed_out && r.secs < 10.0);
        // A background child doesn't keep the run waiting.
        let r = sh("sleep 30 & echo done", 10)?;
        assert!(r.passed() && r.secs < 5.0, "{}", r.secs);
        Ok(())
    }

    #[test]
    fn keeps_long_output_bounded() {
        let mut out = Output::default();
        for _ in 0..300 {
            out.push(&[b'x'; 8192]);
        }
        out.push("é".as_bytes()[..1].as_ref());
        out.push("é".as_bytes()[1..].as_ref());
        let text = out.finish();
        assert!(text.len() < MAX_KEEP + 100 && text.contains("bytes of output dropped"));
        assert!(text.ends_with('é'));
    }
}
