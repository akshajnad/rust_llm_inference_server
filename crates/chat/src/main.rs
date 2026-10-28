//! `rlis-chat`: chat with an rlis server (or any OpenAI-compatible one) in the terminal. Starts
//! `rlis serve` in the background when nothing answers at `--url`, and stops it on exit.
//! `make install` puts it on the PATH as `rlis`, so it also hands the server binary's own
//! subcommands (`rlis serve`, ...) through to it.

mod api;
mod frame;
mod http;
mod input;
mod render;

use std::env;
use std::fmt::Display;
use std::fs::{self, File};
use std::io::{self, Stdout, Write};
use std::net::{Shutdown, TcpStream};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::Parser;
use crossterm::cursor::{MoveTo, MoveToColumn, MoveUp};
use crossterm::event::{
    self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyCode, KeyEventKind, KeyModifiers,
    KeyboardEnhancementFlags, PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use crossterm::style::{Color, Stylize};
use crossterm::terminal::{self, Clear, ClearType};
use crossterm::{execute, queue};
use serde_json::{Value, json};
use unicode_width::UnicodeWidthStr;

use api::Update;
use frame::BORDER;
use input::{Command as SlashCommand, Editor, Input};
use render::Renderer;

/// The model's color: its label, the bar beside its replies, headings and the status meter.
pub const ACCENT: Color = Color::AnsiValue(43);
/// The user's color: their label and the prompt marker.
pub const USER: Color = Color::AnsiValue(141);
/// The prompt card's border, set apart from the transcript's grey ones.
pub const FOCUS: Color = Color::AnsiValue(103);

const COMMANDS: &[SlashCommand] = &[
    ("/clear", "", "start a new conversation"),
    ("/retry", "", "regenerate the last reply"),
    (
        "/system",
        "<prompt|none>",
        "show, set or remove the system prompt",
    ),
    ("/temp", "<0-2>", "show or set the sampling temperature"),
    ("/help", "", "commands, keys and settings"),
    ("/exit", "", "quit"),
];

#[derive(Parser)]
#[command(
    name = "rlis",
    about = "Chat with an rlis server in the terminal",
    after_help = "`rlis serve`, `rlis generate` and `rlis perplexity` run the server binary."
)]
struct Args {
    /// Server to talk to; a local one is started if nothing answers there.
    #[arg(long, default_value = "http://127.0.0.1:8000")]
    url: String,
    #[arg(long)]
    system: Option<String>,
    #[arg(long, default_value_t = 0.6)]
    temperature: f32,
    #[arg(long, default_value_t = 0.9)]
    top_p: f32,
    /// Longest reply; by default a reply may use the rest of the context window.
    #[arg(long)]
    max_tokens: Option<usize>,
    /// Exit instead of starting a server when none is running.
    #[arg(long)]
    no_start: bool,
    /// Extra arguments for `rlis serve` when starting one, e.g. `-- --kv-cache-gb 2`.
    #[arg(last = true)]
    serve_args: Vec<String>,
}

/// The server binary's subcommands, run in place of the chat.
const SERVER_COMMANDS: &[&str] = &["serve", "generate", "perplexity"];

fn main() -> Result<()> {
    if env::args()
        .nth(1)
        .is_some_and(|a| SERVER_COMMANDS.contains(&a.as_str()))
    {
        return run_server_command();
    }
    let args = Args::parse();
    let _server = ensure_server(&args)?;
    let model = model_name(&args.url).unwrap_or_else(|| "unknown model".to_string());
    let _terminal = Terminal::enter()?;
    let mut chat = Chat {
        url: args.url,
        model,
        system: args.system,
        temperature: args.temperature,
        top_p: args.top_p,
        max_tokens: args.max_tokens,
        messages: Vec::new(),
        editor: Editor::new(COMMANDS),
        out: io::stdout(),
    };
    chat.run()
}

/// Raw mode, bracketed paste and (where the terminal supports it) unambiguous modifier keys
/// for the life of the session; restored on drop and on panic.
struct Terminal {
    enhanced: bool,
}

impl Terminal {
    fn enter() -> Result<Self> {
        terminal::enable_raw_mode()?;
        let enhanced = terminal::supports_keyboard_enhancement().unwrap_or(false);
        let mut out = io::stdout();
        execute!(out, EnableBracketedPaste)?;
        if enhanced {
            execute!(
                out,
                PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
            )?;
        }
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            restore(enhanced);
            hook(info);
        }));
        Ok(Self { enhanced })
    }
}

fn restore(enhanced: bool) {
    let mut out = io::stdout();
    if enhanced {
        let _ = execute!(out, PopKeyboardEnhancementFlags);
    }
    let _ = execute!(out, DisableBracketedPaste);
    let _ = terminal::disable_raw_mode();
}

impl Drop for Terminal {
    fn drop(&mut self) {
        restore(self.enhanced);
    }
}

/// A server this process started; killed when dropped.
struct Server(Child);

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn healthy(url: &str) -> bool {
    http::request(url, "GET", "/health", None).is_ok_and(|r| r.status == 200)
}

fn model_name(url: &str) -> Option<String> {
    let text = http::request(url, "GET", "/v1/models", None)
        .ok()?
        .text()
        .ok()?;
    let v: Value = serde_json::from_str(&text).ok()?;
    v["data"][0]["id"].as_str().map(str::to_string)
}

/// The directory `rlis serve` runs in, so its default `models/...` paths resolve: the repo that
/// holds this binary (`<repo>/target/release`), else the current directory.
fn server_dir(exe: &Path) -> PathBuf {
    exe.ancestors()
        .nth(3)
        .filter(|repo| repo.join("models").is_dir())
        .map_or_else(|| env::current_dir().unwrap_or_default(), Path::to_path_buf)
}

/// Replaces this process with the server binary, run from the repo so its default
/// `models/...` paths resolve wherever the command was typed.
fn run_server_command() -> Result<()> {
    let exe = env::current_exe()?.canonicalize()?;
    let bin = server_bin(&exe)?;
    let err = Command::new(&bin)
        .args(env::args_os().skip(1))
        .current_dir(server_dir(&exe))
        .exec();
    Err(err).with_context(|| format!("running {}", bin.display()))
}

fn server_bin(exe: &Path) -> Result<PathBuf> {
    let bin = exe.with_file_name("rlis");
    if !bin.exists() {
        bail!(
            "{} not found; build both binaries with `make chat`",
            bin.display()
        );
    }
    Ok(bin)
}

fn ensure_server(args: &Args) -> Result<Option<Server>> {
    if healthy(&args.url) {
        return Ok(None);
    }
    let (host, port) = http::authority(&args.url)?
        .rsplit_once(':')
        .context("--url needs a port")?;
    if args.no_start || !matches!(host, "127.0.0.1" | "localhost") {
        bail!("no server is answering at {}", args.url);
    }
    // Resolved through the symlink `make install` puts on the PATH, to find the build.
    let exe = env::current_exe()?.canonicalize()?;
    let bin = server_bin(&exe)?;
    let log_path = exe.with_file_name("rlis-chat-server.log");
    let log =
        File::create(&log_path).with_context(|| format!("creating {}", log_path.display()))?;
    let mut cmd = Command::new(&bin);
    cmd.arg("serve");
    if !args.serve_args.iter().any(|a| a.starts_with("--port")) {
        cmd.args(["--port", port]);
    }
    cmd.args(&args.serve_args)
        .current_dir(server_dir(&exe))
        .stdin(Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log);
    let mut server = Server(
        cmd.spawn()
            .with_context(|| format!("starting {}", bin.display()))?,
    );

    let start = Instant::now();
    let mut out = io::stdout();
    while !healthy(&args.url) {
        if let Some(status) = server.0.try_wait()? {
            print!("\r\x1b[2K");
            let log = fs::read_to_string(&log_path).unwrap_or_default();
            let lines: Vec<&str> = log.lines().collect();
            let tail = lines[lines.len().saturating_sub(15)..].join("\n");
            bail!("the server exited ({status}) while starting:\n{tail}");
        }
        if start.elapsed() > Duration::from_secs(300) {
            bail!(
                "the server did not come up in 5 minutes; see {}",
                log_path.display()
            );
        }
        print!(
            "\r{} {}  {}",
            meter(start),
            "starting rlis server".with(ACCENT),
            format!("{}s · loading the model", start.elapsed().as_secs()).dark_grey()
        );
        out.flush()?;
        thread::sleep(Duration::from_millis(120));
    }
    print!("\r\x1b[2K{} {}", meter(start), "warming up".with(ACCENT));
    out.flush()?;
    // The first request after startup compiles GPU pipelines (seconds); pay for it here
    // rather than on the first message.
    let warmup = br#"{"prompt": "Hello", "max_tokens": 1}"#;
    let _ =
        http::request(&args.url, "POST", "/v1/completions", Some(warmup)).and_then(|r| r.text());
    print!("\r\x1b[2K");
    out.flush()?;
    Ok(Some(server))
}

/// A five-cell bar with one lit cell sweeping back and forth.
fn meter(start: Instant) -> String {
    const CELLS: usize = 5;
    let step = (start.elapsed().as_millis() / 90) as usize % (2 * CELLS - 2);
    let lit = if step < CELLS {
        step
    } else {
        2 * CELLS - 2 - step
    };
    (0..CELLS)
        .map(|i| {
            if i == lit {
                "▰".with(ACCENT).to_string()
            } else {
                "▱".dark_grey().to_string()
            }
        })
        .collect()
}

/// Decode speed: tokens after the first over the time since it arrived.
fn rate(tokens: u64, decode: Duration) -> Option<f64> {
    let secs = decode.as_secs_f64();
    (tokens > 1 && secs > 0.0).then(|| (tokens - 1) as f64 / secs)
}

fn term_width() -> usize {
    terminal::size().map_or(80, |(w, _)| w as usize)
}

struct Message {
    role: &'static str,
    content: String,
}

enum Outcome {
    /// What the model wrote (possibly cut short); empty if nothing arrived.
    Reply(String),
    /// The conversation no longer fits the context window.
    TooLong(String),
    Failed(String),
}

struct Chat {
    url: String,
    model: String,
    system: Option<String>,
    temperature: f32,
    top_p: f32,
    max_tokens: Option<usize>,
    messages: Vec<Message>,
    editor: Editor,
    out: Stdout,
}

impl Chat {
    fn run(&mut self) -> Result<()> {
        self.banner()?;
        loop {
            let status = self.status();
            let line = match self.editor.read(&mut self.out, &status)? {
                Input::Line(line) => line,
                Input::Exit => return Ok(()),
            };
            if line.starts_with('/') {
                if !self.command(&line)? {
                    return Ok(());
                }
            } else {
                self.edge(['┌', '┐'], "you", USER, "", BORDER)?;
                for l in frame::wrap(&line, frame::text_width()) {
                    self.row(&l, l.width())?;
                }
                self.edge(['├', '┤'], "rlis", ACCENT, "", BORDER)?;
                self.messages.push(Message {
                    role: "user",
                    content: line,
                });
                self.respond()?;
            }
            self.print("")?;
        }
    }

    /// Runs one reply to the last user message inside the open card and closes it, dropping
    /// the oldest exchanges if the conversation outgrew the context window.
    fn respond(&mut self) -> Result<()> {
        loop {
            match self.generate()? {
                Outcome::Reply(text) if text.is_empty() => {
                    self.messages.pop();
                }
                Outcome::Reply(text) => self.messages.push(Message {
                    role: "assistant",
                    content: text,
                }),
                Outcome::TooLong(_) if self.messages.len() > 2 => {
                    self.messages.drain(..2);
                    self.note("Dropped the oldest exchange to fit the context window")?;
                    continue;
                }
                Outcome::TooLong(message) | Outcome::Failed(message) => {
                    self.messages.pop();
                    self.error(&message)?;
                    self.close()?;
                }
            }
            return Ok(());
        }
    }

    fn body(&self) -> Vec<u8> {
        let mut messages = Vec::new();
        if let Some(s) = &self.system {
            messages.push(json!({"role": "system", "content": s}));
        }
        messages.extend(
            self.messages
                .iter()
                .map(|m| json!({"role": m.role, "content": m.content})),
        );
        let mut body = json!({
            "model": self.model,
            "messages": messages,
            "stream": true,
            "stream_options": {"include_usage": true},
            "temperature": self.temperature,
            "top_p": self.top_p,
        });
        if let Some(n) = self.max_tokens {
            body["max_tokens"] = json!(n);
        }
        serde_json::to_vec(&body).expect("JSON values serialize")
    }

    /// Streams a reply into the open card with a live status line under it: the phase (prefill
    /// until the first token, then decoding) and speed. The card is closed with the reply's
    /// stats, except when nothing was generated, which `respond` reports. Esc or Ctrl-C
    /// interrupts by closing the connection, which also stops generation on the server.
    fn generate(&mut self) -> Result<Outcome> {
        let (tx, rx) = mpsc::channel();
        let (url, body) = (self.url.clone(), self.body());
        thread::spawn(move || api::stream(&url, &body, &tx));

        let width = frame::text_width();
        let mut r = Renderer::new(width);
        let left = frame::left(BORDER);
        let start = Instant::now();
        let mut first_token = None;
        let mut text = String::new();
        let (mut pieces, mut usage, mut finish) = (0u64, None, None);
        let (mut socket, mut interrupted, mut rejected) = (None::<TcpStream>, false, None);
        let mut writing = false;
        let error = loop {
            if event::poll(Duration::from_millis(40))?
                && let Event::Key(k) = event::read()?
                && k.kind != KeyEventKind::Release
                && (k.code == KeyCode::Esc
                    || (k.code == KeyCode::Char('c')
                        && k.modifiers.contains(KeyModifiers::CONTROL)))
            {
                interrupted = true;
                if let Some(s) = &socket {
                    let _ = s.shutdown(Shutdown::Both);
                }
            }
            let mut ended = None;
            for update in rx.try_iter() {
                match update {
                    Update::Connected(s) => {
                        if interrupted {
                            let _ = s.shutdown(Shutdown::Both);
                        }
                        socket = Some(s);
                    }
                    Update::Text(_) if interrupted => {}
                    Update::Text(t) => {
                        first_token.get_or_insert_with(|| start.elapsed());
                        pieces += 1;
                        r.push(&t);
                        text.push_str(&t);
                    }
                    Update::Finish(f) => finish = Some(f),
                    Update::Usage(n) => usage = Some(n),
                    Update::Rejected { message, param } => rejected = Some((message, param)),
                    Update::End(e) => ended = Some(e),
                }
            }

            queue!(self.out, Clear(ClearType::FromCursorDown))?;
            let s = r.take();
            if !s.is_empty() {
                if !writing {
                    write!(self.out, "{left}")?;
                    writing = true;
                }
                self.out.write_all(s.as_bytes())?;
            }
            if let Some(e) = ended {
                break e;
            }
            let (status, status_w) = status_line(start, first_token, pieces, width);
            let status = frame::row(&status, status_w, BORDER, width);
            if writing {
                write!(self.out, "\r\n{status}")?;
                queue!(self.out, MoveUp(1), MoveToColumn(r.col() as u16))?;
            } else {
                write!(self.out, "{status}")?;
                queue!(self.out, MoveToColumn(0))?;
            }
            self.out.flush()?;
        };

        let tail = r.finish();
        if !tail.is_empty() && !writing {
            write!(self.out, "{left}")?;
            writing = true;
        }
        self.out.write_all(tail.as_bytes())?;
        if writing {
            write!(self.out, "\r\n")?;
        }
        self.out.flush()?;

        if let Some((message, param)) = rejected {
            return Ok(if param.as_deref() == Some("messages") {
                Outcome::TooLong(message)
            } else {
                Outcome::Failed(message)
            });
        }
        if interrupted {
            let status = if pieces > 0 {
                format!("interrupted · {pieces} tokens")
            } else {
                "interrupted".to_string()
            };
            self.edge(['└', '┘'], "", BORDER, &status, Color::Yellow)?;
        } else if let Some(e) = error {
            if text.is_empty() {
                return Ok(Outcome::Failed(e));
            }
            self.error(&e)?;
            self.close()?;
        } else {
            let elapsed = start.elapsed();
            let n = usage.unwrap_or(pieces);
            let mut parts = vec![format!("{n} tokens")];
            if let Some(ttft) = first_token {
                if let Some(tps) = rate(n, elapsed.saturating_sub(ttft)) {
                    parts.push(format!("{tps:.0} tok/s"));
                }
                parts.push(format!("{:.2}s to first token", ttft.as_secs_f64()));
            }
            if finish.as_deref() == Some("length") {
                parts.push("stopped at the length limit".to_string());
            }
            let stats = frame::fit(&parts, width.saturating_sub(4));
            self.edge(['└', '┘'], "", BORDER, &stats, Color::DarkGrey)?;
        }
        Ok(Outcome::Reply(text))
    }

    /// Handles a slash command, answering in a card titled with it; false means quit.
    fn command(&mut self, line: &str) -> Result<bool> {
        let (name, arg) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
        let arg = arg.trim();
        match name {
            "/exit" | "/quit" => return Ok(false),
            "/clear" => {
                self.messages.clear();
                queue!(self.out, Clear(ClearType::All), MoveTo(0, 0))?;
                self.banner()?;
                // `run` prints a blank line after every command; the banner ends with one.
                queue!(self.out, MoveUp(1))?;
                return Ok(true);
            }
            _ => {}
        }
        let title = line.lines().next().unwrap_or_default();
        self.edge(['┌', '┐'], title, ACCENT, "", BORDER)?;
        match name {
            "/retry" => {
                if self.messages.last().is_some_and(|m| m.role == "assistant") {
                    self.messages.pop();
                    // The reply closes the card.
                    self.respond()?;
                    return Ok(true);
                }
                self.note("Nothing to retry yet")?;
            }
            "/system" => match arg {
                "" => {
                    let current = self.system.clone().unwrap_or_else(|| "(none)".to_string());
                    self.note(&format!("System prompt: {current}"))?;
                }
                "none" => {
                    self.system = None;
                    self.note("System prompt removed")?;
                }
                s => {
                    self.system = Some(s.to_string());
                    self.note("System prompt set")?;
                }
            },
            "/temp" => match arg.parse::<f32>() {
                Ok(t) if (0.0..=2.0).contains(&t) => {
                    self.temperature = t;
                    self.note(&format!("Temperature set to {t}"))?;
                }
                _ if arg.is_empty() => {
                    self.note(&format!("Temperature is {}", self.temperature))?
                }
                _ => self.error("The temperature must be a number from 0 to 2")?,
            },
            "/help" => self.help()?,
            _ => self.error(&format!("Unknown command {name}; /help lists them"))?,
        }
        self.close()?;
        Ok(true)
    }

    fn banner(&mut self) -> Result<()> {
        let width = term_width();
        // " rlis " and the spaces around it.
        let room = width.saturating_sub(12 + self.model.width());
        let url = if self.url.width() < room {
            self.url.clone()
        } else {
            String::new()
        };
        self.print("")?;
        self.print(format!(
            " {}  {}  {}",
            " rlis ".black().on(ACCENT).bold(),
            self.model.as_str().bold(),
            url.dark_grey()
        ))?;
        let tips = [
            "local inference chat",
            "type / for commands",
            "esc stops a reply",
        ];
        let tips = frame::fit(&tips, width.saturating_sub(4));
        self.print(format!("  {}", tips.dark_grey()))?;
        self.print("")
    }

    /// The settings shown in the prompt card's top edge.
    fn status(&self) -> String {
        let mut parts = vec![self.model.clone(), format!("temp {}", self.temperature)];
        if self.system.is_some() {
            parts.push("system prompt".to_string());
        }
        parts.push(match self.messages.len() {
            0 => "new chat".to_string(),
            1 => "1 message".to_string(),
            n => format!("{n} messages"),
        });
        parts.join(" · ")
    }

    fn help(&mut self) -> Result<()> {
        let label = |c: &str, a: &str| {
            if a.is_empty() {
                c.to_string()
            } else {
                format!("{c} {a}")
            }
        };
        let w = COMMANDS
            .iter()
            .map(|(c, a, _)| label(c, a).width())
            .max()
            .unwrap_or(0);
        self.row(&"commands".with(ACCENT).bold().to_string(), 8)?;
        for (c, a, d) in COMMANDS {
            let gap = " ".repeat(w - label(c, a).width() + 3);
            let a = if a.is_empty() {
                String::new()
            } else {
                format!(" {a}")
            };
            let head = format!("  {}{}{gap}", c.with(ACCENT), a.dark_grey());
            self.columns(&head, 2 + w + 3, d)?;
        }
        self.row("", 0)?;
        self.row(&"keys".with(ACCENT).bold().to_string(), 4)?;
        let keys = [
            ("enter", "send"),
            (
                "\\ then enter",
                "new line (shift+enter and option+enter work in some terminals)",
            ),
            ("esc", "stop a reply, or clear the input"),
            ("↑ ↓", "previous messages, or move through the / menu"),
            ("tab", "complete a command"),
            ("ctrl-c twice", "quit (or ctrl-d on an empty line)"),
        ];
        for (k, d) in keys {
            let head = format!("  {}", format!("{k:<14}").with(ACCENT));
            self.columns(&head, 16, d)?;
        }
        self.row("", 0)?;
        self.note(&format!(
            "temperature {} · top-p {} · {} messages in context",
            self.temperature,
            self.top_p,
            self.messages.len()
        ))
    }

    /// A card row of `head` (`head_w` columns) and then `text`, wrapped beside it.
    fn columns(&mut self, head: &str, head_w: usize, text: &str) -> Result<()> {
        let lines = frame::wrap(text, frame::text_width().saturating_sub(head_w).max(8));
        for (i, l) in lines.iter().enumerate() {
            let head = if i == 0 {
                head.to_string()
            } else {
                " ".repeat(head_w)
            };
            self.row(&format!("{head}{l}"), head_w + l.width())?;
        }
        Ok(())
    }

    /// An edge of a card: its top (with a label), a divider, or its bottom (with a status).
    fn edge(
        &mut self,
        corners: [char; 2],
        label: &str,
        label_color: Color,
        status: &str,
        status_color: Color,
    ) -> Result<()> {
        self.print(frame::edge(
            corners,
            (label, label_color),
            (status, status_color),
            BORDER,
            frame::text_width(),
        ))
    }

    fn close(&mut self) -> Result<()> {
        self.edge(['└', '┘'], "", BORDER, "", BORDER)
    }

    /// A line of the open card: `text` is `w` columns wide and may be styled.
    fn row(&mut self, text: &str, w: usize) -> Result<()> {
        self.print(frame::row(text, w, BORDER, frame::text_width()))
    }

    /// Wrapped text in the open card after a marker.
    fn marked(&mut self, marker: impl Display, s: &str, color: Color) -> Result<()> {
        let lines = frame::wrap(s, frame::text_width() - 2);
        for (i, l) in lines.iter().enumerate() {
            let marker = if i == 0 {
                marker.to_string()
            } else {
                " ".to_string()
            };
            self.row(
                &format!("{marker} {}", l.as_str().with(color)),
                2 + l.width(),
            )?;
        }
        Ok(())
    }

    fn note(&mut self, s: &str) -> Result<()> {
        self.marked("◦".dark_grey(), s, Color::DarkGrey)
    }

    fn error(&mut self, s: &str) -> Result<()> {
        self.marked("✗".red(), s, Color::Red)
    }

    /// Writes a line; raw mode needs the explicit carriage return.
    fn print(&mut self, s: impl Display) -> Result<()> {
        write!(self.out, "{s}\r\n")?;
        self.out.flush()?;
        Ok(())
    }
}

/// The phase, a meter and the numbers so far, fitted to `width` columns; returns the line and
/// its width.
fn status_line(
    start: Instant,
    first_token: Option<Duration>,
    pieces: u64,
    width: usize,
) -> (String, usize) {
    let elapsed = start.elapsed();
    let phase = if first_token.is_some() {
        "decoding"
    } else {
        "prefill"
    };
    let mut parts = vec![format!("{:.1}s", elapsed.as_secs_f64())];
    if let Some(ttft) = first_token {
        parts.push(format!("{pieces} tokens"));
        if let Some(tps) = rate(pieces, elapsed.saturating_sub(ttft)) {
            parts.push(format!("{tps:.0} tok/s"));
        }
    }
    parts.push("esc to stop".to_string());
    // The meter, a space, the phase and two spaces come first.
    let head_w = 8 + phase.width();
    let detail = frame::fit(&parts, width.saturating_sub(head_w));
    let line = format!(
        "{} {}  {}",
        meter(start),
        phase.with(ACCENT),
        detail.as_str().dark_grey()
    );
    (line, head_w + detail.width())
}
