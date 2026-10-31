//! Streaming chat completions on a worker thread: sends the request, parses server-sent events,
//! and reports each piece to the UI over a channel.

use std::net::TcpStream;
use std::sync::mpsc::Sender;

use anyhow::{Result, bail};
use serde_json::Value;

use crate::http;
use crate::native::Native;

pub enum Update {
    /// The request's socket, so the UI can shut it down to interrupt the reply.
    Connected(TcpStream),
    Text(String),
    /// A token not shown (yet): thinking, or a tool call still arriving.
    Thinking,
    Finish(String),
    /// Completion tokens, from the final usage chunk.
    Usage(u64),
    /// The server refused the request; `param` names the offending field.
    Rejected {
        message: String,
        param: Option<String>,
    },
    /// Always the last update, with an error if the stream broke.
    End(Option<String>),
}

/// `native` is for a model that thinks and calls tools natively (see `native`).
pub fn stream(url: &str, body: &[u8], native: bool, tx: &Sender<Update>) {
    let result = run(url, body, native.then(Native::new), tx);
    let _ = tx.send(Update::End(result.err().map(|e| format!("{e:#}"))));
}

fn run(url: &str, body: &[u8], mut native: Option<Native>, tx: &Sender<Update>) -> Result<()> {
    let mut resp = http::request(url, "POST", "/v1/chat/completions", Some(body))?;
    let _ = tx.send(Update::Connected(resp.socket()?));
    if resp.status != 200 {
        let status = resp.status;
        let text = resp.text()?;
        let (message, param) = error_body(&text);
        let message = message.unwrap_or_else(|| format!("HTTP {status}: {}", text.trim()));
        let _ = tx.send(Update::Rejected { message, param });
        return Ok(());
    }
    let mut buf = Vec::new();
    while let Some(bytes) = resp.next()? {
        buf.extend_from_slice(&bytes);
        for data in split_events(&mut buf) {
            for update in parse(&data)? {
                let updates = match (&mut native, update) {
                    (Some(n), Update::Text(t)) => n.feed(&t),
                    (Some(n), Update::Finish(f)) => {
                        let mut held = n.finish();
                        held.push(Update::Finish(f));
                        held
                    }
                    (_, update) => vec![update],
                };
                for update in updates {
                    let _ = tx.send(update);
                }
            }
        }
    }
    Ok(())
}

/// Removes complete events from `buf` and returns their `data:` payloads.
pub fn split_events(buf: &mut Vec<u8>) -> Vec<String> {
    let mut out = Vec::new();
    while let Some(i) = buf.windows(2).position(|w| w == b"\n\n") {
        let event: Vec<u8> = buf.drain(..i + 2).collect();
        let event = String::from_utf8_lossy(&event);
        let data: Vec<&str> = event
            .lines()
            .filter_map(|l| l.strip_prefix("data:"))
            .map(|d| d.strip_prefix(' ').unwrap_or(d))
            .collect();
        if !data.is_empty() {
            out.push(data.join("\n"));
        }
    }
    out
}

pub fn parse(data: &str) -> Result<Vec<Update>> {
    if data == "[DONE]" {
        return Ok(Vec::new());
    }
    let v: Value = serde_json::from_str(data)?;
    if let (Some(message), _) = error_body(data) {
        bail!("{message}");
    }
    let mut out = Vec::new();
    let choice = &v["choices"][0];
    if let Some(t) = choice["delta"]["content"].as_str()
        && !t.is_empty()
    {
        out.push(Update::Text(t.to_string()));
    }
    if let Some(f) = choice["finish_reason"].as_str() {
        out.push(Update::Finish(f.to_string()));
    }
    if let Some(n) = v["usage"]["completion_tokens"].as_u64() {
        out.push(Update::Usage(n));
    }
    Ok(out)
}

/// `{"error": {"message", "param"}}`, as OpenAI-compatible servers report failures.
fn error_body(text: &str) -> (Option<String>, Option<String>) {
    let v: Value = serde_json::from_str(text).unwrap_or_default();
    let field = |k: &str| v["error"][k].as_str().map(str::to_string);
    (field("message"), field("param"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_events_across_reads() {
        let mut buf = b"data: {\"a\":1}\n\ndata: [DO".to_vec();
        assert_eq!(split_events(&mut buf), vec!["{\"a\":1}"]);
        buf.extend_from_slice(b"NE]\n\n: keep-alive\n\n");
        assert_eq!(split_events(&mut buf), vec!["[DONE]"]);
        assert!(buf.is_empty());
    }

    #[test]
    fn parses_chunks() -> Result<()> {
        let text = r#"{"choices":[{"delta":{"content":" sky"},"finish_reason":null,"index":0}]}"#;
        assert!(matches!(&parse(text)?[..], [Update::Text(t)] if t == " sky"));
        let finish = r#"{"choices":[{"delta":{},"finish_reason":"stop","index":0}]}"#;
        assert!(matches!(&parse(finish)?[..], [Update::Finish(f)] if f == "stop"));
        let usage = r#"{"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":9}}"#;
        assert!(matches!(&parse(usage)?[..], [Update::Usage(9)]));
        assert!(parse(r#"{"error":{"message":"boom"}}"#).is_err());
        assert!(parse("[DONE]")?.is_empty());
        Ok(())
    }
}
