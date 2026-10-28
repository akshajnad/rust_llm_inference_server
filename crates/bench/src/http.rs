//! A minimal HTTP/1.1 client: one request per connection (`Connection: close`), written in a
//! single write, with chunked transfer decoding for streamed responses. Enough for
//! OpenAI-compatible servers. A pooled keep-alive client got "connection closed before message
//! completed" on ~20% of requests to llama.cpp's server, which closes idle connections.

use anyhow::{Context, Result};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

pub struct Response {
    pub status: u16,
    stream: TcpStream,
    chunked: bool,
    /// Raw bytes read past the headers, not yet decoded.
    raw: Vec<u8>,
    done: bool,
}

/// `http://host:port` -> `host:port`.
fn authority(base: &str) -> Result<&str> {
    base.strip_prefix("http://")
        .map(|a| a.trim_end_matches('/'))
        .context("only http:// URLs are supported")
}

pub async fn request(
    base: &str,
    method: &str,
    path: &str,
    body: Option<&[u8]>,
) -> Result<Response> {
    let host = authority(base)?;
    let mut stream = TcpStream::connect(host)
        .await
        .with_context(|| format!("connecting to {host}"))?;
    stream.set_nodelay(true)?;
    let mut req =
        format!("{method} {path} HTTP/1.1\r\nHost: {host}\r\nAccept: */*\r\nConnection: close\r\n");
    if let Some(b) = body {
        req.push_str(&format!(
            "Content-Type: application/json\r\nContent-Length: {}\r\n",
            b.len()
        ));
    }
    req.push_str("\r\n");
    let mut bytes = req.into_bytes();
    bytes.extend_from_slice(body.unwrap_or_default());
    stream.write_all(&bytes).await?;

    let mut raw = Vec::new();
    let header_end = loop {
        if let Some(i) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
            break i;
        }
        let mut buf = [0u8; 8192];
        let n = stream.read(&mut buf).await?;
        anyhow::ensure!(n > 0, "connection closed before response headers");
        raw.extend_from_slice(&buf[..n]);
    };
    let head = String::from_utf8_lossy(&raw[..header_end]).to_string();
    let status = head
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .context("bad status line")?;
    let chunked = head.lines().any(|l| {
        l.to_ascii_lowercase().starts_with("transfer-encoding:")
            && l.to_ascii_lowercase().contains("chunked")
    });
    raw.drain(..header_end + 4);
    Ok(Response {
        status,
        stream,
        chunked,
        raw,
        done: false,
    })
}

impl Response {
    /// Next piece of the decoded body, or `None` at the end.
    pub async fn next(&mut self) -> Result<Option<Vec<u8>>> {
        loop {
            if self.done {
                return Ok(None);
            }
            if let Some(out) = self.take_decoded()? {
                return Ok(Some(out));
            }
            let mut buf = [0u8; 16384];
            let n = self.stream.read(&mut buf).await?;
            if n == 0 {
                self.done = true;
                // Without chunking the body runs to EOF.
                return Ok(
                    (!self.chunked && !self.raw.is_empty()).then(|| std::mem::take(&mut self.raw))
                );
            }
            self.raw.extend_from_slice(&buf[..n]);
        }
    }

    fn take_decoded(&mut self) -> Result<Option<Vec<u8>>> {
        if !self.chunked {
            return Ok((!self.raw.is_empty()).then(|| std::mem::take(&mut self.raw)));
        }
        let Some(eol) = self.raw.windows(2).position(|w| w == b"\r\n") else {
            return Ok(None);
        };
        let size_line = String::from_utf8_lossy(&self.raw[..eol]);
        let size = usize::from_str_radix(size_line.split(';').next().unwrap_or("").trim(), 16)
            .with_context(|| format!("bad chunk size {size_line:?}"))?;
        if size == 0 {
            self.done = true;
            return Ok(None);
        }
        if self.raw.len() < eol + 2 + size + 2 {
            return Ok(None);
        }
        let data = self.raw[eol + 2..eol + 2 + size].to_vec();
        self.raw.drain(..eol + 2 + size + 2);
        Ok(Some(data))
    }

    pub async fn text(mut self) -> Result<String> {
        let mut out = Vec::new();
        while let Some(b) = self.next().await? {
            out.extend(b);
        }
        Ok(String::from_utf8_lossy(&out).into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::TcpListener;

    /// A canned chunked response split across awkward read boundaries decodes intact.
    #[tokio::test]
    async fn decodes_chunked_bodies() -> Result<()> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        tokio::spawn(async move {
            let (mut s, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 1024];
            let _ = s.read(&mut buf).await.unwrap();
            let resp = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n6\r\ndata: \r\n5\r\nhello\r\n1\r\n\n\r\n0\r\n\r\n";
            for piece in resp.chunks(7) {
                s.write_all(piece).await.unwrap();
                s.flush().await.unwrap();
            }
        });
        let resp = request(&format!("http://{addr}"), "GET", "/", None).await?;
        assert_eq!(resp.status, 200);
        assert_eq!(resp.text().await?, "data: hello\n");
        Ok(())
    }
}
