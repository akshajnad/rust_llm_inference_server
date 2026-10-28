//! A minimal blocking HTTP/1.1 client, the synchronous twin of `rlis-bench`'s: one request per
//! connection (`Connection: close`) with chunked transfer decoding. The socket can be cloned so
//! another thread can shut it down to abandon a streaming reply.

use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

use anyhow::{Context, Result};

pub struct Response {
    pub status: u16,
    stream: TcpStream,
    chunked: bool,
    /// Raw bytes read past the headers, not yet decoded.
    raw: Vec<u8>,
    done: bool,
}

/// `http://host:port` -> `host:port`.
pub fn authority(base: &str) -> Result<&str> {
    base.strip_prefix("http://")
        .map(|a| a.trim_end_matches('/'))
        .context("only http:// URLs are supported")
}

/// Tries every address the host resolves to: `localhost` may resolve to `::1` first while the
/// server listens on 127.0.0.1.
fn connect(host: &str) -> Result<TcpStream> {
    let mut last = None;
    for addr in host
        .to_socket_addrs()
        .with_context(|| format!("resolving {host}"))?
    {
        match TcpStream::connect_timeout(&addr, Duration::from_secs(2)) {
            Ok(s) => return Ok(s),
            Err(e) => last = Some(e),
        }
    }
    Err(last.map_or_else(|| anyhow::anyhow!("{host} has no address"), Into::into))
        .with_context(|| format!("connecting to {host}"))
}

pub fn request(base: &str, method: &str, path: &str, body: Option<&[u8]>) -> Result<Response> {
    let host = authority(base)?;
    let mut stream = connect(host)?;
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
    stream.write_all(&bytes)?;

    let mut raw = Vec::new();
    let header_end = loop {
        if let Some(i) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
            break i;
        }
        let mut buf = [0u8; 8192];
        let n = stream.read(&mut buf)?;
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
    /// A handle to the socket; shutting it down ends this response's reads.
    pub fn socket(&self) -> Result<TcpStream> {
        Ok(self.stream.try_clone()?)
    }

    /// Next piece of the decoded body, or `None` at the end.
    pub fn next(&mut self) -> Result<Option<Vec<u8>>> {
        loop {
            if self.done {
                return Ok(None);
            }
            if let Some(out) = self.take_decoded()? {
                return Ok(Some(out));
            }
            if self.done {
                return Ok(None);
            }
            let mut buf = [0u8; 16384];
            let n = self.stream.read(&mut buf)?;
            if n == 0 {
                self.done = true;
                anyhow::ensure!(
                    !self.chunked,
                    "connection closed before the response finished"
                );
                // Without chunking the body runs to EOF.
                return Ok((!self.raw.is_empty()).then(|| std::mem::take(&mut self.raw)));
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

    pub fn text(mut self) -> Result<String> {
        let mut out = Vec::new();
        while let Some(b) = self.next()? {
            out.extend(b);
        }
        Ok(String::from_utf8_lossy(&out).into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::thread;

    /// A canned chunked response split across awkward write boundaries decodes intact.
    #[test]
    fn decodes_chunked_bodies() -> Result<()> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let addr = listener.local_addr()?;
        thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            let mut buf = [0u8; 1024];
            let _ = s.read(&mut buf).unwrap();
            let resp = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n6\r\ndata: \r\n5\r\nhello\r\n1\r\n\n\r\n0\r\n\r\n";
            for piece in resp.chunks(7) {
                s.write_all(piece).unwrap();
                s.flush().unwrap();
            }
        });
        let resp = request(&format!("http://{addr}"), "GET", "/", None)?;
        assert_eq!(resp.status, 200);
        assert_eq!(resp.text()?, "data: hello\n");
        Ok(())
    }
}
