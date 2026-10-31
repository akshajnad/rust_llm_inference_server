//! K2 Horizon thinks before it answers and calls tools natively, where the agent reads plain text
//! (`READ path`, `RUN command`, a path and a fenced block). `Native` hides the thinking and turns
//! the calls into that text, so the agent works unchanged; `system` declares the tools.

use crate::api::Update;

/// What K2's chat template puts before the system prompt when given these tools.
const TOOLS: &str = "# Tools
You may call one or more tools to assist with the user query.

Available tools are:

<ifm|tools>
## READ
Read a file, or list a directory ('.' lists the project).

**Parameters**
- `path` *(string, required)* - file or directory

## RUN
Run a shell command in the project's directory.

**Parameters**
- `command` *(string, required)* - the command

## WRITE
Create a file or replace all of it.

**Parameters**
- `path` *(string, required)* - file to write
- `content` *(string, required)* - its whole content

## EDIT
Replace exact lines of an existing file you have read.

**Parameters**
- `path` *(string, required)* - file to edit
- `search` *(string, required)* - lines copied exactly from the file
- `replace` *(string, required)* - lines to put in their place
</ifm|tools>

When calling tools, you MUST follow the tool-call format below:

Wrap all tool calls in a single <ifm|tool_calls></ifm|tool_calls> block. For each call, write the \
function name at the start of <ifm|tool_call>, followed by paired <ifm|arg_key> and <ifm|arg_value> \
tags for each argument:

<ifm|tool_calls>
<ifm|tool_call>$FUNCTION_NAME
<ifm|arg_key>$PARAMETER_NAME</ifm|arg_key>
<ifm|arg_value>$PARAMETER_VALUE</ifm|arg_value>
...
</ifm|tool_call>
</ifm|tool_calls>

String and scalar parameters should be written as plain text. Array and object parameters should \
be written as JSON literals.

";

const THINK_ENDS: [&str; 3] = ["</ifm|think>", "</ifm|think_fast>", "</ifm|think_faster>"];
const CALLS: [&str; 2] = ["<ifm|tool_calls>", "<ifm|tool_call>"];
const CALLS_END: &str = "</ifm|tool_calls>";

/// The agent's system prompt with the tools declared.
pub fn system(prompt: &str) -> String {
    format!("{TOOLS}{prompt}")
}

enum State {
    Thinking,
    Text,
    Calls,
}

pub struct Native {
    state: State,
    /// Text not yet handled: it may end in the start of a tag.
    held: String,
    /// The last character shown, since a request must start on a line of its own.
    last: char,
}

impl Native {
    /// The chat template opens the thinking, so a reply starts inside it.
    pub fn new() -> Self {
        Self {
            state: State::Thinking,
            held: String::new(),
            last: '\n',
        }
    }

    /// One streamed piece of the reply, as what the UI and the agent should see.
    pub fn feed(&mut self, text: &str) -> Vec<Update> {
        self.held.push_str(text);
        let mut out = Vec::new();
        loop {
            match self.state {
                State::Thinking => {
                    out.push(Update::Thinking);
                    match first_of(&self.held, &THINK_ENDS) {
                        Some((at, tag)) => {
                            self.held.drain(..at + tag.len());
                            self.state = State::Text;
                        }
                        None => {
                            let keep = partial_tag(&self.held);
                            self.held.drain(..keep);
                            return out;
                        }
                    }
                }
                State::Text => match first_of(&self.held, &CALLS) {
                    Some((at, _)) => {
                        let text: String = self.held.drain(..at).collect();
                        self.show(&text, &mut out);
                        self.state = State::Calls;
                    }
                    None => {
                        let keep = partial_tag(&self.held);
                        let text: String = self.held.drain(..keep).collect();
                        self.show(&text, &mut out);
                        return out;
                    }
                },
                State::Calls => match self.held.find(CALLS_END) {
                    Some(at) => {
                        let block: String = self.held.drain(..at + CALLS_END.len()).collect();
                        self.calls(&block, &mut out);
                        self.state = State::Text;
                    }
                    None => {
                        out.push(Update::Thinking);
                        return out;
                    }
                },
            }
        }
    }

    /// The end of the reply: whatever was held back, including calls never closed.
    pub fn finish(&mut self) -> Vec<Update> {
        let mut out = Vec::new();
        let held = std::mem::take(&mut self.held);
        match self.state {
            State::Thinking => {}
            State::Text => self.show(&held, &mut out),
            State::Calls => self.calls(&held, &mut out),
        }
        out
    }

    fn show(&mut self, text: &str, out: &mut Vec<Update>) {
        let text = strip_tags(text);
        if let Some(last) = text.chars().last() {
            self.last = last;
            out.push(Update::Text(text));
        }
    }

    fn calls(&mut self, block: &str, out: &mut Vec<Update>) {
        for call in block.split("<ifm|tool_call>").skip(1) {
            let text = as_text(call.split("</ifm|tool_call>").next().unwrap_or(call));
            let text = match self.last {
                '\n' => text,
                _ => format!("\n{text}"),
            };
            self.show(&text, out);
        }
    }
}

/// The earliest of `tags` in `text`, with its position.
fn first_of<'a>(text: &str, tags: &[&'a str]) -> Option<(usize, &'a str)> {
    tags.iter()
        .filter_map(|&t| text.find(t).map(|at| (at, t)))
        .min_by_key(|&(at, _)| at)
}

/// Where `text` stops being safe to act on: from a `<` that may start a tag still arriving.
fn partial_tag(text: &str) -> usize {
    let longest = THINK_ENDS[2].len();
    match text.rfind('<') {
        Some(at) if text.len() - at < longest && !text[at..].contains('>') => at,
        _ => text.len(),
    }
}

/// Drops any `<ifm|...>` tag left in the text; none of them is for the reader.
fn strip_tags(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = first_of(rest, &["<ifm|", "</ifm|"]).map(|(at, _)| at) {
        let Some(end) = rest[at..].find('>') else {
            break;
        };
        out.push_str(&rest[..at]);
        rest = &rest[at + end + 1..];
    }
    out + rest
}

/// The file a change names: K2 sometimes labels the path (`grid:blades.html`, the part of the
/// page it changed), which would create a new file by that name.
fn file_path(path: &str) -> &str {
    path.rsplit(':').next().unwrap_or(path).trim()
}

/// One call, as the agent's text: `NAME\n<ifm|arg_key>k</ifm|arg_key>\n<ifm|arg_value>v</ifm|arg_value>`.
fn as_text(call: &str) -> String {
    let name = call.split("<ifm|arg_key>").next().unwrap_or("").trim();
    let arg = |key: &str| {
        let open = format!("<ifm|arg_key>{key}</ifm|arg_key>");
        let value = call.split(&open).nth(1)?.split_once("<ifm|arg_value>")?.1;
        Some(value.split("</ifm|arg_value>").next()?.to_string())
    };
    let get = |key| arg(key).unwrap_or_default();
    // Its text protocol can slip into a call: `<ifm|tool_call>RUN ls`, with no arguments.
    if !call.contains("<ifm|arg_key>") && name.contains(' ') {
        return format!("{name}\n");
    }
    match name {
        "READ" => format!("READ {}\n", arg("path").unwrap_or_else(|| ".".into())),
        "RUN" => {
            let command = get("command");
            let command = command.trim();
            match command.contains('\n') {
                true => format!("RUN\n```sh\n{command}\n```\n"),
                false => format!("RUN {command}\n"),
            }
        }
        "WRITE" => {
            let content = get("content");
            let newline = if content.ends_with('\n') { "" } else { "\n" };
            format!("{}\n```\n{content}{newline}```\n", file_path(&get("path")))
        }
        "EDIT" => format!(
            "{}\n```\n<<<<<<< SEARCH\n{}\n=======\n{}\n>>>>>>> REPLACE\n```\n",
            file_path(&get("path")),
            get("search"),
            get("replace")
        ),
        other => format!("(unknown tool {other})\n"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feeds `pieces` and joins what is shown.
    fn shown(pieces: &[&str]) -> String {
        let mut n = Native::new();
        let mut updates = Vec::new();
        for p in pieces {
            updates.extend(n.feed(p));
        }
        updates.extend(n.finish());
        updates
            .into_iter()
            .filter_map(|u| match u {
                Update::Text(t) => Some(t),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn hides_thinking_across_pieces() {
        assert_eq!(
            shown(&["plan", " it</if", "m|think>Done", " now"]),
            "Done now"
        );
        assert_eq!(shown(&["x</ifm|think_faster>ok"]), "ok");
        assert_eq!(shown(&["never ends"]), "");
    }

    #[test]
    fn reports_thinking_so_the_ui_can_show_it() {
        let mut n = Native::new();
        assert!(matches!(&n.feed("hmm")[..], [Update::Thinking]));
    }

    #[test]
    fn turns_calls_into_the_agents_text() {
        let call = "<ifm|tool_calls>\n<ifm|tool_call>READ\n<ifm|arg_key>path</ifm|arg_key>\n\
                    <ifm|arg_value>a.py</ifm|arg_value>\n</ifm|tool_call>\n</ifm|tool_calls>";
        assert_eq!(shown(&["</ifm|think>", call]), "READ a.py\n");
        // The tag may arrive split, and a request needs a line of its own.
        let (a, b) = call.split_at(7);
        assert_eq!(
            shown(&["</ifm|think>Reading:", a, b]),
            "Reading:\nREAD a.py\n"
        );
    }

    #[test]
    fn writes_runs_and_edits_read_like_the_protocol() {
        let args = |k: &[(&str, &str)]| {
            k.iter()
                .map(|(k, v)| {
                    format!("<ifm|arg_key>{k}</ifm|arg_key>\n<ifm|arg_value>{v}</ifm|arg_value>\n")
                })
                .collect::<String>()
        };
        let write = format!("WRITE\n{}", args(&[("path", "h.py"), ("content", "x = 1")]));
        assert_eq!(as_text(&write), "h.py\n```\nx = 1\n```\n");
        let run = format!("RUN\n{}", args(&[("command", "ls\npwd")]));
        assert_eq!(as_text(&run), "RUN\n```sh\nls\npwd\n```\n");
        let edit = format!(
            "EDIT\n{}",
            args(&[("path", "h.py"), ("search", "a"), ("replace", "b")])
        );
        assert_eq!(
            as_text(&edit),
            "h.py\n```\n<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE\n```\n"
        );
    }

    #[test]
    fn a_labeled_path_changes_the_file_it_names() {
        let write = "WRITE\n<ifm|arg_key>path</ifm|arg_key>\n\
                     <ifm|arg_value>grid:blades.html</ifm|arg_value>\n\
                     <ifm|arg_key>content</ifm|arg_key>\n<ifm|arg_value>x</ifm|arg_value>\n";
        assert_eq!(as_text(write), "blades.html\n```\nx\n```\n");
        // A read's `file:200` is a line number, not a label.
        let read =
            "READ\n<ifm|arg_key>path</ifm|arg_key>\n<ifm|arg_value>a.py:200</ifm|arg_value>\n";
        assert_eq!(as_text(read), "READ a.py:200\n");
    }

    #[test]
    fn a_request_written_as_a_tool_name_still_runs() {
        assert_eq!(
            as_text("RUN pip install pygame\n"),
            "RUN pip install pygame\n"
        );
        assert_eq!(as_text("READ a.py\n"), "READ a.py\n");
        assert_eq!(as_text("REMOVE\n"), "(unknown tool REMOVE)\n");
    }

    #[test]
    fn keeps_ordinary_angle_brackets_and_drops_stray_tags() {
        assert_eq!(shown(&["</ifm|think>if a < b:"]), "if a < b:");
        assert_eq!(shown(&["</ifm|think>a</ifm|tool_call>b"]), "ab");
    }

    #[test]
    fn a_call_never_closed_still_counts() {
        let call = "<ifm|tool_call>RUN\n<ifm|arg_key>command</ifm|arg_key>\n\
                    <ifm|arg_value>ls</ifm|arg_value>\n";
        assert_eq!(shown(&["</ifm|think>", call]), "RUN ls\n");
    }
}
