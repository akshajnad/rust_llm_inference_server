use std::path::Path;

use anyhow::{Context, Result};
use minijinja::syntax::SyntaxConfig;
use minijinja::{Environment, ErrorKind, Value, context};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// A message as templates see it. K2 Horizon's refuses an assistant turn without a thinking
/// field; earlier turns' thinking is not kept, so it is empty for every model.
#[derive(Serialize)]
struct TemplateMessage<'a> {
    role: &'a str,
    content: &'a str,
    reasoning_content: &'static str,
}

/// The model's own Jinja chat template from `tokenizer_config.json` or `chat_template.jinja`,
/// rendered with the same environment settings and helper functions transformers'
/// `apply_chat_template` provides.
pub struct ChatTemplate {
    env: Environment<'static>,
    bos_token: String,
    eos_token: String,
}

impl ChatTemplate {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(model_dir.join("tokenizer_config.json"))
            .context("reading tokenizer_config.json")?;
        let cfg: serde_json::Value = serde_json::from_str(&text)?;
        let source = match cfg["chat_template"].as_str() {
            Some(s) => s.to_string(),
            None => std::fs::read_to_string(model_dir.join("chat_template.jinja"))
                .context("no chat_template in tokenizer_config.json or chat_template.jinja")?,
        };
        let token = |key: &str| match &cfg[key] {
            serde_json::Value::String(s) => s.clone(),
            v => v["content"].as_str().unwrap_or_default().to_string(),
        };
        Self::new(&source, &token("bos_token"), &token("eos_token"))
    }

    pub fn new(source: &str, bos_token: &str, eos_token: &str) -> Result<Self> {
        let mut env = Environment::new();
        // transformers renders with trim_blocks and lstrip_blocks on.
        env.set_syntax(
            SyntaxConfig::builder()
                .trim_blocks(true)
                .lstrip_blocks(true)
                .build()?,
        );
        env.add_function(
            "raise_exception",
            |msg: String| -> Result<Value, minijinja::Error> {
                Err(minijinja::Error::new(ErrorKind::InvalidOperation, msg))
            },
        );
        env.add_function("strftime_now", |fmt: String| {
            chrono::Local::now().format(&fmt).to_string()
        });
        // Python's string and dict methods (`.get`, `.split`, `.startswith`).
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        // transformers' `{% generation %}` marks a span for masks and renders its body; `if`
        // does the same and trims the same whitespace.
        let source = source
            .replace("{% generation %}", "{% if true %}")
            .replace("{%- endgeneration -%}", "{%- endif -%}");
        env.add_template_owned("chat", source)?;
        Ok(Self {
            env,
            bos_token: bos_token.into(),
            eos_token: eos_token.into(),
        })
    }

    /// Renders the conversation, ending with the assistant header when
    /// `add_generation_prompt` is set. The output already contains BOS, so encode it without
    /// adding special tokens.
    pub fn render(&self, messages: &[ChatMessage], add_generation_prompt: bool) -> Result<String> {
        self.render_with(messages, add_generation_prompt, None)
    }

    /// Like `render`, with a fixed `date_string` for reproducible output.
    pub fn render_with(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        date_string: Option<&str>,
    ) -> Result<String> {
        let tmpl = self.env.get_template("chat")?;
        let messages: Vec<_> = messages
            .iter()
            .map(|m| TemplateMessage {
                role: &m.role,
                content: &m.content,
                reasoning_content: "",
            })
            .collect();
        let mut ctx = context! {
            messages => Value::from(minijinja::value::Serde(messages)),
            add_generation_prompt => add_generation_prompt,
            bos_token => &self.bos_token,
            eos_token => &self.eos_token,
        };
        if let Some(d) = date_string {
            ctx = context! { date_string => d, ..ctx };
        }
        Ok(tmpl.render(ctx)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_simple_template() -> Result<()> {
        let t = ChatTemplate::new(
            "{{ bos_token }}{% for m in messages %}{{ m.role + ': ' + m.content | trim }}\n{% endfor %}\
             {% if add_generation_prompt %}assistant:{% endif %}",
            "<s>",
            "</s>",
        )?;
        let msgs = [ChatMessage {
            role: "user".into(),
            content: " hi ".into(),
        }];
        assert_eq!(t.render(&msgs, true)?, "<s>user: hi\nassistant:");
        Ok(())
    }

    #[test]
    fn generation_tags_render_their_body() -> Result<()> {
        let t = ChatTemplate::new(
            "{% for m in messages %}{{- m.role }}\n    {% generation %}\n    {%- if m.reasoning_content == '' -%}\
             {{- '<t></t>' + m.content }}{%- endif -%}{%- endgeneration -%}{% endfor %}",
            "",
            "",
        )?;
        let msgs = [ChatMessage {
            role: "assistant".into(),
            content: "hi".into(),
        }];
        assert_eq!(t.render(&msgs, false)?, "assistant\n<t></t>hi");
        Ok(())
    }

    #[test]
    fn raise_exception_is_an_error() -> Result<()> {
        let t = ChatTemplate::new("{{ raise_exception('nope') }}", "", "")?;
        assert!(t.render(&[], false).is_err());
        Ok(())
    }
}
