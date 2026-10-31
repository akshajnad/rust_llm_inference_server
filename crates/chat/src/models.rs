//! The models the chat and `rlis code` can start: GGUF files in the repo's `models/` directory
//! (`make models`, `make models-k2`, `make models-qwen-<size>`), each served with a 16,384-token context. `/model`
//! swaps between the ones on disk by stopping the server and starting one with the other model,
//! so only one is in memory at a time.

use std::path::Path;

/// Speeds, footprints and task counts are from an M4 Pro (24 GB); the tasks are `probe.rs`'s.
#[derive(Clone, Copy, PartialEq, Debug, clap::ValueEnum)]
pub enum Model {
    #[value(name = "3b")]
    Coder3B,
    #[value(name = "k2")]
    K2,
    #[value(name = "14b")]
    Coder14B,
    #[value(name = "llama-1b")]
    Llama1B,
}

impl Model {
    pub const ALL: [Model; 4] = [Model::Coder3B, Model::K2, Model::Coder14B, Model::Llama1B];

    /// What `/model` and `--model` call it.
    pub fn name(self) -> &'static str {
        match self {
            Model::Coder3B => "3b",
            Model::K2 => "k2",
            Model::Coder14B => "14b",
            Model::Llama1B => "llama-1b",
        }
    }

    /// Its weights, relative to the repo.
    pub fn gguf(self) -> &'static str {
        match self {
            Model::Coder3B => "models/Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf",
            Model::K2 => "models/K2-Horizon-3.7B-Q4_K_M.gguf",
            Model::Coder14B => "models/Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf",
            Model::Llama1B => "models/Llama-3.2-1B-Instruct-Q8_0.gguf",
        }
    }

    /// The name the server reports for it.
    pub fn served_name(self) -> &'static str {
        match self {
            Model::Coder3B => "qwen2.5-coder-3b-instruct",
            Model::K2 => "k2-horizon-3.7b",
            Model::Coder14B => "qwen2.5-coder-14b-instruct",
            Model::Llama1B => "llama-3.2-1b-instruct",
        }
    }

    /// The command that downloads it.
    pub fn download(self) -> &'static str {
        match self {
            Model::Coder3B => "make models-qwen-3b",
            Model::K2 => "make models-k2",
            Model::Coder14B => "make models-qwen-14b",
            Model::Llama1B => "make models",
        }
    }

    pub fn about(self) -> &'static str {
        match self {
            Model::Coder3B => "Qwen2.5-Coder 3B · 3.9 GB · 61 tok/s · fastest; slips now and then",
            Model::K2 => {
                "K2 Horizon 3.7B · 6.9 GB · 50 tok/s · rlis code's default when on disk; thinks first"
            }
            Model::Coder14B => {
                "Qwen2.5-Coder 14B · 12.9 GB · 21 tok/s · most reliable, at twice K2's memory"
            }
            Model::Llama1B => "Llama 3.2 1B · 3 GB · the chat's model; too small to code well",
        }
    }

    /// Whether its weights and tokenizer are in `repo`.
    pub fn on_disk(self, repo: &Path) -> bool {
        repo.join(self.gguf()).is_file() && repo.join(self.tokenizer()).is_dir()
    }

    pub fn serve_args(self) -> Vec<String> {
        let mut args: Vec<String> = [
            "--model",
            self.gguf(),
            "--tokenizer",
            self.tokenizer(),
            "--served-model-name",
            self.served_name(),
            "--max-seq-len",
            "16384",
        ]
        .map(String::from)
        .into();
        // The 14B's KV cache takes 192 KiB a token and K2's 144 KiB: these hold a whole context.
        match self {
            Model::Coder14B => args.extend(["--kv-cache-gb".to_string(), "3".to_string()]),
            Model::K2 => args.extend(["--kv-cache-gb".to_string(), "2.3".to_string()]),
            _ => {}
        }
        args
    }

    /// The model `name` names: its `/model` name or the name the server reports.
    pub fn parse(name: &str) -> Option<Model> {
        let name = name.trim().to_lowercase();
        Model::ALL
            .into_iter()
            .find(|m| m.name() == name || m.served_name() == name)
    }

    /// The model a server reporting `served` runs, if it is one of these.
    pub fn from_served(served: &str) -> Option<Model> {
        Model::ALL.into_iter().find(|m| m.served_name() == served)
    }

    /// What `rlis code` starts with no `--model`: K2, which slips less than the 3B, when it is
    /// on disk.
    pub fn default_coder(repo: &Path) -> Model {
        match Model::K2.on_disk(repo) {
            true => Model::K2,
            false => Model::Coder3B,
        }
    }

    /// Whether it thinks before answering and calls tools natively (see `native`).
    pub fn native(self) -> bool {
        self == Model::K2
    }

    fn tokenizer(self) -> &'static str {
        match self {
            Model::Coder3B => "models/qwen2.5-coder-3b-instruct",
            Model::K2 => "models/k2-horizon-3.7b",
            Model::Coder14B => "models/qwen2.5-coder-14b-instruct",
            Model::Llama1B => "models/llama-3.2-1b-instruct",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip() {
        for m in Model::ALL {
            assert_eq!(Model::parse(m.name()), Some(m));
            assert_eq!(Model::parse(m.served_name()), Some(m));
            assert_eq!(Model::from_served(m.served_name()), Some(m));
        }
        assert_eq!(Model::parse(" 14B "), Some(Model::Coder14B));
        assert_eq!(Model::parse("70b"), None);
        assert!(
            Model::Coder14B
                .serve_args()
                .contains(&"--kv-cache-gb".to_string())
        );
    }

    #[test]
    fn finds_models_on_disk() -> std::io::Result<()> {
        let repo = std::env::temp_dir().join(format!("rlis-models-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&repo);
        std::fs::create_dir_all(repo.join(Model::Coder3B.tokenizer()))?;
        std::fs::write(repo.join(Model::Coder3B.gguf()), "")?;
        assert!(Model::Coder3B.on_disk(&repo) && !Model::K2.on_disk(&repo));
        assert_eq!(Model::default_coder(&repo), Model::Coder3B);
        // A tokenizer without its weights (a deleted download) is not on disk.
        std::fs::create_dir_all(repo.join(Model::K2.tokenizer()))?;
        assert_eq!(Model::default_coder(&repo), Model::Coder3B);
        std::fs::write(repo.join(Model::K2.gguf()), "")?;
        assert_eq!(Model::default_coder(&repo), Model::K2);
        std::fs::remove_dir_all(&repo)
    }
}
