//! # llama-metal
//!
//! Minimal llama.cpp wrapper with Metal GPU support for local LLM inference.
//!
//! ## Example
//!
//! ```no_run
//! use llama_metal::{LlamaModel, Message, Role};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), llama_metal::Error> {
//!     let model = LlamaModel::load("path/to/model.gguf").await?;
//!     
//!     let messages = vec![
//!         Message::system("You are a helpful assistant."),
//!         Message::user("What is 2+2?"),
//!     ];
//!     
//!     // Non-streaming
//!     let response = model.chat(&messages).await?;
//!     println!("{}", response);
//!     
//!     // Streaming
//!     model.chat_stream(&messages, |token| {
//!         print!("{}", token);
//!     }).await?;
//!     
//!     Ok(())
//! }
//! ```

use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel as LlamaModelInner, Special},
    token::LlamaToken,
};
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::mpsc;

/// Error type for llama-metal operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Failed to initialize backend: {0}")]
    BackendInit(String),
    #[error("Failed to load model: {0}")]
    ModelLoad(String),
    #[error("Inference error: {0}")]
    Inference(String),
}

/// Message role in a conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

/// A chat message.
#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: Role::System, content: content.into() }
    }
    
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: Role::User, content: content.into() }
    }
    
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: Role::Assistant, content: content.into() }
    }
}

/// Configuration for model loading and inference.
#[derive(Debug, Clone)]
pub struct Config {
    /// Number of GPU layers to offload (default: 99 = all)
    pub n_gpu_layers: u32,
    /// Context size (default: 2048)
    pub context_size: u32,
    /// Maximum tokens to generate (default: 512)
    pub max_tokens: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            n_gpu_layers: 99,
            context_size: 2048,
            max_tokens: 512,
        }
    }
}

/// A loaded LLaMA model ready for inference.
pub struct LlamaModel {
    model: Arc<LlamaModelInner>,
    backend: Arc<LlamaBackend>,
    config: Config,
    model_type: ModelType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelType {
    GptOss,
    Llama,
    Qwen,
    Other,
}

impl ModelType {
    fn detect(path: &str) -> Self {
        let p = path.to_lowercase();
        if p.contains("gpt-oss") || p.contains("gptoss") {
            ModelType::GptOss
        } else if p.contains("llama") {
            ModelType::Llama
        } else if p.contains("qwen") {
            ModelType::Qwen
        } else {
            ModelType::Other
        }
    }
}

impl LlamaModel {
    /// Load a model from a GGUF file with default config.
    pub async fn load(path: impl AsRef<Path>) -> Result<Self, Error> {
        Self::load_with_config(path, Config::default()).await
    }

    /// Load a model from a GGUF file with custom config.
    pub async fn load_with_config(path: impl AsRef<Path>, config: Config) -> Result<Self, Error> {
        let path = path.as_ref().to_path_buf();
        let path_str = path.to_string_lossy().to_string();
        let model_type = ModelType::detect(&path_str);
        
        let backend = LlamaBackend::init()
            .map_err(|e| Error::BackendInit(format!("{:?}", e)))?;

        let model_params = LlamaModelParams::default()
            .with_n_gpu_layers(config.n_gpu_layers);

        let model = LlamaModelInner::load_from_file(&backend, &path, &model_params)
            .map_err(|e| Error::ModelLoad(format!("{:?}", e)))?;

        Ok(Self {
            model: Arc::new(model),
            backend: Arc::new(backend),
            config,
            model_type,
        })
    }

    /// Generate a response for the given messages.
    pub async fn chat(&self, messages: &[Message]) -> Result<String, Error> {
        let mut result = String::new();
        self.chat_stream(messages, |token| result.push_str(token)).await?;
        
        if self.model_type == ModelType::GptOss {
            Ok(Self::extract_gpt_oss_answer(&result))
        } else {
            Ok(result)
        }
    }

    /// Generate a streaming response, calling the callback for each token.
    pub async fn chat_stream<F>(&self, messages: &[Message], mut on_token: F) -> Result<String, Error>
    where
        F: FnMut(&str) + Send,
    {
        let prompt = self.format_messages(messages);
        let model = Arc::clone(&self.model);
        let backend = Arc::clone(&self.backend);
        let config = self.config.clone();
        let model_type = self.model_type;

        let (tx, mut rx) = mpsc::unbounded_channel::<String>();

        std::thread::spawn(move || {
            let _ = Self::generate_tokens(&model, &backend, &prompt, &config, model_type, tx);
        });

        let mut result = String::new();
        while let Some(token) = rx.recv().await {
            on_token(&token);
            result.push_str(&token);
        }

        Ok(result)
    }

    fn generate_tokens(
        model: &LlamaModelInner,
        backend: &LlamaBackend,
        prompt: &str,
        config: &Config,
        model_type: ModelType,
        tx: mpsc::UnboundedSender<String>,
    ) -> Result<(), Error> {
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(NonZeroU32::new(config.context_size).unwrap()));

        let mut context = model
            .new_context(backend, ctx_params)
            .map_err(|e| Error::Inference(format!("Context creation failed: {:?}", e)))?;

        let tokens = context.model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| Error::Inference(format!("Tokenization failed: {:?}", e)))?;

        // Process prompt
        let mut prompt_batch = LlamaBatch::new(tokens.len(), 1);
        for (i, &token) in tokens.iter().enumerate() {
            prompt_batch.add(token, i as i32, &[0], true)
                .map_err(|e| Error::Inference(format!("{:?}", e)))?;
        }
        context.decode(&mut prompt_batch)
            .map_err(|e| Error::Inference(format!("{:?}", e)))?;

        // GPT-OSS stop tokens
        let gpt_oss_stops: &[i32] = &[200002, 200007, 200012];
        
        let mut generated = String::new();
        let mut next_pos = tokens.len() as i32;

        for _ in 0..config.max_tokens {
            let logits = context.get_logits();
            let token_idx = logits.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(context.model.token_eos().0 as usize);
            
            let token = LlamaToken(token_idx as i32);

            // Check stop conditions
            if token == context.model.token_eos() {
                break;
            }
            if model_type == ModelType::GptOss && gpt_oss_stops.contains(&token.0) {
                break;
            }

            if let Ok(text) = context.model.token_to_str(token, Special::Plaintext) {
                // Check for ChatML stop sequences
                generated.push_str(&text);
                if generated.contains("<|im_end|>") || generated.contains("<|endoftext|>") {
                    break;
                }
                let _ = tx.send(text);
            }

            let mut batch = LlamaBatch::new(1, 1);
            batch.add(token, next_pos, &[0], true)
                .map_err(|e| Error::Inference(format!("{:?}", e)))?;
            context.decode(&mut batch)
                .map_err(|e| Error::Inference(format!("{:?}", e)))?;
            next_pos += 1;
        }

        Ok(())
    }

    fn format_messages(&self, messages: &[Message]) -> String {
        let mut prompt = String::new();
        
        for msg in messages {
            let role = match msg.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
            };
            prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, msg.content));
        }
        prompt.push_str("<|im_start|>assistant\n");
        prompt
    }

    fn extract_gpt_oss_answer(output: &str) -> String {
        let output = output.trim_start_matches(|c: char| c == '0' || c == '\n' || c.is_whitespace());
        
        let patterns = ["correct answer:", "the answer:", "respond with", "so respond"];
        for pattern in patterns {
            if let Some(pos) = output.to_lowercase().rfind(pattern) {
                let after = &output[pos + pattern.len()..];
                let answer = after.trim()
                    .trim_start_matches(|c: char| c == ':' || c == ' ')
                    .split(|c: char| c == '.' || c == '\n')
                    .next()
                    .unwrap_or("")
                    .trim()
                    .trim_matches('"');
                if !answer.is_empty() && answer.len() < 500 {
                    return answer.to_string();
                }
            }
        }
        output.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let msg = Message::user("Hello");
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.content, "Hello");
    }

    #[test]
    fn test_model_type_detection() {
        assert_eq!(ModelType::detect("gpt-oss-20b.gguf"), ModelType::GptOss);
        assert_eq!(ModelType::detect("llama-3.2.gguf"), ModelType::Llama);
        assert_eq!(ModelType::detect("qwen2.gguf"), ModelType::Qwen);
        assert_eq!(ModelType::detect("other.gguf"), ModelType::Other);
    }
}
