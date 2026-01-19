# llama-metal

Minimal llama.cpp wrapper with Metal GPU support for local LLM inference.

## Features

- Simple API for chat completions
- Streaming support
- Metal GPU acceleration (automatic on macOS)
- GPT-OSS model support with reasoning extraction
- ChatML format support (Llama, Qwen, etc.)

## Usage

```rust
use llama_metal::{LlamaModel, Message};

#[tokio::main]
async fn main() -> Result<(), llama_metal::Error> {
    // Load model
    let model = LlamaModel::load("path/to/model.gguf").await?;
    
    // Create messages
    let messages = vec![
        Message::system("You are a helpful assistant."),
        Message::user("What is 2+2?"),
    ];
    
    // Non-streaming
    let response = model.chat(&messages).await?;
    println!("{}", response);
    
    // Streaming
    model.chat_stream(&messages, |token| {
        print!("{}", token);
    }).await?;
    
    Ok(())
}
```

## Configuration

```rust
use llama_metal::{LlamaModel, Config};

let config = Config {
    n_gpu_layers: 99,    // Layers to offload to GPU
    context_size: 4096,  // Context window size
    max_tokens: 1024,    // Max tokens to generate
};

let model = LlamaModel::load_with_config("model.gguf", config).await?;
```

## Supported Models

- GPT-OSS (with automatic reasoning extraction)
- Llama 2/3
- Qwen/Qwen2
- Any GGUF model using ChatML format

## License

MIT
