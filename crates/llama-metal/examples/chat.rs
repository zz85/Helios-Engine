//! Simple chat example
//!
//! Usage: cargo run --example chat -- /path/to/model.gguf "Your question here"

use llama_metal::{LlamaModel, Message};
use std::env;


#[tokio::main]
async fn main() -> Result<(), llama_metal::Error> {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 3 {
        eprintln!("Usage: {} <model.gguf> <prompt>", args[0]);
        std::process::exit(1);
    }
    
    let model_path = &args[1];
    let prompt = &args[2];
    
    println!("Loading model...");
    let model = LlamaModel::load(model_path).await?;
    
    let messages = vec![
        Message::system("You are a helpful assistant. Be concise."),
        Message::user(prompt),
    ];
    
    // Use non-streaming for clean output (applies GPT-OSS extraction)
    let response = model.chat(&messages).await?;
    println!("Assistant: {}", response);
    Ok(())
}
