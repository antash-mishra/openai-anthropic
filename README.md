# openai-anthropic

[![crates.io](https://img.shields.io/crates/v/openai.svg)](https://crates.io/crates/openai/)
[![Rust workflow](https://github.com/rellfy/openai/actions/workflows/test.yml/badge.svg)](https://github.com/rellfy/openai/actions/workflows/test.yml)

An unofficial Rust library for OpenAI and Anthropic API.

> **Warning**
>
> There may be breaking changes between versions while in alpha.
> See [Implementation Progress](#implementation-progress).


- Anthropic Support added.
- Chat Completion Added (Streaming completion feature to add.)


## Examples

Examples can be found in the `examples` directory.

Please note that examples are not available for all the crate's functionality,
PRs are appreciated to expand the coverage.

Currently, there are examples for the `completions` module and the `chat`
module for both OpenAI and Anthropic APIs. For other modules, refer to the `tests` submodules for some reference.



### Chat Example

```rust
// Relies on OPENAI_KEY and optionally OPENAI_BASE_URL.
let credentials = Credentials::from_env(crate::ApiProvider::OpenAI);
let messages = vec![
    ChatCompletionMessage {
        role: ChatCompletionMessageRole::System,
        content: Some("You are a helpful assistant.".to_string()),
        name: None,
        function_call: None,
    },
    ChatCompletionMessage {
        role: ChatCompletionMessageRole::User,
        content: Some("Tell me a random crab fact".to_string()),
        name: None,
        function_call: None,
    },
];
let chat_completion = ChatCompletion::builder("gpt-4o", messages.clone())
    .credentials(credentials.clone())
    .create()
    .await
    .unwrap();
let returned_message = chat_completion.choices.first().unwrap().message.clone();
// Assistant: Sure! Here's a random crab fact: ...
println!(
    "{:#?}: {}",
    returned_message.role,
    returned_message.content.unwrap().trim()
);
```
### Anthropic Completion Example

```rust
// Relies on ANTHROPIC_KEY and optionally ANTHROPIC_BASE_URL.
let credentials = Credentials::from_env(crate::ApiProvider::Anthropic);
let prompt = "You are a helpful assistant. Tell me a random crab fact.";
let chat_completion = AnthropicChatCompletion::builder(
            "claude-3-5-sonnet-20241022",
            "",
            [ChatCompletionMessage {
                role: ChatCompletionMessageRole::User,
                content: Some("Hello!".to_string()),
                name: None,
                function_call: None,
                tool_call_id: None,
                tool_calls: Vec::new(),
            }]
        )
        .credentials(credentials)
        .temperature(0.0)
        .create()
        .await
        .unwrap();

let returned_message = chat_completion.content
        .first()
        .unwrap()
        .text
        .clone()
        .trim()
// Assistant: Here's a random crab fact: ...
println!("Completion: {}\n Role: {}", returned_message, chat_completion.role);
```


## Implementation Progress

`██████████` Models

`████████░░` Completions (Function calling is supported)

`████████░░` Chat

`██████████` Edits

`░░░░░░░░░░` Images

`█████████░` Embeddings

`░░░░░░░░░░` Audio

`███████░░░` Files

`░░░░░░░░░░` Fine-tunes

`██████████` Moderations

## Contributing

All contributions are welcome. Unit tests are encouraged.

> **Fork Notice**
>
> This package was initially developed by [Valentine Briese](https://github.com/valentinegb/openai).
> As the original repo was archived, this is a fork and continuation of the project.
