//! Given a chat conversation, the model will return a chat completion response.

use super::{anthropic_post, ApiResponseOrError, Credentials, AnthropicUsage, chat::{ChatCompletionMessage, ChatCompletionMessageRole, ChatCompletionResponseFormat, ChatCompletionFunctionDefinition, ToolCall, ChatCompletionFunctionCallDelta}};
use crate::anthropic_request_stream;
use derive_builder::Builder;
use futures_util::StreamExt;
use reqwest::Method;
use reqwest_eventsource::{CannotCloneRequestError, Event, EventSource};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use tokio::sync::mpsc::{channel, Receiver, Sender};

/// A Anthropic Full Chat Completion
pub type AnthropicChatCompletion = AnthropicChatCompletionGeneric<AnthropicChatCompletionContent>;

/// A delta chat completion, which is streamed token by token.
pub type AnthropicChatCompletionDelta = AnthropicChatCompletionGeneric<AnthropicChatCompletionContentDelta>;

#[derive(Deserialize, Clone, Debug, Eq, PartialEq)]
pub struct AnthropicChatCompletionGeneric<C> {
    pub id: String,
    #[serde(rename = "type")]
    pub typ: String,
    pub role: String,
    pub model: String,
    pub content: Vec<C>,
    pub stop_reason: String,
    pub stop_sequence: Option<String>,
    pub usage: Option<AnthropicUsage>,
}

#[derive(Deserialize, Clone, Debug, Eq, PartialEq)]
pub struct AnthropicChatCompletionContent {
    #[serde(rename="type")]
    pub typ: String,
    pub text: String,
}


#[derive(Deserialize, Clone, Debug, Eq, PartialEq)]
pub struct AnthropicChatCompletionContentDelta {
    #[serde(rename="type")]
    pub typ: Option<String>,
    pub text: String,
}


/// Same as ChatCompletionMessage, but received during a response stream.
#[derive(Deserialize, Clone, Debug, Eq, PartialEq)]
pub struct ChatCompletionMessageDelta {
    /// The role of the author of this message.
    pub role: Option<ChatCompletionMessageRole>,
    /// The contents of the message
    pub content: Option<String>,
    /// The name of the user in a multi-user chat
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The function that ChatGPT called
    ///
    /// [API Reference](https://platform.openai.com/docs/api-reference/chat/create#chat/create-function_call)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<ChatCompletionFunctionCallDelta>,
    /// Tool call that this message is responding to.
    /// Required if the role is `Tool`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Tool calls that the assistant is requesting to invoke.
    /// Can only be populated if the role is `Assistant`,
    /// otherwise it should be empty.
    #[serde(
        skip_serializing_if = "<[_]>::is_empty",
        default = "default_tool_calls_deserialization"
    )]
    pub tool_calls: Vec<ToolCall>,
}


#[derive(Serialize, Builder, Debug, Clone)]
#[builder(derive(Clone, Debug, PartialEq))]
#[builder(pattern = "owned")]
#[builder(name = "AnthropicChatCompletionBuilder")]
#[builder(setter(strip_option, into))]
pub struct AnthropicChatCompletionRequest {
    model: String,
    system: Option<String>,
    messages: Vec<ChatCompletionMessage>,

    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,

    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    
    /// How many chat completion choices to generate for each input message.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    n: Option<u8>,
    
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    
    /// Up to 4 sequences where the API will stop generating further tokens.
    #[builder(default)]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop: Vec<String>,
    
    /// This feature is in Beta. If specified, our system will make a best effort to sample deterministically, such that repeated requests with the same seed and parameters should return the same result. Determinism is not guaranteed, and you should refer to the system_fingerprint response parameter to monitor changes in the backend.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
    
    /// The maximum number of tokens allowed for the generated answer. By default, the number of tokens the model can return will be (4096 - prompt tokens).
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<i32>,
    
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,

    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    logit_bias: Option<HashMap<String, f32>>,
    /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. [Learn more](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids).
    #[builder(default)]
    #[serde(skip_serializing_if = "String::is_empty")]
    user: String,
    
    #[builder(default)]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    functions: Vec<ChatCompletionFunctionDefinition>,
    
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    function_call: Option<Value>,
    
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ChatCompletionResponseFormat>,
    /// The credentials to use for this request.
    #[serde(skip_serializing)]
    #[builder(default)]
    credentials: Option<Credentials>,
}

impl<C> AnthropicChatCompletionGeneric<C> {
    /// Creates a new builder for Anthropic chat completion requests
    /// 
    /// # Arguments
    /// * `model` - The model to use (e.g. "claude-2")
    /// * `system` - The system prompt/instructions
    /// * `messages` - Vector of chat messages for the conversation    
    pub fn builder(
        model: &str,
        system: &str,
        messages: impl Into<Vec<ChatCompletionMessage>>,
    ) -> AnthropicChatCompletionBuilder {
        AnthropicChatCompletionBuilder::create_empty()
            .model(model)
            .system(String::from(system))
            .messages(messages)
            .max_tokens(4096)
    }
}


impl AnthropicChatCompletion {
    /// Makes a POST request to create a new chat completion
    /// 
    /// # Arguments
    /// * `request` - The chat completion request parameters
    pub async fn create(request: AnthropicChatCompletionRequest) -> ApiResponseOrError<Self> {
        let credentials_opt = request.credentials.clone();
        anthropic_post("messages", &request, credentials_opt).await
    }
}

impl AnthropicChatCompletionBuilder {
    /// Builds and executes the chat completion request
    pub async fn create(self) -> ApiResponseOrError<AnthropicChatCompletion> {
        let resp = AnthropicChatCompletion::create(self.build().unwrap()).await;
        resp
    }
}

fn default_tool_calls_deserialization() -> Vec<ToolCall> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use dotenvy::dotenv;

    #[tokio::test]
    async fn anthropic_chat() {
        dotenv().ok();
        let credentials = Credentials::from_env(crate::ApiProvider::Anthropic);

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

        assert_eq!(
            chat_completion
                .content
                .first()
                .unwrap()
                .text
                .clone()
                .trim(),
            "Hi there! How can I help you today?"
        );
    }
}
