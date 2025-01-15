use reqwest::multipart::Form;
use reqwest::{header::AUTHORIZATION,header::CONTENT_TYPE, Client, Method, RequestBuilder, Response};
use reqwest_eventsource::{CannotCloneRequestError, EventSource, RequestBuilderExt};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::env;
use std::env::VarError;
use std::sync::{LazyLock, RwLock};

pub mod chat;
pub mod completions;
pub mod edits;
pub mod embeddings;
pub mod files;
pub mod models;
pub mod moderations;
pub mod anthrophic_chat;

pub static DEFAULT_BASE_URL: LazyLock<String> =
    LazyLock::new(|| String::from("https://api.openai.com/v1/"));
static DEFAULT_CREDENTIALS: LazyLock<RwLock<Credentials>> =
    LazyLock::new(|| {
        let provider = ApiProvider::OpenAI;
        RwLock::new(Credentials::from_env(provider))});


// Holds the api provider
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum ApiProvider {
    OpenAI,
    Anthropic,
}

/// Holds the API key and base URL for an OpenAI-compatible API.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Credentials {
    provider: ApiProvider,
    api_key: String,
    base_url: String,
}


impl Credentials {
    /// Creates a new Credentials object for a specific provider.
    pub fn new(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        let base_url = parse_base_url(base_url.into());
        let provider = Self::infer_provider(&base_url);

        Self {
            api_key: api_key.into(),
            base_url,
            provider
        }
    }

    /// Fetches credentials from the environment variables for a specific provider.
    /// # Panics
    /// This function panics if the necessary environment variables are missing.
    pub fn from_env(provider:ApiProvider) -> Credentials {
        let (api_key_var, base_url_var) = match provider {
            ApiProvider::OpenAI => ("OPENAI_KEY", "OPENAI_BASE_URL"),
            ApiProvider::Anthropic => ("ANTHROPIC_KEY", "ANTHROPIC_URL"),
        };
        
        let api_key = env::var(api_key_var)
            .unwrap_or_else(|_| panic!("Environment variable {api_key_var} is not set"));
        
        let base_url_unparsed = env::var(base_url_var)
            .unwrap_or_else(|_| panic!("Environment variable {base_url_var} is not set"));

        let base_url = parse_base_url(base_url_unparsed);

        Credentials { api_key, base_url, provider}

    }

    /// Infers the provider based on the base URL.
    fn infer_provider(base_url: &str) -> ApiProvider {
        if base_url.contains("openai") {
            ApiProvider::OpenAI
        } else if base_url.contains("anthropic") {
            ApiProvider::Anthropic
        } else {
            panic!("Unrecognized base URL: {}", base_url);
        }
    }

    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn provider(&self) -> &ApiProvider {
        &self.provider
    }
}


#[derive(Deserialize, Debug, Clone, Eq, PartialEq)]
pub struct OpenAiError {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

impl OpenAiError {
    fn new(message: String, error_type: String) -> OpenAiError {
        OpenAiError {
            message,
            error_type,
            param: None,
            code: None,
        }
    }
}

impl std::fmt::Display for OpenAiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for OpenAiError {}

#[derive(Deserialize, Clone)]
#[serde(untagged)]
pub enum ApiResponse<T> {
    Err { error: OpenAiError },
    Ok(T),
}

#[derive(Deserialize, Clone, Copy, Debug, Eq, PartialEq)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Deserialize, Clone, Copy, Debug, Eq, PartialEq)]
pub struct AnthropicUsage {
    pub input_tokens: u64,
    pub cache_creation_input_tokens: u64,
    pub cache_read_input_tokens: u64,
    pub output_tokens: u64,
}

pub type ApiResponseOrError<T> = Result<T, OpenAiError>;

impl From<reqwest::Error> for OpenAiError {
    fn from(value: reqwest::Error) -> Self {
        OpenAiError::new(value.to_string(), "reqwest".to_string())
    }
}

impl From<std::io::Error> for OpenAiError {
    fn from(value: std::io::Error) -> Self {
        OpenAiError::new(value.to_string(), "io".to_string())
    }
}

// Adding serde error implementation
impl From<serde_json::Error> for OpenAiError {
    fn from(error: serde_json::Error) -> Self {
        OpenAiError::new(
            error.to_string(),
            "json_parse_error".to_string(),
        )
    }
}

async fn openai_request_json<F, T>(
    method: Method,
    route: &str,
    builder: F,
    credentials_opt: Option<Credentials>,
) -> ApiResponseOrError<T>
where
    F: FnOnce(RequestBuilder) -> RequestBuilder,
    T: DeserializeOwned,
{
    let api_response = openai_request(method, route, builder, credentials_opt)
        .await?
        .json()
        .await?;
    match api_response {
        ApiResponse::Ok(t) => Ok(t),
        ApiResponse::Err { error } => Err(error),
    }
}

async fn openai_request<F>(
    method: Method,
    route: &str,
    builder: F,
    credentials_opt: Option<Credentials>,
) -> ApiResponseOrError<Response>
where
    F: FnOnce(RequestBuilder) -> RequestBuilder,
{
    let client = Client::new();
    
    let credentials =
        credentials_opt.unwrap_or_else(|| DEFAULT_CREDENTIALS.read().unwrap().clone());
    let mut request = client.request(method, format!("{}{route}", credentials.base_url));
    request = builder(request);
    let response = request
        .header(AUTHORIZATION, format!("Bearer {}", credentials.api_key))
        .send()
        .await?;
    Ok(response)
}

async fn openai_request_stream<F>(
    method: Method,
    route: &str,
    builder: F,
    credentials_opt: Option<Credentials>,
) -> Result<EventSource, CannotCloneRequestError>
where
    F: FnOnce(RequestBuilder) -> RequestBuilder,
{
    let client = Client::new();
    let credentials =
        credentials_opt.unwrap_or_else(|| DEFAULT_CREDENTIALS.read().unwrap().clone());
    let mut request = client.request(method, format!("{}{route}", credentials.base_url));
    request = builder(request);
    let stream = request
        .header(AUTHORIZATION, format!("Bearer {}", credentials.api_key))
        .eventsource()?;
    Ok(stream)
}

async fn openai_get<T>(route: &str, credentials_opt: Option<Credentials>) -> ApiResponseOrError<T>
where
    T: DeserializeOwned,
{
    openai_request_json(Method::GET, route, |request| request, credentials_opt).await
}

async fn openai_delete<T>(
    route: &str,
    credentials_opt: Option<Credentials>,
) -> ApiResponseOrError<T>
where
    T: DeserializeOwned,
{
    openai_request_json(Method::DELETE, route, |request| request, credentials_opt).await
}

async fn openai_post<J, T>(
    route: &str,
    json: &J,
    credentials_opt: Option<Credentials>,
) -> ApiResponseOrError<T>
where
    J: Serialize + ?Sized,
    T: DeserializeOwned,
{
    openai_request_json(
        Method::POST,
        route,
        |request| request.json(json),
        credentials_opt,
    )
    .await
}

async fn openai_post_multipart<T>(
    route: &str,
    form: Form,
    credentials_opt: Option<Credentials>,
) -> ApiResponseOrError<T>
where
    T: DeserializeOwned,
{
    openai_request_json(
        Method::POST,
        route,
        |request| request.multipart(form),
        credentials_opt,
    )
    .await
}

async fn anthropic_request_json<F, T>(
    method: Method,
    route: &str,
    builder: F,
    credentials_opt: Option<Credentials>,
) -> ApiResponseOrError<T>
where
    F: FnOnce(RequestBuilder) -> RequestBuilder,
    T: DeserializeOwned,
{
    let response = anthropic_request(method, route, builder, credentials_opt)
        .await?;
        // .json()
        // .await?;
    let text = response.text().await?;
    println!("Raw Response: {}", text);

    // Parse the text
    let api_response = serde_json::from_str(&text)?;

    match api_response {
        ApiResponse::Ok(t) => Ok(t),
        ApiResponse::Err { error } => Err(error),
    }
}

async fn anthropic_request<F>(
    method: Method,
    route: &str,
    builder: F,
    credentials_opt: Option<Credentials>,
) -> ApiResponseOrError<Response>
where
    F: FnOnce(RequestBuilder) -> RequestBuilder,
{
    let client = Client::new();
    let credentials =
        credentials_opt.unwrap_or_else(|| DEFAULT_CREDENTIALS.read().unwrap().clone());
    let mut request = client.request(method, format!("{}{route}", credentials.base_url));
    request = builder(request);
    let response = request
        .header("x-api-key", format!("{}", credentials.api_key))
        .header("anthropic-version", "2023-06-01")
        .header(CONTENT_TYPE, format!("application/json"))
        .send()
        .await?;

    Ok(response)
}

async fn anthropic_request_stream<F>(
    method: Method,
    route: &str,
    builder: F,
    credentials_opt: Option<Credentials>,
) -> Result<EventSource, CannotCloneRequestError>
where
    F: FnOnce(RequestBuilder) -> RequestBuilder,
{
    let client = Client::new();
    let credentials =
        credentials_opt.unwrap_or_else(|| DEFAULT_CREDENTIALS.read().unwrap().clone());
    let mut request = client.request(method, format!("{}{route}", credentials.base_url));
    request = builder(request);
    let stream = request
        .header("x-api-key", format!("{}", credentials.api_key))
        .header("anthropic-version", "2023-06-01")
        .header(CONTENT_TYPE, format!("application/json"))
        .eventsource()?;
    Ok(stream)
}

async fn anthropic_post<J, T>(
    route: &str,
    json: &J,
    credentials_opt: Option<Credentials>,
) -> ApiResponseOrError<T>
where
    J: Serialize + ?Sized,
    T: DeserializeOwned,
{
    let resp = anthropic_request_json(
        Method::POST,
        route,
        |request| request.json(json),
        credentials_opt,
    )
    .await;

    resp
}


/// Sets the key for all OpenAI API functions.
///
/// ## Examples
///
/// Use environment variable `OPENAI_KEY` defined from `.env` file:
///
/// ```rust
/// use openai::set_key;
/// use dotenvy::dotenv;
/// use std::env;
///
/// dotenv().ok();
/// set_key(env::var("OPENAI_KEY").unwrap());
/// ```
#[deprecated(
    since = "1.0.0-alpha.16",
    note = "use the `Credentials` struct instead"
)]
pub fn set_key(value: String) {
    let mut credentials = DEFAULT_CREDENTIALS.write().unwrap();
    credentials.api_key = value;
}

/// Sets the base url for all OpenAI API functions.
///
/// ## Examples
///
/// Use environment variable `OPENAI_BASE_URL` defined from `.env` file:
///
/// ```rust
/// use openai::set_base_url;
/// use dotenvy::dotenv;
/// use std::env;
///
/// dotenv().ok();
/// set_base_url(env::var("OPENAI_BASE_URL").unwrap_or_default());
/// ```
#[deprecated(
    since = "1.0.0-alpha.16",
    note = "use the `Credentials` struct instead"
)]
pub fn set_base_url(mut value: String) {
    if value.is_empty() {
        return;
    }
    value = parse_base_url(value);
    let mut credentials = DEFAULT_CREDENTIALS.write().unwrap();
    credentials.base_url = value;
}

fn parse_base_url(mut value: String) -> String {
    if !value.ends_with('/') {
        value += "/";
    }
    value
}

#[cfg(test)]
pub mod tests {
    pub const DEFAULT_LEGACY_MODEL: &str = "gpt-3.5-turbo-instruct";
}
