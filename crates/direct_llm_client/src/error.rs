use thiserror::Error;

#[derive(Error, Debug)]
pub enum DirectLlmError {
    #[error("Invalid API key for {provider}")]
    InvalidApiKey { provider: String, model_name: String },

    #[error("Rate limited by {provider}")]
    RateLimited { provider: String },

    #[error("Transport error: {0}")]
    Transport(String),

    #[error("Failed to parse provider response: {0}")]
    ParseError(String),

    #[error("Context window exceeded")]
    ContextWindowExceeded,

    #[error("Provider returned an error: status {status}, message: {message}")]
    ProviderError { status: u16, message: String },

    #[error("Stream ended unexpectedly")]
    StreamEndedUnexpectedly,

    #[error("Request could not be cloned for retry")]
    CannotCloneRequest,

    #[error("{0}")]
    Other(String),
}

impl DirectLlmError {
    pub fn from_reqwest_status(status: reqwest::StatusCode, body: String, provider: &str) -> Self {
        match status.as_u16() {
            401 | 403 => DirectLlmError::InvalidApiKey {
                provider: provider.to_string(),
                model_name: String::new(),
            },
            429 => DirectLlmError::RateLimited {
                provider: provider.to_string(),
            },
            _ => DirectLlmError::ProviderError {
                status: status.as_u16(),
                message: body,
            },
        }
    }
}

impl From<reqwest_eventsource::CannotCloneRequestError> for DirectLlmError {
    fn from(_: reqwest_eventsource::CannotCloneRequestError) -> Self {
        DirectLlmError::CannotCloneRequest
    }
}

impl From<reqwest::Error> for DirectLlmError {
    fn from(err: reqwest::Error) -> Self {
        DirectLlmError::Transport(err.to_string())
    }
}