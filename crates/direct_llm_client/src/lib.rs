pub mod adapter;
pub mod anthropic;
pub mod error;
pub mod openai;
pub mod types;

use crate::adapter::ResponseStream;
use crate::anthropic::AnthropicClient;
use crate::error::DirectLlmError;
use crate::openai::OpenAiClient;
use crate::types::{ChatMessage, ProviderConfig, ProviderType, StreamContext, ToolDefinition};

pub struct DirectLlmClient {
    openai_client: OpenAiClient,
    anthropic_client: AnthropicClient,
}

impl DirectLlmClient {
    pub fn new() -> Self {
        let http_client = reqwest::Client::new();
        Self {
            openai_client: OpenAiClient::new(http_client.clone()),
            anthropic_client: AnthropicClient::new(http_client),
        }
    }

    /// Send a chat request with optional tool definitions and get a
    /// ResponseEvent stream in Warp's internal format. This is the drop-in
    /// replacement for `ServerApi::generate_multi_agent_output()`.
    pub async fn chat_stream(
        &self,
        provider_type: ProviderType,
        config: &ProviderConfig,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        ctx: &StreamContext,
    ) -> Result<ResponseStream, DirectLlmError> {
        match provider_type {
            ProviderType::OpenAiCompatible => {
                let stream = self
                    .openai_client
                    .chat_stream(config, messages, tools)
                    .await?;
                Ok(adapter::from_openai_stream(stream, ctx))
            }
            ProviderType::Anthropic => {
                let stream = self
                    .anthropic_client
                    .chat_stream(config, messages, tools)
                    .await?;
                Ok(adapter::from_anthropic_stream(stream, ctx))
            }
        }
    }

    /// Non-streaming chat for simpler use cases.
    pub async fn chat(
        &self,
        provider_type: ProviderType,
        config: &ProviderConfig,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> Result<(Option<String>, Vec<crate::types::ProviderToolCall>), DirectLlmError> {
        match provider_type {
            ProviderType::OpenAiCompatible => {
                self.openai_client.chat(config, messages, tools).await
            }
            ProviderType::Anthropic => {
                self.anthropic_client.chat(config, messages, tools).await
            }
        }
    }
}

/// Determine the provider type and config based on the user's API key settings.
/// Returns None if no custom base URL is configured.
pub fn resolve_provider(
    anthropic_key: Option<&str>,
    anthropic_base_url: Option<&str>,
    openai_key: Option<&str>,
    openai_base_url: Option<&str>,
    model: Option<&str>,
) -> Option<(ProviderType, ProviderConfig)> {
    // Anthropic takes priority if both are configured.
    if let (Some(url), Some(key)) = (anthropic_base_url, anthropic_key) {
        return Some((
            ProviderType::Anthropic,
            ProviderConfig {
                base_url: url.to_string(),
                api_key: key.to_string(),
                model: model
                    .map(String::from)
                    .unwrap_or_else(|| "claude-sonnet-4-20250514".to_string()),
            },
        ));
    }

    if let (Some(url), Some(key)) = (openai_base_url, openai_key) {
        return Some((
            ProviderType::OpenAiCompatible,
            ProviderConfig {
                base_url: url.to_string(),
                api_key: key.to_string(),
                model: model
                    .map(String::from)
                    .unwrap_or_else(|| "gpt-4o".to_string()),
            },
        ));
    }

    None
}