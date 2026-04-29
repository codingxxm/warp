use futures::StreamExt;
use reqwest_eventsource::{Event, EventSource};
use serde::Deserialize;

use crate::error::DirectLlmError;
use crate::types::{ChatMessage, MessageRole, ProviderConfig, ProviderToolCall, ToolDefinition};

#[derive(Debug, Deserialize)]
struct AnthropicStreamMessage {
    id: Option<String>,
    model: Option<String>,
    role: Option<String>,
    content: Option<Vec<AnthropicContentBlock>>,
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "thinking")]
    Thinking { thinking: String },
}

#[derive(Debug, Deserialize)]
struct AnthropicContentBlockStart {
    index: Option<u64>,
    content_block: AnthropicContentBlockStartInner,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum AnthropicContentBlockStartInner {
    #[serde(rename = "text")]
    Text { text: Option<String> },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Option<serde_json::Value>,
    },
    #[serde(rename = "thinking")]
    Thinking { thinking: Option<String> },
}

#[derive(Debug, Deserialize)]
struct AnthropicContentBlockDelta {
    index: Option<u64>,
    delta: AnthropicContentBlockDeltaInner,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum AnthropicContentBlockDeltaInner {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
    #[serde(rename = "thinking_delta")]
    ThinkingDelta { thinking: String },
}

#[derive(Debug, Clone)]
pub enum AnthropicStreamEvent {
    MessageStart {
        message_id: String,
        model: String,
    },
    TextDelta(String),
    ToolCallStart {
        index: u64,
        id: String,
        name: String,
    },
    ToolCallArgumentDelta {
        index: u64,
        partial_json: String,
    },
    ToolCallEnd {
        index: u64,
        id: String,
        name: String,
        arguments: String,
    },
    MessageEnd {
        stop_reason: String,
    },
}

pub struct AnthropicClient {
    http_client: reqwest::Client,
}

impl AnthropicClient {
    pub fn new(http_client: reqwest::Client) -> Self {
        Self { http_client }
    }

    fn build_messages(&self, messages: &[ChatMessage]) -> (Option<String>, Vec<serde_json::Value>) {
        let mut system_prompt = None;
        let mut api_messages = Vec::new();

        for msg in messages {
            if msg.role == MessageRole::System {
                system_prompt = Some(msg.content.clone());
                continue;
            }

            if msg.role == MessageRole::Tool {
                api_messages.push(serde_json::json!({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id.as_deref().unwrap_or(""),
                        "content": msg.content,
                    }],
                }));
                continue;
            }

            if msg.role == MessageRole::Assistant && msg.tool_calls.is_some() {
                let mut content_blocks: Vec<serde_json::Value> = Vec::new();

                if !msg.content.is_empty() {
                    content_blocks.push(serde_json::json!({
                        "type": "text",
                        "text": msg.content,
                    }));
                }

                for tc in msg.tool_calls.as_ref().unwrap() {
                    content_blocks.push(serde_json::json!({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": serde_json::from_str::<serde_json::Value>(&tc.arguments)
                            .unwrap_or(serde_json::Value::Object(serde_json::Map::new())),
                    }));
                }

                api_messages.push(serde_json::json!({
                    "role": "assistant",
                    "content": content_blocks,
                }));
                continue;
            }

            api_messages.push(serde_json::json!({
                "role": match msg.role {
                    MessageRole::User => "user",
                    MessageRole::Assistant => "assistant",
                    _ => "user",
                },
                "content": msg.content,
            }));
        }

        (system_prompt, api_messages)
    }

    fn build_tools(&self, tools: &[ToolDefinition]) -> Vec<serde_json::Value> {
        tools.iter().map(|t| t.anthropic_format()).collect()
    }

    pub async fn chat_stream(
        &self,
        config: &ProviderConfig,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> Result<futures::stream::BoxStream<'static, Result<AnthropicStreamEvent, DirectLlmError>>, DirectLlmError> {
        let url = format!("{}/v1/messages", config.base_url.trim_end_matches('/'));

        let (system_prompt, api_messages) = self.build_messages(messages);
        let json_tools = self.build_tools(tools);

        let mut body = serde_json::json!({
            "model": config.model,
            "messages": api_messages,
            "max_tokens": 16384,
            "stream": true,
        });

        if let Some(system) = system_prompt {
            body["system"] = serde_json::Value::String(system);
        }

        if !json_tools.is_empty() {
            body["tools"] = serde_json::Value::Array(json_tools);
        }

        let request = self
            .http_client
            .post(&url)
            .header("x-api-key", &config.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&body);

        let event_source = EventSource::new(request)?;

        let stream = event_source.filter_map(|result| async move {
            match result {
                Ok(Event::Open) => None,
                Ok(Event::Message(message)) => {
                    let event_type = message.event.clone();
                    let data = message.data.clone();

                    match event_type.as_str() {
                        "message_start" => {
                            match serde_json::from_str::<serde_json::Value>(&data) {
                                Ok(parsed) => {
                                    let msg = &parsed["message"];
                                    let message_id = msg["id"].as_str().unwrap_or("").to_string();
                                    let model = msg["model"].as_str().unwrap_or("").to_string();
                                    Some(Ok(AnthropicStreamEvent::MessageStart { message_id, model }))
                                }
                                Err(e) => Some(Err(DirectLlmError::ParseError(format!(
                                    "Failed to parse Anthropic message_start: {e}"
                                )))),
                            }
                        }
                        "content_block_start" => {
                            match serde_json::from_str::<AnthropicContentBlockStart>(&data) {
                                Ok(parsed) => {
                                    let index = parsed.index.unwrap_or(0);
                                    match parsed.content_block {
                                        AnthropicContentBlockStartInner::Text { text } => {
                                            text.filter(|t| !t.is_empty())
                                                .map(|t| Ok(AnthropicStreamEvent::TextDelta(t)))
                                        }
                                        AnthropicContentBlockStartInner::ToolUse { id, name, .. } => {
                                            Some(Ok(AnthropicStreamEvent::ToolCallStart {
                                                index,
                                                id,
                                                name,
                                            }))
                                        }
                                        AnthropicContentBlockStartInner::Thinking { .. } => None,
                                    }
                                }
                                Err(e) => Some(Err(DirectLlmError::ParseError(format!(
                                    "Failed to parse Anthropic content_block_start: {e}"
                                )))),
                            }
                        }
                        "content_block_delta" => {
                            match serde_json::from_str::<AnthropicContentBlockDelta>(&data) {
                                Ok(parsed) => {
                                    let index = parsed.index.unwrap_or(0);
                                    match parsed.delta {
                                        AnthropicContentBlockDeltaInner::TextDelta { text } => {
                                            Some(Ok(AnthropicStreamEvent::TextDelta(text)))
                                        }
                                        AnthropicContentBlockDeltaInner::InputJsonDelta { partial_json } => {
                                            Some(Ok(AnthropicStreamEvent::ToolCallArgumentDelta {
                                                index,
                                                partial_json,
                                            }))
                                        }
                                        AnthropicContentBlockDeltaInner::ThinkingDelta { .. } => None,
                                    }
                                }
                                Err(e) => Some(Err(DirectLlmError::ParseError(format!(
                                    "Failed to parse Anthropic content_block_delta: {e}"
                                )))),
                            }
                        }
                        "content_block_stop" => None,
                        "message_delta" => {
                            match serde_json::from_str::<serde_json::Value>(&data) {
                                Ok(parsed) => {
                                    let stop_reason = parsed["delta"]["stop_reason"]
                                        .as_str()
                                        .unwrap_or("end_turn")
                                        .to_string();
                                    Some(Ok(AnthropicStreamEvent::MessageEnd { stop_reason }))
                                }
                                Err(e) => Some(Err(DirectLlmError::ParseError(format!(
                                    "Failed to parse Anthropic message_delta: {e}"
                                )))),
                            }
                        }
                        "message_stop" => {
                            Some(Ok(AnthropicStreamEvent::MessageEnd {
                                stop_reason: "end_turn".to_string(),
                            }))
                        }
                        "ping" => None,
                        _ => None,
                    }
                }
                Err(e) => Some(Err(DirectLlmError::Other(format!(
                    "SSE stream error: {e}"
                )))),
            }
        });

        let accumulated = accumulate_anthropic_tool_calls(stream);
        Ok(accumulated.boxed())
    }

    /// Non-streaming chat for simpler use cases.
    pub async fn chat(
        &self,
        config: &ProviderConfig,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> Result<(Option<String>, Vec<ProviderToolCall>), DirectLlmError> {
        let url = format!("{}/v1/messages", config.base_url.trim_end_matches('/'));

        let (system_prompt, api_messages) = self.build_messages(messages);
        let json_tools = self.build_tools(tools);

        let mut body = serde_json::json!({
            "model": config.model,
            "messages": api_messages,
            "max_tokens": 16384,
        });

        if let Some(system) = system_prompt {
            body["system"] = serde_json::Value::String(system);
        }

        if !json_tools.is_empty() {
            body["tools"] = serde_json::Value::Array(json_tools);
        }

        let response = self
            .http_client
            .post(&url)
            .header("x-api-key", &config.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(DirectLlmError::from_reqwest_status(status, body, "Anthropic"));
        }

        let text = response.text().await?;
        let parsed: AnthropicStreamMessage = serde_json::from_str(&text).map_err(|e| {
            DirectLlmError::ParseError(format!("Failed to parse Anthropic response: {e}"))
        })?;

        let mut result_text = None;
        let mut tool_calls = Vec::new();

        if let Some(content) = parsed.content {
            for block in content {
                match block {
                    AnthropicContentBlock::Text { text } => {
                        result_text = Some(text);
                    }
                    AnthropicContentBlock::ToolUse { id, name, input } => {
                        tool_calls.push(ProviderToolCall {
                            id,
                            name,
                            arguments: input.to_string(),
                        });
                    }
                    AnthropicContentBlock::Thinking { .. } => {}
                }
            }
        }

        Ok((result_text, tool_calls))
    }
}

fn accumulate_anthropic_tool_calls(
    stream: impl futures::Stream<Item = Result<AnthropicStreamEvent, DirectLlmError>> + Send + 'static,
) -> impl futures::Stream<Item = Result<AnthropicStreamEvent, DirectLlmError>> + Send + 'static {
    futures::stream::unfold(
        (stream.boxed(), AnthropicToolCallAccumulator::new()),
        |(mut stream, mut acc)| async move {
            loop {
                let item = stream.next().await;
                match item {
                    None => return None,
                    Some(Err(e)) => return Some((Err(e), (stream, acc))),
                    Some(Ok(event)) => match event {
                        AnthropicStreamEvent::MessageStart { message_id, model } => {
                            return Some((
                                Ok(AnthropicStreamEvent::MessageStart { message_id, model }),
                                (stream, acc),
                            ));
                        }
                        AnthropicStreamEvent::TextDelta(text) => {
                            if text.is_empty() {
                                continue;
                            }
                            return Some((
                                Ok(AnthropicStreamEvent::TextDelta(text)),
                                (stream, acc),
                            ));
                        }
                        AnthropicStreamEvent::ToolCallStart { index, id, name } => {
                            acc.start_tool_call(index, id, name);
                            continue;
                        }
                        AnthropicStreamEvent::ToolCallArgumentDelta { index, partial_json } => {
                            acc.append_arguments(index, partial_json);
                            continue;
                        }
                        AnthropicStreamEvent::ToolCallEnd { .. } => {
                            continue;
                        }
                        AnthropicStreamEvent::MessageEnd { stop_reason } => {
                            let completed = acc.flush();
                            if completed.is_empty() {
                                return Some((
                                    Ok(AnthropicStreamEvent::MessageEnd { stop_reason }),
                                    (stream, acc),
                                ));
                            }
                            let mut events: Vec<AnthropicStreamEvent> = completed;
                            events.push(AnthropicStreamEvent::MessageEnd { stop_reason });
                            let first = events.remove(0);
                            acc.pending = events;
                            return Some((Ok(first), (stream, acc)));
                        }
                    },
                }
            }
        },
    )
}

struct AnthropicToolCallAccumulator {
    tool_calls: std::collections::HashMap<u64, (String, String, String)>,
    pending: Vec<AnthropicStreamEvent>,
}

impl AnthropicToolCallAccumulator {
    fn new() -> Self {
        Self {
            tool_calls: std::collections::HashMap::new(),
            pending: Vec::new(),
        }
    }

    fn start_tool_call(&mut self, index: u64, id: String, name: String) {
        self.tool_calls.insert(index, (id, name, String::new()));
    }

    fn append_arguments(&mut self, index: u64, partial_json: String) {
        if let Some((_, _, args)) = self.tool_calls.get_mut(&index) {
            args.push_str(&partial_json);
        }
    }

    fn flush(&mut self) -> Vec<AnthropicStreamEvent> {
        let mut results: Vec<AnthropicStreamEvent> = self.pending.clone();
        self.pending.clear();
        for (index, (id, name, arguments)) in self.tool_calls.drain() {
            results.push(AnthropicStreamEvent::ToolCallEnd {
                index,
                id,
                name,
                arguments,
            });
        }
        results
    }
}