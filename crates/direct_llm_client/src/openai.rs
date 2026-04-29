use futures::StreamExt;
use reqwest_eventsource::{Event, EventSource};
use serde::Deserialize;

use crate::error::DirectLlmError;
use crate::types::{ChatMessage, MessageRole, ProviderConfig, ProviderToolCall, ToolDefinition};

#[derive(Debug, Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    message: OpenAiMessage,
}

#[derive(Debug, Deserialize)]
struct OpenAiMessage {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAiToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OpenAiToolCall {
    id: String,
    function: OpenAiFunction,
}

#[derive(Debug, Deserialize)]
struct OpenAiFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamDelta {
    role: Option<String>,
    content: Option<String>,
    tool_calls: Option<Vec<OpenAiStreamToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamToolCall {
    index: u64,
    id: Option<String>,
    function: Option<OpenAiStreamFunction>,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamFunction {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamChoice {
    delta: OpenAiStreamDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamChunk {
    choices: Vec<OpenAiStreamChoice>,
}

#[derive(Debug, Clone)]
pub enum OpenAiStreamEvent {
    TextDelta(String),
    ToolCallStart {
        index: u64,
        id: String,
        name: String,
    },
    ToolCallArgumentDelta {
        index: u64,
        arguments: String,
    },
    ToolCallEnd {
        index: u64,
        id: String,
        name: String,
        arguments: String,
    },
    Done,
}

pub struct OpenAiClient {
    http_client: reqwest::Client,
}

impl OpenAiClient {
    pub fn new(http_client: reqwest::Client) -> Self {
        Self { http_client }
    }

    fn build_messages(&self, messages: &[ChatMessage]) -> Vec<serde_json::Value> {
        messages
            .iter()
            .map(|msg| {
                let role = match msg.role {
                    MessageRole::System => "system",
                    MessageRole::User => "user",
                    MessageRole::Assistant => "assistant",
                    MessageRole::Tool => "tool",
                };

                if msg.role == MessageRole::Tool {
                    let mut val = serde_json::json!({
                        "role": "tool",
                        "content": msg.content,
                        "tool_call_id": msg.tool_call_id.as_deref().unwrap_or(""),
                    });
                    if let Some(name) = &msg.name {
                        val["name"] = serde_json::Value::String(name.clone());
                    }
                    val
                } else if msg.role == MessageRole::Assistant && msg.tool_calls.is_some() {
                    let tool_calls: Vec<serde_json::Value> = msg
                        .tool_calls
                        .as_ref()
                        .unwrap()
                        .iter()
                        .map(|tc| {
                            serde_json::json!({
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": tc.arguments,
                                }
                            })
                        })
                        .collect();
                    serde_json::json!({
                        "role": "assistant",
                        "content": msg.content,
                        "tool_calls": tool_calls,
                    })
                } else {
                    serde_json::json!({
                        "role": role,
                        "content": msg.content,
                    })
                }
            })
            .collect()
    }

    fn build_tools(&self, tools: &[ToolDefinition]) -> Vec<serde_json::Value> {
        tools.iter().map(|t| t.openai_format()).collect()
    }

    pub async fn chat_stream(
        &self,
        config: &ProviderConfig,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> Result<futures::stream::BoxStream<'static, Result<OpenAiStreamEvent, DirectLlmError>>, DirectLlmError> {
        let url = format!("{}/chat/completions", config.base_url.trim_end_matches('/'));

        let json_messages = self.build_messages(messages);
        let json_tools = self.build_tools(tools);

        let mut body = serde_json::json!({
            "model": config.model,
            "messages": json_messages,
            "stream": true,
        });

        if !json_tools.is_empty() {
            body["tools"] = serde_json::Value::Array(json_tools);
        }

        let request = self
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", config.api_key))
            .header("Content-Type", "application/json")
            .json(&body);

        let event_source = EventSource::new(request)?;

        let stream = event_source.filter_map(|result| async move {
            match result {
                Ok(Event::Open) => None,
                Ok(Event::Message(message)) => {
                    if message.data == "[DONE]" {
                        return Some(Ok(OpenAiStreamEvent::Done));
                    }

                    let chunk: OpenAiStreamChunk = match serde_json::from_str(&message.data) {
                        Ok(c) => c,
                        Err(e) => {
                            return Some(Err(DirectLlmError::ParseError(format!(
                                "Failed to parse OpenAI stream chunk: {e}"
                            ))));
                        }
                    };

                    for choice in &chunk.choices {
                        if let Some(reason) = &choice.finish_reason {
                            if reason == "stop" || reason == "end_turn" {
                                return Some(Ok(OpenAiStreamEvent::Done));
                            }
                            // "tool_calls" finish_reason means all tool calls are complete.
                            // We should flush accumulated tool calls.
                            if reason == "tool_calls" {
                                // The tool call accumulator is handled externally.
                                // Just signal Done — the accumulator will flush.
                                return Some(Ok(OpenAiStreamEvent::Done));
                            }
                        }

                        if let Some(content) = &choice.delta.content {
                            if !content.is_empty() {
                                return Some(Ok(OpenAiStreamEvent::TextDelta(content.clone())));
                            }
                        }

                        if let Some(tool_calls) = &choice.delta.tool_calls {
                            for tc in tool_calls {
                                // Tool call start: has id + name
                                if let (Some(id), Some(func)) = (&tc.id, &tc.function) {
                                    if let Some(name) = &func.name {
                                        return Some(Ok(OpenAiStreamEvent::ToolCallStart {
                                            index: tc.index,
                                            id: id.clone(),
                                            name: name.clone(),
                                        }));
                                    }
                                    // Argument delta after start (id present but no name)
                                    if let Some(arguments) = &func.arguments {
                                        return Some(Ok(OpenAiStreamEvent::ToolCallArgumentDelta {
                                            index: tc.index,
                                            arguments: arguments.clone(),
                                        }));
                                    }
                                }
                                // Argument delta without id (continuing a tool call)
                                else if let Some(func) = &tc.function {
                                    if let Some(arguments) = &func.arguments {
                                        return Some(Ok(OpenAiStreamEvent::ToolCallArgumentDelta {
                                            index: tc.index,
                                            arguments: arguments.clone(),
                                        }));
                                    }
                                }
                            }
                        }
                    }

                    None // Empty delta, skip
                }
                Err(e) => Some(Err(DirectLlmError::Other(format!(
                    "SSE stream error: {e}"
                )))),
            }
        });

        // Accumulate partial tool calls and emit complete ToolCallEnd events
        let accumulated = accumulate_tool_calls(stream);
        Ok(accumulated.boxed())
    }

    /// Non-streaming chat for simpler use cases.
    pub async fn chat(
        &self,
        config: &ProviderConfig,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> Result<(Option<String>, Vec<ProviderToolCall>), DirectLlmError> {
        let url = format!("{}/chat/completions", config.base_url.trim_end_matches('/'));

        let json_messages = self.build_messages(messages);
        let json_tools = self.build_tools(tools);

        let mut body = serde_json::json!({
            "model": config.model,
            "messages": json_messages,
        });

        if !json_tools.is_empty() {
            body["tools"] = serde_json::Value::Array(json_tools);
        }

        let response = self
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", config.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(DirectLlmError::from_reqwest_status(
                status,
                body,
                "OpenAI-compatible",
            ));
        }

        let text = response.text().await?;
        let parsed: OpenAiResponse = serde_json::from_str(&text).map_err(|e| {
            DirectLlmError::ParseError(format!("Failed to parse OpenAI response: {e}"))
        })?;

        if parsed.choices.is_empty() {
            return Ok((None, Vec::new()));
        }

        let choice = &parsed.choices[0];
        let content = choice.message.content.clone();
        let tool_calls = choice
            .message
            .tool_calls
            .as_ref()
            .map(|tcs| {
                tcs.iter()
                    .map(|tc| ProviderToolCall {
                        id: tc.id.clone(),
                        name: tc.function.name.clone(),
                        arguments: tc.function.arguments.clone(),
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok((content, tool_calls))
    }
}

fn accumulate_tool_calls(
    stream: impl futures::Stream<Item = Result<OpenAiStreamEvent, DirectLlmError>> + Send + 'static,
) -> impl futures::Stream<Item = Result<OpenAiStreamEvent, DirectLlmError>> + Send + 'static {
    futures::stream::unfold(
        (stream.boxed(), ToolCallAccumulator::new()),
        |(mut stream, mut acc)| async move {
            loop {
                let item = stream.next().await;
                match item {
                    None => return None,
                    Some(Err(e)) => return Some((Err(e), (stream, acc))),
                    Some(Ok(event)) => match event {
                        OpenAiStreamEvent::TextDelta(text) => {
                            if text.is_empty() {
                                continue;
                            }
                            return Some((
                                Ok(OpenAiStreamEvent::TextDelta(text)),
                                (stream, acc),
                            ));
                        }
                        OpenAiStreamEvent::ToolCallStart { index, id, name } => {
                            acc.start_tool_call(index, id, name);
                            continue;
                        }
                        OpenAiStreamEvent::ToolCallArgumentDelta { index, arguments } => {
                            acc.append_arguments(index, arguments);
                            continue;
                        }
                        OpenAiStreamEvent::ToolCallEnd { .. } => {
                            // We emit our own ToolCallEnd events
                            continue;
                        }
                        OpenAiStreamEvent::Done => {
                            let completed = acc.flush();
                            if completed.is_empty() {
                                return Some((
                                    Ok(OpenAiStreamEvent::Done),
                                    (stream, acc),
                                ));
                            }
                            // Emit first completed tool call, then Done on next call
                            let mut events: Vec<OpenAiStreamEvent> = completed;
                            events.push(OpenAiStreamEvent::Done);
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

struct ToolCallAccumulator {
    tool_calls: std::collections::HashMap<u64, (String, String, String)>,
    pending: Vec<OpenAiStreamEvent>,
}

impl ToolCallAccumulator {
    fn new() -> Self {
        Self {
            tool_calls: std::collections::HashMap::new(),
            pending: Vec::new(),
        }
    }

    fn start_tool_call(&mut self, index: u64, id: String, name: String) {
        self.tool_calls.insert(index, (id, name, String::new()));
    }

    fn append_arguments(&mut self, index: u64, arguments: String) {
        if let Some((_, _, args)) = self.tool_calls.get_mut(&index) {
            args.push_str(&arguments);
        }
    }

    fn flush(&mut self) -> Vec<OpenAiStreamEvent> {
        let mut results: Vec<OpenAiStreamEvent> = self.pending.clone();
        self.pending.clear();
        for (index, (id, name, arguments)) in self.tool_calls.drain() {
            results.push(OpenAiStreamEvent::ToolCallEnd {
                index,
                id,
                name,
                arguments,
            });
        }
        results
    }
}