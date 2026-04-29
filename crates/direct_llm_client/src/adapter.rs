use std::sync::Arc;

use futures::StreamExt;
use warp_multi_agent_api as api;

use crate::anthropic::AnthropicStreamEvent;
use crate::error::DirectLlmError;
use crate::openai::OpenAiStreamEvent;
use crate::types::StreamContext;

// Short aliases for deeply nested protobuf modules
use api::response_event as re;
use api::response_event::stream_finished as sf;
use api::client_action as ca;
use api::message as msg;
use api::message::tool_call as tc;

pub type ResponseStream = futures::stream::BoxStream<'static, Result<api::ResponseEvent, Arc<DirectLlmError>>>;

pub fn from_openai_stream(
    stream: futures::stream::BoxStream<'static, Result<OpenAiStreamEvent, DirectLlmError>>,
    ctx: &StreamContext,
) -> ResponseStream {
    let (sender, receiver) = futures::channel::mpsc::unbounded();
    let state = AdapterState::new(ctx);

    tokio::spawn(async move {
        let mut state = state;
        let mut stream = stream;

        while let Some(item) = stream.next().await {
            match item {
                Err(e) => {
                    let _ = sender.unbounded_send(Err(Arc::new(e)));
                    break;
                }
                Ok(event) => {
                    let events = convert_openai_event(event, &mut state);
                    for ev in events {
                        if sender.unbounded_send(Ok(ev)).is_err() {
                            break;
                        }
                    }
                }
            }
        }

        if !state.has_emitted_finished {
            let _ = sender.unbounded_send(Ok(make_stream_finished(
                sf::Reason::Done(sf::Done {}),
            )));
        }
    });

    receiver.boxed()
}

pub fn from_anthropic_stream(
    stream: futures::stream::BoxStream<'static, Result<AnthropicStreamEvent, DirectLlmError>>,
    ctx: &StreamContext,
) -> ResponseStream {
    let (sender, receiver) = futures::channel::mpsc::unbounded();
    let state = AdapterState::new(ctx);

    tokio::spawn(async move {
        let mut state = state;
        let mut stream = stream;

        while let Some(item) = stream.next().await {
            match item {
                Err(e) => {
                    let _ = sender.unbounded_send(Err(Arc::new(e)));
                    break;
                }
                Ok(event) => {
                    let events = convert_anthropic_event(event, &mut state);
                    for ev in events {
                        if sender.unbounded_send(Ok(ev)).is_err() {
                            break;
                        }
                    }
                }
            }
        }

        if !state.has_emitted_finished {
            let _ = sender.unbounded_send(Ok(make_stream_finished(
                sf::Reason::Done(sf::Done {}),
            )));
        }
    });

    receiver.boxed()
}

struct AdapterState {
    conversation_id: String,
    task_id: String,
    request_id: String,
    current_message_id: Option<String>,
    has_emitted_init: bool,
    has_emitted_finished: bool,
}

impl AdapterState {
    fn new(ctx: &StreamContext) -> Self {
        Self {
            conversation_id: ctx.conversation_id.clone(),
            task_id: ctx.task_id.clone(),
            request_id: uuid::Uuid::new_v4().to_string(),
            current_message_id: None,
            has_emitted_init: false,
            has_emitted_finished: false,
        }
    }

    fn ensure_init(&mut self) -> Vec<api::ResponseEvent> {
        if self.has_emitted_init {
            return Vec::new();
        }
        self.has_emitted_init = true;

        let mut events = Vec::new();

        // StreamInit event
        events.push(api::ResponseEvent {
            r#type: Some(re::Type::Init(re::StreamInit {
                conversation_id: self.conversation_id.clone(),
                request_id: self.request_id.clone(),
                run_id: String::new(),
            })),
        });

        // CreateTask action — the conversation model requires this before AddMessagesToTask
        events.push(api::ResponseEvent {
            r#type: Some(re::Type::ClientActions(re::ClientActions {
                actions: vec![api::ClientAction {
                    action: Some(ca::Action::CreateTask(ca::CreateTask {
                        task: Some(api::Task {
                            id: self.task_id.clone(),
                            description: String::new(),
                            dependencies: None,
                            messages: Vec::new(),
                            summary: String::new(),
                            server_data: String::new(),
                        }),
                    })),
                }],
            })),
        });

        events
    }
}

fn make_stream_finished(reason: sf::Reason) -> api::ResponseEvent {
    api::ResponseEvent {
        r#type: Some(re::Type::Finished(re::StreamFinished {
            reason: Some(reason),
            token_usage: Vec::new(),
            should_refresh_model_config: false,
            request_cost: None,
            conversation_usage_metadata: None,
        })),
    }
}

fn make_add_agent_output(state: &AdapterState, text: String) -> api::ResponseEvent {
    let message_id = state
        .current_message_id
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    api::ResponseEvent {
        r#type: Some(re::Type::ClientActions(re::ClientActions {
            actions: vec![api::ClientAction {
                action: Some(ca::Action::AddMessagesToTask(
                    ca::AddMessagesToTask {
                        task_id: state.task_id.clone(),
                        messages: vec![api::Message {
                            id: message_id,
                            task_id: state.task_id.clone(),
                            request_id: state.request_id.clone(),
                            timestamp: None,
                            server_message_data: String::new(),
                            citations: Vec::new(),
                            message: Some(msg::Message::AgentOutput(
                                msg::AgentOutput { text },
                            )),
                        }],
                    },
                )),
            }],
        })),
    }
}

fn make_append_agent_output(state: &AdapterState, text: String) -> api::ResponseEvent {
    let message_id = state.current_message_id.clone().unwrap_or_default();

    api::ResponseEvent {
        r#type: Some(re::Type::ClientActions(re::ClientActions {
            actions: vec![api::ClientAction {
                action: Some(ca::Action::AppendToMessageContent(
                    ca::AppendToMessageContent {
                        task_id: state.task_id.clone(),
                        message: Some(api::Message {
                            id: message_id,
                            task_id: state.task_id.clone(),
                            request_id: state.request_id.clone(),
                            timestamp: None,
                            server_message_data: String::new(),
                            citations: Vec::new(),
                            message: Some(msg::Message::AgentOutput(
                                msg::AgentOutput { text },
                            )),
                        }),
                        mask: Some(prost_types::FieldMask {
                            paths: vec!["agent_output.text".to_string()],
                        }),
                    },
                )),
            }],
        })),
    }
}

fn make_tool_call_message(state: &AdapterState, tool_call: msg::ToolCall) -> api::ResponseEvent {
    let tool_message_id = uuid::Uuid::new_v4().to_string();

    api::ResponseEvent {
        r#type: Some(re::Type::ClientActions(re::ClientActions {
            actions: vec![api::ClientAction {
                action: Some(ca::Action::AddMessagesToTask(
                    ca::AddMessagesToTask {
                        task_id: state.task_id.clone(),
                        messages: vec![api::Message {
                            id: tool_message_id,
                            task_id: state.task_id.clone(),
                            request_id: state.request_id.clone(),
                            timestamp: None,
                            server_message_data: String::new(),
                            citations: Vec::new(),
                            message: Some(msg::Message::ToolCall(tool_call)),
                        }],
                    },
                )),
            }],
        })),
    }
}

fn map_tool_call(name: &str, arguments_json: &str) -> tc::Tool {
    let args: serde_json::Value = serde_json::from_str(arguments_json)
        .unwrap_or(serde_json::Value::Null);

    match name {
        "run_command" | "run_shell_command" => {
            let command = args["command"].as_str().unwrap_or("").to_string();
            let is_read_only = args["is_read_only"].as_bool().unwrap_or(false);
            let risk_category = if is_read_only {
                api::RiskCategory::ReadOnly as i32
            } else {
                api::RiskCategory::NontrivialLocalChange as i32
            };
            tc::Tool::RunShellCommand(tc::RunShellCommand {
                command,
                is_read_only,
                uses_pager: false,
                citations: Vec::new(),
                is_risky: false,
                risk_category,
                wait_until_complete_value: None,
            })
        }
        "search_codebase" => {
            let query = args["query"].as_str().unwrap_or("").to_string();
            let path_filters = args["path_filters"]
                .as_array()
                .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default();
            let codebase_path = args["codebase_path"].as_str().unwrap_or("").to_string();
            tc::Tool::SearchCodebase(tc::SearchCodebase {
                query,
                path_filters,
                codebase_path,
            })
        }
        "read_files" => {
            let mut files: Vec<tc::read_files::File> = Vec::new();
            if let Some(arr) = args["files"].as_array() {
                for v in arr {
                    let name = v["name"].as_str().unwrap_or("").to_string();
                    if !name.is_empty() {
                        files.push(tc::read_files::File {
                            name,
                            line_ranges: Vec::new(),
                        });
                    }
                }
            } else if let Some(arr) = args["file_paths"].as_array() {
                for v in arr {
                    if let Some(s) = v.as_str() {
                        if !s.is_empty() {
                            files.push(tc::read_files::File {
                                name: s.to_string(),
                                line_ranges: Vec::new(),
                            });
                        }
                    }
                }
            }
            tc::Tool::ReadFiles(tc::ReadFiles { files })
        }
        "apply_file_diffs" | "request_file_edits" => {
            tc::Tool::ApplyFileDiffs(tc::ApplyFileDiffs {
                summary: args["summary"].as_str().unwrap_or("").to_string(),
                diffs: Vec::new(),
                new_files: Vec::new(),
                deleted_files: Vec::new(),
                v4a_updates: Vec::new(),
            })
        }
        "grep" => {
            let mut queries: Vec<String> = Vec::new();
            if let Some(arr) = args["queries"].as_array() {
                for v in arr {
                    if let Some(s) = v.as_str() {
                        queries.push(s.to_string());
                    }
                }
            } else if let Some(s) = args["pattern"].as_str() {
                queries.push(s.to_string());
            }
            let path = args["path"].as_str().unwrap_or("").to_string();
            tc::Tool::Grep(tc::Grep { queries, path })
        }
        "file_glob" | "file_glob_v2" => {
            let mut patterns: Vec<String> = Vec::new();
            if let Some(arr) = args["patterns"].as_array() {
                for v in arr {
                    if let Some(s) = v.as_str() {
                        patterns.push(s.to_string());
                    }
                }
            } else if let Some(s) = args["pattern"].as_str() {
                patterns.push(s.to_string());
            }
            let search_dir = args["search_dir"].as_str().unwrap_or("").to_string();
            let max_matches = args["max_matches"].as_i64().unwrap_or(0) as i32;
            let max_depth = args["max_depth"].as_i64().unwrap_or(0) as i32;
            let min_depth = args["min_depth"].as_i64().unwrap_or(0) as i32;
            tc::Tool::FileGlobV2(tc::FileGlobV2 {
                patterns,
                search_dir,
                max_matches,
                max_depth,
                min_depth,
            })
        }
        "create_documents" => {
            let mut new_documents: Vec<tc::create_documents::NewDocument> = Vec::new();
            if let Some(arr) = args["new_documents"].as_array() {
                for v in arr {
                    let content = v["content"].as_str().unwrap_or("").to_string();
                    let title = v["title"].as_str().unwrap_or("").to_string();
                    new_documents.push(tc::create_documents::NewDocument { content, title });
                }
            }
            tc::Tool::CreateDocuments(tc::CreateDocuments { new_documents })
        }
        "edit_documents" => {
            tc::Tool::EditDocuments(tc::EditDocuments {
                diffs: Vec::new(),
            })
        }
        "read_documents" => {
            let mut documents: Vec<tc::read_documents::Document> = Vec::new();
            if let Some(arr) = args["documents"].as_array() {
                for v in arr {
                    let document_id = v["document_id"].as_str().unwrap_or("").to_string();
                    documents.push(tc::read_documents::Document {
                        document_id,
                        line_ranges: Vec::new(),
                    });
                }
            }
            tc::Tool::ReadDocuments(tc::ReadDocuments { documents })
        }
        "suggest_new_conversation" => {
            let message_id = args["message_id"].as_str().unwrap_or("").to_string();
            tc::Tool::SuggestNewConversation(tc::SuggestNewConversation {
                message_id,
            })
        }
        "suggest_prompt" => {
            tc::Tool::SuggestPrompt(tc::SuggestPrompt {
                display_mode: None,
                is_trigger_irrelevant: false,
            })
        }
        "read_shell_command_output" => {
            let command_id = args["command_id"].as_str().unwrap_or("").to_string();
            tc::Tool::ReadShellCommandOutput(tc::ReadShellCommandOutput {
                command_id,
                delay: None,
            })
        }
        "write_to_long_running_shell_command" => {
            let input = args["input"].as_str().unwrap_or("").to_string().into_bytes();
            let command_id = args["command_id"].as_str().unwrap_or("").to_string();
            tc::Tool::WriteToLongRunningShellCommand(
                tc::WriteToLongRunningShellCommand {
                    input,
                    mode: None,
                    command_id,
                },
            )
        }
        "init_project" => tc::Tool::InitProject(tc::InitProject {}),
        "open_code_review" => tc::Tool::OpenCodeReview(tc::OpenCodeReview {}),
        "subagent" => {
            let task_id = args["task_id"].as_str().unwrap_or("").to_string();
            let payload = args["payload"].as_str().unwrap_or("").to_string();
            tc::Tool::Subagent(tc::Subagent {
                task_id,
                payload,
                metadata: None,
            })
        }
        _ => tc::Tool::Server(tc::Server {
            payload: arguments_json.to_string(),
        }),
    }
}

fn convert_openai_event(event: OpenAiStreamEvent, state: &mut AdapterState) -> Vec<api::ResponseEvent> {
    let mut events = Vec::new();

    match event {
        OpenAiStreamEvent::TextDelta(text) => {
            if text.is_empty() {
                return events;
            }

            events.extend(state.ensure_init());

            if state.current_message_id.is_none() {
                state.current_message_id = Some(uuid::Uuid::new_v4().to_string());
                events.push(make_add_agent_output(state, text));
            } else {
                events.push(make_append_agent_output(state, text));
            }
        }
        OpenAiStreamEvent::ToolCallStart { .. } => {
            state.current_message_id = None;
        }
        OpenAiStreamEvent::ToolCallArgumentDelta { .. } => {}
        OpenAiStreamEvent::ToolCallEnd {
            id,
            name,
            arguments,
            ..
        } => {
            events.extend(state.ensure_init());

            let tool_call = msg::ToolCall {
                tool_call_id: id,
                tool: Some(map_tool_call(&name, &arguments)),
            };
            state.current_message_id = None;
            events.push(make_tool_call_message(state, tool_call));
        }
        OpenAiStreamEvent::Done => {
            state.has_emitted_finished = true;
            events.push(make_stream_finished(sf::Reason::Done(sf::Done {})));
        }
    }

    events
}

fn convert_anthropic_event(event: AnthropicStreamEvent, state: &mut AdapterState) -> Vec<api::ResponseEvent> {
    let mut events = Vec::new();

    match event {
        AnthropicStreamEvent::MessageStart { .. } => {
            events.extend(state.ensure_init());
        }
        AnthropicStreamEvent::TextDelta(text) => {
            if text.is_empty() {
                return events;
            }

            events.extend(state.ensure_init());

            if state.current_message_id.is_none() {
                state.current_message_id = Some(uuid::Uuid::new_v4().to_string());
                events.push(make_add_agent_output(state, text));
            } else {
                events.push(make_append_agent_output(state, text));
            }
        }
        AnthropicStreamEvent::ToolCallStart { .. } => {
            state.current_message_id = None;
        }
        AnthropicStreamEvent::ToolCallArgumentDelta { .. } => {}
        AnthropicStreamEvent::ToolCallEnd {
            id,
            name,
            arguments,
            ..
        } => {
            events.extend(state.ensure_init());

            let tool_call = msg::ToolCall {
                tool_call_id: id,
                tool: Some(map_tool_call(&name, &arguments)),
            };
            state.current_message_id = None;
            events.push(make_tool_call_message(state, tool_call));
        }
        AnthropicStreamEvent::MessageEnd { stop_reason } => {
            state.has_emitted_finished = true;
            let reason = match stop_reason.as_str() {
                "max_tokens" => sf::Reason::MaxTokenLimit(sf::ReachedMaxTokenLimit {}),
                _ => sf::Reason::Done(sf::Done {}),
            };
            events.push(make_stream_finished(reason));
        }
    }

    events
}

pub fn error_to_stream_finished(error: DirectLlmError) -> api::ResponseEvent {
    let reason = match &error {
        DirectLlmError::InvalidApiKey { provider, model_name } => {
            let provider_val = match provider.as_str() {
                "Anthropic" => api::LlmProvider::Anthropic as i32,
                "OpenAI-compatible" => api::LlmProvider::Openai as i32,
                _ => api::LlmProvider::Unknown as i32,
            };
            sf::Reason::InvalidApiKey(sf::InvalidApiKey {
                provider: provider_val,
                model_name: model_name.clone(),
            })
        }
        DirectLlmError::RateLimited { .. } => {
            sf::Reason::QuotaLimit(sf::QuotaLimit {})
        }
        DirectLlmError::ContextWindowExceeded => {
            sf::Reason::ContextWindowExceeded(sf::ContextWindowExceeded {})
        }
        _ => {
            sf::Reason::InternalError(sf::InternalError {
                message: error.to_string(),
            })
        }
    };

    make_stream_finished(reason)
}