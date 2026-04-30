use warp_multi_agent_api as api;

use direct_llm_client::types::{ChatMessage, MessageRole, ProviderConfig, ProviderType, ToolDefinition, StreamContext};

use crate::server::server_api::AIApiError;

/// Decide whether to route AI requests through the direct client.
/// Returns the provider config if direct routing should be used, None otherwise.
pub fn should_route_direct(
    direct_api_enabled: bool,
    api_keys: &ai::api_keys::ApiKeys,
    model: Option<&str>,
) -> Option<(ProviderType, ProviderConfig)> {
    if !direct_api_enabled {
        return None;
    }

    let is_oss = warp_core::channel::ChannelState::channel() == warp_core::channel::Channel::Oss;
    if !is_oss {
        return None;
    }

    direct_llm_client::resolve_provider(
        api_keys.anthropic.as_deref(),
        api_keys.anthropic_base_url.as_deref(),
        api_keys.openai.as_deref(),
        api_keys.openai_base_url.as_deref(),
        model,
    )
}

/// Convert Warp's AIAgentInput messages into ChatMessage format
/// for direct LLM API calls.
pub fn convert_input_to_chat_messages(
    input: &[crate::ai::agent::AIAgentInput],
    tasks: &[api::Task],
) -> Vec<ChatMessage> {
    let mut messages = Vec::new();

    // Add conversation history from existing tasks
    for task in tasks {
        for msg in &task.messages {
            if let Some(msg_type) = &msg.message {
                match msg_type {
                    api::message::Message::AgentOutput(output) => {
                        messages.push(ChatMessage {
                            role: MessageRole::Assistant,
                            content: output.text.clone(),
                            tool_calls: None,
                            tool_call_id: None,
                            name: None,
                        });
                    }
                    api::message::Message::ToolCall(tool_call) => {
                        let tool_name = extract_tool_name(tool_call);
                        messages.push(ChatMessage {
                            role: MessageRole::Assistant,
                            content: String::new(),
                            tool_calls: Some(vec![direct_llm_client::types::ProviderToolCall {
                                id: tool_call.tool_call_id.clone(),
                                name: tool_name,
                                arguments: extract_tool_args(tool_call),
                            }]),
                            tool_call_id: None,
                            name: None,
                        });
                    }
                    api::message::Message::ToolCallResult(result) => {
                        messages.push(ChatMessage {
                            role: MessageRole::Tool,
                            content: extract_tool_result_text(result),
                            tool_calls: None,
                            tool_call_id: Some(result.tool_call_id.clone()),
                            name: None,
                        });
                    }
                    api::message::Message::UserQuery(query) => {
                        messages.push(ChatMessage {
                            role: MessageRole::User,
                            content: query.query.clone(),
                            tool_calls: None,
                            tool_call_id: None,
                            name: None,
                        });
                    }
                    _ => {}
                }
            }
        }
    }

    // Add new input messages
    for ai_input in input {
        match ai_input {
            crate::ai::agent::AIAgentInput::UserQuery { query, .. } => {
                messages.push(ChatMessage {
                    role: MessageRole::User,
                    content: query.clone(),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                });
            }
            crate::ai::agent::AIAgentInput::ActionResult { result, .. } => {
                let result_text = format_tool_action_result(result);
                let call_id = format!("action_{}", result.id);
                messages.push(ChatMessage {
                    role: MessageRole::Tool,
                    content: result_text,
                    tool_calls: None,
                    tool_call_id: Some(call_id),
                    name: None,
                });
            }
            _ => {}
        }
    }

    messages
}

fn extract_tool_name(tool_call: &api::message::ToolCall) -> String {
    if let Some(tool) = &tool_call.tool {
        match tool {
            api::message::tool_call::Tool::RunShellCommand(_) => "run_command",
            api::message::tool_call::Tool::SearchCodebase(_) => "search_codebase",
            api::message::tool_call::Tool::ReadFiles(_) => "read_files",
            api::message::tool_call::Tool::ApplyFileDiffs(_) => "apply_file_diffs",
            api::message::tool_call::Tool::Grep(_) => "grep",
            api::message::tool_call::Tool::FileGlobV2(_) => "file_glob_v2",
            api::message::tool_call::Tool::CreateDocuments(_) => "create_documents",
            api::message::tool_call::Tool::EditDocuments(_) => "edit_documents",
            api::message::tool_call::Tool::ReadDocuments(_) => "read_documents",
            api::message::tool_call::Tool::WriteToLongRunningShellCommand(_) => "write_to_long_running_shell_command",
            api::message::tool_call::Tool::ReadShellCommandOutput(_) => "read_shell_command_output",
            api::message::tool_call::Tool::SuggestNewConversation(_) => "suggest_new_conversation",
            api::message::tool_call::Tool::SuggestPrompt(_) => "suggest_prompt",
            api::message::tool_call::Tool::InitProject(_) => "init_project",
            api::message::tool_call::Tool::OpenCodeReview(_) => "open_code_review",
            api::message::tool_call::Tool::Subagent(_) => "subagent",
            api::message::tool_call::Tool::Server(s) => &s.payload,
            _ => "unknown",
        }.to_string()
    } else {
        "unknown".to_string()
    }
}

fn extract_tool_args(tool_call: &api::message::ToolCall) -> String {
    if let Some(tool) = &tool_call.tool {
        match tool {
            api::message::tool_call::Tool::RunShellCommand(cmd) => {
                serde_json::json!({"command": cmd.command, "is_read_only": cmd.is_read_only}).to_string()
            }
            api::message::tool_call::Tool::Grep(g) => {
                serde_json::json!({"queries": g.queries, "path": g.path}).to_string()
            }
            api::message::tool_call::Tool::SearchCodebase(s) => {
                serde_json::json!({"query": s.query, "codebase_path": s.codebase_path}).to_string()
            }
            api::message::tool_call::Tool::Server(s) => s.payload.clone(),
            _ => "{}".to_string(),
        }
    } else {
        "{}".to_string()
    }
}

fn extract_tool_result_text(result: &api::message::ToolCallResult) -> String {
    // ToolCallResult contains a result oneof that's hard to fully enumerate.
    // For MVP, we just provide the tool_call_id and a generic "completed" text.
    format!("Tool call {} completed", result.tool_call_id)
}

fn format_tool_action_result(result: &crate::ai::agent::AIAgentActionResult) -> String {
    use crate::ai::agent::AIAgentActionResultType as ResultType;
    match &result.result {
        ResultType::RequestCommandOutput(cmd) => {
            match cmd {
                crate::ai::agent::RequestCommandOutputResult::Completed { output, .. } => {
                    output.clone()
                }
                crate::ai::agent::RequestCommandOutputResult::LongRunningCommandSnapshot { grid_contents, .. } => {
                    grid_contents.clone()
                }
                _ => "Command completed".to_string(),
            }
        }
        ResultType::ReadFiles(read_result) => {
            match read_result {
                crate::ai::agent::ReadFilesResult::Success { files } => {
                    files.iter()
                        .filter_map(|f| match &f.content {
                            crate::ai::agent::AnyFileContent::StringContent(s) => {
                                Some(format!("--- {} ---\n{}", f.file_name, s))
                            }
                            crate::ai::agent::AnyFileContent::BinaryContent(_) => {
                                Some(format!("--- {} --- (binary file)", f.file_name))
                            }
                        })
                        .collect::<Vec<_>>()
                        .join("\n\n")
                }
                crate::ai::agent::ReadFilesResult::Error(e) => format!("Read files error: {}", e),
                _ => "Read files cancelled".to_string(),
            }
        }
        ResultType::Grep(grep_result) => {
            match grep_result {
                crate::ai::agent::GrepResult::Success { matched_files } => {
                    let lines: Vec<String> = matched_files.iter()
                        .flat_map(|mf| {
                            mf.matched_lines.iter()
                                .map(|lm| format!("{}:{}", mf.file_path, lm.line_number))
                        })
                        .collect();
                    lines.join("\n")
                }
                crate::ai::agent::GrepResult::Error(e) => format!("Grep error: {}", e),
                _ => "Grep cancelled".to_string(),
            }
        }
        ResultType::FileGlobV2(glob_result) => {
            match glob_result {
                crate::ai::agent::FileGlobV2Result::Success { matched_files, .. } => {
                    let paths: Vec<String> = matched_files.iter()
                        .map(|m| m.file_path.clone())
                        .collect();
                    serde_json::json!({"paths": paths}).to_string()
                }
                crate::ai::agent::FileGlobV2Result::Error(e) => format!("File glob error: {}", e),
                _ => "File glob cancelled".to_string(),
            }
        }
        ResultType::SearchCodebase(search_result) => {
            match search_result {
                crate::ai::agent::SearchCodebaseResult::Success { files } => {
                    files.iter()
                        .map(|f| format!("--- {} ---\n{}", f.file_name, match &f.content {
                            crate::ai::agent::AnyFileContent::StringContent(s) => s.clone(),
                            crate::ai::agent::AnyFileContent::BinaryContent(_) => "(binary)".to_string(),
                        }))
                        .collect::<Vec<_>>()
                        .join("\n\n")
                }
                crate::ai::agent::SearchCodebaseResult::Failed { message, .. } => {
                    format!("Search failed: {}", message)
                }
                _ => "Search cancelled".to_string(),
            }
        }
        ResultType::CreateDocuments(docs_result) => {
            match docs_result {
                crate::ai::agent::CreateDocumentsResult::Success { created_documents } => {
                    let ids: Vec<String> = created_documents.iter()
                        .map(|d| d.document_id.to_string())
                        .collect();
                    serde_json::json!({"created": ids}).to_string()
                }
                crate::ai::agent::CreateDocumentsResult::Error(e) => format!("Create docs error: {}", e),
                _ => "Create docs cancelled".to_string(),
            }
        }
        ResultType::EditDocuments(docs_result) => {
            match docs_result {
                crate::ai::agent::EditDocumentsResult::Success { updated_documents } => {
                    let ids: Vec<String> = updated_documents.iter()
                        .map(|d| d.document_id.to_string())
                        .collect();
                    serde_json::json!({"edited": ids}).to_string()
                }
                crate::ai::agent::EditDocumentsResult::Error(e) => format!("Edit docs error: {}", e),
                _ => "Edit docs cancelled".to_string(),
            }
        }
        ResultType::ReadDocuments(docs_result) => {
            match docs_result {
                crate::ai::agent::ReadDocumentsResult::Success { documents } => {
                    documents.iter()
                        .map(|d| format!("--- {} ---\n{}", d.document_id, d.content))
                        .collect::<Vec<_>>()
                        .join("\n\n")
                }
                crate::ai::agent::ReadDocumentsResult::Error(e) => format!("Read docs error: {}", e),
                _ => "Read docs cancelled".to_string(),
            }
        }
        _ => "Action completed".to_string(),
    }
}

/// Convert supported tool types to provider tool definitions for direct LLM calls.
pub fn convert_supported_tools_to_tool_definitions(
    supported_tools: &[api::ToolType],
) -> Vec<ToolDefinition> {
    supported_tools.iter().filter_map(|tool| {
        match tool {
            api::ToolType::RunShellCommand => Some(ToolDefinition {
                name: "run_command".to_string(),
                description: "Execute a shell command and return its output.".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The shell command to execute"},
                        "is_read_only": {"type": "boolean", "description": "Whether the command is read-only (no side effects)"}
                    },
                    "required": ["command"]
                }),
            }),
            api::ToolType::ReadFiles => Some(ToolDefinition {
                name: "read_files".to_string(),
                description: "Read the contents of files at the given paths.".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "file_paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of file paths to read"
                        }
                    },
                    "required": ["file_paths"]
                }),
            }),
            api::ToolType::ApplyFileDiffs => Some(ToolDefinition {
                name: "request_file_edits".to_string(),
                description: "Apply edits to files using diffs.".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string", "description": "Summary of the changes"}
                    },
                    "required": ["summary"]
                }),
            }),
            api::ToolType::SearchCodebase => Some(ToolDefinition {
                name: "search_codebase".to_string(),
                description: "Search the codebase for relevant code.".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "codebase_path": {"type": "string", "description": "Path to the codebase root"}
                    },
                    "required": ["query"]
                }),
            }),
            api::ToolType::Grep => Some(ToolDefinition {
                name: "grep".to_string(),
                description: "Search file contents using pattern matching.".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Search patterns"
                        },
                        "path": {"type": "string", "description": "Directory to search in"}
                    },
                    "required": ["queries"]
                }),
            }),
            api::ToolType::FileGlobV2 => Some(ToolDefinition {
                name: "file_glob_v2".to_string(),
                description: "Find files matching glob patterns.".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "patterns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Glob patterns to match"
                        },
                        "search_dir": {"type": "string", "description": "Directory to search in"}
                    },
                    "required": ["patterns"]
                }),
            }),
            api::ToolType::CreateDocuments => Some(ToolDefinition {
                name: "create_documents".to_string(),
                description: "Create new documents/files.".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "new_documents": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "content": {"type": "string"}
                                },
                                "required": ["title", "content"]
                            },
                            "description": "Documents to create"
                        }
                    },
                    "required": ["new_documents"]
                }),
            }),
            api::ToolType::EditDocuments => Some(ToolDefinition {
                name: "edit_documents".to_string(),
                description: "Edit existing documents/files.".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "document_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "IDs of documents to edit"
                        }
                    },
                    "required": ["document_ids"]
                }),
            }),
            api::ToolType::ReadDocuments => Some(ToolDefinition {
                name: "read_documents".to_string(),
                description: "Read existing documents/files.".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "documents": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "document_id": {"type": "string"}
                                },
                                "required": ["document_id"]
                            },
                            "description": "Documents to read"
                        }
                    },
                    "required": ["documents"]
                }),
            }),
            api::ToolType::WriteToLongRunningShellCommand => Some(ToolDefinition {
                name: "write_to_long_running_shell_command".to_string(),
                description: "Write input to a long-running shell command.".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "command_id": {"type": "string", "description": "ID of the running command"},
                        "input": {"type": "string", "description": "Input to write to the command"}
                    },
                    "required": ["command_id", "input"]
                }),
            }),
            api::ToolType::ReadShellCommandOutput => Some(ToolDefinition {
                name: "read_shell_command_output".to_string(),
                description: "Read output from a long-running shell command.".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "command_id": {"type": "string", "description": "ID of the command to read output from"}
                    },
                    "required": ["command_id"]
                }),
            }),
            api::ToolType::SuggestNewConversation => Some(ToolDefinition {
                name: "suggest_new_conversation".to_string(),
                description: "Suggest starting a new conversation.".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "message_id": {"type": "string"}
                    }
                }),
            }),
            api::ToolType::SuggestPrompt => Some(ToolDefinition {
                name: "suggest_prompt".to_string(),
                description: "Suggest a follow-up prompt to the user.".to_string(),
                parameters: serde_json::json!({"type": "object"}),
            }),
            api::ToolType::InitProject => Some(ToolDefinition {
                name: "init_project".to_string(),
                description: "Initialize a project.".to_string(),
                parameters: serde_json::json!({"type": "object"}),
            }),
            api::ToolType::OpenCodeReview => Some(ToolDefinition {
                name: "open_code_review".to_string(),
                description: "Open a code review.".to_string(),
                parameters: serde_json::json!({"type": "object"}),
            }),
            api::ToolType::Subagent => Some(ToolDefinition {
                name: "subagent".to_string(),
                description: "Spawn a sub-agent for a subtask.".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string"},
                        "payload": {"type": "string"}
                    },
                    "required": ["task_id", "payload"]
                }),
            }),
            _ => None,
        }
    }).collect()
}

/// Build a system prompt for direct LLM calls from session context.
pub fn build_system_prompt(params: &super::RequestParams) -> String {
    let mut parts = Vec::new();

    parts.push("You are an AI assistant helping a user in a terminal environment. You can execute commands, read and edit files, search code, and more.".to_string());

    if let Some(cwd) = params.session_context.current_working_directory() {
        parts.push(format!("Current working directory: {}", cwd));
    }

    if let Some(shell) = params.session_context.shell() {
        parts.push(format!("Shell: {:?}", shell));
    }

    parts.push("When you need to perform actions, use the available tools. Always explain what you're doing before taking actions.".to_string());

    parts.join("\n\n")
}

/// Get the StreamContext (task_id and conversation_id) for a direct LLM call.
pub fn get_stream_context(params: &super::RequestParams) -> StreamContext {
    // Use the conversation model's root task ID if available.
    // On first query the root is optimistic and compute_active_tasks() returns empty,
    // so we rely on root_task_id from the controller instead of params.tasks.
    let task_id = params.root_task_id.clone()
        .unwrap_or_else(|| {
            let id = uuid::Uuid::new_v4().to_string();
            log::warn!("DirectLLM: no root_task_id, generating random task_id={}", id);
            id
        });

    let conversation_id = params.conversation_token
        .as_ref()
        .map(|t| t.as_str().to_string())
        .unwrap_or_else(|| {
            let id = uuid::Uuid::new_v4().to_string();
            log::warn!("DirectLLM: no conversation_token, generating random conversation_id={}", id);
            id
        });

    StreamContext { task_id, conversation_id }
}

/// Convert DirectLlmError to AIApiError for stream error mapping.
pub fn direct_error_to_api_error(err: &direct_llm_client::error::DirectLlmError) -> AIApiError {
    match err {
        direct_llm_client::error::DirectLlmError::RateLimited { .. } => AIApiError::QuotaLimit,
        direct_llm_client::error::DirectLlmError::Transport(e) => {
            AIApiError::Other(anyhow::anyhow!("Direct API transport error: {}", e))
        }
        direct_llm_client::error::DirectLlmError::InvalidApiKey { provider, model_name } => {
            AIApiError::Other(anyhow::anyhow!("Invalid API key for {} (model: {})", provider, model_name))
        }
        direct_llm_client::error::DirectLlmError::ContextWindowExceeded => {
            AIApiError::Other(anyhow::anyhow!("Context window exceeded"))
        }
        direct_llm_client::error::DirectLlmError::ProviderError { status, message } => {
            AIApiError::ErrorStatus(
                http::StatusCode::from_u16(*status).unwrap_or(http::StatusCode::INTERNAL_SERVER_ERROR),
                message.clone(),
            )
        }
        other => AIApiError::Other(anyhow::anyhow!("Direct API error: {}", other)),
    }
}

