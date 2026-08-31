use std::{
    collections::{BTreeMap, HashMap},
    env, fmt, fs,
    io::{self, Write},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
        mpsc::{self, Receiver, Sender},
    },
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, bail};
use chrono::Utc;
use crossterm::{
    clipboard::CopyToClipboard,
    event::{
        self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
        Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, KeyboardEnhancementFlags,
        MouseButton, MouseEventKind, PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
    },
    execute,
};
use directories::ProjectDirs;
use futures::StreamExt;
use ratatui::{
    Frame, Terminal,
    backend::Backend,
    buffer::Buffer,
    layout::{
        Alignment,
        Constraint::{Length, Min},
        Layout, Margin, Rect,
    },
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, BorderType, Padding, Paragraph, Wrap},
};
use reqwest::header::{HeaderName, HeaderValue};
use rig::{
    agent::{
        Agent, AgentBuilder, AgentHook, CompletionCallAction, CompletionCallEvent, HookContext,
        MultiTurnStreamItem, PromptResponse, RequestPatch,
        tool::{DynamicTool, ToolExecutionError, ToolOutput},
    },
    client::CompletionClient,
    completion::{Message as RigMessage, Usage as RigUsage},
    message::{Reasoning, ReasoningContent, ToolResultContent, UserContent},
    providers::openai,
    streaming::StreamedAssistantContent,
};
use rmcp::{
    ClientHandler, ClientLifecycleMode, ClientServiceExt, RoleClient,
    model::{
        CallToolRequest, CallToolRequestParams, CallToolResult, CancelTaskParams,
        CancelledNotificationParam, ClientCapabilities, ClientInfo, ClientRequest, ContentBlock,
        GetTaskParams, Implementation, JsonObject, ProgressNotificationParam, ProgressToken,
        ProtocolVersion, RequestId, ResourceContents, ServerResult, TASKS_EXTENSION_ID,
        TaskPayload, Tool as McpTool,
    },
    service::{NotificationContext, Peer, PeerRequestOptions, RunningService},
    transport::{
        StreamableHttpClientTransport, TokioChildProcess,
        streamable_http_client::StreamableHttpClientTransportConfig,
    },
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::process::Command;
use tui_prompts::{State, TextState};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};
use url::Url;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

const DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1";
const DEFAULT_MODEL: &str = "openrouter/free";
const DEFAULT_MCP_CONFIG: &str = "mcp.json";
const MAX_TOOL_ROUNDS: usize = 64;
const TOOL_CONCURRENCY: usize = 16;
const MCP_RPC_TIMEOUT: Duration = Duration::from_secs(120);
const MAX_RAW_MCP_RESULT_BYTES: usize = 4 * 1024 * 1024;
const MAX_MODEL_TOOL_RESULT_BYTES: usize = 10 * 1024;
const MAX_HISTORY_TOOL_RESULT_BYTES: usize = 1024;
const MAX_UI_TOOL_RESULT_BYTES: usize = 8 * 1024;
const MAX_TOOL_OUTPUT_PREVIEW_BYTES: usize = 8 * 1024;
const MAX_TOOL_OUTPUT_PREVIEW_LINES: usize = 200;
const LIGHT_GRAY: Color = Color::Rgb(120, 120, 120);
static TOOL_SEQUENCE: AtomicU64 = AtomicU64::new(1);

type ToolId = u64;
type ProgressRoutes = HashMap<ProgressToken, (ToolId, Sender<ToolEvent>)>;

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Session {
    history: Vec<RigMessage>,
    messages: Vec<SavedMessage>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum SavedMessage {
    User {
        text: String,
    },
    Assistant {
        text: String,
        metrics: Option<String>,
    },
    Thinking {
        id: String,
        text: Option<String>,
        duration_ms: u64,
    },
    Tools {
        tools: Vec<SavedTool>,
    },
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct SavedTool {
    label: String,
    args: Value,
    output: Option<Value>,
    error: Option<String>,
    duration_ms: u64,
}

#[derive(Clone)]
struct MarkdownStyle;

impl tui_markdown::StyleSheet for MarkdownStyle {
    fn code(&self) -> Style {
        Style::new().fg(LIGHT_GRAY)
    }
}

struct Settings {
    api_key: String,
    base_url: String,
    model: String,
    model_context_window: Option<u64>,
    system: Option<String>,
    agent_name: Option<String>,
    agent_description: Option<String>,
    additional_params: Value,
    mcp_config: Option<String>,
}

impl Settings {
    fn load() -> Result<Self> {
        let AgentFile {
            name: agent_name,
            description: agent_description,
            base_url,
            model,
            model_context_window,
            params,
            prompt,
        } = load_agent()?.unwrap_or_default();
        let base_url = base_url.map_or_else(
            || env_or("OPENAI_BASE_URL", DEFAULT_BASE_URL),
            |value| nonempty("agent base_url", value),
        )?;
        let model = model.map_or_else(
            || env_or("OPENAI_MODEL", DEFAULT_MODEL),
            |value| nonempty("agent model", value),
        )?;
        let model_context_window = match model_context_window {
            Some(value) => Some(positive_u64("agent model_context_window", value)?),
            None => env_u64("MODEL_CONTEXT_WINDOW")?,
        };
        Ok(Self {
            api_key: env_or("OPENAI_API_KEY", "")?,
            base_url,
            model,
            model_context_window,
            system: (!prompt.is_empty()).then_some(prompt),
            agent_name,
            agent_description,
            additional_params: Value::Object(params.into_iter().collect()),
            mcp_config: env_path("MCP_CONFIG")?,
        })
    }

    fn mcp_path(&self) -> (&str, bool) {
        self.mcp_config
            .as_deref()
            .map_or((DEFAULT_MCP_CONFIG, true), |path| (path, false))
    }

    fn session_label(&self, tools: usize) -> String {
        match tools {
            0 => self.model.clone(),
            1 => format!("{} · 1 tool", self.model),
            _ => format!("{} · {tools} tools", self.model),
        }
    }
}

#[derive(Default, Deserialize)]
struct AgentFile {
    name: Option<String>,
    description: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
    model_context_window: Option<u64>,
    #[serde(flatten)]
    params: BTreeMap<String, Value>,
    #[serde(skip)]
    prompt: String,
}

// ---------- MCP ----------

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct McpConfig {
    #[serde(default, rename = "mcpServers")]
    servers: BTreeMap<String, McpServer>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum McpServer {
    Stdio(StdioServer),
    Http(HttpServer),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StdioServer {
    command: String,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    env: HashMap<String, String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HttpServer {
    url: String,
    #[serde(default)]
    headers: HashMap<String, String>,
}

#[derive(Debug)]
struct McpConnection {
    name: String,
    service: RunningService<RoleClient, McpClient>,
    protocol: ProtocolVersion,
    tasks: bool,
}

#[derive(Debug)]
struct ToolSpec {
    server: usize,
    alias: String,
    name: String,
    label: String,
    description: String,
    parameters: Value,
    output_schema: Option<Value>,
}

#[derive(Debug)]
enum ActiveCall {
    Request {
        peer: Peer<RoleClient>,
        request_id: RequestId,
    },
    Task {
        peer: Peer<RoleClient>,
        task_id: String,
    },
}

#[derive(Debug)]
struct McpToolResult {
    ui: Value,
    model: ToolOutput,
    is_error: bool,
}

#[derive(Clone, Default, Debug)]
struct ProgressRouter {
    active: Arc<Mutex<ProgressRoutes>>,
}

impl ProgressRouter {
    fn register(&self, token: ProgressToken, id: ToolId, events: Sender<ToolEvent>) {
        self.active
            .lock()
            .unwrap_or_else(|lock| lock.into_inner())
            .insert(token, (id, events));
    }

    fn remove(&self, token: &ProgressToken) {
        self.active
            .lock()
            .unwrap_or_else(|lock| lock.into_inner())
            .remove(token);
    }

    fn handle(&self, params: ProgressNotificationParam) {
        let route = self
            .active
            .lock()
            .unwrap_or_else(|lock| lock.into_inner())
            .get(&params.progress_token)
            .cloned();
        if let Some((id, events)) = route {
            let status = params
                .message
                .filter(|message| !message.trim().is_empty())
                .or_else(|| {
                    params
                        .total
                        .filter(|total| *total > 0.0)
                        .map(|total| format!("{:.0}/{total:.0}", params.progress))
                });
            let Some(status) = status else { return };
            let _ = events.send(ToolEvent::Progress {
                id,
                status: Some(status),
            });
        }
    }
}

#[derive(Clone, Debug)]
struct McpClient {
    info: ClientInfo,
    progress: ProgressRouter,
}

impl ClientHandler for McpClient {
    fn get_info(&self) -> ClientInfo {
        self.info.clone()
    }

    async fn on_progress(
        &self,
        params: ProgressNotificationParam,
        _context: NotificationContext<RoleClient>,
    ) {
        self.progress.handle(params);
    }
}

#[derive(Default, Debug)]
struct McpHost {
    services: Vec<McpConnection>,
    tools: Vec<ToolSpec>,
    active: Mutex<HashMap<ToolId, ActiveCall>>,
    progress: ProgressRouter,
    // Incremented whenever the user aborts a turn. A call checks the epoch when
    // transitioning from an ordinary request to an MCP Task, closing a small race
    // where cancellation can arrive between the two protocol states.
    cancel_epoch: AtomicU64,
}

impl McpHost {
    async fn load(path: &str, optional: bool) -> Result<Self> {
        let config = match fs::read_to_string(path) {
            Ok(text) => {
                serde_json::from_str(&text).with_context(|| format!("invalid MCP config {path}"))?
            }
            Err(error) if optional && error.kind() == io::ErrorKind::NotFound => {
                McpConfig::default()
            }
            Err(error) => return Err(error).with_context(|| format!("failed to read {path}")),
        };

        let mut host = Self::default();
        for (name, server) in config.servers {
            host.connect(&name, server).await?;
        }
        Ok(host)
    }

    async fn connect(&mut self, server_name: &str, config: McpServer) -> Result<()> {
        // This client intentionally targets the current MCP revision. Discover keeps
        // protocol negotiation explicit and avoids accidentally opting into the
        // incompatible pre-2026 experimental Tasks design.
        let lifecycle = ClientLifecycleMode::Discover {
            preferred_versions: vec![ProtocolVersion::V_2026_07_28],
        };
        let client = McpClient {
            info: mcp_client_info(),
            progress: self.progress.clone(),
        };

        let service = match config {
            McpServer::Stdio(StdioServer { command, args, env }) => {
                let mut command = Command::new(expand(&command)?);
                command.args(
                    args.into_iter()
                        .map(|arg| expand(&arg))
                        .collect::<Result<Vec<_>>>()?,
                );
                command.envs(
                    env.into_iter()
                        .map(|(key, value)| Ok((key, expand(&value)?)))
                        .collect::<Result<Vec<_>>>()?,
                );
                let transport = TokioChildProcess::new(command)
                    .with_context(|| format!("failed to start MCP server {server_name}"))?;
                client.serve_with_lifecycle(transport, lifecycle).await
            }
            McpServer::Http(HttpServer { url, headers }) => {
                let url = expand(&url)?;
                validate_mcp_url(&url)?;
                let config = StreamableHttpClientTransportConfig::with_uri(url)
                    .custom_headers(mcp_headers(headers)?);
                client
                    .serve_with_lifecycle(
                        StreamableHttpClientTransport::from_config(config),
                        lifecycle,
                    )
                    .await
            }
        }
        .with_context(|| format!("failed to connect to MCP server {server_name}"))?;

        let info = service
            .peer_info()
            .context("MCP server omitted negotiated information")?;
        let protocol = info.protocol_version.clone();
        let tasks = info.capabilities.supports_tasks();
        let index = self.services.len();

        if info.capabilities.tools.is_some() {
            for tool in service
                .list_all_tools()
                .await
                .with_context(|| format!("failed to list tools from MCP server {server_name}"))?
            {
                self.add_tool(index, server_name, tool)?;
            }
        }

        self.services.push(McpConnection {
            name: server_name.to_owned(),
            service,
            protocol,
            tasks,
        });
        Ok(())
    }

    fn add_tool(&mut self, server: usize, server_name: &str, tool: McpTool) -> Result<()> {
        let alias = model_tool_name(server, server_name, &tool.name);
        if self.tools.iter().any(|tool| tool.alias == alias) {
            bail!("MCP tool name collision after model-safe normalization: {alias}");
        }

        self.tools.push(ToolSpec {
            server,
            alias,
            name: tool.name.to_string(),
            label: format!("{server_name}/{}", tool.name),
            description: tool.description.as_deref().unwrap_or_default().to_owned(),
            parameters: Value::Object((*tool.input_schema).clone()),
            output_schema: tool
                .output_schema
                .map(|schema| Value::Object((*schema).clone())),
        });
        Ok(())
    }

    fn set_active(&self, id: ToolId, call: ActiveCall) {
        self.active
            .lock()
            .unwrap_or_else(|lock| lock.into_inner())
            .insert(id, call);
    }

    fn clear_active(&self, id: ToolId) {
        self.active
            .lock()
            .unwrap_or_else(|lock| lock.into_inner())
            .remove(&id);
    }

    fn was_cancelled(&self, epoch: u64) -> bool {
        self.cancel_epoch.load(Ordering::Acquire) != epoch
    }

    async fn call(
        &self,
        id: ToolId,
        server: usize,
        name: &str,
        arguments: Value,
        output_schema: Option<&Value>,
        events: &Sender<ToolEvent>,
    ) -> Result<McpToolResult> {
        let epoch = self.cancel_epoch.load(Ordering::Acquire);
        let connection = self
            .services
            .get(server)
            .with_context(|| format!("unknown MCP server index {server}"))?;
        let arguments: JsonObject =
            serde_json::from_value(arguments).context("MCP tool arguments must be an object")?;
        let request = CallToolRequestParams::new(name.to_owned()).with_arguments(arguments);
        let options = PeerRequestOptions::with_timeout(MCP_RPC_TIMEOUT)
            .reset_timeout_on_progress()
            .with_max_total_timeout(MCP_RPC_TIMEOUT.saturating_mul(10));

        let handle = connection
            .service
            .send_cancellable_request(
                ClientRequest::CallToolRequest(CallToolRequest::new(request)),
                options,
            )
            .await
            .with_context(|| format!("failed to send MCP tool {name}"));
        let handle = match handle {
            Ok(handle) => handle,
            Err(error) => return Err(error),
        };
        let progress_token = handle.progress_token.clone();
        self.progress
            .register(progress_token.clone(), id, events.clone());
        let peer = handle.peer.clone();
        let request_id = handle.id.clone();

        if self.was_cancelled(epoch) {
            cancel_request(&peer, request_id).await;
            self.progress.remove(&progress_token);
            bail!("MCP tool {name} was cancelled");
        }

        self.set_active(
            id,
            ActiveCall::Request {
                peer: peer.clone(),
                request_id: request_id.clone(),
            },
        );
        if self.was_cancelled(epoch) {
            cancel_request(&peer, request_id).await;
            self.clear_active(id);
            self.progress.remove(&progress_token);
            bail!("MCP tool {name} was cancelled");
        }

        let response = match handle
            .await_response()
            .await
            .with_context(|| format!("MCP tool {name} failed"))
        {
            Ok(response) => response,
            Err(error) => {
                self.clear_active(id);
                self.progress.remove(&progress_token);
                return Err(error);
            }
        };

        let result = match response {
            ServerResult::CallToolResult(result) => encode_tool_result(result, output_schema),
            ServerResult::CreateTaskResult(result) => {
                if !connection.tasks {
                    Err(anyhow!(
                        "MCP server returned a Task without declaring the Tasks extension"
                    ))
                } else {
                    let task = result.task;
                    let task_id = task.task_id;
                    let poll_interval_ms = task.poll_interval_ms;
                    let status = task
                        .status_message
                        .filter(|message| !message.trim().is_empty())
                        .unwrap_or_else(|| "Task started".into());
                    let _ = events.send(ToolEvent::Status {
                        id,
                        status: Some(status.clone()),
                    });

                    if self.was_cancelled(epoch) {
                        let _ = peer
                            .cancel_task(CancelTaskParams::new(task_id.clone()))
                            .await;
                        Err(anyhow!("MCP tool {name} was cancelled"))
                    } else {
                        self.set_active(
                            id,
                            ActiveCall::Task {
                                peer: peer.clone(),
                                task_id: task_id.clone(),
                            },
                        );
                        if self.was_cancelled(epoch) {
                            let _ = peer.cancel_task(CancelTaskParams::new(task_id)).await;
                            Err(anyhow!("MCP tool {name} was cancelled"))
                        } else {
                            Self::wait_for_task(
                                &peer,
                                id,
                                name,
                                task_id,
                                (poll_interval_ms, Some(status)),
                                output_schema,
                                events,
                            )
                            .await
                        }
                    }
                }
            }
            ServerResult::InputRequiredResult(_) => Err(anyhow!(
                "MCP tool {name} requires interactive client input; this client does not advertise elicitation or other MRTR input capabilities"
            )),
            _ => Err(anyhow!(
                "MCP tool {name} returned a response type that is invalid for tools/call"
            )),
        };

        self.clear_active(id);
        self.progress.remove(&progress_token);
        result
    }

    async fn wait_for_task(
        peer: &Peer<RoleClient>,
        id: ToolId,
        name: &str,
        task_id: String,
        initial: (Option<u64>, Option<String>),
        output_schema: Option<&Value>,
        events: &Sender<ToolEvent>,
    ) -> Result<McpToolResult> {
        let (initial_poll_interval_ms, mut last_status) = initial;
        let mut poll_interval = task_poll_interval(initial_poll_interval_ms);

        loop {
            tokio::time::sleep(poll_interval).await;
            let result = tokio::time::timeout(
                MCP_RPC_TIMEOUT,
                peer.get_task(GetTaskParams::new(task_id.clone())),
            )
            .await
            .with_context(|| format!("MCP task {task_id} for tool {name} timed out"))?
            .with_context(|| format!("failed to get MCP task {task_id} for tool {name}"))?;

            poll_interval = task_poll_interval(result.task.task.poll_interval_ms);
            let status = result
                .task
                .task
                .status_message
                .clone()
                .filter(|message| !message.trim().is_empty())
                .or_else(|| match &result.task.payload {
                    TaskPayload::Working => Some("Task running".into()),
                    TaskPayload::InputRequired { .. } => Some("Task requires input".into()),
                    _ => None,
                });
            if status != last_status {
                last_status = status.clone();
                let _ = events.send(ToolEvent::Status { id, status });
            }

            match result.task.payload {
                TaskPayload::Working => {}
                TaskPayload::Completed { result } => {
                    let result = serde_json::from_value::<CallToolResult>(Value::Object(result))
                        .context("MCP task returned an invalid tool result")?;
                    return encode_tool_result(result, output_schema);
                }
                TaskPayload::InputRequired { input_requests } => {
                    let count = input_requests.len();
                    let cancellation = peer
                        .cancel_task(CancelTaskParams::new(task_id.clone()))
                        .await;
                    if let Err(error) = cancellation {
                        bail!(
                            "MCP task {task_id} for tool {name} requires {count} client input request(s), and cancelling the unsupported interactive task failed: {error}"
                        );
                    }
                    bail!(
                        "MCP task {task_id} for tool {name} requires {count} client input request(s); interactive Task input is not supported by this client"
                    );
                }
                TaskPayload::Failed { error } => {
                    bail!(
                        "MCP task {task_id} for tool {name} failed: {}",
                        Value::Object(error)
                    );
                }
                TaskPayload::Cancelled => {
                    bail!("MCP task {task_id} for tool {name} was cancelled");
                }
                _ => bail!("MCP task {task_id} for tool {name} returned an unknown status"),
            }
        }
    }

    async fn cancel_calls(&self) {
        // Bump first so a request that is just about to become a Task observes the
        // cancellation even if it was not present in the map snapshot below.
        self.cancel_epoch.fetch_add(1, Ordering::AcqRel);
        let active =
            std::mem::take(&mut *self.active.lock().unwrap_or_else(|lock| lock.into_inner()));

        for call in active.into_values() {
            match call {
                ActiveCall::Request { peer, request_id } => {
                    cancel_request(&peer, request_id).await;
                }
                ActiveCall::Task { peer, task_id } => {
                    let _ = peer.cancel_task(CancelTaskParams::new(task_id)).await;
                }
            }
        }
    }

    fn dynamic_tools(self: &Arc<Self>, events: Sender<ToolEvent>) -> Vec<DynamicTool> {
        self.tools
            .iter()
            .map(|spec| {
                let host = Arc::clone(self);
                let events = events.clone();
                let server = spec.server;
                let name = spec.name.clone();
                let label = spec.label.clone();
                let output_schema = spec.output_schema.clone();

                DynamicTool::new(
                    spec.alias.clone(),
                    spec.description.clone(),
                    spec.parameters.clone(),
                    move |_, args| {
                        let host = Arc::clone(&host);
                        let events = events.clone();
                        let name = name.clone();
                        let label = label.clone();
                        let output_schema = output_schema.clone();

                        Box::pin(async move {
                            let id = TOOL_SEQUENCE.fetch_add(1, Ordering::Relaxed);
                            let started = Instant::now();
                            let _ = events.send(ToolEvent::Start {
                                id,
                                label,
                                args: args.clone(),
                                started,
                            });

                            let result = host
                                .call(id, server, &name, args, output_schema.as_ref(), &events)
                                .await;
                            let finished = Instant::now();
                            let (error, output, model_result) = match result {
                                Ok(result) => {
                                    let output = Some(result.ui);
                                    if result.is_error {
                                        let error = mcp_tool_error_message(&result.model);
                                        (
                                            Some(error.clone()),
                                            output,
                                            Err(ToolExecutionError::other(error)),
                                        )
                                    } else {
                                        (None, output, Ok(result.model))
                                    }
                                }
                                Err(error) => {
                                    let error = format!("{error:#}");
                                    (
                                        Some(error.clone()),
                                        None,
                                        Err(ToolExecutionError::other(error)),
                                    )
                                }
                            };

                            let _ = events.send(ToolEvent::Finish {
                                id,
                                error,
                                output,
                                duration: finished.duration_since(started),
                                finished,
                            });
                            model_result
                        })
                    },
                )
            })
            .collect()
    }

    fn summary(&self) -> String {
        if self.services.is_empty() {
            return "No MCP servers connected.".into();
        }

        let mut output = format!(
            "**MCP** · {} server{} · {} tool{}",
            self.services.len(),
            if self.services.len() == 1 { "" } else { "s" },
            self.tools.len(),
            if self.tools.len() == 1 { "" } else { "s" },
        );
        for (index, connection) in self.services.iter().enumerate() {
            let tools = self
                .tools
                .iter()
                .filter(|tool| tool.server == index)
                .count();
            output.push_str(&format!(
                "\n\n- **{}** · {tools} tool{} · MCP {}{}",
                connection.name,
                if tools == 1 { "" } else { "s" },
                connection.protocol.as_str(),
                if connection.tasks { " · Tasks" } else { "" },
            ));
        }
        output
    }

    async fn shutdown(mut self) -> Result<()> {
        let mut errors = Vec::new();
        for connection in &mut self.services {
            match connection
                .service
                .close_with_timeout(Duration::from_secs(3))
                .await
            {
                Ok(Some(_)) => {}
                Ok(None) => {
                    errors.push(format!("MCP server {} shutdown timed out", connection.name))
                }
                Err(error) => errors.push(format!(
                    "MCP server {} shutdown failed: {error}",
                    connection.name
                )),
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            bail!(errors.join("; "))
        }
    }
}

fn mcp_client_info() -> ClientInfo {
    let mut capabilities = ClientCapabilities::default();
    capabilities.extensions = Some(BTreeMap::from([(
        TASKS_EXTENSION_ID.to_owned(),
        JsonObject::new(),
    )]));
    ClientInfo::new(
        capabilities,
        Implementation::new(env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION")),
    )
}

fn encode_tool_result(
    result: CallToolResult,
    output_schema: Option<&Value>,
) -> Result<McpToolResult> {
    validate_structured_output(&result, output_schema)?;
    project_tool_result(&result)
}

fn validate_structured_output(
    result: &CallToolResult,
    output_schema: Option<&Value>,
) -> Result<()> {
    if result.is_error == Some(true) {
        return Ok(());
    }
    let Some(schema) = output_schema else {
        return Ok(());
    };
    let structured = result.structured_content.as_ref().context(
        "MCP server omitted structuredContent required by the tool's declared outputSchema",
    )?;
    let validator = jsonschema::options()
        .with_draft(jsonschema::Draft::Draft202012)
        .offline()
        .build(schema)
        .map_err(|_| anyhow!("MCP tool declared an invalid outputSchema"))?;
    if !validator.is_valid(structured) {
        bail!(
            "MCP server returned structuredContent that does not match its declared outputSchema"
        );
    }
    Ok(())
}

fn task_poll_interval(ms: Option<u64>) -> Duration {
    const DEFAULT: Duration = Duration::from_secs(1);
    const MINIMUM: Duration = Duration::from_millis(100);
    ms.map(Duration::from_millis)
        .unwrap_or(DEFAULT)
        .max(MINIMUM)
}

async fn cancel_request(peer: &Peer<RoleClient>, request_id: RequestId) {
    let _ = peer
        .notify_cancelled(CancelledNotificationParam::new(
            Some(request_id),
            Some("stopped by user".into()),
        ))
        .await;
}

fn mcp_tool_error_message(output: &ToolOutput) -> String {
    let rendered = output.render();
    let detail = if rendered.trim().is_empty() {
        "MCP tool error".into()
    } else {
        truncate_head_tail(&rendered, MAX_TOOL_OUTPUT_PREVIEW_BYTES, "tool error")
    };
    format!("MCP tool returned isError=true: {detail}")
}

// ---------- Model ----------

enum ToolEvent {
    Start {
        id: ToolId,
        label: String,
        args: Value,
        started: Instant,
    },
    Status {
        id: ToolId,
        status: Option<String>,
    },
    Progress {
        id: ToolId,
        status: Option<String>,
    },
    Finish {
        id: ToolId,
        error: Option<String>,
        output: Option<Value>,
        duration: Duration,
        finished: Instant,
    },
}

fn build_agent(
    settings: &Settings,
    mcp: &Arc<McpHost>,
    events: Sender<ToolEvent>,
) -> Result<Agent> {
    let client = openai::CompletionsClient::builder()
        .api_key(settings.api_key.clone())
        .base_url(settings.base_url.clone())
        .build()?;
    let mut builder = AgentBuilder::new(client.responses_api().completion_model(&settings.model))
        .default_max_turns(MAX_TOOL_ROUNDS)
        .additional_params(settings.additional_params.clone());
    if let Some(name) = &settings.agent_name {
        builder = builder.name(name);
    }
    if let Some(description) = &settings.agent_description {
        builder = builder.description(description);
    }
    if let Some(system) = &settings.system {
        builder = builder.preamble(system);
    }
    let tools = mcp.dynamic_tools(events);
    Ok(if tools.is_empty() {
        builder.build()
    } else {
        builder.dynamic_tools(tools).build()
    })
}

#[derive(Clone, Copy)]
struct CompactToolHistory;

impl AgentHook for CompactToolHistory {
    async fn on_completion_call(
        &self,
        _context: &HookContext,
        event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        let history = compact_tool_history(event.history, event.turn > 1);
        if history == event.history {
            CompletionCallAction::Continue
        } else {
            CompletionCallAction::patch(RequestPatch::new().history(history))
        }
    }
}

fn compact_tool_history(history: &[RigMessage], preserve_current_turn: bool) -> Vec<RigMessage> {
    let current_turn = preserve_current_turn
        .then(|| history.iter().rposition(is_user_prompt_message))
        .flatten();
    let latest = (preserve_current_turn && current_turn.is_none())
        .then(|| history.iter().rposition(is_tool_result_message))
        .flatten();
    history
        .iter()
        .enumerate()
        .map(|(index, message)| {
            if Some(index) == latest
                || current_turn
                    .is_some_and(|start| index > start && is_tool_result_message(message))
            {
                return message.clone();
            }
            let RigMessage::User { content } = message else {
                return message.clone();
            };
            RigMessage::User {
                content: content
                    .iter()
                    .map(|content| match content {
                        UserContent::ToolResult(result) => {
                            let mut result = result.clone();
                            result.content = compact_tool_result_content(&result.content);
                            UserContent::ToolResult(result)
                        }
                        content => content.clone(),
                    })
                    .collect(),
            }
        })
        .collect()
}

fn is_user_prompt_message(message: &RigMessage) -> bool {
    matches!(
        message,
        RigMessage::User { content }
            if content.iter().any(|item| !matches!(item, UserContent::ToolResult(_)))
    )
}

fn is_tool_result_message(message: &RigMessage) -> bool {
    matches!(
        message,
        RigMessage::User { content }
            if content.iter().any(|item| matches!(item, UserContent::ToolResult(_)))
    )
}

fn compact_tool_result_content(content: &[ToolResultContent]) -> Vec<ToolResultContent> {
    content
        .iter()
        .map(|content| match content {
            ToolResultContent::Text(text) => ToolResultContent::text(truncate_head_tail(
                &text.text,
                MAX_HISTORY_TOOL_RESULT_BYTES,
                "older tool result",
            )),
            ToolResultContent::Json { value } => {
                ToolResultContent::json(limit_json(value.clone(), MAX_HISTORY_TOOL_RESULT_BYTES))
            }
            ToolResultContent::Image(_) => {
                ToolResultContent::text("[image omitted from older tool result]")
            }
        })
        .collect()
}

async fn run_turn_with_events<B: Backend, F>(
    terminal: &mut Terminal<B>,
    ui: &mut Ui,
    agent: &Agent,
    input: &str,
    history: &mut Vec<RigMessage>,
    events: &Receiver<ToolEvent>,
    poll_event: &mut F,
) -> Result<PromptResponse>
where
    <B as Backend>::Error: Send + Sync + 'static,
    F: FnMut() -> Result<Option<Event>>,
{
    ui.begin(Instant::now());
    let mut escape = false;
    let stream = agent
        .runner(input)
        .add_hook(CompactToolHistory)
        .history(history.clone())
        .tool_concurrency(TOOL_CONCURRENCY)
        .stream();
    tokio::pin!(stream);

    let mut interval = tokio::time::interval(Duration::from_millis(80));
    let mut message = None;
    let mut stream = loop {
        tokio::select! {
            stream = &mut stream => break stream,
            _ = interval.tick() => {
                drain_tool_events(ui, events, &mut message);
                drain_busy_events(terminal, ui, poll_event, &mut escape)?;
                draw(terminal, ui)?;
            }
        }
    };

    let mut final_response = None;
    loop {
        tokio::select! {
            item = stream.next() => match item {
                Some(Ok(MultiTurnStreamItem::FinalResponse(response))) => {
                    final_response = Some(response)
                }
                Some(Ok(item)) => {
                    drain_tool_events(ui, events, &mut message);
                    match item {
                        MultiTurnStreamItem::StreamAssistantItem(
                            StreamedAssistantContent::Text(text),
                        ) => {
                            ui.phase("Responding");
                            ui.assistant_delta(&mut message, &text.text);
                        }
                        MultiTurnStreamItem::StreamAssistantItem(
                            StreamedAssistantContent::ReasoningDelta { id, reasoning, .. },
                        ) => {
                            message = None;
                            ui.phase("Thinking");
                            ui.reasoning_delta(&id, &reasoning);
                        }
                        MultiTurnStreamItem::StreamAssistantItem(
                            StreamedAssistantContent::Reasoning { reasoning, id },
                        ) => {
                            message = None;
                            ui.phase("Thinking");
                            ui.reasoning(&id, &reasoning);
                        }
                        MultiTurnStreamItem::CompletionCall(call)
                            if call.usage.input_tokens > 0 =>
                        {
                            ui.context = call.usage.input_tokens;
                        }
                        MultiTurnStreamItem::StreamAssistantItem(
                            StreamedAssistantContent::Final(stream_final),
                        ) if stream_final.usage.input_tokens > 0 => {
                            ui.context = stream_final.usage.input_tokens;
                        }
                        _ => {}
                    }
                    draw(terminal, ui)?;
                }
                Some(Err(error)) => {
                    drain_tool_events(ui, events, &mut message);
                    ui.finish_thinking_at(Instant::now());
                    return Err(error.into());
                }
                None => break,
            },
            _ = interval.tick() => {
                drain_tool_events(ui, events, &mut message);
                drain_busy_events(terminal, ui, poll_event, &mut escape)?;
                draw(terminal, ui)?;
            }
        }
    }

    let response = finalize_stream(final_response, ui, events, &mut message)?;
    if let Some(messages) = response.messages() {
        history.extend_from_slice(messages);
    }
    Ok(response)
}

fn drain_busy_events<B: Backend, F>(
    terminal: &mut Terminal<B>,
    ui: &mut Ui,
    poll_event: &mut F,
    escape: &mut bool,
) -> Result<()>
where
    B::Error: Send + Sync + 'static,
    F: FnMut() -> Result<Option<Event>>,
{
    for _ in 0..64 {
        let Some(event) = poll_event()? else { break };
        match event {
            Event::Paste(text) => ui.insert_input(&text),
            Event::Key(key) if key.kind == KeyEventKind::Press => {
                if key.code != KeyCode::Esc {
                    *escape = false;
                }
                match key.code {
                    KeyCode::Enter => {}
                    KeyCode::Esc if *escape => return Err(Stop.into()),
                    KeyCode::Esc => *escape = true,
                    KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        if ui.input.value().is_empty() {
                            return Err(Stop.into());
                        }
                        ui.clear_input();
                    }
                    _ if handle_common_key(ui, &key) => {}
                    _ => ui.input.handle_key_event(key),
                }
            }
            Event::Mouse(mouse) => handle_mouse(ui, mouse.kind, mouse.column, mouse.row)?,
            Event::Resize(_, _) => terminal.autoresize()?,
            _ => {}
        }
    }
    Ok(())
}

// ---------- TUI ----------

const COMMAND_HELP: &str = "\
## Commands

| Command | Action |
| --- | --- |
| `/copy` | copy the latest assistant response |
| `/export` | save the conversation as Markdown |
| `/clear` | clear the conversation |
| `/help` | show this help |
| `/mcp` | show MCP servers/tools |
| `/exit` `/quit` | save the session and exit |

## Keys

| Key | Action |
| --- | --- |
| `Enter` | send |
| `Shift+Enter` | new line |
| `▸` | toggle its details |
| `drag` | copy selection |
| `↑/↓` | scroll |
| `Esc Esc` | stop the current turn |
| `Ctrl+C` | clear input, or exit when empty";
const CONTENT_GUTTER: u16 = 3;
const INPUT_GUTTER: u16 = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CommandAction {
    Prompt,
    Handled,
    Exit,
}

#[derive(Clone)]
enum Message {
    User(String),
    Assistant {
        text: String,
        metrics: Option<String>,
    },
    Thinking {
        id: String,
        text: Option<String>,
        started: Option<Instant>,
        duration: Option<Duration>,
        open: bool,
    },
    Tools(Vec<ToolView>),
    Info(String),
    Error(String),
}

#[derive(Clone)]
struct ToolView {
    id: ToolId,
    started: Option<Instant>,
    label: String,
    detail: String,
    status: Option<String>,
    args: Value,
    output: Option<Value>,
    output_open: bool,
    output_preview: Option<String>,
    open: bool,
    state: ToolState,
}

#[derive(Clone)]
enum ToolState {
    Pending,
    Done(Duration),
    Failed(String, Duration),
}

#[derive(Clone, Copy)]
enum ChatRow {
    Thinking(usize),
    Tool(usize),
    ToolOutput(usize),
}

struct Ui {
    messages: Vec<Message>,
    input: TextState<'static>,
    activity: Option<(&'static str, Instant)>,
    thinking_since: Option<Instant>,
    session_label: String,
    context: u64,
    context_window: Option<u64>,
    scroll: u16,
    scroll_max: u16,
    follow_tail: bool,
    chat_area: Rect,
    chat_scroll: u16,
    chat_rows: Vec<Option<ChatRow>>,
    selection: Option<((u16, u16), (u16, u16))>,
    screen: Buffer,
}

struct ChatState {
    ui: Ui,
    history: Vec<RigMessage>,
}

impl ChatState {
    fn new(settings: &Settings, mcp: &McpHost, resumed: Option<Session>) -> Self {
        let label = settings.session_label(mcp.tools.len());
        match resumed {
            None => Self {
                ui: Ui::new(label, settings.model_context_window),
                history: Vec::new(),
            },
            Some(session) => Self {
                ui: Ui::from_saved(label, settings.model_context_window, session.messages),
                history: session.history,
            },
        }
    }
}

impl Ui {
    fn new(session_label: String, context_window: Option<u64>) -> Self {
        Self {
            messages: Vec::new(),
            input: TextState::new(),
            activity: None,
            thinking_since: None,
            session_label,
            context: 0,
            context_window,
            scroll: 0,
            scroll_max: 0,
            follow_tail: true,
            chat_area: Rect::default(),
            chat_scroll: 0,
            chat_rows: Vec::new(),
            selection: None,
            screen: Buffer::empty(Rect::default()),
        }
    }

    fn from_saved(
        session_label: String,
        context_window: Option<u64>,
        messages: Vec<SavedMessage>,
    ) -> Self {
        let mut ui = Self::new(session_label, context_window);
        ui.messages = messages
            .into_iter()
            .map(|message| match message {
                SavedMessage::User { text } => Message::User(text),
                SavedMessage::Assistant { text, metrics } => Message::Assistant { text, metrics },
                SavedMessage::Thinking {
                    id,
                    text,
                    duration_ms,
                } => Message::Thinking {
                    id,
                    text,
                    started: None,
                    duration: Some(Duration::from_millis(duration_ms)),
                    open: false,
                },
                SavedMessage::Tools { tools } => {
                    Message::Tools(tools.into_iter().map(ToolView::from).collect())
                }
            })
            .collect();
        ui
    }

    fn push(&mut self, message: Message) -> usize {
        self.messages.push(message);
        self.messages.len() - 1
    }

    fn assistant(&mut self, text: impl Into<String>) -> usize {
        self.push(Message::Assistant {
            text: text.into(),
            metrics: None,
        })
    }

    fn assistant_delta(&mut self, index: &mut Option<usize>, delta: &str) {
        let i = *index.get_or_insert_with(|| self.assistant(""));
        if let Message::Assistant { text, .. } = &mut self.messages[i] {
            text.push_str(delta);
        }
    }

    fn tools(&self) -> impl Iterator<Item = &ToolView> {
        self.messages
            .iter()
            .filter_map(|message| match message {
                Message::Tools(tools) => Some(tools.iter()),
                _ => None,
            })
            .flatten()
    }

    fn tools_mut(&mut self) -> impl Iterator<Item = &mut ToolView> {
        self.messages
            .iter_mut()
            .filter_map(|message| match message {
                Message::Tools(tools) => Some(tools.iter_mut()),
                _ => None,
            })
            .flatten()
    }

    fn clear_input(&mut self) {
        self.input.value_mut().clear();
        *self.input.position_mut() = 0;
    }

    fn insert_input(&mut self, text: &str) {
        let mut chars = text.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch == '\r' {
                if chars.peek() == Some(&'\n') {
                    chars.next();
                }
                self.input.push('\n');
            } else {
                self.input.push(ch);
            }
        }
    }

    fn reasoning_delta(&mut self, id: &str, delta: &str) {
        if delta.is_empty() {
            return;
        }
        if let Some(index) = self.messages.iter().rposition(
            |message| matches!(message, Message::Thinking { id: current, .. } if current == id),
        ) {
            if let Message::Thinking { text, .. } = &mut self.messages[index] {
                text.get_or_insert_default().push_str(delta);
            }
        } else {
            let started = self.begin_reasoning();
            self.push(Message::Thinking {
                id: id.to_owned(),
                text: Some(delta.to_owned()),
                started: Some(started),
                duration: None,
                open: false,
            });
        }
    }

    fn reasoning(&mut self, id: &str, reasoning: &Reasoning) {
        let text = reasoning_text(reasoning);
        if let Some(index) = self.messages.iter().rposition(
            |message| matches!(message, Message::Thinking { id: current, .. } if current == id),
        ) {
            if let Message::Thinking { text: current, .. } = &mut self.messages[index] {
                *current = text;
            }
        } else {
            let started = self.begin_reasoning();
            self.push(Message::Thinking {
                id: id.to_owned(),
                text,
                started: Some(started),
                duration: None,
                open: false,
            });
        }
    }

    fn begin_reasoning(&mut self) -> Instant {
        let now = Instant::now();
        if self
            .messages
            .iter()
            .rev()
            .any(|message| matches!(message, Message::Thinking { duration: None, .. }))
        {
            self.finish_thinking_at(now);
            self.thinking_since = Some(now);
        }
        self.thinking_since.unwrap_or(now)
    }

    fn toggle_thinking(&mut self, index: usize) -> bool {
        let Some(Message::Thinking { text, open, .. }) = self.messages.get_mut(index) else {
            return false;
        };
        if thinking_parts(text.as_deref()).1.is_none() {
            return false;
        }
        *open = !*open;
        true
    }

    fn toggle_tool(&mut self, index: usize) -> bool {
        self.tools_mut().nth(index).is_some_and(|tool| {
            tool.open = !tool.open;
            true
        })
    }

    fn toggle_tool_output(&mut self, index: usize) -> bool {
        let opening = self
            .tools()
            .nth(index)
            .and_then(|tool| (tool.open && tool.output.is_some()).then_some(!tool.output_open));
        let Some(opening) = opening else {
            return false;
        };
        if opening {
            for tool in self.tools_mut() {
                tool.output_open = false;
            }
        }

        let Some(tool) = self.tools_mut().nth(index) else {
            return false;
        };
        if opening {
            tool.output_preview = tool.output.as_ref().map(tool_output_preview);
        }
        tool.output_open = opening;
        true
    }

    fn begin(&mut self, started: Instant) {
        self.activity = Some(("Thinking", started));
        self.thinking_since = Some(started);
    }

    fn phase(&mut self, label: &'static str) {
        self.phase_at(label, Instant::now());
    }

    fn phase_at(&mut self, label: &'static str, at: Instant) {
        let current = self.activity.map(|(phase, _)| phase);
        if current == Some(label) {
            return;
        }
        if current == Some("Thinking") {
            self.finish_thinking_at(at);
        }
        self.thinking_since = (label == "Thinking").then_some(at);
        if let Some((phase, _)) = &mut self.activity {
            *phase = label;
        }
    }

    fn finish_thinking_at(&mut self, at: Instant) {
        if let Some(Message::Thinking {
            started, duration, ..
        }) = self
            .messages
            .iter_mut()
            .rev()
            .find(|message| matches!(message, Message::Thinking { duration: None, .. }))
        {
            *duration = Some(
                started
                    .and_then(|started| at.checked_duration_since(started))
                    .unwrap_or_default(),
            );
        }
        self.thinking_since = None;
    }

    fn tool_mut(&mut self, id: ToolId) -> Option<&mut ToolView> {
        self.tools_mut().find(|tool| tool.id == id)
    }

    fn tool_status(&mut self, id: ToolId, status: Option<String>) {
        if let Some(tool) = self.tool_mut(id) {
            tool.status = status;
        }
    }

    fn finish_tool(
        &mut self,
        id: ToolId,
        error: Option<String>,
        output: Option<Value>,
        duration: Duration,
    ) {
        if let Some(tool) = self.tool_mut(id) {
            tool.status = None;
            tool.output = output;
            tool.output_open = false;
            tool.output_preview = None;
            tool.state = error.map_or(ToolState::Done(duration), |error| {
                ToolState::Failed(error, duration)
            });
        }
    }

    fn has_pending_tools(&self) -> bool {
        self.tools()
            .any(|tool| matches!(&tool.state, ToolState::Pending))
    }

    fn fail_pending_tools(&mut self, reason: &str) {
        for tool in self.tools_mut() {
            if matches!(&tool.state, ToolState::Pending) {
                tool.status = None;
                tool.state = ToolState::Failed(
                    reason.to_owned(),
                    tool.started
                        .map_or(Duration::ZERO, |started| started.elapsed()),
                );
            }
        }
    }

    fn metrics(&mut self, start: usize, response: &PromptResponse, elapsed: Duration) {
        let usage = response.usage();
        let context = response
            .completion_calls()
            .last()
            .map_or(usage.input_tokens, |call| call.usage.input_tokens);
        self.context = context;
        let summary = usage_summary(&usage, elapsed);
        if let Some(Message::Assistant { metrics, .. }) = self.messages[start..]
            .iter_mut()
            .rev()
            .find(|message| matches!(message, Message::Assistant { .. }))
        {
            *metrics = Some(summary);
        }
    }

    fn context_title(&self) -> String {
        let Some(window) = self.context_window else {
            return format!("ctx {}", compact_count(self.context));
        };
        let percent = self.context.saturating_mul(100).saturating_add(window / 2) / window;
        format!("ctx {} ({percent}%)", context_compact(self.context))
    }

    fn scroll_up(&mut self, amount: u16) {
        if self.scroll_max == 0 {
            return;
        }
        if self.follow_tail {
            self.scroll = self.scroll_max;
        }
        self.follow_tail = false;
        self.scroll = self.scroll.saturating_sub(amount);
    }

    fn scroll_down(&mut self, amount: u16) {
        self.scroll = self.scroll.saturating_add(amount).min(self.scroll_max);
        self.follow_tail = self.scroll == self.scroll_max;
    }

    fn follow_tail(&mut self) {
        self.follow_tail = true;
        self.scroll = self.scroll_max;
    }

    fn move_input_vertical(&mut self, down: bool) -> bool {
        if !self.input.value().contains('\n') {
            return false;
        }
        let lengths = self
            .input
            .value()
            .split('\n')
            .map(|line| line.chars().count())
            .collect::<Vec<_>>();
        let mut line = 0;
        let mut column = self.input.position();
        while column > lengths[line] {
            column -= lengths[line] + 1;
            line += 1;
        }
        let target = if down {
            (line + 1).min(lengths.len() - 1)
        } else {
            line.saturating_sub(1)
        };
        *self.input.position_mut() =
            lengths[..target].iter().sum::<usize>() + target + column.min(lengths[target]);
        true
    }

    fn chat_row_at(&self, column: u16, row: u16) -> Option<ChatRow> {
        if !self.chat_area.contains((column, row).into()) {
            return None;
        }
        let row = usize::from(self.chat_scroll.saturating_add(row - self.chat_area.y));
        self.chat_rows.get(row).copied().flatten()
    }

    fn last_assistant(&self) -> Option<&str> {
        self.messages
            .iter()
            .rev()
            .find_map(|message| match message {
                Message::Assistant { text, .. } if !text.is_empty() => Some(text.as_str()),
                _ => None,
            })
    }

    fn transcript(&self) -> Option<String> {
        let mut text = String::new();
        for message in &self.messages {
            if !text.is_empty() {
                text.push_str("\n\n---\n\n");
            }
            match message {
                Message::User(body) => text.push_str(&format!("## User\n\n{body}")),
                Message::Assistant {
                    text: body,
                    metrics,
                } if !body.is_empty() => {
                    let header = metrics.as_deref().map_or_else(
                        || self.session_label.clone(),
                        |metrics| format!("{} · {metrics}", self.session_label),
                    );
                    text.push_str(&format!("## Assistant ({header})\n\n{body}"));
                }
                Message::Thinking {
                    text: Some(body),
                    duration,
                    ..
                } => {
                    let elapsed = duration
                        .map(compact_duration)
                        .unwrap_or_else(|| "in progress".into());
                    text.push_str(&format!("## Assistant Thinking ({elapsed})\n\n{body}"));
                }
                Message::Tools(tools) => {
                    for (index, tool) in tools.iter().enumerate() {
                        if index > 0 {
                            text.push_str("\n\n");
                        }
                        text.push_str(&format!(
                            "**Tool: {}**\n\n**Input:**\n```json\n{}\n```",
                            tool.label,
                            pretty_json(&tool.args)
                        ));
                        if let Some(output) = &tool.output {
                            text.push_str(&format!(
                                "\n\n**Output (bounded UI projection, not an exact model-request trace):**\n```json\n{}\n```",
                                pretty_json(output)
                            ));
                        } else if let ToolState::Failed(error, _) = &tool.state {
                            text.push_str(&format!("\n\n**Error:**\n```text\n{error}\n```"));
                        }
                    }
                }
                _ => {
                    text.truncate(text.trim_end_matches("\n\n---\n\n").len());
                }
            }
        }
        (!text.is_empty()).then_some(text)
    }

    fn saved_messages(&self) -> Result<Vec<SavedMessage>> {
        self.messages
            .iter()
            .filter_map(|message| match message {
                Message::User(text) => Some(Ok(SavedMessage::User { text: text.clone() })),
                Message::Assistant { text, metrics } => Some(Ok(SavedMessage::Assistant {
                    text: text.clone(),
                    metrics: metrics.clone(),
                })),
                Message::Thinking {
                    id,
                    text,
                    duration: Some(duration),
                    ..
                } => Some(Ok(SavedMessage::Thinking {
                    id: id.clone(),
                    text: text.clone(),
                    duration_ms: duration_ms(*duration),
                })),
                Message::Thinking { .. } => Some(Err(anyhow!(
                    "cannot save a session with unfinished reasoning"
                ))),
                Message::Tools(tools) => Some(
                    tools
                        .iter()
                        .map(SavedTool::try_from)
                        .collect::<Result<Vec<_>>>()
                        .map(|tools| SavedMessage::Tools { tools }),
                ),
                Message::Info(_) | Message::Error(_) => None,
            })
            .collect()
    }
}

impl TryFrom<&ToolView> for SavedTool {
    type Error = anyhow::Error;

    fn try_from(tool: &ToolView) -> Result<Self> {
        let (error, duration) = match &tool.state {
            ToolState::Pending => bail!("cannot save a session with pending tools"),
            ToolState::Done(duration) => (None, *duration),
            ToolState::Failed(error, duration) => (Some(error.clone()), *duration),
        };
        Ok(Self {
            label: tool.label.clone(),
            args: tool.args.clone(),
            output: tool.output.clone(),
            error,
            duration_ms: duration_ms(duration),
        })
    }
}

impl From<SavedTool> for ToolView {
    fn from(tool: SavedTool) -> Self {
        let duration = Duration::from_millis(tool.duration_ms);
        let state = tool.error.map_or(ToolState::Done(duration), |error| {
            ToolState::Failed(error, duration)
        });
        let id = TOOL_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        Self {
            id,
            started: None,
            detail: tool_detail(&tool.args),
            label: tool.label,
            status: None,
            args: tool.args,
            output: tool.output,
            output_open: false,
            output_preview: None,
            open: false,
            state,
        }
    }
}

fn duration_ms(duration: Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}

fn pretty_json(value: &Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
}

impl ToolView {
    fn pending(id: ToolId, label: &str, args: &Value, started: Instant) -> Self {
        Self {
            id,
            started: Some(started),
            label: label.to_owned(),
            detail: tool_detail(args),
            status: None,
            args: args.clone(),
            output: None,
            output_open: false,
            output_preview: None,
            open: false,
            state: ToolState::Pending,
        }
    }

    fn elapsed(&self) -> Duration {
        match &self.state {
            ToolState::Pending => self
                .started
                .map_or(Duration::ZERO, |started| started.elapsed()),
            ToolState::Done(duration) | ToolState::Failed(_, duration) => *duration,
        }
    }

    fn finished_at(&self, now: Instant) -> Option<Instant> {
        match &self.state {
            ToolState::Pending => Some(now),
            ToolState::Done(duration) | ToolState::Failed(_, duration) => self
                .started
                .and_then(|started| started.checked_add(*duration)),
        }
    }
}

fn drain_tool_events(ui: &mut Ui, events: &Receiver<ToolEvent>, message: &mut Option<usize>) {
    for event in events.try_iter() {
        match event {
            ToolEvent::Start {
                id,
                label,
                args,
                started,
            } => {
                *message = None;
                ui.phase_at("Working", started);
                let tool = ToolView::pending(id, &label, &args, started);
                match ui.messages.last_mut() {
                    Some(Message::Tools(tools)) => tools.push(tool),
                    _ => {
                        ui.push(Message::Tools(vec![tool]));
                    }
                }
            }
            ToolEvent::Status { id, status } | ToolEvent::Progress { id, status } => {
                ui.tool_status(id, status);
            }
            ToolEvent::Finish {
                id,
                error,
                output,
                duration,
                finished,
            } => {
                ui.finish_tool(id, error, output, duration);
                if !ui.has_pending_tools() {
                    ui.phase_at("Thinking", finished);
                }
            }
        }
    }
}

fn finalize_stream(
    final_response: Option<PromptResponse>,
    ui: &mut Ui,
    events: &Receiver<ToolEvent>,
    message: &mut Option<usize>,
) -> Result<PromptResponse> {
    drain_tool_events(ui, events, message);
    ui.finish_thinking_at(Instant::now());
    final_response.context("stream ended without a final response")
}

#[derive(Debug)]
struct DrawError(String);

impl fmt::Display for DrawError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "draw failure: {}", self.0)
    }
}

impl std::error::Error for DrawError {}

#[derive(Debug)]
struct Stop;

impl fmt::Display for Stop {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("stopped")
    }
}

impl std::error::Error for Stop {}

fn draw<B: Backend>(terminal: &mut Terminal<B>, ui: &mut Ui) -> Result<()>
where
    <B as Backend>::Error: Send + Sync + 'static,
{
    let frame = terminal
        .draw(|frame| render(frame, ui))
        .map_err(|error| DrawError(error.to_string()))
        .map_err(anyhow::Error::from)?;
    ui.screen = frame.buffer.clone();
    Ok(())
}

fn render(frame: &mut Frame, ui: &mut Ui) {
    let area = frame.area();
    let activity_height = if ui.activity.is_some() { 2 } else { 0 };
    let input_height = (ui.input.value().matches('\n').count() + 1).min(6) as u16 + 4;
    let [_, header, chat, activity, input] = Layout::vertical([
        Length(1),
        Length(2),
        Min(3),
        Length(activity_height),
        Length(input_height),
    ])
    .areas(area);
    let header = header.inner(Margin::new(CONTENT_GUTTER, 0));
    let chat = chat.inner(Margin::new(CONTENT_GUTTER, 0));
    let activity = activity.inner(Margin::new(CONTENT_GUTTER, 0));
    let input = input.inner(Margin::new(INPUT_GUTTER, 0));
    let [title, label_area] = Layout::horizontal([Length(3), Min(1)]).areas(header);
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "thx",
            Style::default().add_modifier(Modifier::BOLD),
        ))),
        title,
    );
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            ui.session_label.clone(),
            Style::default().fg(LIGHT_GRAY),
        )))
        .alignment(Alignment::Right),
        label_area,
    );
    let (text, chat_rows) = history_content(ui, chat.width);
    ui.scroll_max = history_max_scroll(chat, &text);
    ui.scroll = if ui.follow_tail {
        ui.scroll_max
    } else {
        ui.scroll.min(ui.scroll_max)
    };
    ui.chat_area = chat;
    ui.chat_scroll = ui.scroll;
    ui.chat_rows = chat_rows;
    if ui.messages.is_empty() {
        frame.render_widget(empty_logo(chat), chat);
    } else {
        frame.render_widget(
            Paragraph::new(text)
                .wrap(Wrap { trim: false })
                .scroll((ui.scroll, 0)),
            chat,
        );
    }
    if ui.activity.is_some() {
        frame.render_widget(
            Paragraph::new(Text::from(vec![Line::default(), activity_line(ui)])),
            activity,
        );
    }
    render_input(frame, input, ui);
    if let Some((start, end)) = ui.selection {
        highlight_selection(frame.buffer_mut(), start, end);
    }
}

fn empty_logo(area: Rect) -> Paragraph<'static> {
    const ART: [&str; 3] = ["╺┳╸╻ ╻╻ ╻", " ┃ ┣━┫┏╋┛", " ╹ ╹ ╹╹ ╹"];
    let width = ART
        .iter()
        .map(|line| line.chars().count())
        .max()
        .unwrap_or(0) as u16;
    let height = ART.len() as u16;
    let top = area.height.saturating_sub(height) / 2;
    let left = area.width.saturating_sub(width) / 2;
    let mut lines = vec![Line::default(); top as usize];
    lines.extend(ART.into_iter().map(|line| {
        Line::from(Span::styled(
            format!("{}{}", " ".repeat(left as usize), line),
            Style::default().fg(LIGHT_GRAY),
        ))
    }));
    Paragraph::new(lines)
}

fn selection_bounds(start: (u16, u16), end: (u16, u16)) -> ((u16, u16), (u16, u16)) {
    if (start.1, start.0) <= (end.1, end.0) {
        (start, end)
    } else {
        (end, start)
    }
}

fn highlight_selection(buffer: &mut Buffer, start: (u16, u16), end: (u16, u16)) {
    let (start, end) = selection_bounds(start, end);
    let area = *buffer.area();
    if area.is_empty() {
        return;
    }
    for y in start.1..=end.1.min(area.bottom().saturating_sub(1)) {
        let from = if y == start.1 { start.0 } else { area.x };
        let to = if y == end.1 {
            end.0
        } else {
            area.right().saturating_sub(1)
        };
        for x in from.max(area.x)..=to.min(area.right().saturating_sub(1)) {
            if let Some(cell) = buffer.cell_mut((x, y)) {
                cell.set_style(Style::default().add_modifier(Modifier::REVERSED));
            }
        }
    }
}

fn selected_text(buffer: &Buffer, start: (u16, u16), end: (u16, u16)) -> String {
    let (start, end) = selection_bounds(start, end);
    let area = *buffer.area();
    if area.is_empty() {
        return String::new();
    }
    let mut lines = Vec::new();
    for y in start.1..=end.1.min(area.bottom().saturating_sub(1)) {
        let from = if y == start.1 { start.0 } else { area.x };
        let to = if y == end.1 {
            end.0
        } else {
            area.right().saturating_sub(1)
        };
        let from = from.max(area.x);
        let line = (from..=to.min(area.right().saturating_sub(1)))
            .filter_map(|x| buffer.cell((x, y)))
            .map(|cell| cell.symbol())
            .collect::<String>();
        let gutter = area.x.saturating_add(CONTENT_GUTTER);
        let line = if from < gutter {
            let mut content = line.as_str();
            for _ in 0..gutter.saturating_sub(from) {
                content = content.strip_prefix(' ').unwrap_or(content);
            }
            content.to_owned()
        } else {
            line
        };
        lines.push(line.trim_end().to_owned());
    }
    lines.join("\n").trim_end().to_owned()
}

fn handle_mouse(ui: &mut Ui, kind: MouseEventKind, column: u16, row: u16) -> Result<()> {
    match kind {
        MouseEventKind::ScrollUp => ui.scroll_up(3),
        MouseEventKind::ScrollDown => ui.scroll_down(3),
        MouseEventKind::Down(MouseButton::Left) => {
            ui.selection = Some(((column, row), (column, row)))
        }
        MouseEventKind::Drag(MouseButton::Left) => {
            if let Some((_, end)) = &mut ui.selection {
                *end = (column, row);
            }
        }
        MouseEventKind::Up(MouseButton::Left) => {
            if let Some((start, end)) = ui.selection.take() {
                if start == end {
                    match ui.chat_row_at(column, row) {
                        Some(ChatRow::Thinking(index)) => {
                            ui.toggle_thinking(index);
                        }
                        Some(ChatRow::Tool(tool)) => {
                            ui.toggle_tool(tool);
                        }
                        Some(ChatRow::ToolOutput(tool)) => {
                            ui.toggle_tool_output(tool);
                        }
                        None => {}
                    }
                } else {
                    let text = selected_text(&ui.screen, start, end);
                    if !text.is_empty() {
                        copy_to_clipboard(&text)?;
                    }
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn drain_queued_scroll_events<G>(
    ui: &mut Ui,
    poll_event: &mut G,
    pending: &mut Option<Event>,
) -> Result<()>
where
    G: FnMut() -> Result<Option<Event>>,
{
    loop {
        match poll_event()? {
            Some(Event::Mouse(mouse))
                if matches!(
                    mouse.kind,
                    MouseEventKind::ScrollUp | MouseEventKind::ScrollDown
                ) =>
            {
                handle_mouse(ui, mouse.kind, mouse.column, mouse.row)?;
            }
            Some(event) => {
                *pending = Some(event);
                break;
            }
            None => break,
        }
    }
    Ok(())
}

fn render_input(frame: &mut Frame, area: Rect, ui: &Ui) {
    let busy = ui.activity.is_some();
    let mut block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(LIGHT_GRAY))
        .padding(Padding::new(1, 1, 1, 1));
    if ui.context > 0 {
        block = block.title(
            Line::from(Span::styled(
                format!(" {} ", ui.context_title()),
                Style::default().fg(LIGHT_GRAY),
            ))
            .right_aligned(),
        );
    }
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.width == 0 || inner.height == 0 {
        return;
    }

    let value = ui.input.value();
    if value.is_empty() {
        frame.render_widget(
            Paragraph::new(Span::styled(
                "Ask anything…",
                Style::default().fg(LIGHT_GRAY),
            )),
            inner,
        );
    } else {
        let before = value.chars().take(ui.input.position()).collect::<String>();
        let cursor_row = before.matches('\n').count();
        let cursor = UnicodeWidthStr::width(before.rsplit('\n').next().unwrap_or_default());
        let scroll_x = cursor.saturating_sub(usize::from(inner.width.saturating_sub(1)));
        let scroll_y = cursor_row.saturating_sub(usize::from(inner.height.saturating_sub(1)));
        frame.render_widget(
            Paragraph::new(value).scroll((
                scroll_y.min(u16::MAX as usize) as u16,
                scroll_x.min(u16::MAX as usize) as u16,
            )),
            inner,
        );
        if !busy && inner.width > 0 {
            frame.set_cursor_position((
                inner.x
                    + cursor
                        .saturating_sub(scroll_x)
                        .min(usize::from(inner.width - 1)) as u16,
                inner.y
                    + cursor_row
                        .saturating_sub(scroll_y)
                        .min(usize::from(inner.height - 1)) as u16,
            ));
        }
    }
    if value.is_empty() && !busy && inner.width > 0 {
        frame.set_cursor_position((inner.x, inner.y));
    }
}

fn reasoning_text(reasoning: &Reasoning) -> Option<String> {
    let collect = |summary| {
        reasoning
            .content
            .iter()
            .filter_map(|content| match content {
                ReasoningContent::Summary(text) if summary => Some(text.as_str()),
                ReasoningContent::Text { text, .. } if !summary => Some(text.as_str()),
                _ => None,
            })
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    };
    let summary = collect(true);
    if !summary.trim().is_empty() {
        return Some(summary);
    }
    let text = collect(false);
    (!text.trim().is_empty()).then_some(text)
}

fn thinking_parts(text: Option<&str>) -> (Option<&str>, Option<&str>) {
    const MAX_TITLE_CHARS: usize = 80;
    let Some(text) = text.map(str::trim).filter(|text| !text.is_empty()) else {
        return (None, None);
    };
    let (first, rest) = text.split_once('\n').unwrap_or((text, ""));
    let first = first.trim();
    let title = first
        .strip_prefix("**")
        .and_then(|title| title.strip_suffix("**"))
        .map(str::trim)
        .filter(|title| !title.is_empty() && title.chars().count() <= MAX_TITLE_CHARS);
    if let Some(title) = title {
        let body = rest.trim();
        return (Some(title), (!body.is_empty()).then_some(body));
    }
    (None, Some(text))
}

fn thinking_lines(
    text: Option<&str>,
    open: bool,
    started: Option<Instant>,
    duration: Option<Duration>,
) -> Vec<Line<'static>> {
    let (title, body) = thinking_parts(text);
    let marker = match (body.is_some(), open) {
        (true, true) => "▾",
        (true, false) => "▸",
        (false, _) => "•",
    };
    let elapsed = duration.unwrap_or_else(|| started.map_or(Duration::ZERO, |at| at.elapsed()));
    let label = title.map_or_else(
        || "Thinking".to_owned(),
        |title| format!("Thinking · {title}"),
    );
    let mut lines = vec![Line::from(vec![Span::styled(
        format!("{marker} {label} · {}", compact_duration(elapsed)),
        Style::default().fg(LIGHT_GRAY),
    )])];
    if open && let Some(body) = body {
        lines.push(Line::default());
        lines.extend(body.lines().map(|line| {
            Line::from(vec![
                Span::raw("    "),
                Span::styled(line.to_owned(), Style::default().fg(LIGHT_GRAY)),
            ])
        }));
    }
    lines
}

fn activity_line(ui: &Ui) -> Line<'static> {
    let Some((label, started)) = ui.activity else {
        return Line::default();
    };
    let elapsed = started.elapsed();
    Line::from(vec![
        Span::styled(animation_frame(elapsed).to_string(), Style::default()),
        Span::styled(
            format!(" {label} {}", compact_duration(elapsed)),
            Style::default().fg(LIGHT_GRAY),
        ),
    ])
}

fn history_content(ui: &Ui, width: u16) -> (Text<'static>, Vec<Option<ChatRow>>) {
    let mut lines = Vec::<(Line<'static>, Option<ChatRow>)>::new();
    let mut tool_index = 0;
    for (message_index, message) in ui.messages.iter().enumerate() {
        match message {
            Message::User(text) => {
                lines.extend(
                    user_message_lines(text, width)
                        .into_iter()
                        .map(|line| (line, None)),
                );
            }
            Message::Assistant { text, metrics } => {
                lines.extend(
                    assistant_message_lines(text)
                        .into_iter()
                        .map(|line| (line, None)),
                );
                if let Some(metrics) = metrics {
                    lines.extend([
                        (Line::default(), None),
                        (
                            Line::from(Span::styled(
                                metrics.clone(),
                                Style::default().fg(LIGHT_GRAY),
                            )),
                            None,
                        ),
                    ]);
                }
            }
            Message::Thinking {
                text,
                started,
                duration,
                open,
                ..
            } => {
                let collapsible = thinking_parts(text.as_deref()).1.is_some();
                lines.extend(
                    thinking_lines(text.as_deref(), *open, *started, *duration)
                        .into_iter()
                        .map(|line| {
                            let row = collapsible.then_some(ChatRow::Thinking(message_index));
                            (line, row)
                        }),
                );
            }
            Message::Tools(tools) => {
                lines.extend(tool_group_content(tools, &mut tool_index));
            }
            Message::Info(text) => {
                lines.push((
                    Line::from(vec![
                        Span::styled("✓ ", Style::default()),
                        Span::styled(text.clone(), Style::default().fg(LIGHT_GRAY)),
                    ]),
                    None,
                ));
            }
            Message::Error(text) => {
                for (index, line) in text.lines().enumerate() {
                    lines.push((
                        Line::from(vec![
                            Span::styled(
                                if index == 0 { "× " } else { "  " },
                                Style::default().add_modifier(Modifier::BOLD),
                            ),
                            Span::styled(line.to_owned(), Style::default()),
                        ]),
                        None,
                    ));
                }
            }
        }
        lines.push((Line::default(), None));
    }

    let mut rows = Vec::new();
    for (line, tool) in &lines {
        let height = Paragraph::new(line.clone())
            .wrap(Wrap { trim: false })
            .line_count(width.max(1))
            .max(1);
        rows.extend(std::iter::repeat_n(*tool, height));
    }
    (
        Text::from(lines.into_iter().map(|(line, _)| line).collect::<Vec<_>>()),
        rows,
    )
}

fn assistant_message_lines(message: &str) -> Vec<Line<'static>> {
    let options = tui_markdown::Options::new(MarkdownStyle);
    tui_markdown::from_str_with_options(message, &options)
        .lines
        .into_iter()
        .map(owned_line)
        .collect()
}

fn owned_line(line: Line<'_>) -> Line<'static> {
    let mut owned = Line::from(
        line.spans
            .into_iter()
            .map(|span| Span::styled(span.content.into_owned(), span.style))
            .collect::<Vec<_>>(),
    );
    owned.style = line.style;
    owned.alignment = line.alignment;
    owned
}

fn user_message_lines(message: &str, width: u16) -> Vec<Line<'static>> {
    let width = usize::from(width);
    if width == 0 {
        return Vec::new();
    }

    let max_bubble = width.saturating_sub(2).max(1);
    let side = if max_bubble > 4 { 2 } else { 0 };
    let inner = max_bubble.saturating_sub(side * 2).max(1);
    let wrapped = message
        .split('\n')
        .flat_map(|source| wrap_cells(source, inner))
        .collect::<Vec<_>>();
    let content = wrapped
        .iter()
        .map(|line| UnicodeWidthStr::width(line.as_str()))
        .max()
        .unwrap_or(0);
    let bubble_width = (content + side * 2).min(max_bubble).max(1);
    let indent = width.saturating_sub(bubble_width);
    let style = Style::default().fg(Color::White).bg(Color::Rgb(60, 60, 60));
    let line = |body: String| {
        Line::from(vec![
            Span::raw(" ".repeat(indent)),
            Span::styled(body, style),
        ])
    };

    let mut lines = vec![line(" ".repeat(bubble_width))];
    for text in wrapped {
        let used = UnicodeWidthStr::width(text.as_str());
        lines.push(line(format!(
            "{}{}{}",
            " ".repeat(side),
            text,
            " ".repeat(bubble_width.saturating_sub(side + used)),
        )));
    }
    lines.push(line(" ".repeat(bubble_width)));
    lines
}

fn wrap_cells(text: &str, width: usize) -> Vec<String> {
    if text.is_empty() {
        return vec![String::new()];
    }
    let mut lines = Vec::new();
    let mut line = String::new();
    let mut used = 0;
    for ch in text.chars() {
        let cells = UnicodeWidthChar::width(ch).unwrap_or(0);
        if used + cells > width && !line.is_empty() {
            lines.push(std::mem::take(&mut line));
            used = 0;
        }
        line.push(ch);
        used += cells;
    }
    lines.push(line);
    lines
}

fn tool_group_content(
    tools: &[ToolView],
    tool_index: &mut usize,
) -> Vec<(Line<'static>, Option<ChatRow>)> {
    let now = Instant::now();
    let live_elapsed = tools
        .iter()
        .filter_map(|tool| tool.started)
        .min()
        .zip(tools.iter().filter_map(|tool| tool.finished_at(now)).max())
        .map_or(Duration::ZERO, |(started, finished)| {
            finished.checked_duration_since(started).unwrap_or_default()
        });
    let elapsed = if live_elapsed.is_zero() {
        tools
            .iter()
            .map(ToolView::elapsed)
            .max()
            .unwrap_or_default()
    } else {
        live_elapsed
    };
    let failed = tools
        .iter()
        .any(|tool| matches!(&tool.state, ToolState::Failed(_, _)));
    let pending = tools
        .iter()
        .any(|tool| matches!(&tool.state, ToolState::Pending));
    let icon = if failed {
        '×'
    } else if pending {
        '○'
    } else {
        '●'
    };
    let header = format!(
        "{icon} {} tool call{} · {}",
        tools.len(),
        if tools.len() == 1 { "" } else { "s" },
        compact_duration(elapsed),
    );
    let header_style = if failed {
        Style::default().add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(LIGHT_GRAY).add_modifier(Modifier::BOLD)
    };
    let mut lines = vec![(Line::from(Span::styled(header, header_style)), None)];
    let last = tools.len().saturating_sub(1);

    for (i, tool) in tools.iter().enumerate() {
        let index = *tool_index;
        *tool_index += 1;
        let error = match &tool.state {
            ToolState::Failed(error, _) => Some(error.as_str()),
            _ => None,
        };
        let branch = if i == last { "└ " } else { "├ " };
        let child = if i == last { "   " } else { "│  " };
        let action_style = if error.is_some() {
            Style::default().add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(LIGHT_GRAY).add_modifier(Modifier::BOLD)
        };
        let mut spans = vec![
            Span::styled(branch, Style::default().fg(LIGHT_GRAY)),
            Span::styled(
                if tool.open { "▾ " } else { "▸ " },
                Style::default().fg(LIGHT_GRAY),
            ),
            Span::styled(tool.label.clone(), action_style),
        ];
        if !tool.detail.is_empty() {
            spans.push(Span::styled(" · ", Style::default().fg(LIGHT_GRAY)));
            spans.push(Span::styled(
                tool.detail.clone(),
                Style::default().fg(LIGHT_GRAY),
            ));
        }
        if let Some(status) = &tool.status {
            spans.push(Span::styled(" · ", Style::default().fg(LIGHT_GRAY)));
            spans.push(Span::styled(
                inline(status),
                Style::default().fg(LIGHT_GRAY),
            ));
        }
        spans.push(Span::styled(
            format!(" · {}", compact_duration(tool.elapsed())),
            Style::default().fg(LIGHT_GRAY),
        ));
        lines.push((Line::from(spans), Some(ChatRow::Tool(index))));

        if tool.open {
            let json = tool_output_preview(&tool.args);
            lines.extend(json.lines().map(|line| {
                (
                    Line::from(vec![
                        Span::styled(child, Style::default().fg(LIGHT_GRAY)),
                        Span::styled(format!("  {line}"), Style::default().fg(LIGHT_GRAY)),
                    ]),
                    Some(ChatRow::Tool(index)),
                )
            }));
            if tool.output.is_some() {
                lines.push((
                    Line::from(vec![
                        Span::styled(child, Style::default().fg(LIGHT_GRAY)),
                        Span::styled(
                            if tool.output_open {
                                "  ▾ Output"
                            } else {
                                "  ▸ Output"
                            },
                            Style::default().fg(LIGHT_GRAY).add_modifier(Modifier::BOLD),
                        ),
                    ]),
                    Some(ChatRow::ToolOutput(index)),
                ));
                if tool.output_open
                    && let Some(output) = &tool.output_preview
                {
                    lines.extend(output.lines().map(|line| {
                        (
                            Line::from(vec![
                                Span::styled(child, Style::default().fg(LIGHT_GRAY)),
                                Span::styled(
                                    format!("    {line}"),
                                    Style::default().fg(LIGHT_GRAY),
                                ),
                            ]),
                            Some(ChatRow::ToolOutput(index)),
                        )
                    }));
                }
            }
        }
        if let Some(error) = error {
            for (line_index, line) in error.lines().enumerate() {
                lines.push((
                    Line::from(vec![
                        Span::styled(child, Style::default().fg(LIGHT_GRAY)),
                        Span::styled(
                            if line_index == 0 {
                                "└ failed: "
                            } else {
                                "  "
                            },
                            Style::default().add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(line.to_owned(), Style::default()),
                    ]),
                    Some(ChatRow::Tool(index)),
                ));
            }
        }
    }
    lines
}

fn tool_detail(args: &Value) -> String {
    let Value::Object(args) = args else {
        return scalar_detail(args).map_or_else(String::new, |(_, value)| inline(&value));
    };
    args.iter()
        .filter_map(|(key, value)| scalar_detail(value).map(|(score, value)| (score, key, value)))
        .max_by(|a, b| a.0.cmp(&b.0).then_with(|| a.2.len().cmp(&b.2.len())))
        .map_or_else(String::new, |(_, key, value)| {
            inline(&format!("{key} {value}"))
        })
}

fn scalar_detail(value: &Value) -> Option<(u8, String)> {
    match value {
        Value::String(value) if !value.trim().is_empty() => {
            let value = inline(value);
            let target = value.starts_with("http://")
                || value.starts_with("https://")
                || value.starts_with('/')
                || value.starts_with("./")
                || value.starts_with("../");
            Some((if target { 3 } else { 2 }, value))
        }
        Value::Array(values) if !values.is_empty() => {
            let values = values
                .iter()
                .map(scalar_detail)
                .collect::<Option<Vec<_>>>()?;
            let score = values.iter().map(|(score, _)| *score).max().unwrap_or(0) + 1;
            Some((
                score,
                values
                    .into_iter()
                    .map(|(_, value)| value)
                    .collect::<Vec<_>>()
                    .join(", "),
            ))
        }
        Value::Number(value) => Some((0, value.to_string())),
        Value::Bool(value) => Some((0, value.to_string())),
        _ => None,
    }
}

fn inline(text: &str) -> String {
    const MAX_INLINE_CHARS: usize = 60;
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut chars = text.chars();
    let preview = chars.by_ref().take(MAX_INLINE_CHARS).collect::<String>();
    if chars.next().is_none() {
        return text;
    }
    format!("{preview}…")
}

fn project_tool_result(result: &CallToolResult) -> Result<McpToolResult> {
    let raw_bytes = serde_json::to_vec(result)
        .context("failed to encode MCP tool result")?
        .len();
    if raw_bytes > MAX_RAW_MCP_RESULT_BYTES {
        bail!(
            "MCP tool result exceeded raw safety limit ({} bytes > {} bytes)",
            raw_bytes,
            MAX_RAW_MCP_RESULT_BYTES
        );
    }

    let structured = result.structured_content.clone();
    let mut content = Vec::new();
    for block in &result.content {
        if let Some(text) = block.as_text()
            && structured.as_ref().is_some_and(|value| {
                serde_json::from_str::<Value>(&text.text).ok().as_ref() == Some(value)
            })
        {
            continue;
        }
        content.push(project_content_block(block));
    }

    let semantic = match (structured, content.as_slice()) {
        (Some(value), []) => value,
        (None, [only]) if only.get("type").and_then(Value::as_str) == Some("text") => only
            .get("text")
            .cloned()
            .unwrap_or_else(|| Value::String(String::new())),
        (structured, content) => {
            let mut value = serde_json::Map::new();
            if let Some(structured) = structured {
                value.insert("structuredContent".into(), structured);
            }
            if !content.is_empty() {
                value.insert("content".into(), Value::Array(content.to_vec()));
            }
            Value::Object(value)
        }
    };

    let model_value = limit_json(semantic.clone(), MAX_MODEL_TOOL_RESULT_BYTES);
    let model = match model_value {
        Value::String(text) => ToolOutput::text(truncate_head_tail(
            &text,
            MAX_MODEL_TOOL_RESULT_BYTES,
            "model tool result",
        )),
        value => ToolOutput::json(value),
    };
    Ok(McpToolResult {
        ui: limit_json(semantic, MAX_UI_TOOL_RESULT_BYTES),
        model,
        is_error: result.is_error.unwrap_or(false),
    })
}

fn project_content_block(block: &ContentBlock) -> Value {
    match block {
        ContentBlock::Text(text) => serde_json::json!({
            "type": "text",
            "text": text.text,
        }),
        ContentBlock::Image(image) => {
            Value::String(format!("[MCP image omitted: {}]", inline(&image.mime_type)))
        }
        ContentBlock::Audio(audio) => {
            Value::String(format!("[MCP audio omitted: {}]", inline(&audio.mime_type)))
        }
        ContentBlock::Resource(resource) => match &resource.resource {
            ResourceContents::TextResourceContents {
                uri,
                mime_type,
                text,
                ..
            } => serde_json::json!({
                "type": "resource",
                "uri": uri,
                "mimeType": mime_type,
                "text": text,
            }),
            ResourceContents::BlobResourceContents { uri, mime_type, .. } => serde_json::json!({
                "type": "resource",
                "uri": uri,
                "mimeType": mime_type,
                "content": "[MCP binary resource omitted]",
            }),
            _ => Value::String("[MCP resource omitted]".into()),
        },
        ContentBlock::ResourceLink(resource) => serde_json::json!({
            "type": "resource_link",
            "uri": resource.uri,
            "name": resource.name,
            "title": resource.title,
            "description": resource.description,
            "mimeType": resource.mime_type,
            "size": resource.size,
        }),
        _ => Value::String("[unsupported MCP content omitted]".into()),
    }
}

fn limit_json(value: Value, max_bytes: usize) -> Value {
    if json_fits(&value, max_bytes) {
        return value;
    }

    project_json_value(&value, max_bytes)
}

fn project_json_value(value: &Value, max_bytes: usize) -> Value {
    if json_fits(value, max_bytes) {
        return value.clone();
    }

    match value {
        Value::String(text) => project_json_string(text, max_bytes),
        Value::Array(values) => project_json_array(values, max_bytes),
        Value::Object(values) => project_json_object(values, max_bytes),
        _ => Value::Null,
    }
}

fn project_json_string(text: &str, max_bytes: usize) -> Value {
    let mut low = 0;
    let mut high = max_bytes;
    let mut best = Value::String(String::new());
    while low <= high {
        let limit = low + (high - low) / 2;
        let candidate = Value::String(truncate_head_tail(text, limit, "JSON string"));
        if json_fits(&candidate, max_bytes) {
            best = candidate;
            low = limit.saturating_add(1);
        } else if limit == 0 {
            break;
        } else {
            high = limit - 1;
        }
    }
    best
}

fn project_json_array(values: &[Value], max_bytes: usize) -> Value {
    const MAX_ITEMS: usize = 8;

    let sample_len = values.len().min(MAX_ITEMS);
    let head = sample_len / 2;
    let tail = sample_len - head;
    let mut selected = values[..head].iter().collect::<Vec<_>>();
    selected.extend(values[values.len().saturating_sub(tail)..].iter());
    let initially_omitted = values.len().saturating_sub(selected.len());

    for keep in (0..=selected.len()).rev() {
        let kept_head = keep / 2;
        let kept_tail = keep - kept_head;
        let chosen = selected[..kept_head]
            .iter()
            .chain(selected[selected.len().saturating_sub(kept_tail)..].iter())
            .copied()
            .collect::<Vec<_>>();
        let omitted = initially_omitted + selected.len() - chosen.len();
        if let Some(candidate) = fit_json_children(&chosen, max_bytes, |children| {
            let mut result = children;
            if omitted > 0 {
                result.insert(
                    kept_head,
                    serde_json::json!({"__thx_omitted_items": omitted}),
                );
            }
            Value::Array(result)
        }) {
            return candidate;
        }
    }
    Value::Array(Vec::new())
}

fn project_json_object(values: &serde_json::Map<String, Value>, max_bytes: usize) -> Value {
    const MAX_SCALARS: usize = 8;
    const MAX_STRUCTURES: usize = 4;

    let mut scalars = values
        .iter()
        .filter(|(_, value)| !value.is_array() && !value.is_object())
        .collect::<Vec<_>>();
    scalars.sort_by_key(|(key, value)| (json_len(value), key.as_str()));
    let significant_scalar = scalars.pop();
    let mut structures = values
        .iter()
        .filter(|(_, value)| value.is_array() || value.is_object())
        .collect::<Vec<_>>();
    structures.sort_by_key(|(key, _)| key.as_str());

    let mut selected = scalars
        .into_iter()
        .take(MAX_SCALARS.saturating_sub(1))
        .collect::<Vec<_>>();
    selected.extend(significant_scalar);
    selected.extend(structures.into_iter().take(MAX_STRUCTURES));
    selected.sort_by_key(|(key, _)| key.as_str());

    for keep in (0..=selected.len()).rev() {
        let chosen = selected[..keep]
            .iter()
            .map(|(key, value)| (key.as_str(), *value))
            .collect::<Vec<_>>();
        let omitted = values.len().saturating_sub(chosen.len());
        let children = chosen.iter().map(|(_, value)| *value).collect::<Vec<_>>();
        if let Some(candidate) = fit_json_children(&children, max_bytes, |children| {
            let mut result = serde_json::Map::new();
            for ((key, _), value) in chosen.iter().zip(children) {
                result.insert((*key).to_owned(), value);
            }
            if omitted > 0 {
                let mut marker = "__thx_omitted_fields".to_owned();
                while values.contains_key(&marker) {
                    marker.insert(0, '_');
                }
                result.insert(marker, Value::from(omitted));
            }
            Value::Object(result)
        }) {
            return candidate;
        }
    }
    Value::Object(serde_json::Map::new())
}

fn fit_json_children<F>(children: &[&Value], max_bytes: usize, build: F) -> Option<Value>
where
    F: Fn(Vec<Value>) -> Value,
{
    let mut low = 0;
    let mut high = max_bytes;
    let mut best = None;
    while low <= high {
        let child_budget = low + (high - low) / 2;
        let candidate = build(
            children
                .iter()
                .map(|value| project_json_value(value, child_budget))
                .collect(),
        );
        if json_fits(&candidate, max_bytes) {
            best = Some(candidate);
            low = child_budget.saturating_add(1);
        } else if child_budget == 0 {
            break;
        } else {
            high = child_budget - 1;
        }
    }
    best
}

fn json_len(value: &Value) -> usize {
    serde_json::to_vec(value).map_or(usize::MAX, |encoded| encoded.len())
}

fn json_fits(value: &Value, max_bytes: usize) -> bool {
    struct LimitedWriter(usize);

    impl Write for LimitedWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if bytes.len() > self.0 {
                return Err(io::Error::other("JSON byte budget exceeded"));
            }
            self.0 -= bytes.len();
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    serde_json::to_writer(LimitedWriter(max_bytes), value).is_ok()
}

fn truncate_head_tail(text: &str, max_bytes: usize, label: &str) -> String {
    if text.len() <= max_bytes {
        return text.to_owned();
    }
    let marker = format!("\n\n[thx: {} bytes omitted from {label}]\n\n", text.len());
    if marker.len() >= max_bytes {
        return truncate_utf8(&marker, max_bytes).to_owned();
    }
    let available = max_bytes - marker.len();
    let head_bytes = available / 4;
    let tail_bytes = available - head_bytes;
    let head = truncate_utf8(text, head_bytes);
    let mut tail_start = text.len().saturating_sub(tail_bytes);
    while !text.is_char_boundary(tail_start) {
        tail_start += 1;
    }
    let omitted = text
        .len()
        .saturating_sub(head.len() + text[tail_start..].len());
    format!(
        "{head}\n\n[thx: {omitted} bytes omitted from {label}]\n\n{}",
        &text[tail_start..]
    )
}

fn tool_output_preview(value: &Value) -> String {
    let rendered = pretty_json(value);
    let mut end = truncate_utf8(&rendered, MAX_TOOL_OUTPUT_PREVIEW_BYTES).len();
    let content_lines = MAX_TOOL_OUTPUT_PREVIEW_LINES.saturating_sub(1);
    if let Some(line_end) = rendered
        .match_indices('\n')
        .nth(content_lines.saturating_sub(1))
        .map(|(index, _)| index)
    {
        end = end.min(line_end);
    }
    if end == rendered.len() {
        rendered
    } else {
        format!(
            "{}\n… [output preview truncated]",
            rendered[..end].trim_end()
        )
    }
}

fn truncate_utf8(text: &str, max_bytes: usize) -> &str {
    let mut end = text.len().min(max_bytes);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

fn history_max_scroll(area: Rect, text: &Text<'_>) -> u16 {
    let paragraph = Paragraph::new(text.clone()).wrap(Wrap { trim: false });
    let trailing_blank = usize::from(text.lines.last().is_some_and(|line| line.spans.is_empty()));
    let lines = paragraph
        .line_count(area.width.max(1))
        .saturating_sub(trailing_blank)
        .min(u16::MAX as usize) as u16;
    lines.saturating_sub(area.height)
}

fn handle_command(
    command: &str,
    ui: &mut Ui,
    history: &mut Vec<RigMessage>,
    mcp: &McpHost,
) -> CommandAction {
    match command {
        "/exit" | "/quit" => CommandAction::Exit,
        "/clear" => {
            ui.messages.clear();
            history.clear();
            ui.activity = None;
            ui.thinking_since = None;
            CommandAction::Handled
        }
        "/help" => {
            ui.assistant(COMMAND_HELP);
            CommandAction::Handled
        }
        "/mcp" => {
            ui.assistant(mcp.summary());
            CommandAction::Handled
        }
        "/copy" => {
            match ui.last_assistant() {
                Some(text) => match copy_to_clipboard(text) {
                    Ok(()) => ui.push(Message::Info("Copied last assistant response.".into())),
                    Err(error) => {
                        ui.push(Message::Error(format!("Clipboard copy failed: {error}")))
                    }
                },
                None => ui.push(Message::Error("Nothing to copy yet.".into())),
            };
            CommandAction::Handled
        }
        "/export" => {
            match ui.transcript() {
                Some(text) => match export_transcript(&text) {
                    Ok(path) => ui.push(Message::Info(format!(
                        "Exported conversation to {}.",
                        path.file_name()
                            .and_then(|name| name.to_str())
                            .unwrap_or("Markdown file")
                    ))),
                    Err(error) => ui.push(Message::Error(format!(
                        "Conversation export failed: {error}"
                    ))),
                },
                None => ui.push(Message::Error("Nothing to export yet.".into())),
            };
            CommandAction::Handled
        }
        command if command.starts_with('/') => {
            ui.push(Message::Error(format!(
                "Unknown command `{command}`. Type `/help` for available commands."
            )));
            CommandAction::Handled
        }
        _ => CommandAction::Prompt,
    }
}

fn handle_ctrl_c(ui: &mut Ui) -> CommandAction {
    if ui.input.value().is_empty() {
        CommandAction::Exit
    } else {
        ui.clear_input();
        CommandAction::Handled
    }
}

fn resume_arg() -> Result<Option<String>> {
    resume_arg_from(env::args_os().skip(1))
}

fn resume_arg_from(mut args: impl Iterator<Item = std::ffi::OsString>) -> Result<Option<String>> {
    let Some(arg) = args.next() else {
        return Ok(None);
    };
    if arg != "--resume" {
        bail!("usage: thx [--resume <session-id>]");
    }
    let id = args
        .next()
        .context("missing session id after --resume")?
        .into_string()
        .map_err(|_| anyhow!("session id must be valid UTF-8"))?;
    if args.next().is_some() {
        bail!("usage: thx [--resume <session-id>]");
    }
    validate_session_id(&id)?;
    Ok(Some(id))
}

fn validate_session_id(id: &str) -> Result<()> {
    if id.is_empty() || !id.bytes().all(|byte| byte.is_ascii_digit() || byte == b'-') {
        bail!("invalid session id");
    }
    Ok(())
}

fn session_dir() -> Result<PathBuf> {
    let project = ProjectDirs::from("", "", "thx")
        .context("failed to resolve the platform application state directory")?;
    Ok(project
        .state_dir()
        .unwrap_or_else(|| project.data_local_dir())
        .join("sessions"))
}

fn session_path_in(directory: &Path, id: &str) -> Result<PathBuf> {
    validate_session_id(id)?;
    Ok(directory.join(format!("{id}.json")))
}

fn next_session_path(directory: &Path, timestamp: &str) -> Result<(String, PathBuf)> {
    validate_session_id(timestamp)?;
    for index in 1_u64.. {
        let id = if index == 1 {
            timestamp.to_owned()
        } else {
            format!("{timestamp}-{index}")
        };
        let path = session_path_in(directory, &id)?;
        if !path.exists() {
            return Ok((id, path));
        }
    }
    bail!("failed to choose a session filename")
}

fn save_session(path: &Path, session: &Session) -> Result<()> {
    let json = serde_json::to_vec_pretty(session).context("failed to serialize session")?;
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(path)
        .with_context(|| format!("failed to create session {}", path.display()))?;
    file.write_all(&json)
        .with_context(|| format!("failed to write session {}", path.display()))
}

fn load_session(id: &str) -> Result<Session> {
    load_session_from(&session_dir()?, id)
}

fn load_session_from(directory: &Path, id: &str) -> Result<Session> {
    let path = session_path_in(directory, id)?;
    let json =
        fs::read(&path).with_context(|| format!("failed to read session {}", path.display()))?;
    serde_json::from_slice(&json)
        .with_context(|| format!("failed to deserialize session {}", path.display()))
}

fn autosave_session(session: &Session) -> Result<String> {
    let timestamp = Utc::now().format("%Y%m%d-%H%M%S").to_string();
    autosave_session_in(session, &session_dir()?, &timestamp)
}

fn autosave_session_in(session: &Session, directory: &Path, timestamp: &str) -> Result<String> {
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder
        .create(directory)
        .with_context(|| format!("failed to create session directory {}", directory.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(directory, fs::Permissions::from_mode(0o700)).with_context(|| {
            format!(
                "failed to restrict session directory {}",
                directory.display()
            )
        })?;
    }
    let (id, path) = next_session_path(directory, timestamp)?;
    save_session(&path, session)?;
    Ok(id)
}

fn completed_session(history: Vec<RigMessage>, ui: &Ui) -> Result<Option<Session>> {
    if history.is_empty() {
        Ok(None)
    } else {
        Ok(Some(Session {
            history,
            messages: ui.saved_messages()?,
        }))
    }
}

// ---------- App ----------

#[tokio::main]
async fn main() -> Result<()> {
    let resume = resume_arg()?;
    let resumed = resume.as_deref().map(load_session).transpose()?;
    if let Err(error) = dotenvy::dotenv()
        && !error.not_found()
    {
        return Err(error.into());
    }
    let settings = Settings::load()?;
    let (mcp_path, optional) = settings.mcp_path();
    let mcp = Arc::new(McpHost::load(mcp_path, optional).await?);
    let (event_tx, event_rx) = mpsc::channel();
    let agent = build_agent(&settings, &mcp, event_tx)?;

    let mut terminal = ratatui::try_init()?;
    let result: Result<Option<Session>> = async {
        execute!(
            io::stdout(),
            EnableBracketedPaste,
            EnableMouseCapture,
            PushKeyboardEnhancementFlags(
                KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
                    | KeyboardEnhancementFlags::REPORT_EVENT_TYPES
            )
        )?;
        terminal.autoresize()?;
        terminal.clear()?;
        chat(
            &mut terminal,
            &agent,
            &mcp,
            &event_rx,
            ChatState::new(&settings, &mcp, resumed),
            &mut || event::read().map_err(Into::into),
            &mut || {
                if event::poll(Duration::ZERO)? {
                    event::read().map(Some).map_err(Into::into)
                } else {
                    Ok(None)
                }
            },
        )
        .await
    }
    .await;
    let save = result
        .as_ref()
        .ok()
        .and_then(|session| session.as_ref())
        .map(autosave_session);
    let _ = execute!(
        io::stdout(),
        DisableBracketedPaste,
        DisableMouseCapture,
        PopKeyboardEnhancementFlags
    );
    ratatui::restore();

    match result {
        Ok(_) => {}
        Err(error) => {
            if let Err(shutdown) = shutdown_host(agent, mcp).await {
                return Err(error.context(format!("MCP shutdown also failed: {shutdown:#}")));
            }
            return Err(error);
        }
    }

    if let Some(save) = save {
        match save {
            Ok(id) => println!("Continue with:\n  thx --resume {id}"),
            Err(error) => eprintln!("Warning: session was not saved:\n  {error:#}"),
        }
    }

    shutdown_host(agent, mcp).await
}

async fn shutdown_host(agent: Agent, mcp: Arc<McpHost>) -> Result<()> {
    drop(agent);
    match Arc::try_unwrap(mcp) {
        Ok(host) => host.shutdown().await,
        Err(_) => Err(anyhow::anyhow!(
            "internal error: MCP host still referenced during shutdown"
        )),
    }
}

async fn chat<B: Backend, F, G>(
    terminal: &mut Terminal<B>,
    agent: &Agent,
    mcp: &McpHost,
    events: &Receiver<ToolEvent>,
    state: ChatState,
    read_event: &mut F,
    poll_event: &mut G,
) -> Result<Option<Session>>
where
    B::Error: Send + Sync + 'static,
    F: FnMut() -> Result<Event>,
    G: FnMut() -> Result<Option<Event>>,
{
    let ChatState {
        mut ui,
        mut history,
    } = state;
    let mut pending_event = None;

    loop {
        draw(terminal, &mut ui)?;
        let event = pending_event.take().map_or_else(&mut *read_event, Ok)?;
        match event {
            Event::Paste(text) => {
                ui.insert_input(&text);
            }
            Event::Mouse(mouse) => {
                handle_mouse(&mut ui, mouse.kind, mouse.column, mouse.row)?;
                if matches!(
                    mouse.kind,
                    MouseEventKind::ScrollUp | MouseEventKind::ScrollDown
                ) {
                    drain_queued_scroll_events(&mut ui, poll_event, &mut pending_event)?;
                }
            }
            Event::Resize(_, _) => {
                terminal.autoresize()?;
            }
            Event::Key(key) if key.kind == KeyEventKind::Press => match key.code {
                KeyCode::Esc => {}
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    if handle_ctrl_c(&mut ui) == CommandAction::Exit {
                        break;
                    }
                }
                _ if handle_common_key(&mut ui, &key) => {}
                KeyCode::Enter if !ui.input.value().trim().is_empty() => {
                    let input = std::mem::take(ui.input.value_mut());
                    *ui.input.position_mut() = 0;
                    let command = input.trim();
                    ui.follow_tail();
                    match handle_command(command, &mut ui, &mut history, mcp) {
                        CommandAction::Exit => break,
                        CommandAction::Handled => continue,
                        CommandAction::Prompt => {}
                    }

                    ui.push(Message::User(input.clone()));
                    let turn_messages = ui.messages.len();
                    let started = Instant::now();
                    let result = run_turn_with_events(
                        terminal,
                        &mut ui,
                        agent,
                        &input,
                        &mut history,
                        events,
                        poll_event,
                    )
                    .await;
                    ui.activity = None;
                    ui.thinking_since = None;
                    match result {
                        Ok(response) => {
                            ui.metrics(turn_messages, &response, started.elapsed());
                        }
                        Err(error) => {
                            mcp.cancel_calls().await;
                            if error.downcast_ref::<DrawError>().is_some() {
                                return Err(error);
                            }
                            if error.downcast_ref::<Stop>().is_some() {
                                ui.fail_pending_tools("cancelled");
                                ui.push(Message::Info("Stopped current turn.".into()));
                            } else {
                                ui.fail_pending_tools("aborted");
                                ui.push(Message::Error(format!("{error:#}")));
                            }
                        }
                    }
                }
                _ => ui.input.handle_key_event(key),
            },
            _ => {}
        }
    }
    completed_session(history, &ui)
}

fn handle_common_key(ui: &mut Ui, key: &KeyEvent) -> bool {
    match key.code {
        KeyCode::Up if !ui.move_input_vertical(false) => ui.scroll_up(1),
        KeyCode::Down if !ui.move_input_vertical(true) => ui.scroll_down(1),
        KeyCode::PageUp => ui.scroll_up(8),
        KeyCode::PageDown => ui.scroll_down(8),
        KeyCode::Enter | KeyCode::Char('\n' | '\r')
            if key.modifiers.contains(KeyModifiers::SHIFT) =>
        {
            ui.insert_input("\n")
        }
        _ => return false,
    }
    true
}

fn copy_to_clipboard(text: &str) -> Result<()> {
    execute!(io::stdout(), CopyToClipboard::to_clipboard_from(text))?;
    Ok(())
}

fn export_transcript(text: &str) -> Result<PathBuf> {
    let directory = env::current_dir().context("failed to resolve current directory")?;
    export_transcript_to(&directory, text)
}

fn export_transcript_to(directory: &Path, text: &str) -> Result<PathBuf> {
    let mut index = 1;
    loop {
        let name = if index == 1 {
            "thx-export.md".into()
        } else {
            format!("thx-export-{index}.md")
        };
        let path = directory.join(name);
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(mut file) => {
                file.write_all(text.as_bytes())
                    .with_context(|| format!("failed to write {}", path.display()))?;
                return Ok(path);
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => index += 1,
            Err(error) => {
                return Err(error).with_context(|| format!("failed to create {}", path.display()));
            }
        }
    }
}

// ---------- Helpers ----------

fn env_value(name: &str) -> Result<Option<String>> {
    match env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(error).with_context(|| format!("failed to read {name}")),
    }
}

fn env_or(name: &str, default: &str) -> Result<String> {
    let value = env_value(name)?.unwrap_or_else(|| default.into());
    nonempty(name, value)
}

fn nonempty(name: &str, value: String) -> Result<String> {
    if value.trim().is_empty() {
        bail!("{name} must not be empty");
    }
    Ok(value)
}

fn env_u64(name: &str) -> Result<Option<u64>> {
    env_value(name)?
        .map(|value| {
            let value = value
                .trim()
                .parse::<u64>()
                .with_context(|| format!("{name} must be a positive integer"))?;
            positive_u64(name, value)
        })
        .transpose()
}

fn positive_u64(name: &str, value: u64) -> Result<u64> {
    if value == 0 {
        bail!("{name} must be greater than zero");
    }
    Ok(value)
}

fn env_path(name: &str) -> Result<Option<String>> {
    match env_value(name)? {
        Some(value) if value.trim().is_empty() => bail!("{name} must not be empty"),
        Some(value) => expand(&value).map(Some),
        None => Ok(None),
    }
}

fn usage_summary(usage: &RigUsage, elapsed: Duration) -> String {
    if !usage.has_values() {
        return compact_duration(elapsed);
    }
    let mut parts = vec![
        format!("input-total {}", compact_count(usage.input_tokens)),
        format!("output-total {}", compact_count(usage.output_tokens)),
    ];
    if usage.tool_use_prompt_tokens > 0 {
        parts.push(format!(
            "tool-prompt {}",
            compact_count(usage.tool_use_prompt_tokens),
        ));
    }
    if usage.reasoning_tokens > 0 {
        parts.push(format!("∵ {}", compact_count(usage.reasoning_tokens)));
    }
    parts.push(format!("⧖ {}", compact_duration(elapsed)));
    parts.join(" · ")
}

fn compact_count(value: u64) -> String {
    let (value, suffix) = if value >= 1_000_000 {
        (value as f64 / 1_000_000.0, "m")
    } else if value >= 1_000 {
        (value as f64 / 1_000.0, "k")
    } else {
        return value.to_string();
    };
    format!("{value:.1}{suffix}").replace(".0", "")
}

fn context_compact(value: u64) -> String {
    if value >= 1_000_000 {
        format!("{:.1}M", value as f64 / 1_000_000.0)
    } else if value >= 1_000 {
        format!("{:.1}K", value as f64 / 1_000.0)
    } else {
        value.to_string()
    }
}

fn compact_duration(duration: Duration) -> String {
    if duration < Duration::from_secs(10) {
        let tenths = duration.as_millis() / 100;
        format!("{}.{:01}s", tenths / 10, tenths % 10)
    } else if duration < Duration::from_secs(60) {
        format!("{}s", duration.as_secs())
    } else {
        format!("{}m {}s", duration.as_secs() / 60, duration.as_secs() % 60)
    }
}

fn animation_frame(elapsed: Duration) -> char {
    const FRAMES: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
    FRAMES[(elapsed.as_millis() as usize / 80) % FRAMES.len()]
}

fn mcp_headers(headers: HashMap<String, String>) -> Result<HashMap<HeaderName, HeaderValue>> {
    headers
        .into_iter()
        .map(|(name, value)| {
            Ok((
                name.parse().context("invalid MCP header name")?,
                expand(&value)?
                    .parse()
                    .context("invalid MCP header value")?,
            ))
        })
        .collect()
}

fn expand(value: &str) -> Result<String> {
    shellexpand::full(value)
        .map(|value| value.into_owned())
        .map_err(anyhow::Error::from)
}

fn load_agent() -> Result<Option<AgentFile>> {
    let path = match env_path("THX_AGENT_FILE")? {
        Some(path) => path,
        None => match env_value("THX_AGENT")? {
            Some(name) => {
                let name = nonempty("THX_AGENT", name)?;
                format!(".agents/agents/{name}.md")
            }
            None => return Ok(None),
        },
    };
    let text = fs::read_to_string(&path).with_context(|| format!("failed to read {path}"))?;
    parse_agent(&text)
        .with_context(|| format!("invalid agent file {path}"))
        .map(Some)
}

fn parse_agent(text: &str) -> Result<AgentFile> {
    let mut offset = 0;
    let mut frontmatter_start = None;
    let mut frontmatter_end = None;
    let mut prompt_start = None;
    for (index, line) in text.split_inclusive('\n').enumerate() {
        let marker = line.trim_end_matches(['\r', '\n']);
        if index == 0 {
            if marker != "---" {
                bail!("agent file must start with YAML frontmatter");
            }
            frontmatter_start = Some(line.len());
        } else if marker == "---" {
            frontmatter_end = Some(offset);
            prompt_start = Some(offset + line.len());
            break;
        }
        offset += line.len();
    }
    let start = frontmatter_start.context("missing YAML frontmatter")?;
    let end = frontmatter_end.context("unterminated YAML frontmatter")?;
    let mut agent: AgentFile =
        serde_yaml_ng::from_str(&text[start..end]).context("invalid YAML frontmatter")?;
    agent.prompt = text[prompt_start.unwrap_or(text.len())..].trim().to_owned();
    if agent.prompt.is_empty() {
        bail!("agent system prompt must not be empty");
    }
    Ok(agent)
}

fn validate_mcp_url(url: &str) -> Result<()> {
    let url = Url::parse(url).context("invalid MCP URL")?;
    if !url.username().is_empty() || url.password().is_some() {
        bail!("MCP URL must not contain embedded credentials; use headers instead");
    }
    let local = match url.host() {
        Some(url::Host::Domain(d)) => d == "localhost",
        Some(url::Host::Ipv4(ip)) => ip.is_loopback(),
        Some(url::Host::Ipv6(ip)) => ip.is_loopback(),
        None => false,
    };
    if url.scheme() != "https" && !(url.scheme() == "http" && local) {
        bail!("MCP HTTP must use HTTPS except on localhost");
    }
    Ok(())
}

fn model_tool_name(index: usize, server: &str, tool: &str) -> String {
    let safe = |s: &str| {
        s.chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || "_-".contains(c) {
                    c
                } else {
                    '_'
                }
            })
            .collect::<String>()
    };
    let mut name = format!("mcp_{index}_{}_{}", safe(server), safe(tool));
    name.truncate(64);
    name
}
