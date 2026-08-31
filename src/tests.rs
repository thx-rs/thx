use super::*;

use std::{
    collections::VecDeque,
    ffi::OsString,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering as AtomicOrdering},
    },
};

use ratatui::{Terminal, backend::TestBackend, buffer::Buffer, layout::Rect, text::Text};
use rig::message::{AssistantContent, ToolCall, ToolFunction};
use rig::test_utils::{MockCompletionModel, MockStreamEvent};
use rmcp::model::NumberOrString;
use serde_json::json;
use serial_test::serial;
use tempfile::tempdir;

static ENV_LOCK: Mutex<()> = Mutex::new(());

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

struct EnvGuard(Vec<(&'static str, Option<OsString>)>);

impl EnvGuard {
    fn new(vars: &[(&'static str, Option<&str>)]) -> Self {
        let old = vars
            .iter()
            .map(|(name, _)| (*name, env::var_os(name)))
            .collect::<Vec<_>>();
        for (name, value) in vars {
            unsafe {
                match value {
                    Some(value) => env::set_var(name, value),
                    None => env::remove_var(name),
                }
            }
        }
        Self(old)
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (name, value) in self.0.drain(..) {
            unsafe {
                match value {
                    Some(value) => env::set_var(name, value),
                    None => env::remove_var(name),
                }
            }
        }
    }
}

fn test_settings() -> Settings {
    Settings {
        api_key: "test-key".into(),
        base_url: "https://example.com/v1".into(),
        model: "test-model".into(),
        model_context_window: None,
        system: None,
        agent_name: None,
        agent_description: None,
        additional_params: json!({}),
        mcp_config: Some("missing-mcp.json".into()),
    }
}

fn terminal() -> Terminal<TestBackend> {
    Terminal::new(TestBackend::new(100, 30)).expect("test terminal")
}

fn plain_text(text: &Text<'_>) -> String {
    text.lines
        .iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<Vec<_>>()
                .join("")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn tool(id: ToolId, action: &str, state: ToolState) -> ToolView {
    ToolView {
        id,
        started: Some(Instant::now()),
        label: format!("server/{action}"),
        detail: String::new(),
        status: None,
        args: json!({}),
        output: None,
        output_open: false,
        output_preview: None,
        open: false,
        state,
    }
}

// ---------- Input ----------

#[test]
fn input_insert_preserves_multiline_paste() {
    let mut ui = Ui::new("test".into(), None);
    ui.insert_input("first\r\nsecond\rthird");
    assert_eq!(ui.input.value(), "first\nsecond\nthird");
}

#[test]
fn input_insert_respects_cursor_position() {
    let mut ui = Ui::new("test".into(), None);
    ui.insert_input("ac");
    ui.input.move_left();
    ui.insert_input("b");
    assert_eq!(ui.input.value(), "abc");
}

#[tokio::test]
async fn shift_enter_inserts_newline_and_enter_submits() {
    let model = MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::text("answer"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let agent = AgentBuilder::new(model).build();
    let settings = test_settings();
    let mcp = McpHost::default();
    let (_tx, rx) = mpsc::channel();
    let mut terminal = terminal();
    let mut events = VecDeque::from([
        Event::Paste("first".into()),
        Event::Key(event::KeyEvent::new(KeyCode::Enter, KeyModifiers::SHIFT)),
        Event::Paste("second".into()),
        Event::Key(event::KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
        Event::Key(event::KeyEvent::new(
            KeyCode::Char('c'),
            KeyModifiers::CONTROL,
        )),
    ]);

    let session = chat(
        &mut terminal,
        &agent,
        &mcp,
        &rx,
        ChatState::new(&settings, &mcp, None),
        &mut || events.pop_front().context("missing test event"),
        &mut || Ok(None),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(matches!(
        &session.messages[0],
        SavedMessage::User { text } if text == "first\nsecond"
    ));
}

#[tokio::test]
async fn bracketed_paste_never_submits() {
    let model = MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::text("unused"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let agent = AgentBuilder::new(model).build();
    let settings = test_settings();
    let mcp = McpHost::default();
    let (_tx, rx) = mpsc::channel();
    let mut terminal = terminal();
    let ctrl_c = || {
        Event::Key(event::KeyEvent::new(
            KeyCode::Char('c'),
            KeyModifiers::CONTROL,
        ))
    };
    let mut events = VecDeque::from([Event::Paste("first\r\nsecond".into()), ctrl_c(), ctrl_c()]);

    let session = chat(
        &mut terminal,
        &agent,
        &mcp,
        &rx,
        ChatState::new(&settings, &mcp, None),
        &mut || events.pop_front().context("missing test event"),
        &mut || Ok(None),
    )
    .await
    .unwrap();
    assert!(session.is_none());
}

// ---------- Configuration ----------

#[test]
fn session_label_shows_model_and_tool_count() {
    let settings = test_settings();
    assert_eq!(settings.session_label(0), "test-model");
    assert_eq!(settings.session_label(1), "test-model · 1 tool");
    assert_eq!(settings.session_label(3), "test-model · 3 tools");
}

#[test]
#[serial]
fn settings_load_uses_defaults() {
    let _lock = env_lock();
    let _env = EnvGuard::new(&[
        ("OPENAI_API_KEY", Some("secret")),
        ("OPENAI_BASE_URL", None),
        ("OPENAI_MODEL", None),
        ("MODEL_CONTEXT_WINDOW", None),
        ("MCP_CONFIG", None),
        ("THX_AGENT", None),
        ("THX_AGENT_FILE", None),
    ]);

    let settings = Settings::load().unwrap();
    assert_eq!(settings.api_key, "secret");
    assert_eq!(settings.base_url, DEFAULT_BASE_URL);
    assert_eq!(settings.model, DEFAULT_MODEL);
    assert_eq!(settings.model_context_window, None);
    assert_eq!(settings.system, None);
    assert_eq!(settings.mcp_config, None);
    assert_eq!(settings.mcp_path(), (DEFAULT_MCP_CONFIG, true));
}

#[test]
#[serial]
fn settings_load_reads_agent_configuration() {
    let _lock = env_lock();
    let dir = tempdir().unwrap();
    let file = dir.path().join("leader.md");
    fs::write(
        &file,
        "---\nname: leader\ndescription: Leads\nbase_url: https://agent.example/v1\nmodel: agent-model\nmodel_context_window: 200000\nother_param: hello\n---\n\nAgent prompt.\n",
    )
    .unwrap();
    let _env = EnvGuard::new(&[
        ("OPENAI_API_KEY", Some("secret")),
        ("OPENAI_BASE_URL", Some("https://ignored.example/v1")),
        ("OPENAI_MODEL", Some("ignored-model")),
        ("MODEL_CONTEXT_WINDOW", Some("not-used")),
        ("MCP_CONFIG", Some("custom-mcp.json")),
        ("THX_AGENT", None),
        ("THX_AGENT_FILE", Some(file.to_str().unwrap())),
    ]);

    let settings = Settings::load().unwrap();
    assert_eq!(settings.base_url, "https://agent.example/v1");
    assert_eq!(settings.model, "agent-model");
    assert_eq!(settings.model_context_window, Some(200_000));
    assert_eq!(settings.system.as_deref(), Some("Agent prompt."));
    assert_eq!(settings.agent_name.as_deref(), Some("leader"));
    assert_eq!(settings.agent_description.as_deref(), Some("Leads"));
    assert_eq!(settings.additional_params["other_param"], "hello");
    assert_eq!(settings.mcp_path(), ("custom-mcp.json", false));
}

#[test]
fn parse_agent_preserves_unknown_model_parameters() {
    let agent = parse_agent(
        "---\nname: leader\nmodel: openai/gpt-5\nmodel_context_window: 400000\nreasoning_effort: high\ncount: 3\n---\n\nSystem prompt.\n",
    )
    .unwrap();
    assert_eq!(agent.name.as_deref(), Some("leader"));
    assert_eq!(agent.model.as_deref(), Some("openai/gpt-5"));
    assert_eq!(agent.model_context_window, Some(400_000));
    assert!(!agent.params.contains_key("model_context_window"));
    assert_eq!(agent.params["reasoning_effort"], "high");
    assert_eq!(agent.params["count"], 3);
    assert_eq!(agent.prompt, "System prompt.");
}

#[test]
fn parse_agent_requires_frontmatter_and_prompt() {
    assert!(parse_agent("Prompt only").is_err());
    assert!(parse_agent("---\nname: x\n---\n\n").is_err());
    assert!(parse_agent("---\nname: x\nPrompt").is_err());
}

#[test]
#[serial]
fn expand_uses_standard_shell_syntax() {
    let _lock = env_lock();
    let _env = EnvGuard::new(&[("THX_EXPAND", Some("value"))]);
    assert_eq!(expand("${THX_EXPAND}/x").unwrap(), "value/x");
    assert_eq!(expand("$THX_EXPAND/x").unwrap(), "value/x");
    assert!(!expand("~/x").unwrap().starts_with('~'));
}

#[test]
#[serial]
fn expand_rejects_missing_shell_variable() {
    let _lock = env_lock();
    let _env = EnvGuard::new(&[("THX_MISSING_EXPAND", None)]);
    assert!(expand("${THX_MISSING_EXPAND}").is_err());
}

#[test]
#[serial]
fn settings_load_reads_context_window_from_env() {
    let _lock = env_lock();
    let _env = EnvGuard::new(&[
        ("OPENAI_API_KEY", Some("secret")),
        ("OPENAI_BASE_URL", None),
        ("OPENAI_MODEL", None),
        ("MODEL_CONTEXT_WINDOW", Some("400000")),
        ("MCP_CONFIG", None),
        ("THX_AGENT", None),
        ("THX_AGENT_FILE", None),
    ]);

    assert_eq!(
        Settings::load().unwrap().model_context_window,
        Some(400_000)
    );
}

#[test]
#[serial]
fn settings_load_rejects_invalid_context_window() {
    let _lock = env_lock();
    let _env = EnvGuard::new(&[
        ("OPENAI_API_KEY", Some("secret")),
        ("OPENAI_BASE_URL", None),
        ("OPENAI_MODEL", None),
        ("MODEL_CONTEXT_WINDOW", Some("0")),
        ("MCP_CONFIG", None),
        ("THX_AGENT", None),
        ("THX_AGENT_FILE", None),
    ]);

    assert!(Settings::load().is_err());
}

// ---------- MCP configuration ----------

#[test]
fn validates_secure_mcp_urls() {
    assert!(validate_mcp_url("https://example.com/mcp").is_ok());
    assert!(validate_mcp_url("http://localhost:3000/mcp").is_ok());
    assert!(validate_mcp_url("http://127.0.0.1:3000/mcp").is_ok());
    assert!(validate_mcp_url("http://[::1]:3000/mcp").is_ok());
    assert!(validate_mcp_url("http://example.com/mcp").is_err());
    assert!(validate_mcp_url("ws://localhost/mcp").is_err());
}

#[test]
fn parses_stdio_and_http_mcp_servers() {
    let config: McpConfig = serde_json::from_value(json!({
        "mcpServers": {
            "stdio": {"command":"server", "args":["a"], "env":{"A":"B"}},
            "http": {"url":"https://example.com/mcp", "headers":{"X-Test":"yes"}}
        }
    }))
    .unwrap();
    assert_eq!(config.servers.len(), 2);
    assert!(matches!(&config.servers["stdio"], McpServer::Stdio(_)));
    assert!(matches!(&config.servers["http"], McpServer::Http(_)));
}

#[test]
fn unsupported_mcp_fields_are_rejected() {
    let value = json!({
        "mcpServers": {"remote": {"url":"https://example.com/mcp", "oauth":{}}}
    });
    assert!(serde_json::from_value::<McpConfig>(value).is_err());
}

#[tokio::test]
async fn missing_default_mcp_config_is_optional_but_explicit_path_is_not() {
    let host = McpHost::load("definitely-missing-default-mcp.json", true)
        .await
        .unwrap();
    assert!(host.services.is_empty());
    assert!(host.tools.is_empty());

    assert!(
        McpHost::load("definitely-missing-explicit-mcp.json", false)
            .await
            .is_err()
    );
}

#[test]
fn model_tool_names_are_safe_unique_and_bounded() {
    assert_eq!(
        model_tool_name(2, "my server", "read/file"),
        "mcp_2_my_server_read_file"
    );
    assert_ne!(
        model_tool_name(0, "server", "tool"),
        model_tool_name(1, "server", "tool")
    );
    let name = model_tool_name(999, &"s".repeat(100), &"t".repeat(100));
    assert!(name.len() <= 64);
    assert!(
        name.chars()
            .all(|c| c.is_ascii_alphanumeric() || "_-".contains(c))
    );
}

#[test]
#[serial]
fn mcp_headers_validate_and_expand_values() {
    let _lock = env_lock();
    let _env = EnvGuard::new(&[("THX_TOKEN", Some("Bearer abc"))]);
    let headers = mcp_headers(HashMap::from([(
        "authorization".into(),
        "${THX_TOKEN}".into(),
    )]))
    .unwrap();
    assert_eq!(
        headers
            .get(&HeaderName::from_static("authorization"))
            .unwrap(),
        &HeaderValue::from_static("Bearer abc")
    );
    assert!(mcp_headers(HashMap::from([("bad header".into(), "x".into())])).is_err());
}

#[test]
fn build_agent_from_settings() {
    let host = Arc::new(McpHost::default());
    let (tx, _rx) = mpsc::channel();
    assert!(build_agent(&test_settings(), &host, tx).is_ok());
}

async fn tasks_request_branch(info: ClientInfo) -> (bool, ServerResult) {
    use rmcp::{
        model::{
            ClientJsonRpcMessage, CreateTaskResult, DiscoverResult, GetMeta, ServerCapabilities,
            ServerJsonRpcMessage, Task, TaskStatus,
        },
        transport::{IntoTransport, Transport},
    };

    let (server_transport, client_transport) = tokio::io::duplex(16 * 1024);
    let server = tokio::spawn(async move {
        let mut server = IntoTransport::<rmcp::RoleServer, _, _>::into_transport(server_transport);
        let ClientJsonRpcMessage::Request(discover) = server.receive().await.unwrap() else {
            panic!("expected discover request")
        };
        server
            .send(ServerJsonRpcMessage::response(
                ServerResult::DiscoverResult(DiscoverResult::new(
                    vec![ProtocolVersion::V_2026_07_28],
                    ServerCapabilities::builder()
                        .enable_tools()
                        .enable_tasks()
                        .build(),
                )),
                discover.id,
            ))
            .await
            .unwrap();

        let ClientJsonRpcMessage::Request(call) = server.receive().await.unwrap() else {
            panic!("expected tools/call request")
        };
        let meta = call.request.get_meta();
        assert_eq!(meta.protocol_version(), Some(ProtocolVersion::V_2026_07_28));
        assert!(meta.get_progress_token().is_some());
        let opted_in = meta
            .client_capabilities()
            .is_some_and(|capabilities| capabilities.supports_tasks());
        let result = if opted_in {
            ServerResult::CreateTaskResult(CreateTaskResult::new(Task::new(
                "task-1",
                TaskStatus::Working,
                "2026-08-31T00:00:00Z",
                "2026-08-31T00:00:00Z",
            )))
        } else {
            ServerResult::CallToolResult(CallToolResult::success(vec![
                rmcp::model::ContentBlock::text("ordinary"),
            ]))
        };
        server
            .send(ServerJsonRpcMessage::response(result, call.id))
            .await
            .unwrap();
        opted_in
    });

    let client = McpClient {
        info,
        progress: ProgressRouter::default(),
    }
    .serve_with_lifecycle(
        client_transport,
        ClientLifecycleMode::Discover {
            preferred_versions: vec![ProtocolVersion::V_2026_07_28],
        },
    )
    .await
    .unwrap();
    let request = CallToolRequestParams::new("test").with_arguments(JsonObject::new());
    let response = client
        .send_cancellable_request(
            ClientRequest::CallToolRequest(CallToolRequest::new(request)),
            PeerRequestOptions::no_options(),
        )
        .await
        .unwrap()
        .await_response()
        .await
        .unwrap();
    let opted_in = server.await.unwrap();
    client.cancel().await.unwrap();
    (opted_in, response)
}

#[tokio::test]
async fn tasks_opt_in_is_present_on_each_call_and_controls_server_branch() {
    let (opted_in, response) = tasks_request_branch(mcp_client_info()).await;
    assert!(opted_in);
    assert!(matches!(response, ServerResult::CreateTaskResult(_)));

    let (opted_in, response) = tasks_request_branch(ClientInfo::default()).await;
    assert!(!opted_in);
    assert!(matches!(response, ServerResult::CallToolResult(_)));
}

#[derive(Clone)]
struct CompletedTaskServer {
    is_error: bool,
}

impl rmcp::ServerHandler for CompletedTaskServer {
    fn get_info(&self) -> rmcp::model::ServerInfo {
        rmcp::model::ServerInfo::new(
            rmcp::model::ServerCapabilities::builder()
                .enable_tools()
                .enable_tasks()
                .build(),
        )
    }

    async fn call_tool(
        &self,
        _request: CallToolRequestParams,
        _context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<rmcp::model::CallToolResponse, rmcp::ErrorData> {
        let task = rmcp::model::Task::new(
            "task-1",
            rmcp::model::TaskStatus::Working,
            "2026-08-31T00:00:00Z",
            "2026-08-31T00:00:00Z",
        )
        .with_poll_interval_ms(1);
        Ok(rmcp::model::CallToolResponse::Task(
            rmcp::model::CreateTaskResult::new(task),
        ))
    }

    async fn get_task(
        &self,
        _request: GetTaskParams,
        _context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<rmcp::model::GetTaskResult, rmcp::ErrorData> {
        let result = if self.is_error {
            CallToolResult::error(vec![rmcp::model::ContentBlock::text("task failed")])
        } else {
            CallToolResult::success(vec![rmcp::model::ContentBlock::text("task complete")])
        };
        let result = serde_json::to_value(result)
            .unwrap()
            .as_object()
            .unwrap()
            .clone();
        let task = rmcp::model::Task::new(
            "task-1",
            rmcp::model::TaskStatus::Completed,
            "2026-08-31T00:00:00Z",
            "2026-08-31T00:00:01Z",
        );
        Ok(rmcp::model::GetTaskResult::new(
            rmcp::model::DetailedTask::new(task, TaskPayload::Completed { result }),
        ))
    }
}

async fn completed_task_result(is_error: bool) -> McpToolResult {
    use rmcp::ServiceExt;

    let router = ProgressRouter::default();
    let (server_transport, client_transport) = tokio::io::duplex(16 * 1024);
    let server = tokio::spawn(async move {
        let service = CompletedTaskServer { is_error }
            .serve(server_transport)
            .await
            .unwrap();
        service.waiting().await.unwrap();
    });
    let client = McpClient {
        info: mcp_client_info(),
        progress: router.clone(),
    }
    .serve_with_lifecycle(
        client_transport,
        ClientLifecycleMode::Discover {
            preferred_versions: vec![ProtocolVersion::V_2026_07_28],
        },
    )
    .await
    .unwrap();
    let host = McpHost {
        services: vec![McpConnection {
            name: "tasks".into(),
            service: client,
            protocol: ProtocolVersion::V_2026_07_28,
            tasks: true,
        }],
        progress: router,
        ..McpHost::default()
    };
    let (tx, _rx) = mpsc::channel();
    let result = host.call(9, 0, "task", json!({}), None, &tx).await.unwrap();
    host.shutdown().await.unwrap();
    server.await.unwrap();
    result
}

#[tokio::test]
async fn completed_task_preserves_normal_and_tool_error_results() {
    let success = completed_task_result(false).await;
    assert!(!success.is_error);
    assert_eq!(success.model.as_text(), Some("task complete"));

    let failure = completed_task_result(true).await;
    assert!(failure.is_error);
    assert_eq!(failure.model.as_text(), Some("task failed"));
}

#[tokio::test]
async fn actual_request_progress_token_routes_server_notifications() {
    use rmcp::{
        model::{
            ClientJsonRpcMessage, DiscoverResult, GetMeta, ServerCapabilities,
            ServerJsonRpcMessage, ServerNotification,
        },
        transport::{IntoTransport, Transport},
    };

    let router = ProgressRouter::default();
    let (server_transport, client_transport) = tokio::io::duplex(16 * 1024);
    let server = tokio::spawn(async move {
        let mut transport =
            IntoTransport::<rmcp::RoleServer, _, _>::into_transport(server_transport);
        let ClientJsonRpcMessage::Request(discover) = transport.receive().await.unwrap() else {
            panic!("expected discover request")
        };
        transport
            .send(ServerJsonRpcMessage::response(
                ServerResult::DiscoverResult(DiscoverResult::new(
                    vec![ProtocolVersion::V_2026_07_28],
                    ServerCapabilities::builder().enable_tools().build(),
                )),
                discover.id,
            ))
            .await
            .unwrap();

        let ClientJsonRpcMessage::Request(call) = transport.receive().await.unwrap() else {
            panic!("expected tools/call request")
        };
        let token = call
            .request
            .get_meta()
            .get_progress_token()
            .expect("request progress token");
        for message in ["phase one", "phase two"] {
            transport
                .send(ServerJsonRpcMessage::notification(
                    ServerNotification::ProgressNotification(
                        rmcp::model::ProgressNotification::new(
                            ProgressNotificationParam::new(token.clone(), 1.0)
                                .with_message(message),
                        ),
                    ),
                ))
                .await
                .unwrap();
        }
        transport
            .send(ServerJsonRpcMessage::response(
                ServerResult::CallToolResult(CallToolResult::success(vec![
                    rmcp::model::ContentBlock::text("complete"),
                ])),
                call.id,
            ))
            .await
            .unwrap();
    });

    let client = McpClient {
        info: mcp_client_info(),
        progress: router.clone(),
    }
    .serve_with_lifecycle(
        client_transport,
        ClientLifecycleMode::Discover {
            preferred_versions: vec![ProtocolVersion::V_2026_07_28],
        },
    )
    .await
    .unwrap();
    let host = McpHost {
        services: vec![McpConnection {
            name: "test".into(),
            service: client,
            protocol: ProtocolVersion::V_2026_07_28,
            tasks: false,
        }],
        progress: router,
        ..McpHost::default()
    };
    let (tx, rx) = mpsc::channel();
    let result = host.call(7, 0, "test", json!({}), None, &tx).await.unwrap();
    assert_eq!(result.model.as_text(), Some("complete"));
    let progress = rx
        .try_iter()
        .filter_map(|event| match event {
            ToolEvent::Progress { id, status } => Some((id, status)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        progress,
        vec![(7, Some("phase one".into())), (7, Some("phase two".into()))]
    );
    server.await.unwrap();
    host.shutdown().await.unwrap();
}

// ---------- Parallel tool state ----------

fn tool_group_lines(tools: &[ToolView]) -> Vec<Line<'static>> {
    let mut index = 0;
    tool_group_content(tools, &mut index)
        .into_iter()
        .map(|(line, _)| line)
        .collect()
}

#[test]
fn tool_events_are_correlated_by_id_and_can_finish_out_of_order() {
    let mut ui = Ui::new("test-model".into(), None);
    let (tx, rx) = mpsc::channel();
    let started = Instant::now();

    tx.send(ToolEvent::Start {
        id: 10,
        label: "server/first".into(),
        args: json!({"path":"/a"}),
        started,
    })
    .unwrap();
    tx.send(ToolEvent::Start {
        id: 20,
        label: "server/second".into(),
        args: json!({"path":"/b"}),
        started,
    })
    .unwrap();
    tx.send(ToolEvent::Status {
        id: 20,
        status: Some("Building 3/10".into()),
    })
    .unwrap();
    tx.send(ToolEvent::Finish {
        id: 10,
        error: None,
        output: Some(json!({"ok":true})),
        duration: Duration::from_millis(40),
        finished: Instant::now(),
    })
    .unwrap();

    let mut message = None;
    drain_tool_events(&mut ui, &rx, &mut message);

    let Message::Tools(tools) = &ui.messages[0] else {
        panic!("expected tool group");
    };
    let first = tools.iter().find(|tool| tool.id == 10).unwrap();
    let second = tools.iter().find(|tool| tool.id == 20).unwrap();
    assert!(matches!(&first.state, ToolState::Done(_)));
    assert!(matches!(&second.state, ToolState::Pending));
    assert_eq!(second.status.as_deref(), Some("Building 3/10"));
    assert!(ui.has_pending_tools());
}

#[test]
fn finishing_one_parallel_tool_does_not_finish_another() {
    let mut ui = Ui::new("test-model".into(), None);
    ui.push(Message::Tools(vec![
        tool(1, "a", ToolState::Pending),
        tool(2, "b", ToolState::Pending),
    ]));
    ui.finish_tool(2, None, Some(json!(2)), Duration::from_millis(10));

    let Message::Tools(tools) = &ui.messages[0] else {
        unreachable!()
    };
    assert!(matches!(&tools[0].state, ToolState::Pending));
    assert!(matches!(&tools[1].state, ToolState::Done(_)));
}

#[test]
fn cancelling_marks_every_pending_tool() {
    let mut ui = Ui::new("test-model".into(), None);
    ui.push(Message::Tools(vec![
        tool(1, "a", ToolState::Pending),
        tool(2, "b", ToolState::Done(Duration::from_millis(1))),
        tool(3, "c", ToolState::Pending),
    ]));
    ui.fail_pending_tools("cancelled");

    let Message::Tools(tools) = &ui.messages[0] else {
        unreachable!()
    };
    assert!(matches!(&tools[0].state, ToolState::Failed(error, _) if error == "cancelled"));
    assert!(matches!(&tools[1].state, ToolState::Done(_)));
    assert!(matches!(&tools[2].state, ToolState::Failed(error, _) if error == "cancelled"));
    assert!(!ui.has_pending_tools());
}

#[test]
fn parallel_tool_group_uses_wall_clock_max_not_sum() {
    let tools = vec![
        tool(1, "a", ToolState::Done(Duration::from_secs(2))),
        tool(2, "b", ToolState::Done(Duration::from_secs(3))),
    ];
    let rendered = plain_text(&Text::from(tool_group_lines(&tools)));
    assert!(rendered.starts_with("● 2 tool calls · 3.0s"), "{rendered}");
    assert!(!rendered.contains("5.0s"), "{rendered}");
}

#[test]
fn task_status_is_visible_on_its_tool() {
    let mut tool = tool(1, "build", ToolState::Pending);
    tool.status = Some("Compiling 47/83".into());
    let rendered = plain_text(&Text::from(tool_group_lines(&[tool])));
    assert!(rendered.contains("Compiling 47/83"), "{rendered}");
}

#[test]
fn tool_output_is_nested_and_bounded() {
    let mut tool = tool(1, "fetch", ToolState::Done(Duration::from_millis(7)));
    tool.args = json!({"url":"https://example.com"});
    tool.output = Some(json!({"status":200,"body":"ok"}));
    tool.open = true;

    let collapsed = plain_text(&Text::from(tool_group_lines(&[tool.clone()])));
    assert!(collapsed.contains("▸ Output"));
    assert!(!collapsed.contains("\"status\""));

    tool.output_preview = tool.output.as_ref().map(tool_output_preview);
    tool.output_open = true;
    let expanded = plain_text(&Text::from(tool_group_lines(&[tool])));
    assert!(expanded.contains("\"status\": 200"));
}

#[test]
fn tool_detail_prefers_paths_and_urls() {
    assert_eq!(
        tool_detail(&json!({"max_length":1000,"url":"https://example.com"})),
        "url https://example.com"
    );
    assert_eq!(tool_detail(&json!({"paths":["/a","/b"]})), "paths /a, /b");
}

#[test]
fn tool_result_projection_deduplicates_structured_compatibility_text() {
    let result = CallToolResult::structured(json!({"answer":42}));
    let projected = project_tool_result(&result).unwrap();
    assert_eq!(projected.model.as_json(), Some(&json!({"answer":42})));
    assert_eq!(projected.ui, json!({"answer":42}));
}

#[test]
fn tool_result_projection_preserves_additional_text() {
    let mut result = CallToolResult::structured(json!({"answer":42}));
    result
        .content
        .push(rmcp::model::ContentBlock::text("Additional note"));
    let projected = project_tool_result(&result).unwrap();
    let value = projected.model.as_json().unwrap();
    assert_eq!(value["structuredContent"], json!({"answer":42}));
    assert_eq!(value["content"][0]["text"], "Additional note");
}

#[test]
fn tool_result_projection_preserves_error_state_and_small_text() {
    let result = CallToolResult::error(vec![rmcp::model::ContentBlock::text("failed")]);
    let projected = project_tool_result(&result).unwrap();
    assert!(projected.is_error);
    assert_eq!(projected.model.as_text(), Some("failed"));
    assert!(mcp_tool_error_message(&projected.model).contains("failed"));
}

#[test]
fn large_text_projection_keeps_head_and_tail() {
    let log = format!("BEGIN\n{}\nFINAL ERROR", "x".repeat(50 * 1024));
    let result = CallToolResult::success(vec![rmcp::model::ContentBlock::text(log)]);
    let projected = project_tool_result(&result).unwrap();
    let model = projected.model.as_text().unwrap();
    assert!(model.contains("BEGIN"));
    assert!(model.contains("FINAL ERROR"));
    assert!(model.contains("bytes omitted"));
    assert!(model.len() <= MAX_MODEL_TOOL_RESULT_BYTES);
}

#[test]
fn structured_projection_remains_valid_bounded_json() {
    let result = CallToolResult::structured(json!({"log": "🙂".repeat(20_000)}));
    let projected = project_tool_result(&result).unwrap();
    let value = projected.model.as_json().unwrap();
    assert!(serde_json::to_vec(value).unwrap().len() <= MAX_MODEL_TOOL_RESULT_BYTES);
    assert!(value["log"].as_str().unwrap().contains("bytes omitted"));

    let preview = tool_output_preview(&json!({"text":"🙂".repeat(MAX_TOOL_OUTPUT_PREVIEW_BYTES)}));
    assert!(preview.contains("output preview truncated"));
}

#[test]
fn structured_projections_apply_one_global_budget_with_array_sampling() {
    let rows = (0..1_000)
        .map(|index| {
            json!({
                "index": index,
                "payload": format!("HEAD-{index}|{}|TAIL-{index}", "🙂".repeat(400)),
            })
        })
        .collect::<Vec<_>>();
    let result = CallToolResult::structured(json!({
        "requestId": "stress-123",
        "status": "complete",
        "rows": rows,
    }));
    let projected = project_tool_result(&result).unwrap();
    let projections = [
        (
            projected.model.as_json().unwrap().clone(),
            MAX_MODEL_TOOL_RESULT_BYTES,
        ),
        (projected.ui, MAX_UI_TOOL_RESULT_BYTES),
        (
            limit_json(
                result.structured_content.unwrap(),
                MAX_HISTORY_TOOL_RESULT_BYTES,
            ),
            MAX_HISTORY_TOOL_RESULT_BYTES,
        ),
    ];

    for (projection, budget) in projections {
        let encoded = serde_json::to_vec(&projection).unwrap();
        assert!(encoded.len() <= budget, "{} > {budget}", encoded.len());
        let decoded: Value = serde_json::from_slice(&encoded).unwrap();
        let rendered = decoded.to_string();
        assert!(rendered.contains("stress-123"), "{rendered}");
        assert!(rendered.contains("HEAD-0"), "{rendered}");
        assert!(rendered.contains("TAIL-999"), "{rendered}");
        assert!(rendered.matches("__thx_omitted_items").count() <= 1);
        assert!(rendered.matches("bytes omitted").count() <= 8);
    }
}

#[test]
fn array_projection_does_not_grow_with_omission_markers() {
    let large_item = "x".repeat(1_000);
    for count in [100, 10_000] {
        let projection = limit_json(json!(vec![&large_item; count]), 512);
        let encoded = serde_json::to_vec(&projection).unwrap();
        assert!(encoded.len() <= 512, "{} > 512", encoded.len());
        assert!(serde_json::from_slice::<Value>(&encoded).is_ok());
        assert_eq!(
            projection
                .to_string()
                .matches("__thx_omitted_items")
                .count(),
            1
        );
    }
}

#[test]
fn newest_tool_result_batch_stays_current_then_becomes_historical() {
    let large = "start".to_owned() + &"x".repeat(20_000) + "end";
    let RigMessage::User { content: first } =
        RigMessage::tool_result("latest-a", "read", large.clone())
    else {
        unreachable!()
    };
    let RigMessage::User { content: second } =
        RigMessage::tool_result("latest-b", "read", large.clone())
    else {
        unreachable!()
    };
    let latest_batch = RigMessage::User {
        content: first.into_iter().chain(second).collect(),
    };
    let history = vec![
        RigMessage::tool_result("old", "read", large.clone()),
        RigMessage::assistant("between"),
        latest_batch.clone(),
    ];
    let compacted = compact_tool_history(&history, true);
    let RigMessage::User { content: old } = &compacted[0] else {
        panic!("expected old tool result")
    };
    let UserContent::ToolResult(old) = &old[0] else {
        panic!("expected tool result content")
    };
    assert!(old.content[0].as_text().unwrap().len() <= MAX_HISTORY_TOOL_RESULT_BYTES);
    assert!(old.content[0].as_text().unwrap().ends_with("end"));
    assert_eq!(compacted[2], latest_batch);

    let historical = compact_tool_history(&history, false);
    let RigMessage::User { content } = &historical[2] else {
        panic!("expected historical parallel batch")
    };
    assert_eq!(content.len(), 2);
    assert!(content.iter().all(|content| {
        let UserContent::ToolResult(result) = content else {
            return false;
        };
        let text = result.content[0].as_text().unwrap();
        text.len() <= MAX_HISTORY_TOOL_RESULT_BYTES && text.ends_with("end")
    }));
}

#[test]
fn small_historical_tool_results_remain_unchanged() {
    let history = vec![RigMessage::tool_result(
        "small",
        "lookup",
        "useful exact value",
    )];
    assert_eq!(compact_tool_history(&history, false), history);
}

#[test]
fn output_schema_validates_structured_content() {
    let schema = json!({
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
        "additionalProperties": false
    });
    let valid = CallToolResult::structured(json!({"answer":42}));
    assert!(validate_structured_output(&valid, Some(&schema)).is_ok());

    let invalid = CallToolResult::structured(json!({"answer":"forty-two"}));
    let error = validate_structured_output(&invalid, Some(&schema))
        .unwrap_err()
        .to_string();
    assert_eq!(
        error,
        "MCP server returned structuredContent that does not match its declared outputSchema"
    );
    assert!(error.len() < 128);
}

#[test]
fn output_schema_requires_structured_content_only_for_success() {
    let schema = json!({"type":"string"});
    let missing = CallToolResult::success(vec![rmcp::model::ContentBlock::text("plain")]);
    assert!(
        validate_structured_output(&missing, None).is_ok(),
        "tools without outputSchema may return unstructured output"
    );
    assert_eq!(
        validate_structured_output(&missing, Some(&schema))
            .unwrap_err()
            .to_string(),
        "MCP server omitted structuredContent required by the tool's declared outputSchema"
    );

    let error = CallToolResult::error(vec![rmcp::model::ContentBlock::text("failed")]);
    assert!(validate_structured_output(&error, Some(&schema)).is_ok());
}

#[test]
fn output_schema_supports_2020_12_root_types_and_composition() {
    for (schema, value) in [
        (
            json!({"type":"array","items":{"type":"integer"}}),
            json!([1, 2]),
        ),
        (json!({"type":"string","minLength":2}), json!("ok")),
        (json!({"type":"number","minimum":1}), json!(1.5)),
        (json!({"type":"boolean"}), json!(true)),
        (json!({"type":"null"}), Value::Null),
        (
            json!({"oneOf":[{"type":"string"},{"type":"integer"}]}),
            json!(7),
        ),
    ] {
        let result = CallToolResult::structured(value);
        assert!(validate_structured_output(&result, Some(&schema)).is_ok());
    }
}

#[test]
fn output_schema_resolves_local_defs_but_rejects_external_refs() {
    let local = json!({
        "$defs":{"item":{"type":"string","pattern":"^[a-z]+$"}},
        "type":"array",
        "items":{"$ref":"#/$defs/item"}
    });
    assert!(
        validate_structured_output(
            &CallToolResult::structured(json!(["alpha", "beta"])),
            Some(&local)
        )
        .is_ok()
    );

    for reference in [
        "https://127.0.0.1:9/schema.json",
        "file:///definitely-not-readable-by-thx/schema.json",
    ] {
        let schema = json!({"$ref":reference});
        assert!(
            validate_structured_output(&CallToolResult::structured(json!({})), Some(&schema))
                .is_err()
        );
    }
}

#[test]
fn output_schema_rejects_invalid_and_handles_deep_recursive_schemas() {
    let invalid = json!({"type":42});
    assert_eq!(
        validate_structured_output(&CallToolResult::structured(json!(42)), Some(&invalid))
            .unwrap_err()
            .to_string(),
        "MCP tool declared an invalid outputSchema"
    );

    let recursive = json!({
        "$defs":{"node":{"oneOf":[
            {"type":"null"},
            {"type":"object","properties":{"next":{"$ref":"#/$defs/node"}},"required":["next"]}
        ]}},
        "$ref":"#/$defs/node"
    });
    let mut value = Value::Null;
    for _ in 0..64 {
        value = json!({"next":value});
    }
    assert!(
        validate_structured_output(&CallToolResult::structured(value), Some(&recursive)).is_ok()
    );
}

#[test]
fn historical_object_compaction_keeps_metadata_and_a_significant_scalar() {
    let large = format!(
        "BEGIN-UNIQUE|{}MIDDLE-UNIQUE{}|END-UNIQUE",
        "x".repeat(4_000),
        "y".repeat(4_000)
    );
    let value = json!({
        "a_compact_fact":"useful",
        "b_count":17,
        "z_arbitrary_payload_name":large,
    });
    let compact = limit_json(value, MAX_HISTORY_TOOL_RESULT_BYTES);
    let encoded = serde_json::to_vec(&compact).unwrap();
    let rendered = compact.to_string();

    assert!(encoded.len() <= MAX_HISTORY_TOOL_RESULT_BYTES);
    assert_eq!(compact["a_compact_fact"], "useful");
    assert!(rendered.contains("BEGIN-UNIQUE"), "{rendered}");
    assert!(rendered.contains("END-UNIQUE"), "{rendered}");
    assert!(rendered.contains("bytes omitted"), "{rendered}");
    assert!(!rendered.contains("MIDDLE-UNIQUE"), "{rendered}");
}

#[test]
fn non_text_projection_omits_binary_payloads_and_keeps_resource_metadata() {
    let result = CallToolResult::success(vec![
        rmcp::model::ContentBlock::image("IMAGE_BASE64_SECRET", "image/png"),
        rmcp::model::ContentBlock::audio("AUDIO_BASE64_SECRET", "audio/wav"),
        rmcp::model::ContentBlock::resource(ResourceContents::blob(
            "BLOB_BASE64_SECRET",
            "file:///artifact.bin",
        )),
        rmcp::model::ContentBlock::resource_link(
            rmcp::model::Resource::new("file:///report.txt", "report").with_mime_type("text/plain"),
        ),
    ]);
    let projected = project_tool_result(&result).unwrap();
    let model = projected.model.render();
    for secret in [
        "IMAGE_BASE64_SECRET",
        "AUDIO_BASE64_SECRET",
        "BLOB_BASE64_SECRET",
    ] {
        assert!(!model.contains(secret));
    }
    assert!(model.contains("MCP image omitted: image/png"));
    assert!(model.contains("MCP audio omitted: audio/wav"));
    assert!(model.contains("file:///artifact.bin"));
    assert!(model.contains("file:///report.txt"));
}

#[test]
fn progress_routes_only_known_tokens_to_tool_state() {
    let router = ProgressRouter::default();
    let (tx, rx) = mpsc::channel();
    let known = ProgressToken(NumberOrString::String("known".into()));
    router.register(known.clone(), 7, tx);

    router.handle(
        ProgressNotificationParam::new(ProgressToken(NumberOrString::Number(99)), 1.0)
            .with_message("ignored"),
    );
    assert!(rx.try_recv().is_err());

    router.handle(ProgressNotificationParam::new(known, 1.0).with_message("Compiling 1/2"));
    assert!(matches!(
        rx.try_recv(),
        Ok(ToolEvent::Progress { id: 7, status: Some(status) }) if status == "Compiling 1/2"
    ));
}

// ---------- Real Rig runner semantics ----------

async fn run_turn<B: Backend>(
    terminal: &mut Terminal<B>,
    ui: &mut Ui,
    agent: &Agent,
    input: &str,
    history: &mut Vec<RigMessage>,
    events: &Receiver<ToolEvent>,
) -> Result<PromptResponse>
where
    <B as Backend>::Error: Send + Sync + 'static,
{
    run_turn_with_events(terminal, ui, agent, input, history, events, &mut || {
        Ok(None)
    })
    .await
}

#[tokio::test]
async fn run_turn_updates_ui_and_history() {
    let model = MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::text("hello"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let agent = AgentBuilder::new(model).build();
    let mut terminal = terminal();
    let mut ui = Ui::new("test-model".into(), None);
    let mut history = Vec::new();
    let (_tx, rx) = mpsc::channel();

    let response = run_turn(&mut terminal, &mut ui, &agent, "hi", &mut history, &rx)
        .await
        .unwrap();
    assert_eq!(response.output(), "hello");
    assert_eq!(ui.last_assistant(), Some("hello"));
    assert!(!history.is_empty());
}

#[tokio::test]
async fn run_turn_surfaces_model_errors() {
    let model = MockCompletionModel::from_stream_turns([vec![MockStreamEvent::error("boom")]]);
    let agent = AgentBuilder::new(model).build();
    let mut terminal = terminal();
    let mut ui = Ui::new("test-model".into(), None);
    let mut history = Vec::new();
    let (_tx, rx) = mpsc::channel();

    let error = run_turn(&mut terminal, &mut ui, &agent, "hi", &mut history, &rx)
        .await
        .err()
        .unwrap();
    assert!(error.to_string().contains("boom"));
}

#[tokio::test]
async fn multiple_tool_calls_execute_concurrently() {
    let running = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let tool = {
        let running = Arc::clone(&running);
        let peak = Arc::clone(&peak);
        DynamicTool::new(
            "wait",
            "Wait briefly",
            json!({"type":"object"}),
            move |_, args| {
                let running = Arc::clone(&running);
                let peak = Arc::clone(&peak);
                Box::pin(async move {
                    let current = running.fetch_add(1, AtomicOrdering::SeqCst) + 1;
                    peak.fetch_max(current, AtomicOrdering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    running.fetch_sub(1, AtomicOrdering::SeqCst);
                    Ok::<ToolOutput, ToolExecutionError>(ToolOutput::json(args))
                })
            },
        )
    };

    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("c1", "wait", json!({"n":1})),
            MockStreamEvent::tool_call("c2", "wait", json!({"n":2})),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        vec![
            MockStreamEvent::text("done"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ]);
    let agent = AgentBuilder::new(model)
        .dynamic_tool(tool)
        .default_max_turns(8)
        .build();
    let mut terminal = terminal();
    let mut ui = Ui::new("test-model".into(), None);
    let mut history = Vec::new();
    let (_tx, rx) = mpsc::channel();

    let response = run_turn(
        &mut terminal,
        &mut ui,
        &agent,
        "use wait twice",
        &mut history,
        &rx,
    )
    .await
    .unwrap();
    assert_eq!(response.output(), "done");
    assert!(peak.load(AtomicOrdering::SeqCst) >= 2, "tools ran serially");
}

#[tokio::test]
async fn earlier_tool_results_stay_full_until_the_user_turn_finishes() {
    let exact = "EXACT-VALUE-FROM-TOOL-A";
    let large = format!("{}{}{}", "a".repeat(900), exact, "z".repeat(7_000));
    let tool_a = DynamicTool::new("tool_a", "First", json!({"type":"object"}), {
        let large = large.clone();
        move |_, _| {
            let large = large.clone();
            Box::pin(async move { Ok::<ToolOutput, ToolExecutionError>(ToolOutput::text(large)) })
        }
    });
    let tool_b = DynamicTool::new("tool_b", "Second", json!({"type":"object"}), |_, _| {
        Box::pin(async {
            Ok::<ToolOutput, ToolExecutionError>(ToolOutput::text("second operation complete"))
        })
    });
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("a", "tool_a", json!({})),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        vec![
            MockStreamEvent::tool_call("b", "tool_b", json!({})),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        vec![
            MockStreamEvent::text(exact),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        vec![
            MockStreamEvent::text("resumed"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ]);
    let requests = model.clone();
    let agent = AgentBuilder::new(model)
        .dynamic_tools(vec![tool_a, tool_b])
        .default_max_turns(8)
        .build();
    let mut terminal = terminal();
    let mut ui = Ui::new("test-model".into(), None);
    let mut history = Vec::new();
    let (_tx, rx) = mpsc::channel();

    let response = run_turn(
        &mut terminal,
        &mut ui,
        &agent,
        "Use A, then B, then return A's exact value",
        &mut history,
        &rx,
    )
    .await
    .unwrap();
    assert_eq!(response.output(), exact);
    run_turn(
        &mut terminal,
        &mut ui,
        &agent,
        "Continue the saved session",
        &mut history,
        &rx,
    )
    .await
    .unwrap();

    let requests = requests.requests();
    let encoded = requests
        .iter()
        .map(|request| serde_json::to_vec(&request.chat_history).unwrap())
        .collect::<Vec<_>>();
    let tool_a_result = |index: usize| {
        requests[index]
            .chat_history
            .iter()
            .filter_map(|message| match message {
                RigMessage::User { content } => Some(content),
                _ => None,
            })
            .flatten()
            .find_map(|content| match content {
                UserContent::ToolResult(result) if result.name == "tool_a" => {
                    Some(serde_json::to_string(&result.content).unwrap())
                }
                _ => None,
            })
            .unwrap()
    };
    assert!(tool_a_result(1).contains(exact));
    assert!(tool_a_result(2).contains(exact));
    assert!(!tool_a_result(3).contains(exact));
    assert!(encoded[3].len() < encoded[2].len());

    let json = serde_json::to_vec(&Session {
        history: history.clone(),
        messages: ui.saved_messages().unwrap(),
    })
    .unwrap();
    let resumed: Session = serde_json::from_slice(&json).unwrap();
    assert_eq!(resumed.history, history);
}

#[test]
fn finalize_stream_requires_final_response() {
    let mut ui = Ui::new("test-model".into(), None);
    let (_tx, rx) = mpsc::channel();
    let mut message = None;
    assert!(finalize_stream(None, &mut ui, &rx, &mut message).is_err());
}

// ---------- TUI ----------

#[test]
fn command_dispatch_distinguishes_prompts_and_commands() {
    let mut ui = Ui::new("test-model".into(), None);
    let mut history = Vec::new();
    let mcp = McpHost::default();

    assert_eq!(
        handle_command("hello", &mut ui, &mut history, &mcp),
        CommandAction::Prompt
    );
    assert_eq!(
        handle_command("/mcp", &mut ui, &mut history, &mcp),
        CommandAction::Handled
    );
    assert_eq!(
        handle_command("/exit", &mut ui, &mut history, &mcp),
        CommandAction::Exit
    );
}

#[test]
fn help_documents_copy_and_export() {
    assert!(COMMAND_HELP.contains("| `/copy` | copy the latest assistant response |"));
    assert!(COMMAND_HELP.contains("| `/export` | save the conversation as Markdown |"));
    assert!(COMMAND_HELP.contains("| `/exit` `/quit` | save the session and exit |"));
}

#[test]
fn assistant_inline_code_is_light_gray() {
    let lines = assistant_message_lines("Use `code` here");
    let code = lines
        .iter()
        .flat_map(|line| &line.spans)
        .find(|span| span.content == "code")
        .expect("inline code span");
    assert_eq!(code.style.fg, Some(LIGHT_GRAY));
}

#[test]
fn export_transcript_writes_markdown_without_overwriting_existing_exports() {
    let dir = tempdir().unwrap();
    let first = export_transcript_to(dir.path(), "first").unwrap();
    let second = export_transcript_to(dir.path(), "second").unwrap();

    assert_eq!(
        first.file_name().unwrap().to_str().unwrap(),
        "thx-export.md"
    );
    assert_eq!(
        second.file_name().unwrap().to_str().unwrap(),
        "thx-export-2.md"
    );
    assert_eq!(fs::read_to_string(first).unwrap(), "first");
    assert_eq!(fs::read_to_string(second).unwrap(), "second");
    assert!(fs::read_dir(dir.path()).unwrap().all(|entry| {
        entry
            .unwrap()
            .path()
            .extension()
            .and_then(|ext| ext.to_str())
            != Some("json")
    }));
}

// ---------- Sessions ----------

fn representative_session() -> Session {
    let call = ToolCall::from_wire(
        "call-1",
        ToolFunction::new("lookup".into(), json!({"query":"rust"})),
    );
    let history = vec![
        RigMessage::user("question"),
        RigMessage::Assistant {
            id: Some("response-1".into()),
            content: vec![
                AssistantContent::Reasoning(
                    Reasoning::new("considering").with_id("reason-1".into()),
                ),
                AssistantContent::ToolCall(call),
                AssistantContent::text("answer"),
            ],
        },
        RigMessage::tool_result("call-1", "lookup", "result"),
    ];
    Session {
        history,
        messages: vec![
            SavedMessage::User {
                text: "question".into(),
            },
            SavedMessage::Thinking {
                id: "reason-1".into(),
                text: Some("considering".into()),
                duration_ms: 12,
            },
            SavedMessage::Tools {
                tools: vec![SavedTool {
                    label: "server/lookup".into(),
                    args: json!({"query":"rust"}),
                    output: Some(json!({"value":"result"})),
                    error: None,
                    duration_ms: 8,
                }],
            },
            SavedMessage::Assistant {
                text: "answer".into(),
                metrics: Some("1.0s".into()),
            },
        ],
    }
}

#[test]
fn session_round_trip_preserves_rig_history_and_visible_messages() {
    let session = representative_session();
    let json = serde_json::to_vec_pretty(&session).unwrap();
    let restored: Session = serde_json::from_slice(&json).unwrap();
    assert_eq!(restored, session);
}

#[test]
fn validates_session_ids() {
    assert!(validate_session_id("20260830-051923").is_ok());
    assert!(validate_session_id("20260830-051923-2").is_ok());
    for invalid in ["", "session", "x.json", "../x", "1/2", "1\\2", "1_2"] {
        assert!(validate_session_id(invalid).is_err(), "accepted {invalid}");
    }
}

#[test]
fn resume_resolution_uses_only_the_session_directory_and_id() {
    let state = tempdir().unwrap();
    let sessions = state.path().join("sessions");
    assert_eq!(
        session_path_in(&sessions, "20260830-051923").unwrap(),
        sessions.join("20260830-051923.json")
    );
    assert!(session_path_in(&sessions, "../outside").is_err());
}

#[test]
fn autosave_collisions_advance_without_overwriting() {
    let state = tempdir().unwrap();
    let session = representative_session();
    let timestamp = "20260830-051923";
    let first = autosave_session_in(&session, state.path(), timestamp).unwrap();
    let second = autosave_session_in(&session, state.path(), timestamp).unwrap();
    let third = autosave_session_in(&session, state.path(), timestamp).unwrap();
    assert_eq!(
        (first.as_str(), second.as_str(), third.as_str()),
        (timestamp, "20260830-051923-2", "20260830-051923-3")
    );
    assert_eq!(load_session_from(state.path(), timestamp).unwrap(), session);
}

#[test]
#[cfg(unix)]
fn session_directory_and_files_are_private() {
    use std::os::unix::fs::PermissionsExt;

    let state = tempdir().unwrap();
    let sessions = state.path().join("sessions");
    let id = autosave_session_in(&representative_session(), &sessions, "20260830-051923").unwrap();
    let directory_mode = fs::metadata(&sessions).unwrap().permissions().mode() & 0o777;
    let file_mode = fs::metadata(session_path_in(&sessions, &id).unwrap())
        .unwrap()
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(directory_mode, 0o700);
    assert_eq!(file_mode, 0o600);
}

#[test]
fn exit_commands_and_empty_ctrl_c_share_exit_action() {
    let mcp = McpHost::default();
    for command in ["/exit", "/quit"] {
        let mut ui = Ui::new("test".into(), None);
        let mut history = Vec::new();
        assert_eq!(
            handle_command(command, &mut ui, &mut history, &mcp),
            CommandAction::Exit
        );
    }
    let mut ui = Ui::new("test".into(), None);
    assert_eq!(handle_ctrl_c(&mut ui), CommandAction::Exit);
}

#[test]
fn ctrl_c_with_input_clears_without_exit_or_session() {
    let mut ui = Ui::new("test".into(), None);
    ui.insert_input("draft");
    assert_eq!(handle_ctrl_c(&mut ui), CommandAction::Handled);
    assert!(ui.input.value().is_empty());
    assert!(completed_session(Vec::new(), &ui).unwrap().is_none());
}

#[test]
fn empty_conversation_produces_no_session_for_every_exit_mechanism() {
    let mcp = McpHost::default();
    for command in [Some("/exit"), Some("/quit"), None] {
        let mut ui = Ui::new("test".into(), None);
        let mut history = Vec::new();
        let action = match command {
            Some(command) => handle_command(command, &mut ui, &mut history, &mcp),
            None => handle_ctrl_c(&mut ui),
        };
        assert_eq!(action, CommandAction::Exit);
        assert!(completed_session(history, &ui).unwrap().is_none());
    }
}

#[test]
fn restored_startup_state_is_idle_and_has_no_pending_tools() {
    let session = representative_session();
    let ui = Ui::from_saved("current-model".into(), Some(200_000), session.messages);
    assert!(ui.input.value().is_empty());
    assert!(ui.activity.is_none());
    assert!(ui.thinking_since.is_none());
    assert!(!ui.has_pending_tools());
    assert!(ui.follow_tail);
    assert_eq!(ui.last_assistant(), Some("answer"));
}

#[test]
fn missing_and_malformed_sessions_return_clear_errors() {
    let state = tempdir().unwrap();
    let missing = load_session_from(state.path(), "20260830-051923")
        .unwrap_err()
        .to_string();
    assert!(missing.contains("failed to read session"), "{missing}");

    let path = state.path().join("20260830-051923.json");
    fs::write(&path, b"not json").unwrap();
    let malformed = load_session_from(state.path(), "20260830-051923")
        .unwrap_err()
        .to_string();
    assert!(
        malformed.contains("failed to deserialize session"),
        "{malformed}"
    );
}

#[test]
fn resume_cli_accepts_exactly_one_valid_id() {
    let parse = |args: &[&str]| resume_arg_from(args.iter().map(|arg| OsString::from(*arg)));
    assert_eq!(parse(&[]).unwrap(), None);
    assert_eq!(
        parse(&["--resume", "20260830-051923"]).unwrap().as_deref(),
        Some("20260830-051923")
    );
    assert!(parse(&["--resume"]).is_err());
    assert!(parse(&["--resume", "./file.json"]).is_err());
    assert!(parse(&["--resume", "1", "extra"]).is_err());
    assert!(parse(&["session-id"]).is_err());
}

#[test]
fn assistant_delta_builds_one_message() {
    let mut ui = Ui::new("test-model".into(), None);
    let mut index = None;
    ui.assistant_delta(&mut index, "hel");
    ui.assistant_delta(&mut index, "lo");
    assert_eq!(ui.last_assistant(), Some("hello"));
}

#[test]
fn transcript_contains_user_assistant_and_tools() {
    let mut ui = Ui::new("test-model".into(), None);
    ui.push(Message::User("question".into()));
    ui.assistant("answer");
    let mut view = tool(1, "read", ToolState::Done(Duration::from_millis(2)));
    view.args = json!({"path":"/tmp/a"});
    view.output = Some(json!({"text":"ok"}));
    ui.push(Message::Tools(vec![view]));

    let text = ui.transcript().unwrap();
    assert!(text.contains("## User\n\nquestion"));
    assert!(text.contains("## Assistant (test-model)\n\nanswer"));
    assert!(text.contains("**Tool: server/read**"));
    assert!(text.contains("/tmp/a"));
    assert!(text.contains("bounded UI projection, not an exact model-request trace"));
}

#[test]
fn user_messages_wrap_by_terminal_cell_width() {
    let lines = user_message_lines("hello world", 8);
    assert!(lines.len() >= 3);
    for line in lines {
        let width = line
            .spans
            .iter()
            .map(|span| UnicodeWidthStr::width(span.content.as_ref()))
            .sum::<usize>();
        assert_eq!(width, 8);
    }
}

#[test]
fn user_message_bubble_is_dark_gray_backed_with_white_text() {
    let lines = user_message_lines("hello", 12);
    assert!(lines.iter().all(|line| {
        let (indent, bubble) = line.spans.split_at(1);
        let indent = &indent[0];
        indent.style.bg.is_none()
            && bubble.iter().all(|span| {
                span.style.bg == Some(Color::Rgb(60, 60, 60)) && span.style.fg == Some(Color::White)
            })
            && UnicodeWidthStr::width(indent.content.as_ref()) > 0
    }));
}

#[test]
fn selection_is_order_independent() {
    assert_eq!(selection_bounds((5, 3), (2, 1)), ((2, 1), (5, 3)));
    assert_eq!(selection_bounds((1, 1), (2, 2)), ((1, 1), (2, 2)));
}

#[test]
fn selected_text_reads_buffer_region() {
    let area = Rect::new(0, 0, 8, 2);
    let mut buffer = Buffer::empty(area);
    buffer.set_string(0, 0, "hello   ", Style::default());
    buffer.set_string(0, 1, "world   ", Style::default());
    assert_eq!(selected_text(&buffer, (0, 0), (7, 1)), "hello\nworld");
}

#[test]
fn context_title_uses_configured_window() {
    let mut ui = Ui::new("test-model".into(), Some(400_000));
    ui.context = 100_000;
    assert_eq!(ui.context_title(), "ctx 100.0K (25%)");
}

#[test]
fn context_title_without_window_omits_percentage() {
    let mut ui = Ui::new("test-model".into(), None);
    ui.context = 100_000;
    assert_eq!(ui.context_title(), "ctx 100k");
}

#[test]
fn usage_summary_labels_aggregate_run_totals() {
    let mut usage = RigUsage::new();
    usage.input_tokens = 12_000;
    usage.output_tokens = 300;
    let summary = usage_summary(&usage, Duration::from_secs(1));
    assert!(summary.contains("input-total 12k"), "{summary}");
    assert!(summary.contains("output-total 300"), "{summary}");
}

#[test]
fn compact_formatters_are_readable() {
    assert_eq!(compact_count(999), "999");
    assert_eq!(compact_count(1_000), "1k");
    assert_eq!(compact_count(1_500_000), "1.5m");
    assert_eq!(compact_duration(Duration::from_millis(1200)), "1.2s");
    assert_eq!(compact_duration(Duration::from_secs(65)), "1m 5s");
}

// ---------- Optional end-to-end ----------

#[tokio::test]
#[ignore = "real API request; requires .env and consumes credits"]
#[serial]
async fn e2e_real_api() -> Result<()> {
    dotenvy::dotenv().ok();
    let settings = Settings::load()?;
    let host = Arc::new(McpHost::default());
    let (tx, rx) = mpsc::channel();
    let agent = build_agent(&settings, &host, tx)?;
    let mut terminal = terminal();
    let mut ui = Ui::new(settings.session_label(0), settings.model_context_window);
    let mut history = Vec::new();
    let response = tokio::time::timeout(
        Duration::from_secs(180),
        run_turn(
            &mut terminal,
            &mut ui,
            &agent,
            "Reply with exactly THX_OK and nothing else.",
            &mut history,
            &rx,
        ),
    )
    .await
    .context("real API e2e timed out")??;
    assert!(response.output().contains("THX_OK"));
    Ok(())
}
