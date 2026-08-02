use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use agent_core::{CheckCodeTool, EditFileTool};
use agent_types::{AgentError, Tool, ToolCtx};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

fn temp_workspace(tag: &str) -> PathBuf {
    let mut root = std::env::temp_dir();
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    root.push(format!(
        "agent_core_task13_3_{tag}_{}_{unique}",
        std::process::id()
    ));
    std::fs::create_dir_all(&root).unwrap();
    root
}

fn tool_ctx(root: &Path) -> ToolCtx {
    ToolCtx {
        project_root: root.to_path_buf(),
        cancel: CancellationToken::new(),
        approval_provider: None,
    }
}

fn tool_error(result: agent_types::Result<String>) -> String {
    match result {
        Err(AgentError::Tool { name, reason }) => {
            assert!(name == "check_code" || name == "edit_file");
            reason
        }
        other => panic!("expected tool validation error, got {other:?}"),
    }
}

#[test]
fn check_code_schema_exposes_only_closed_checker_enum() {
    // **Validates: Requirements 2.29**
    let schema = CheckCodeTool.schema().input_schema;
    assert_eq!(schema["type"], "object");
    assert_eq!(schema["additionalProperties"], false);
    assert!(schema["properties"].get("command").is_none());
    assert_eq!(
        schema["properties"]["checker"]["enum"],
        json!(["auto", "rust", "typescript", "python", "node_build"])
    );
}

#[tokio::test]
async fn edit_file_rejects_malformed_new_str_without_touching_file() {
    // **Validates: Requirements 2.34, 3.8**
    let root = temp_workspace("malformed_edit");
    let path = root.join("edit.txt");
    let original = "keep DELETE keep";
    std::fs::write(&path, original).unwrap();
    let ctx = tool_ctx(&root);

    let malformed = [
        None,
        Some(Value::Null),
        Some(json!(0)),
        Some(json!(false)),
        Some(json!({})),
    ];
    for new_str in malformed {
        std::fs::write(&path, original).unwrap();
        let mut input = json!({"path": "edit.txt", "old_str": "DELETE"});
        if let Some(value) = new_str {
            input["new_str"] = value;
        }
        let reason = tool_error(EditFileTool.invoke(input, &ctx).await);
        assert!(reason.contains("missing required string field 'new_str'"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), original);
    }

    std::fs::remove_dir_all(root).ok();
}
#[tokio::test]
async fn edit_file_preserves_explicit_empty_string_deletion() {
    // **Validates: Requirements 2.34, 3.8**
    let root = temp_workspace("empty_delete");
    let path = root.join("edit.txt");
    std::fs::write(&path, "keep DELETE keep").unwrap();

    let output = EditFileTool
        .invoke(
            json!({"path": "edit.txt", "old_str": "DELETE", "new_str": ""}),
            &tool_ctx(&root),
        )
        .await
        .unwrap();

    assert!(output.contains("Edited edit.txt"));
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "keep  keep");
    std::fs::remove_dir_all(root).ok();
}

#[tokio::test]
async fn check_code_rejects_legacy_command_without_touching_canary() {
    // **Validates: Requirements 2.29**
    let root = temp_workspace("legacy_checker");
    let canary = root.join("checker-canary.txt");
    std::fs::write(&canary, "must remain").unwrap();
    #[cfg(windows)]
    let command = "del /Q checker-canary.txt";
    #[cfg(not(windows))]
    let command = "rm -f checker-canary.txt";

    let reason = tool_error(
        CheckCodeTool
            .invoke(json!({"command": command}), &tool_ctx(&root))
            .await,
    );

    assert!(reason.contains("legacy field 'command' is no longer supported"));
    assert!(reason.contains("use 'checker'"));
    assert!(reason.contains("'bash' tool"));
    assert_eq!(std::fs::read_to_string(&canary).unwrap(), "must remain");
    std::fs::remove_dir_all(root).ok();
}

#[tokio::test]
async fn check_code_rejects_malformed_or_unknown_selector_before_execution() {
    // **Validates: Requirements 2.29**
    let root = temp_workspace("malformed_checker");
    let ctx = tool_ctx(&root);

    for input in [
        json!({"checker": null}),
        json!({"checker": "shell"}),
        json!({"arguments": ["--help"]}),
        json!([]),
    ] {
        let reason = tool_error(CheckCodeTool.invoke(input, &ctx).await);
        assert!(
            reason.contains("must be a JSON string")
                || reason.contains("unknown checker")
                || reason.contains("unsupported field")
                || reason.contains("must be a JSON object")
        );
    }

    assert!(std::fs::read_dir(&root).unwrap().next().is_none());
    std::fs::remove_dir_all(root).ok();
}

#[tokio::test]
async fn allowed_rust_checker_and_native_auto_detection_still_run() {
    // **Validates: Requirements 2.29, 3.9**
    let root = temp_workspace("allowed_rust");
    std::fs::create_dir(root.join("src")).unwrap();
    std::fs::write(
        root.join("Cargo.toml"),
        "[package]\nname='task13_3_checker_fixture'\nversion='0.0.0'\nedition='2021'\n",
    )
    .unwrap();
    std::fs::write(root.join("src/lib.rs"), "pub fn valid() -> bool { true }\n").unwrap();
    let ctx = tool_ctx(&root);

    let explicit = CheckCodeTool
        .invoke(json!({"checker": "rust"}), &ctx)
        .await
        .unwrap();
    assert!(explicit.contains("check command: cargo check --message-format short"));
    assert!(explicit.contains("exit_code: 0"));
    assert!(explicit.contains("No errors — check passed."));

    let detected = CheckCodeTool.invoke(json!({}), &ctx).await.unwrap();
    assert!(detected.contains("check command: cargo check --message-format short"));
    assert!(detected.contains("exit_code: 0"));
    assert!(detected.contains("No errors — check passed."));

    std::fs::remove_dir_all(root).ok();
}
