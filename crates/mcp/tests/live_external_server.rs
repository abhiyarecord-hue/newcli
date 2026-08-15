//! Live interoperability against a real third-party MCP server.
//!
//! Every other MCP test in this workspace drives a fake peer we wrote, so it can
//! only prove that the client is self-consistent. This one talks to the official
//! `@modelcontextprotocol/server-filesystem` over real stdio JSON-RPC, which is
//! the only way to find a mismatch between our reading of the protocol and an
//! independent implementation of it.
//!
//! Opt-in by design. It is skipped unless `AGENT_LIVE_TESTS=1`, because it needs
//! network access on the first run and a Node toolchain, neither of which the
//! blocking offline gate is allowed to depend on. CI sets `AGENT_LIVE_TESTS=0`,
//! so this never runs there and its absence is recorded as `not_run` rather than
//! being quietly counted as coverage.
//!
//! Run it with:
//!   AGENT_LIVE_TESTS=1 cargo test -p mcp --test live_external_server -- --nocapture

use std::path::Path;

use mcp::McpClient;
use serde_json::json;

/// The server package is resolved through `npx`, which on Windows is a shim
/// script rather than an executable, so the extension is required there.
fn npx_command() -> &'static str {
    if cfg!(windows) {
        "npx.cmd"
    } else {
        "npx"
    }
}

fn live_enabled() -> bool {
    std::env::var("AGENT_LIVE_TESTS").ok().as_deref() == Some("1")
}

/// Forward slashes are used for the tool arguments. The filesystem server
/// normalizes them on Windows, and avoiding backslashes keeps the JSON payload
/// free of escaping that would obscure a genuine failure.
fn tool_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

#[tokio::test(flavor = "multi_thread")]
async fn real_filesystem_mcp_server_discovery_and_invocation() {
    if !live_enabled() {
        eprintln!("skipped: set AGENT_LIVE_TESTS=1 to run live MCP interoperability");
        return;
    }

    let workspace = tempfile::tempdir().expect("temp workspace");
    let note = workspace.path().join("note.txt");
    let contents = "live-mcp-interop-marker";
    std::fs::write(&note, contents).expect("seed file");

    let root = tool_path(workspace.path());
    let client = McpClient::connect(
        npx_command(),
        &["-y", "@modelcontextprotocol/server-filesystem", &root],
        "fs",
    )
    .await
    .expect("connect to the real filesystem MCP server");

    // 1. Discovery must survive an independently written server's schemas.
    let discovered = client.discovered_tools();
    assert!(
        !discovered.is_empty(),
        "a real server advertised no usable tools; schema validation may be too strict"
    );

    let remote_names: Vec<&str> = discovered
        .iter()
        .map(|tool| tool.remote_name.as_str())
        .collect();
    for expected in ["read_text_file", "list_directory", "write_file"] {
        assert!(
            remote_names.contains(&expected),
            "expected the real server to advertise {expected}, saw {remote_names:?}"
        );
    }

    // 2. The identity split is the defect class this client was hardened against:
    //    the model sees a namespaced local alias, while the wire must carry the
    //    server's exact name. Previously only a fake peer proved this.
    let read_tool = discovered
        .iter()
        .find(|tool| tool.remote_name == "read_text_file")
        .expect("read_text_file present");
    assert_ne!(
        read_tool.local_name, read_tool.remote_name,
        "the local alias must be namespaced, not identical to the remote name"
    );
    assert!(
        read_tool.local_name.contains("fs"),
        "the local alias should carry the configured server name, got {}",
        read_tool.local_name
    );

    // 3. Invocation must use the remote name and return the real file content.
    let output = client
        .call_tool(&read_tool.remote_name, json!({ "path": tool_path(&note) }))
        .await
        .expect("read_text_file via the real server");
    assert!(
        output.contains(contents),
        "tool output did not contain the seeded marker; got {output:?}"
    );

    // 4. A listing proves arguments and results round-trip for a second shape.
    let listing = client
        .call_tool("list_directory", json!({ "path": root }))
        .await
        .expect("list_directory via the real server");
    assert!(
        listing.contains("note.txt"),
        "directory listing missing the seeded file; got {listing:?}"
    );

    // 5. A protocol-level error from a real server must surface as an error or an
    //    error payload rather than being reported as a successful empty result.
    let missing = client
        .call_tool(
            "read_text_file",
            json!({ "path": format!("{root}/definitely-absent.txt") }),
        )
        .await;
    match missing {
        Err(_) => {}
        Ok(text) => {
            let lowered = text.to_lowercase();
            assert!(
                lowered.contains("error")
                    || lowered.contains("not found")
                    || lowered.contains("enoent"),
                "a missing file must not look like a successful read; got {text:?}"
            );
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn real_server_process_is_terminated_when_client_is_dropped() {
    if !live_enabled() {
        eprintln!("skipped: set AGENT_LIVE_TESTS=1 to run live MCP interoperability");
        return;
    }

    let workspace = tempfile::tempdir().expect("temp workspace");
    let root = tool_path(workspace.path());

    let client = McpClient::connect(
        npx_command(),
        &["-y", "@modelcontextprotocol/server-filesystem", &root],
        "fs",
    )
    .await
    .expect("connect to the real filesystem MCP server");
    assert!(!client.discovered_tools().is_empty());

    // Dropping the client must take the spawned server tree down with it. This is
    // asserted against a fake peer elsewhere; here it is checked against a real
    // Node process tree, which is what actually leaks in production if the
    // supervisor is wrong. The temporary directory cannot be removed on Windows
    // while a child still holds it, so a clean teardown is the observable signal.
    drop(client);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    let path = workspace.path().to_path_buf();
    workspace
        .close()
        .unwrap_or_else(|error| panic!("server tree still holds {}: {error}", path.display()));
}
