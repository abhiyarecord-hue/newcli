//! Local stdio MCP peer used only by Task 9 preservation tests.

use std::fs;
use std::io::{self, BufRead, Write};
use std::net::TcpListener;

use serde_json::{json, Value};

fn read_json(lines: &mut impl Iterator<Item = io::Result<String>>) -> Option<Value> {
    lines
        .next()
        .map(|line| serde_json::from_str(&line.unwrap()).unwrap())
}

/// Read the next *request*, skipping JSON-RPC notifications.
///
/// A notification has no `id` and expects no reply, so a compliant server
/// ignores it and keeps waiting. Reading strictly line by line would mistake the
/// spec-required `notifications/initialized` for the following request.
fn read_json_request(lines: &mut impl Iterator<Item = io::Result<String>>) -> Option<Value> {
    loop {
        let message = read_json(lines)?;
        if message.get("id").is_some() {
            return Some(message);
        }
    }
}

fn respond(id: Value, result: Value) {
    let mut stdout = io::stdout().lock();
    serde_json::to_writer(
        &mut stdout,
        &json!({"jsonrpc":"2.0", "id":id, "result":result}),
    )
    .unwrap();
    stdout.write_all(b"\n").unwrap();
    stdout.flush().unwrap();
}

fn main() {
    let port_file = std::env::args().nth(1).expect("port-file argument");
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    fs::write(
        &port_file,
        listener.local_addr().unwrap().port().to_string(),
    )
    .unwrap();

    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    let initialize = read_json_request(&mut lines).expect("initialize request");
    respond(initialize["id"].clone(), json!({"capabilities":{}}));

    let list = read_json_request(&mut lines).expect("tools/list request");
    respond(
        list["id"].clone(),
        json!({"tools":[
            {"name":"remote_search", "description":"searches locally", "inputSchema":{"type":"object"}},
            {"name":"invalid", "description":"must be rejected", "inputSchema":"not-an-object"}
        ]}),
    );

    while let Some(call) = read_json(&mut lines) {
        let name = call["params"]["name"].as_str().unwrap_or("");
        respond(
            call["id"].clone(),
            json!({"content":[{"type":"text", "text":name}]}),
        );
    }

    drop(listener);
}
