//! Local stdio peer used only by Task 2 protocol exploration tests.

use std::fs;
use std::io::{self, BufRead, Write};
use std::thread;
use std::time::Duration;

use serde_json::{json, Value};

fn main() {
    let mut args = std::env::args().skip(1);
    let protocol = args.next().expect("protocol mode");
    let mode = args.next().expect("peer behavior");
    let extra = args.next();
    match protocol.as_str() {
        "mcp" => run_mcp(&mode),
        "lsp" => run_lsp(&mode, extra.as_deref()),
        other => panic!("unknown protocol {other}"),
    }
}

fn write_json_line(value: &Value) {
    let stdout = io::stdout();
    let mut out = stdout.lock();
    serde_json::to_writer(&mut out, value).unwrap();
    out.write_all(b"\n").unwrap();
    out.flush().unwrap();
}

fn read_json_line(lines: &mut impl Iterator<Item = io::Result<String>>) -> Value {
    serde_json::from_str(&lines.next().expect("request line").expect("read request")).unwrap()
}

/// Read the next *request*, skipping notifications.
///
/// A JSON-RPC notification has no `id` and expects no reply. A real server
/// ignores it and keeps waiting for the next request. This peer previously read
/// strictly line by line, so once the client began sending the spec-required
/// `notifications/initialized`, the notification was mistaken for the following
/// request and answered with a null id. Skipping notifications here is what a
/// compliant server does.
fn read_json_request(lines: &mut impl Iterator<Item = io::Result<String>>) -> Value {
    loop {
        let message = read_json_line(lines);
        if message.get("id").is_some() {
            return message;
        }
    }
}

/// Reject a handshake that omits the fields the MCP specification requires.
///
/// This mirrors what a spec-compliant server does, and it exists because the
/// permissive version of this peer hid a real defect: the client sent only
/// `capabilities`, which every real server rejects with `-32603`, making all of
/// them unreachable. Validating here means that regression fails in the offline
/// suite instead of waiting for a live run.
fn assert_valid_initialize(init: &Value) {
    let params = init
        .get("params")
        .unwrap_or_else(|| panic!("initialize has no params: {init}"));
    assert!(
        params
            .get("protocolVersion")
            .and_then(Value::as_str)
            .is_some(),
        "initialize must send a string protocolVersion: {init}"
    );
    let client_info = params
        .get("clientInfo")
        .unwrap_or_else(|| panic!("initialize must send clientInfo: {init}"));
    assert!(
        client_info.get("name").and_then(Value::as_str).is_some(),
        "clientInfo must carry a name: {init}"
    );
    assert!(
        client_info.get("version").and_then(Value::as_str).is_some(),
        "clientInfo must carry a version: {init}"
    );
}

fn response(id: Value, result: Value) -> Value {
    json!({"jsonrpc":"2.0", "id":id, "result":result})
}

fn run_mcp(mode: &str) {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    let init = read_json_request(&mut lines);
    assert_valid_initialize(&init);
    match mode {
        "silent" => {
            thread::sleep(Duration::from_secs(30));
            return;
        }
        "oversized" => {
            let stdout = io::stdout();
            let mut out = stdout.lock();
            out.write_all(&vec![b'x'; 1025]).unwrap();
            out.flush().unwrap();
            thread::sleep(Duration::from_secs(30));
            return;
        }
        "malformed" => {
            let mut stdout = io::stdout().lock();
            stdout.write_all(b"not-json\n").unwrap();
            stdout.flush().unwrap();
            return;
        }
        _ => write_json_line(&response(init["id"].clone(), json!({"capabilities":{}}))),
    }

    let list = read_json_request(&mut lines);
    if mode == "notification" {
        write_json_line(&json!({"jsonrpc":"2.0", "method":"progress", "params":{"step":1}}));
    }
    write_json_line(&response(
        list["id"].clone(),
        json!({"tools":[{
            "name":"search", "description":"remote search", "inputSchema":{"type":"object"}
        }]}),
    ));

    match mode {
        "reorder" => {
            let first = read_json_line(&mut lines);
            let second = read_json_line(&mut lines);
            write_json_line(&response(
                second["id"].clone(),
                json!({"content":[{"type":"text","text":"second"}]}),
            ));
            write_json_line(&response(
                first["id"].clone(),
                json!({"content":[{"type":"text","text":"first"}]}),
            ));
            return;
        }
        "request-silent" => {
            let _call = read_json_line(&mut lines);
            thread::sleep(Duration::from_secs(30));
            return;
        }
        "eof-pending" => {
            let _first = read_json_line(&mut lines);
            let _second = read_json_line(&mut lines);
            return;
        }
        "unknown-duplicate" => {
            let first = read_json_line(&mut lines);
            write_json_line(&response(
                json!(999_999),
                json!({"content":[{"type":"text","text":"unknown"}]}),
            ));
            write_json_line(&response(
                first["id"].clone(),
                json!({"content":[{"type":"text","text":"first"}]}),
            ));
            write_json_line(&response(
                first["id"].clone(),
                json!({"content":[{"type":"text","text":"duplicate"}]}),
            ));
            let second = read_json_line(&mut lines);
            write_json_line(&response(
                second["id"].clone(),
                json!({"content":[{"type":"text","text":"second"}]}),
            ));
            return;
        }
        _ => {}
    }

    if let Some(Ok(line)) = lines.next() {
        let call: Value = serde_json::from_str(&line).unwrap();
        let id = if mode == "string-id" {
            Value::String(call["id"].as_i64().unwrap().to_string())
        } else {
            call["id"].clone()
        };
        let name = call["params"]["name"].as_str().unwrap_or("");
        write_json_line(&response(
            id,
            json!({"content":[{"type":"text","text":name}]}),
        ));
    }
}
fn read_lsp_message(input: &mut impl BufRead) -> Value {
    let mut length = None;
    loop {
        let mut line = String::new();
        input.read_line(&mut line).unwrap();
        if line.trim().is_empty() {
            break;
        }
        if let Some(raw) = line.trim().strip_prefix("Content-Length:") {
            length = raw.trim().parse::<usize>().ok();
        }
    }
    let mut body = vec![0; length.expect("Content-Length")];
    input.read_exact(&mut body).unwrap();
    serde_json::from_slice(&body).unwrap()
}

fn write_lsp_message(value: &Value, duplicate: bool) {
    let body = serde_json::to_vec(value).unwrap();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    write!(out, "Content-Length: {}\r\n", body.len()).unwrap();
    if duplicate {
        write!(out, "Content-Length: {}\r\n", body.len()).unwrap();
    }
    write!(out, "\r\n").unwrap();
    out.write_all(&body).unwrap();
    out.flush().unwrap();
}

fn run_lsp(mode: &str, capture_path: Option<&str>) {
    let stdin = io::stdin();
    let mut input = stdin.lock();
    let init = read_lsp_message(&mut input);
    if let Some(path) = capture_path {
        fs::write(path, init["params"]["rootUri"].as_str().unwrap()).unwrap();
    }

    match mode {
        "eof" => return,
        "malformed-length" => {
            print!("Content-Length: nope\r\n\r\n");
            io::stdout().flush().unwrap();
            return;
        }
        "oversized" => {
            print!("Content-Length: 16777217\r\n\r\n");
            io::stdout().flush().unwrap();
            thread::sleep(Duration::from_secs(30));
            return;
        }
        "duplicate" => write_lsp_message(
            &response(init["id"].clone(), json!({"capabilities":{}})),
            true,
        ),
        _ => write_lsp_message(
            &response(init["id"].clone(), json!({"capabilities":{}})),
            false,
        ),
    }

    let _initialized = read_lsp_message(&mut input);
    thread::sleep(Duration::from_secs(30));
}
