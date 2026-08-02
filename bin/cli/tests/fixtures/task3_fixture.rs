//! Inert local process fixture for Task 3 exploration tests.

use std::fs::{self, OpenOptions};
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

fn main() {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("normal") => {
            println!("normal-stdout");
            eprintln!("normal-stderr");
        }
        Some("flood") => flood(),
        Some("sleep-ms") => {
            let millis = args
                .next()
                .expect("sleep duration")
                .parse::<u64>()
                .expect("integer milliseconds");
            thread::sleep(Duration::from_millis(millis));
        }
        // Observe concurrency directly instead of inferring it from wall-clock
        // time. Each invocation publishes a "live" marker, waits for a sibling's
        // live marker to appear, records whether it saw one, then removes its
        // own marker. Two invocations that genuinely overlap each see the other;
        // invocations that are serialized cannot, because the earlier one has
        // already removed its marker. This is independent of machine speed.
        Some("rendezvous") => {
            let dir = PathBuf::from(args.next().expect("rendezvous directory"));
            let id = args.next().expect("rendezvous id");
            let wait_ms = args
                .next()
                .expect("rendezvous wait")
                .parse::<u64>()
                .expect("integer milliseconds");

            let own_marker = format!("live-{id}");
            fs::write(dir.join(&own_marker), b"1").expect("publish live marker");

            let deadline = Instant::now() + Duration::from_millis(wait_ms);
            let mut saw_peer = false;
            while Instant::now() < deadline {
                let peer_present =
                    fs::read_dir(&dir)
                        .into_iter()
                        .flatten()
                        .flatten()
                        .any(|entry| {
                            let name = entry.file_name().to_string_lossy().into_owned();
                            name.starts_with("live-") && name != own_marker
                        });
                if peer_present {
                    saw_peer = true;
                    break;
                }
                thread::sleep(Duration::from_millis(25));
            }

            if saw_peer {
                fs::write(dir.join(format!("sawpeer-{id}")), b"1").expect("record peer marker");
            }
            // Remove the marker last so a serialized successor cannot observe it.
            let _ = fs::remove_file(dir.join(&own_marker));
        }
        Some("descendant-parent") | Some("eval-descendant-parent") => {
            let mode = std::env::args().nth(1).expect("fixture mode");
            let marker = args.next().expect("marker path");
            let exe = std::env::current_exe().expect("fixture executable");
            // Deliberately never waited on: this fixture exists to leave a live
            // descendant behind so the executor's process-tree termination can be
            // observed. Reaping it here would defeat the test.
            #[allow(clippy::zombie_processes)]
            let _descendant = Command::new(exe)
                .arg("descendant")
                .arg(marker)
                .spawn()
                .expect("spawn descendant");
            let delay = if mode == "eval-descendant-parent" {
                Duration::from_millis(1500)
            } else {
                Duration::from_secs(10)
            };
            thread::sleep(delay);
        }
        Some("descendant") => {
            let marker = args.next().expect("marker path");
            append(&marker, "started\n");
            thread::sleep(Duration::from_millis(700));
            append(&marker, "survived\n");
        }
        Some("mcp-start") => run_mcp(&args.next().expect("marker path")),
        other => panic!("unknown fixture mode: {other:?}"),
    }
}

fn append(path: &str, text: &str) {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap();
    file.write_all(text.as_bytes()).unwrap();
    file.flush().unwrap();
}
fn flood() {
    let chunk = vec![b'x'; 64 * 1024];
    let mut stdout = io::stdout().lock();
    let mut stderr = io::stderr().lock();
    for _ in 0..32 {
        stdout.write_all(&chunk).unwrap();
        stderr.write_all(&chunk).unwrap();
    }
    stdout.flush().unwrap();
    stderr.flush().unwrap();
}

fn write_json_line(value: &Value) {
    let mut out = io::stdout().lock();
    serde_json::to_writer(&mut out, value).unwrap();
    out.write_all(b"\n").unwrap();
    out.flush().unwrap();
}

fn run_mcp(marker: &str) {
    fs::write(marker, "spawned without trust decision\n").unwrap();
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    let init: Value = serde_json::from_str(&lines.next().unwrap().unwrap()).unwrap();
    write_json_line(&json!({
        "jsonrpc": "2.0", "id": init["id"], "result": {"capabilities": {}}
    }));
    let list: Value = serde_json::from_str(&lines.next().unwrap().unwrap()).unwrap();
    write_json_line(&json!({
        "jsonrpc": "2.0", "id": list["id"], "result": {"tools": []}
    }));
    for line in lines {
        if line.is_err() {
            break;
        }
    }
}
