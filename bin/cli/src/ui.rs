use std::io::{IsTerminal, Write};

#[derive(Clone, Default)]
pub struct UsageStats {
    pub turn_in: u64,
    pub turn_out: u64,
    pub turn_total: u64,
    pub turn_calls: u64,
    pub session_in: u64,
    pub session_out: u64,
    pub session_total: u64,
    pub session_calls: u64,
}

impl UsageStats {
    pub fn start_turn(&mut self) {
        self.turn_in = 0;
        self.turn_out = 0;
        self.turn_total = 0;
        self.turn_calls = 0;
    }

    pub fn api_call(&mut self) {
        self.turn_calls += 1;
        self.session_calls += 1;
    }

    pub fn add_tokens(&mut self, input: u32, output: u32, total: u32) {
        self.turn_in += input as u64;
        self.turn_out += output as u64;
        self.turn_total += total as u64;
        self.session_in += input as u64;
        self.session_out += output as u64;
        self.session_total += total as u64;
    }
}

fn color(code: &str, text: impl AsRef<str>) -> String {
    if std::env::var_os("NO_COLOR").is_none() && std::io::stdout().is_terminal() {
        format!("\x1b[{code}m{}\x1b[0m", text.as_ref())
    } else {
        text.as_ref().to_string()
    }
}

fn short(value: &str, max: usize) -> String {
    let mut chars = value.chars();
    let text: String = chars.by_ref().take(max).collect();
    if chars.next().is_some() {
        format!("{text}…")
    } else {
        text
    }
}
/// Ask the user to pick between Vibe (free-flow chat) and RustySpec
/// (structured 7-stage workflow) before starting the session.
pub fn mode_select() {
    let cyan = |s: &str| color("1;38;5;45", s);
    let gold = |s: &str| color("1;38;5;214", s);
    println!();
    println!(
        "{}",
        color(
            "38;5;240",
            "       ┌────────────────────────────────────────────────────────────┐"
        )
    );
    println!(
        "       │  {}                                              │",
        color("1;97", "Choose a session mode")
    );
    println!(
        "{}",
        color(
            "38;5;240",
            "       ├────────────────────────────────────────────────────────────┤"
        )
    );
    println!(
        "       │  {}  free-flow chat, quick tasks                          │",
        cyan("[1] VIBE")
    );
    println!(
        "       │  {}  structured spec → plan → code workflow          │",
        gold("[2] RUSTYSPEC")
    );
    println!(
        "{}",
        color(
            "38;5;240",
            "       └────────────────────────────────────────────────────────────┘"
        )
    );
    print!("       Pick 1 or 2 (default 1): ");
    std::io::stdout().flush().ok();
}

pub fn banner(
    provider: &str,
    model: &str,
    workspace: &str,
    api_ready: bool,
    memory: bool,
    git: bool,
) {
    let cyan = |s: &str| color("1;38;5;45", s);
    let gold = |s: &str| color("1;38;5;214", s);
    println!();
    println!(
        "{}",
        cyan("     ███████╗██████╗ ██╗     ██╗ █████╗ ███╗   ██╗")
    );
    println!(
        "{}",
        cyan("     ██╔════╝██╔══██╗██║     ██║██╔══██╗████╗  ██║")
    );
    println!(
        "{}",
        gold("     ███████╗██████╔╝██║     ██║███████║██╔██╗ ██║")
    );
    println!(
        "{}",
        gold("     ╚════██║██╔══██╗██║██   ██║██╔══██║██║╚██╗██║")
    );
    println!(
        "{}",
        gold("     ███████║██║  ██║██║╚█████╔╝██║  ██║██║ ╚████║")
    );
    println!(
        "{}",
        color(
            "38;5;240",
            "     ╚══════╝╚═╝  ╚═╝╚═╝ ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝ ░"
        )
    );
    println!("{}", color("1;97", "                       D E V   A I"));
    println!(
        "{}",
        color(
            "38;5;240",
            "       ┌────────────────────────────────────────────────────────────┐░"
        )
    );
    println!("       │ {}  {}", color("38;5;45", "PROVIDER"), provider);
    println!(
        "       │ {}     {}",
        color("38;5;45", "MODEL"),
        short(model, 46)
    );
    println!(
        "       │ {}       {}",
        color("38;5;45", "API"),
        if api_ready {
            color("1;32", "● ready")
        } else {
            color("1;31", "○ key missing")
        }
    );
    println!(
        "       │ {} {}",
        color("38;5;45", "WORKSPACE"),
        short(workspace, 44)
    );
    println!(
        "       │ {}    {}   {}  {}",
        color("38;5;45", "SYSTEM"),
        if git { "git ✓" } else { "git —" },
        if memory { "memory ✓" } else { "memory —" },
        color("38;5;214", "autopilot ●")
    );
    println!(
        "{}",
        color(
            "38;5;240",
            "       └────────────────────────────────────────────────────────────┘░"
        )
    );
    println!(
        "{}",
        color(
            "38;5;236",
            "        ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░"
        )
    );
    println!(
        "       {}  /remember  /undo  /clear  /quit\n",
        color("38;5;244", "COMMANDS")
    );
}

pub fn prompt_start() {
    eprintln!(
        "{}",
        color(
            "38;5;45",
            "  ╭─ YOU ───────────────────────────────────────────────────────────╮"
        )
    );
    eprint!("  {} {} ", color("38;5;45", "│"), color("1;38;5;214", "❯"));
    std::io::stderr().flush().ok();
}

pub fn prompt_end() {
    eprintln!(
        "{}",
        color(
            "38;5;45",
            "  ╰─────────────────────────────────────────────────────────────────╯"
        )
    );
}

pub fn turn_started() {
    eprintln!(
        "  {} {}",
        color("38;5;45", "╭─"),
        color("1;97", "SRIJAN IS WORKING")
    );
}
pub fn api_call(number: u64) {
    eprintln!(
        "  {} {}",
        color("38;5;240", "├─"),
        color("38;5;141", format!("◇ API request #{number}"))
    );
}

pub fn thinking(text: &str) {
    let line = short(text.lines().next().unwrap_or("thinking"), 58);
    eprint!(
        "\r\x1b[2K  {} {}",
        color("38;5;240", "├─"),
        color("38;5;244", format!("◈ {line}"))
    );
    std::io::stderr().flush().ok();
}

pub fn tool_started(name: &str) {
    eprintln!(
        "\r\x1b[2K  {} {}",
        color("38;5;240", "├─"),
        color("1;38;5;214", format!("⚙ {name}"))
    );
}

pub fn tool_done(name: &str) {
    eprintln!(
        "  {} {}",
        color("38;5;240", "│ "),
        color("1;32", format!("✓ {name}"))
    );
}

pub fn turn_ended() {
    eprintln!("\r\x1b[2K  {}", color("38;5;45", "╰─ work complete"));
}

fn tokens(value: u64) -> String {
    if value >= 1_000_000 {
        format!("{:.2}M", value as f64 / 1_000_000.0)
    } else if value >= 1_000 {
        format!("{:.1}K", value as f64 / 1_000.0)
    } else {
        value.to_string()
    }
}

pub fn answer(text: &str) {
    println!(
        "{}",
        color(
            "38;5;214",
            "  ╭─ SRIJAN ────────────────────────────────────────────────────────╮"
        )
    );
    if text.is_empty() {
        println!(
            "  {} {}",
            color("38;5;214", "│"),
            color("38;5;244", "Task completed without a text response.")
        );
    } else {
        for line in text.lines() {
            println!("  {} {line}", color("38;5;214", "│"));
        }
    }
    println!(
        "{}",
        color(
            "38;5;214",
            "  ╰─────────────────────────────────────────────────────────────────╯"
        )
    );
}

pub fn error(text: &str) {
    eprintln!(
        "{}",
        color(
            "1;31",
            "  ╭─ ERROR ─────────────────────────────────────────────────────────╮"
        )
    );
    eprintln!("  {} {text}", color("1;31", "│"));
    eprintln!(
        "{}",
        color(
            "1;31",
            "  ╰─────────────────────────────────────────────────────────────────╯"
        )
    );
}

pub fn usage(stats: &UsageStats) {
    let unavailable = stats.turn_total == 0 && stats.turn_calls > 0;
    eprintln!(
        "{}",
        color(
            "38;5;240",
            "  ╭─ USAGE ─────────────────────────────────────────────────────────╮"
        )
    );
    eprintln!(
        "  {} {}  API {:>2}  ↑ {:>7}  ↓ {:>7}  Σ {:>7}{}",
        color("38;5;240", "│"),
        color("38;5;45", "TURN   "),
        stats.turn_calls,
        tokens(stats.turn_in),
        tokens(stats.turn_out),
        tokens(stats.turn_total),
        if unavailable {
            "  (provider did not report tokens)"
        } else {
            ""
        }
    );
    eprintln!(
        "  {} {}  API {:>2}  ↑ {:>7}  ↓ {:>7}  Σ {:>7}",
        color("38;5;240", "│"),
        color("38;5;214", "SESSION"),
        stats.session_calls,
        tokens(stats.session_in),
        tokens(stats.session_out),
        tokens(stats.session_total)
    );
    eprintln!(
        "{}",
        color(
            "38;5;240",
            "  ╰─────────────────────────────────────────────────────────────────╯"
        )
    );
}
