# Security Policy

## Status

This project is **early alpha**. It executes AI-generated code and shell commands. Use it in a trusted environment (ideally a container or VM) and review actions before approving.

## Security Model

- **PathJail** — All file access is confined to the workspace root. Directory traversal (`..`), absolute-path escapes, and symlink escapes are rejected.
- **SecretLeakHook** — Blocks AWS keys, API tokens, and private keys from being written or passed to shell commands (fail-closed).
- **DestructiveCommandHook** — Blocks `rm -rf /`, `mkfs`, forced git pushes.
- **User Approval** — The agent asks for confirmation before running risky commands (installs, deletions, network downloads, git push).
- **NetGuard** — `web_fetch` is HTTPS-only, domain-allowlisted, blocks private/loopback/link-local IPs, and pins the validated IP to prevent DNS-rebinding.

## Known Limitations

- **Process-based sandbox, not a true VM.** `ProcessFallback` runs commands with a cleared environment and timeouts, but does NOT provide kernel-level isolation. For untrusted workloads, run the whole CLI inside a container.
- **Command approval uses a blocklist**, which is not exhaustive. Sophisticated bypasses (e.g. `xargs`, script piping) are possible. Do not rely on it as the sole defense — run in a sandboxed environment.

## Reporting a Vulnerability

Please open a private security advisory on GitHub or email the maintainer. Do not open a public issue for security vulnerabilities.

We aim to acknowledge reports within 7 days.
