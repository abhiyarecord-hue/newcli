# Rust AI Coding Agent — Usage Guide

## Quick Start (3 Steps)

### Step 1: Build (sirf ek baar)
```powershell
cd C:\Users\Acer\Downloads\rustcoddingcli-main-20260711T040518Z-2-001\rustcoddingcli-main
cargo build --release -p cli
```

### Step 2: CLI ko accessible jagah copy karo (optional, ek baar)
```powershell
Copy-Item ".\target\release\cli.exe" "C:\Users\Acer\cli.exe"
```

### Step 3: Kisi bhi folder mein use karo
```powershell
# 1. API key set karo (Vertex AI Express Mode - Cloud Console se)
$env:GEMINI_API_KEY = "tumhari-api-key"

# 2. Apna project folder banao ya usme jao
mkdir C:\Users\Acer\Projects\mera-project
cd C:\Users\Acer\Projects\mera-project

# 3. CLI chalao
& "C:\Users\Acer\cli.exe" chat
```

---

## API Key Kahan Se Milegi

1. https://console.cloud.google.com jao
2. APIs & Services > Library > "Vertex AI API" enable karo
3. APIs & Services > Credentials > API Key banao
4. Woh key `$env:GEMINI_API_KEY` mein daalo

Billing startup credits se katega (free tier nahi).

---

## Model Switch Karna

CLI chalane se pehle `$env:GEMINI_MODEL` set karo:

| Model | ID | Best For |
|-------|-----|----------|
| **3.5 Flash** (default) | `gemini-3.5-flash` | Fast, daily coding, high volume |
| **3.5 Pro** | `gemini-3.5-pro-preview` | Hardest tasks, 2M context window |
| **3.1 Pro** | `gemini-3.1-pro` | Stable reasoning, complex problems |
| **3.1 Flash-Lite** | `gemini-3.1-flash-lite` | Cheapest, ultra fast, simple tasks |

```powershell
$env:GEMINI_MODEL = "gemini-3.5-pro-preview"
& "C:\Users\Acer\cli.exe" chat
```

Agar set nahi kiya toh default `gemini-3.5-flash` chalega.

---

## Workspace (Kidhar Files Banegi)

Agent usi folder mein files banata hai jahaan se tum CLI chalate ho.

**Method 1:** Pehle `cd` karo us folder mein
```powershell
cd C:\Users\Acer\Projects\snake-game
& "C:\Users\Acer\cli.exe" chat
```

**Method 2:** `--workspace` flag use karo
```powershell
& "C:\Users\Acer\cli.exe" chat --workspace "C:\Users\Acer\Projects\snake-game"
```

---

## VS Code Mein Use Karna (Recommended)

1. VS Code mein `File > Open Folder` → apna project folder kholo
2. Terminal kholo: `Ctrl + `` `
3. Yeh paste karo:
```powershell
$env:GEMINI_API_KEY = "tumhari-api-key"
& "C:\Users\Acer\cli.exe" chat
```
4. Agent jo files banayega woh Explorer panel mein real-time dikhenge

---

## Available Tools (Agent Ke Paas)

| Tool | Kaam |
|------|------|
| `read_file` | File padhna (optional line range) |
| `write_file` | File banana / overwrite karna |
| `list_files` | Directory listing |
| `search_text` | Recursive text search across files |
| `bash` | Shell command chalana (sandboxed) |

Agent khud decide karta hai kaunsa tool kab use karna hai.

---

## Special Commands (Chat Ke Andar)

| Command | Kaam |
|---------|------|
| `/quit` | Exit |
| `/exit` | Exit |
| `Ctrl+C` | Force quit |

---

## Token Tracking

Har turn ke baad dikhega:
```
[tokens: prompt=629, output=150, total=900 | session: 1800]
```
- `prompt` = input tokens (system + tools + history)
- `output` = generated tokens
- `total` = prompt + output + thinking
- `session` = poore session ka cumulative total

---

## AI Studio (Free Tier) Pe Wapas Jaana

Agar kabhi free tier test karna ho (20 req/day limit):
```powershell
$env:GEMINI_USE_AI_STUDIO = "1"
$env:GEMINI_API_KEY = "AI-Studio-wali-key"
& "C:\Users\Acer\cli.exe" chat
```

---

## Rebuild Karna (Code Change Ke Baad)

```powershell
cd C:\Users\Acer\Downloads\rustcoddingcli-main-20260711T040518Z-2-001\rustcoddingcli-main
cargo build --release -p cli
Copy-Item ".\target\release\cli.exe" "C:\Users\Acer\cli.exe" -Force
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "not recognized" error | Full path use karo: `& "C:\Users\Acer\cli.exe" chat` |
| 429 Too Many Requests | Free tier quota khatam. Vertex AI key use karo ya wait karo |
| "error sending request" | Internet check karo, ya API key galat hai |
| Files nahi dikh rahi VS Code mein | VS Code mein sahi folder open karo (`File > Open Folder`) |
| Agent plan banata hai par file nahi likhta | Dubara bolo: "file banao, write_file use karo" |

---

## Example Prompts

```
You> ek snake game banao browser ke liye
You> Cargo.toml padho aur batao kitne crates hain
You> src folder mein "TODO" search karo
You> cargo test chala ke batao results
You> ek REST API server banao Rust mein with actix-web
You> is project ka README.md likh do
```
