# Rust AI Coding Agent — Usage Guide

## Quick Start (3 Steps)

### Step 1: Build (sirf ek baar)
```powershell
cd C:\path\to\rustcoddingcli-main
cargo build --release -p cli
```
Binary banti hai `.\target\release\srijan.exe`.

> Command ka naam **`srijan`** hai, `cli` nahi. PowerShell me `cli` uska apna
> built-in shortcut hai (`Clear-Item` ke liye) aur wo hataya nahi ja sakta, to
> `cli chat` hamare program tak pahunchta hi nahi.

### Step 2: PATH me daalo (ek baar)
```powershell
$rel = (Resolve-Path .\target\release).Path
$u = [Environment]::GetEnvironmentVariable('Path','User')
[Environment]::SetEnvironmentVariable('Path', "$u;$rel", 'User')
```
Iske baad **naya** terminal kholna zaroori hai — PATH ka badlav pehle se khule
window par lagu nahi hota.

Binary ko kisi doosri jagah copy karne se bachein. Antivirus naye banaye gaye
unsigned executable ko `AppData\Local\Programs` jaisi jagah se chup-chaap hata
deta hai; build folder aam taur par chhoda jaata hai.

### Step 3: Provider set karo aur chalao
```powershell
# Ek baar (har naye terminal me apne aap lagega)
setx LLM_PROVIDER "openai"
setx LLM_MODEL    "gpt-5.6-sol"
setx OPENAI_BASE_URL "https://<tumhara-endpoint>/openai/v1"
setx LLM_API_KEY  "<tumhari-key>"

# Naya terminal kholo, phir apne project me jao
cd C:\Users\Acer\Projects\mera-project
srijan chat
```

Shuru me jo box dikhta hai usme `PROVIDER`, `MODEL` aur `API ● ready` dekh lein.
Agar `MODEL` purana dikhe, to terminal purana hai — naya kholein.

---

## Kaam ke commands

| Command | Kaam |
|---|---|
| `srijan chat` | Baat-cheet karke code likhwana/sudharwana |
| `srijan chat -m spec` | 7-charan wala structured workflow |
| `srijan index` | Project ka index banana (ek baar) |
| `srijan search "kuch"` | Index me dhundhna |
| `srijan spec specify --from-file req.md` | Zaroorat ek file se dena (lambi prompt ke liye behtar) |
| `srijan --help` | Poori list |

`spec` ke saat charan: `specify → clarify → plan → tasks → tests → implement →
analyze`. **Code `implement` par banta hai**, pehle charan par nahi.

---

## API key kahan se

Koi bhi OpenAI-compatible endpoint chalta hai — OpenAI, Azure AI Foundry,
Mistral, DeepSeek, Together, OpenRouter, ya local Ollama/LM Studio/vLLM
(inme key ki zaroorat nahi). `LLM_PROVIDER`, `OPENAI_BASE_URL` aur `LLM_API_KEY`
usi hisaab se set karein.

Embeddings alag se set hoti hain (`EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`,
`EMBEDDING_BASE_URL`), kyunki chat aur embedding ke provider alag ho sakte hain.

---

## Dhyan rakhne wali baatein

- `chat` khud files likhta aur badalta hai. Chalane se pehle `git commit` kar
  lein, taaki `git checkout .` se wapas laya ja sake.
- `srijan serve` par abhi authentication nahi hai. Use na chalayein.
- Command execution sandbox nahi hai.

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
| `bash` | Shell command chalana (best-effort policy + user approval, **sandbox isolation nahi**) |

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
| Search sirf keyword results de raha hai, semantic nahi | Message padho: agar `BM25-only:` se shuru ho, embedding provider/model/dimension badla hai. Wahi embedding config ke saath dubara index karo |
| Windows par "file in use" / sharing violation | Editor band karo jo file khuli rakhta hai, ya workspace ko antivirus real-time scan se exclude karo. Write bounded retry karta hai, chupke se skip nahi karta |
| MCP server dubara approval maang raha hai | Uske command/args/env badalne se purani approval invalid ho jaati hai. Ye by design hai |

---

## Upgrade Karte Waqt (Migration)

Purane workspace ko naye build ke saath chalane par 4 cheezein migrate hoti hain. **Koi bhi migration
destructive nahi hai — fail hone par original data waisa hi rehta hai.**

| Kya | Kya hota hai |
|-----|--------------|
| Chat history | `.agent/HISTORY.jsonl` ek baar padh ke `.agent/HISTORY.v2.json` banta hai. Purani file **backup ke roop mein rakhi jaati hai**, delete nahi hoti |
| Long-term memory | `.agent/MEMORY.md`, `SOUL.md`, `HEARTBEAT.md` **bilkul nahi chhede jaate** — alag contract hai. `/clear` bhi inhe nahi hatata |
| Search index | Embedding provider/model/dimension badla ho to **poora re-index chahiye**. Tab tak keyword (BM25) search chalta rehta hai |
| Spec `tests` artifact | Ab directory hai (`tests/`). Purani `tests` regular file ko `tests.backup` naam se rename kiya jaata hai |

Poori technical detail, exact bounds, aur durability ki limits README ke "Migration and Compatibility"
aur "Bounds and Durability" sections mein hain.

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
