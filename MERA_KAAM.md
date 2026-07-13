# मेरा काम — क्या करना है, कैसे करना है

यह फाइल बताती है कि प्रोजेक्ट को दुनिया के सामने लाने के लिए **तुम्हें** क्या करना है (जो काम कोड से बाहर हैं और सिर्फ तुम कर सकते हो)।

---

## ✅ जो हो चुका है (कोड में)

- 9 tools (read_file, write_file, edit_file, list_files, search_text, bash, web_fetch, check_code, dispatch_subagent)
- 6 AI providers (Gemini, OpenAI, Anthropic, Mistral, DeepSeek, Ollama)
- Git checkpoints + /undo
- Self-healing edit loop
- CRDT concurrent editing
- Session memory (/remember)
- Security (PathJail, secret detection, user approval)
- Hinglish mode
- IPC server (cli serve)
- MCP support
- 154 tests pass
- CI pipeline, CONTRIBUTING.md, SECURITY.md
- GitHub pe push ho chuka: https://github.com/abhiyarecord-hue/newcli

---

## 📋 तुम्हारा काम (Priority Order)

### 1. क्रेट्स को crates.io पर publish करो (Ecosystem lock-in)

**क्यों:** लोग तुम्हारे crates के ऊपर अपने agents बनाएंगे। यह closed tools कभी नहीं दे सकते।

**कैसे:**
```powershell
# Step 1: crates.io par account banao
# https://crates.io par jao → GitHub se login karo

# Step 2: API token lo
# https://crates.io/settings/tokens → "New Token" → copy karo

# Step 3: Terminal me login karo
cargo login <tumhara-token>

# Step 4: Har crate ko order me publish karo (dependencies pehle)
# Order: agent-types → runtime-core → llm-client → ... → agent-core → cli
cd crates/agent-types
cargo publish
# (thoda wait karo, phir agla)
```

**Note:** Har crate ke Cargo.toml me `description`, `license`, `repository` fields add karne padenge publish se pehle. Bata dena to main add kar dunga.

---

### 2. 60-सेकंड का Demo Video/GIF बनाओ

**क्यों:** Repo ki pehli impression yahi hoti hai. README ke top pe GIF = 10x zyada stars.

**कैसे:**
- Windows par **ScreenToGif** (free) download karo: https://www.screentogif.com/
- Terminal record karo jisme:
  1. `cli chat` chalao
  2. "ek snake game banao" bolo
  3. Agent files banata dikhe (tools chalte hue)
  4. Browser me game khulta dikhe
- GIF ko repo me `demo.gif` naam se daalo
- README ke top me add karo: `![Demo](demo.gif)`

**Bata dena** GIF ban jaye to main README me embed kar dunga.

---

### 3. SWE-bench Score निकालो (Credibility)

**क्यों:** Ek real number ("resolves X% of SWE-bench Lite") = instant credibility.

**कैसे:**
- `swe bench test/` folder me 300 cases already hain
- Chhote scale se shuru karo (5-10 Python cases)
- Har case: repo clone karo → agent ko bug fix karne do → test chalao
- Yeh time-consuming hai (repos clone + Python setup)
- **Realistic:** Pehle 10-20 cases pe score nikaalo, README me likho "Early results: X/20 on SWE-bench Lite subset"

**Bata dena** jab ready ho to main runner ko poori tarah wire kar dunga.

---

### 4. Prebuilt Binaries + Installer (Easy install)

**क्यों:** Log `cargo build` nahi karna chahte — ek command me install chahiye.

**कैसे:**
- GitHub Releases page pe jao: repo → Releases → "Create a new release"
- Tag: `v0.1.0-alpha`
- CI already banaya hai — usme release binaries build karne ka step add kar sakte hain
- Users phir download karke chala sakenge

**Bata dena** to main CI me release-binary build step add kar dunga (Linux/Mac/Windows .exe).

---

### 5. VS Code Extension (बड़ा काम, बाद में)

**क्यों:** Diff preview + inline suggestions = Cursor/Kiro jaisa experience.

**कैसे:**
- `cli serve --port 9527` backend already ready hai
- Ek alag folder me TypeScript extension banao
- Extension `cli serve` ko spawn kare, IPC se connect ho
- **Yeh 1-2 hafte ka alag project hai** — abhi mat karo, pehle CLI launch karo

**Bata dena** jab ready ho to main extension ka scaffold bana dunga.

---

## 🎯 सबसे पहले क्या करो (मेरी सलाह)

| Order | काम | समय | कौन करेगा |
|-------|-----|-----|-----------|
| 1 | GitHub release banao (v0.1.0-alpha) | 10 min | तुम |
| 2 | Demo GIF banao | 30 min | तुम |
| 3 | crates.io publish (metadata add karke) | 1 ghanta | तुम + मैं |
| 4 | SWE-bench 10-20 cases pe score | 2-3 ghante | तुम + मैं |
| 5 | Prebuilt binaries CI me | 1 ghanta | मैं (tum trigger karo) |
| 6 | VS Code extension | 1-2 hafte | बाद में |

---

## 💡 Launch करने से पहले Checklist

- [ ] README me demo GIF hai
- [ ] GitHub release v0.1.0-alpha bana hai
- [ ] crates.io pe publish (optional but strong)
- [ ] SWE-bench ka koi number hai (chhota bhi chalega)
- [ ] "alpha" tag clearly mentioned hai (jhoothe claims nahi)
- [ ] Reddit r/rust, Hacker News, Twitter pe share karo — **Hinglish-first angle highlight karo**

---

## 📢 कहाँ Share करें (Marketing)

1. **r/rust** (Reddit) — "I built a Rust-native AI coding agent" — Rust community loves Rust tools
2. **Hacker News** — "Show HN: NewGen CLI"
3. **Twitter/X** — Indian dev community ko tag karo, Hinglish angle
4. **r/LocalLLaMA** — Offline/Ollama support highlight karo (yeh community usse pyaar karti hai)
5. **Dev.to / Hashnode** — Ek blog post likho "Why I built a Hinglish-first AI coding agent"

**Sabse strong angle:** "Fully offline + Hinglish-first AI coding agent in Rust" — yeh do cheezein koi aur nahi de raha.

---

## ⚠️ Jo abhi NAHI karna (Honesty)

- "Cursor/Claude Code se behtar" mat likhna — abhi alpha hai
- SWE-bench score jhootha mat daalna — jo actual hai wahi likhna
- MicroVM claim mat karna — abhi process-based hai (SECURITY.md me sach likha hai)

**Yaad rakho:** Honest alpha > jhootha "production-ready". Community trust sabse important hai.
