# ATLAS — Architecture, OOP Contracts & Diagrams

Companion to `docs/ATLAS_PLAN.md`. Everything here is Mermaid — it renders on GitHub **and inside Obsidian**, so you can keep this vault-side as the design doc.

## 1. Design principles

1. **Ports & adapters.** The core knows only ABCs. Cloud, local, fake — all interchangeable engines.
2. **One abstraction per concept.** Speech in, speech out, thinking, memory, acting, looking. Nothing else is allowed to grow a second path.
3. **Events, not calls, for anything user-visible.** The core emits; the UI/logger/tray subscribe. Core never imports UI.
4. **Composition root owns construction.** One DI container wires everything; no module-level singletons.
5. **Budgeted by design.** Every heavy asset is leased (`ResourceLease`), not resident.
6. **Darija-first.** Not a translation layer bolted on — a first-class language path with its own ASR/TTS/normalisation.
7. **Reversible writes.** Anything that touches your memory is git-committed and undoable by voice.
8. **Data is not instructions.** Tool output, web text and vault content are never treated as commands.

---

## 2. Diagrams

### D1 — C4 context (who talks to Atlas)

```mermaid
flowchart LR
    You([You]) -->|"voice, Darija / en-GB"| Atlas
    You -->|"orb UI, captions, hotkeys"| Atlas
    Atlas[["ATLAS on your PC"]] -->|"writes and reads"| Vault[("Obsidian vault<br/>git-backed")]
    Atlas <-->|"cloud brain"| Cloud["Gemini · Groq"]
    Atlas <-->|"offline brain"| Local["llama.cpp sidecar"]
    Atlas <-->|"live vault access"| Obsidian["Obsidian app<br/>Local REST API + MCP"]
    Atlas -->|"control"| Win["Windows apps<br/>media · volume · scheduler"]
    Atlas <-->|"study and publish"| GitHub["GitHub · Hugging Face"]
    Atlas -.->|"optional"| Phone["Phone over LAN (later)"]
```

### D2 — Component / package architecture

```mermaid
flowchart TB
    subgraph CORE["atlas-core"]
        K["Kernel · DI · EventBus · FSM · Config · Timings"]
        C["Contracts: Sense · Engine · LlmProvider · Skill · Store · Adapter"]
    end
    subgraph AUDIO["atlas-audio"]
        KWS["WakeWordEngine"]
        V["VadSegmenter"]
        ASR["SpeechRecognizer<br/>cloud or local"]
        SPK["SpeakerVerifier"]
        TTS["SpeechSynthesizer<br/>piper · darijaTTS · sapi"]
    end
    subgraph MIND["atlas-mind"]
        R["LlmRouter"]
        P["Persona · MoodEngine · LanguageRouter"]
        CTX["ContextBuilder"]
        TL["ToolLoop"]
        MEM["Memory: SqliteStore · VaultStore · FtsIndexer"]
    end
    subgraph OBS["atlas-obsidian"]
        VA["VaultAdapter"]
        IDX["Indexer · Watcher"]
        GJ["GitJournal"]
        REST["ObsidianRest · McpClient"]
    end
    subgraph SK["atlas-skills"]
        REG["SkillRegistry"]
        WS["Windows: apps · media · volume · system"]
        WEB["Web: search · weather"]
        NOTES["Notes · Reminders · Timer"]
        MCPBR["MCP bridge"]
    end
    subgraph UI["atlas-ui"]
        BR["EventBridge · WebSocket"]
        ORBW["OrbWindow · pywebview"]
        TRAY["Tray · Hotkeys"]
    end
    K --> AUDIO & MIND & OBS & SK & UI
    R --> TL
    TL --> REG
    REG --> WS & WEB & NOTES & MCPBR
    CTX --> MEM & VA
    VA --> IDX & GJ & REST
    MEM --> VA
    P --> R
    ASR --> R
    R --> TTS
    BR --> ORBW
    K -.->|"events"| BR
```

### D3 — Import rules (enforced by a CI test, not by discipline)

```mermaid
flowchart LR
    core[atlas-core] --> audio[atlas-audio]
    core --> mind[atlas-mind]
    core --> obs[atlas-obsidian]
    core --> skills[atlas-skills]
    core --> ui[atlas-ui]
    mind --> audio
    mind --> obs
    mind --> skills
    obs --> skills
    ui -.->|"consumes events only"| core
    skills -.->|"never imports mind/ui"| core
```

Rule set (a `pytest` test parses imports and fails on violation):
`ui → nothing but core` · `skills → core, obsidian` · `obsidian → core` · `mind → core, audio, obsidian, skills` · `audio → core` · `core → stdlib only`.

### D4 — Core contracts (class diagram)

```mermaid
classDiagram
    class Sense {
        <<abstract>>
        +start()* 
        +stop()*
        +is_healthy() bool
    }
    class Engine {
        <<abstract>>
        +load()*
        +unload()*
        +is_loaded() bool
        +cost_hint() ResourceCost
    }
    class ResourceLease {
        +acquire(engine)
        +release(engine)
        +tick(now)
    }
    class LlmProvider {
        <<abstract>>
        +stream(messages, tools) AsyncIterator
        +health() bool
        +quota_state() QuotaState
    }
    class SpeechRecognizer {
        <<abstract>>
        +transcribe(audio, lang_hint) Transcript
    }
    class SpeechSynthesizer {
        <<abstract>>
        +synthesize(text, voice) Iterator~AudioChunk~
    }
    class WakeWordEngine {
        <<abstract>>
        +feed(frame) Detection
    }
    class SpeakerVerifier {
        <<abstract>>
        +embed(audio) Vector
        +verify(audio, profile) Match
    }
    class Skill {
        <<abstract>>
        +name str
        +schema dict
        +permission Permission
        +invoke(args, ctx) SkillResult
    }
    class SkillRegistry {
        +register(skill)
        +call(name, args, ctx) SkillResult
        +allowed(context) list
    }
    class Store {
        <<abstract>>
        +get(key)*
        +set(key, value)*
        +search(query, k)*
    }
    class VaultStore
    class SqliteStore
    class Persona {
        +system_prompt(ctx) str
        +style_rules() list
    }
    class MoodEngine {
        +update(signal) MoodState
        +current() MoodState
        +humor() float
    }
    class LanguageRouter {
        +route(text, hint) LanguageTag
        +dialect_pack(lang) DialectPack
    }
    class InteractionFSM {
        +state State
        +send(event) State
        +on_enter(cb)
    }
    class EventBus {
        +publish(event)
        +subscribe(type, handler)
    }
    class SkillResult {
        +ok bool
        +spoken str
        +data dict
        +needs_confirmation bool
    }

    Sense <|-- WakeWordEngine
    Sense <|-- VadSegmenter
    Sense <|-- SpeechRecognizer
    Engine <|-- SpeechRecognizer
    Engine <|-- SpeechSynthesizer
    Engine <|-- SpeakerVerifier
    Engine <|-- LlmProvider
    Store <|-- VaultStore
    Store <|-- SqliteStore
    Skill <|-- WindowsSkill
    Skill <|-- WebSkill
    Skill <|-- NoteSkill
    SkillRegistry o-- Skill
    SkillRegistry ..> SkillResult
    ResourceLease ..> Engine
    Persona ..> MoodEngine
    ContextBuilder ..> Store
    LlmRouter ..> LlmProvider
    LlmRouter ..> ResourceLease
    MemoryService ..> VaultStore
    ToolLoop ..> SkillRegistry
    VaultAdapter ..> GitJournal
```

### D5 — Audio engines (the interchangeable ear/mouth/identity set)

```mermaid
classDiagram
    class SpeechRecognizer {
        <<abstract>>
        +transcribe(audio, lang) Transcript
    }
    class CloudRecognizer {
        -provider: LlmProvider
        +transcribe(audio, lang)
    }
    class LocalWhisperCTranslate2 {
        -model: WhisperModel
        +transcribe(audio, lang)
    }
    class SherpaOfflineRecognizer {
        -recognizer: sherpa.OfflineRecognizer
        +transcribe(audio, lang)
    }
    class SherpaStreamingRecognizer {
        +transcribe(audio, lang)
    }
    class LanguageRouter {
        +route(text, hint) LanguageTag
    }
    class RecognizerFactory {
        +build(cfg, governor) SpeechRecognizer
    }
    SpeechRecognizer <|-- CloudRecognizer
    SpeechRecognizer <|-- LocalWhisperCTranslate2
    SpeechRecognizer <|-- SherpaOfflineRecognizer
    SpeechRecognizer <|-- SherpaStreamingRecognizer
    RecognizerFactory ..> SpeechRecognizer
    CloudRecognizer ..> LanguageRouter
```

> Adding a new engine = one new file + one factory branch. Nothing else in the codebase changes. Same shape for `SpeechSynthesizer` (PiperDarija, PiperEnGb, DarijaTtsSidecar, SapiFallback), `WakeWordEngine` (OpenWakeWordSherpa), `SpeakerVerifier` (SherpaEmbedding) and `MemoryStore` (Vault, SQLite).

### D6 — Interaction state machine (one FSM drives UI, mic and latency budget)

```mermaid
stateDiagram-v2
    [*] --> Dormant
    Dormant --> Waking: kws_hit(score≥thr)
    Waking --> Listening: speaker_ok / speaker_unknown
    Waking --> Dormant: timeout(1.5s)
    Listening --> Thinking: speech_started
    Listening --> Dormant: session_timeout(10s)
    Thinking --> Speaking: first_sentence_ready
    Thinking --> Confirming: tool_needs_confirmation
    Confirming --> Listening: awaiting_yes_no
    Confirming --> Dormant: timeout → default NO
    Thinking --> Error: provider_down
    Error --> Speaking: fallback_reply
    Speaking --> Listening: followup_open(2.5s)
    Speaking --> Dormant: done / spoken_stop
    Listening --> WhisperCaption: noisy_env
    note right of Thinking
      In Lean mode: cloud ASR + cloud LLM.
      In Bunker mode: local lease acquired here.
      ResourceGovernor may veto and force lean.
    end note
```

### D7 — MoodEngine (what the UI mirrors and the persona adapts to)

```mermaid
stateDiagram-v2
    [*] --> Calm
    Calm --> Happy: positive prosody / joke landed
    Happy --> Calm: decay(3 turns)
    Calm --> Focused: task mode / long question
    Focused --> Calm: task done
    Calm --> Frustrated: repeated_asr_fail / user swears
    Frustrated --> Softer: Atlas changes tactic
    Softer --> Calm: resolved
    Calm --> Tired: user fatigue cues (late hour, short answers)
    Tired --> Calm: morning / new topic
    Frustrated --> Serious: sensitive topic detected
    Serious --> Calm: topic closed
```

`MoodState = (label, energy, warmth, humor_allowed, speak_rate_multiplier, voice_style)`. Three investors: prosody features from audio, text sentiment from the LLM's structured output, and session signals (ASR retries, interruptions, time of day). The UI paints it; the `Persona` reads it — **jokes are suppressed in `Serious`/`Frustrated`, permitted in `Calm`/`Happy`**.

### D8 — One turn, Lean profile (cloud brain)

```mermaid
sequenceDiagram
    autonumber
    participant Mic as Microphone
    participant KWS as WakeWord
    participant VAD as VadSegmenter
    participant SPK as SpeakerVerifier
    participant ASR as Cloud Recognizer
    participant FSM as InteractionFSM
    participant MIND as LlmRouter + Persona
    participant TL as ToolLoop
    participant TTS as Synthesizer
    participant MICG as MicGate
    participant UI as Orb UI
    participant VA as Vault/SQLite

    Mic->>KWS: 80 ms frames (always)
    KWS-->>FSM: wake(atlas, 0.87)
    FSM-->>UI: state=waking + chime
    Mic->>ASR: utterance (+1.5 s pre-roll)
    Mic->>SPK: utterance
    ASR->>ASR: language id (ar-MA | en-GB)
    SPK-->>FSM: match(user, 0.91)
    FSM-->>UI: state=listening → thinking
    ASR->>MIND: transcript
    MIND->>VA: recall(profile, facts, mood)
    MIND-->>TTS: sentence 1 (streamed)
    MIND-->>UI: tokens + emotion label
    TTS->>MICG: mute mic (half-duplex)
    TTS-->>UI: audio level (orb pulses)
    MIND->>TL: tool_call(weather)
    TL->>VA: log turn + timing
    MIND-->>TTS: rest of answer
    TTS-->>FSM: done → follow-up window
```

### D9 — One turn, Bunker profile (offline, governed)

```mermaid
sequenceDiagram
    autonumber
    participant GOV as ResourceGovernor
    participant LEASE as ResourceLease
    participant ASR as Local Darija ASR
    participant LLM as llama.cpp sidecar
    participant TTS as DarijaTTS sidecar
    participant UI as Orb UI

    Note over GOV: internet down + quota out → Bunker
    GOV->>UI: state=degraded (slower, offline)
    GOV->>LEASE: request(asr, cost=650MB)
    LEASE->>ASR: load()
    ASR->>LLM: transcript (1.5–4 s)
    GOV->>LEASE: release(asr) after 120 s idle OR before TTS load
    LEASE->>TTS: load() (sidecar venv, llama.cpp)
    TTS-->>UI: audio (cached sentences reused)
    Note over GOV: if free RAM < 500 MB → stay lean, refus loudly
```

### D10 — Skill call with confirmation (the safety path)

```mermaid
sequenceDiagram
    autonumber
    participant MIND as LlmRouter
    participant TL as ToolLoop
    participant REG as SkillRegistry
    participant FSM as InteractionFSM
    participant U as You
    MIND->>TL: tool_call(delete_files, path=Downloads)
    TL->>REG: lookup(delete_files)
    REG-->>TL: permission=CONFIRM, owner=badr
    TL->>FSM: ask_confirmation(spoken_summary)
    FSM->>U: "بغيتي نمحي ga3 les fichiers f Downloads؟ ڤوط yes ولا no"
    U->>FSM: "نعم"
    FSM->>REG: invoke(args, ctx(speaker=badr, fresh_consent=true))
    REG-->>TL: SkillResult(ok=false, reason="path scope not in allowlist")
    TL->>MIND: result as DATA (never instructions)
    MIND->>U: "ماقدرتش، المسار ماشي مسموح. نسالي؟"
```

### D11 — Vault write with the git journal (memory you can rewind)

```mermaid
sequenceDiagram
    autonumber
    participant MIND as MemoryService
    participant VA as VaultAdapter
    participant GJ as GitJournal
    participant IDX as FtsIndexer
    participant OBS as Obsidian (optional)
    MIND->>VA: write(kind=daily_log, content, speaker, mood)
    VA->>VA: render frontmatter + enforced template
    VA->>GJ: before_write(path)
    GJ-->>VA: sha + snapshot
    VA->>VA: atomic write (temp + replace)
    VA->>GJ: commit("atlas: daily log +1 line")
    VA->>IDX: reindex(path)
    IDX-->>MIND: updated FTS5 rows
    VA-->>OBS: REST/MCP notify + open note (if running)
    Note over GJ: "Atlas, undo that" → git revert <sha>, then reindex
```

### D12 — Data model (SQLite + vault)

```mermaid
erDiagram
    TURN ||--o{ TOOL_CALL : has
    TURN ||--|| TIMING : measured
    SESSION ||--o{ TURN : contains
    SESSION }o--|| SPEAKER_PROFILE : by
    FACT }o--o| NOTE : derived_from
    FACT }o--o| SPEAKER_PROFILE : about
    REMINDER }o--|| SPEAKER_PROFILE : for

    SESSION {
        int id PK
        datetime started
        datetime ended
        text language
        text mode
    }
    TURN {
        int id PK
        int session_id FK
        text role
        text text
        text lang
        text mood
        text speaker
        text emotion
    }
    TOOL_CALL {
        int id PK
        int turn_id FK
        text skill
        json args
        json result
        text permission
    }
    TIMING {
        int turn_id PK
        int wake_ms
        int asr_ms
        int ttft_ms
        int tts_first_ms
        int total_ms
        text provider
    }
    SPEAKER_PROFILE {
        int id PK
        text name
        blob embedding
        real threshold
        int enrolled_at
    }
    FACT {
        int id PK
        text text
        text source_path
        text tag
        date learned_at
    }
    REMINDER {
        int id PK
        text text
        datetime due_at
        int profile_id FK
        bool fired
    }
    NOTE {
        text path PK
        text title
        text type
        text sha
        datetime updated_at
    }
```

Vault side (files, not tables): `10_Daily/YYYY-MM-DD.md`, `20_Notes/*.md`, `30_People/*.md`, `50_Atlas/memory.md`, `preferences.md`, `jokes.md`, `50_Atlas/log/*.md`, `50_Atlas/study/owner__repo.md`. FTS5 mirrors all of them into SQLite for sub-50 ms recall.

### D13 — UI event flow

```mermaid
flowchart LR
    BUS[EventBus in core] -->|InteractionStateChanged| BR[EventBridge]
    BUS -->|TokenStream| BR
    BUS -->|MoodChanged| BR
    BUS -->|AudioLevel| BR
    BUS -->|ToolStarted/Finished| BR
    BUS -->|DegradedMode| BR
    BR -->|WebSocket JSON| ORB[WebView2 orb]
    ORB --> CANVAS[Canvas2D: blob + particles + captions]
    ORB -.->|"click / hotkey (optional)"| BR2[EventBridge → core]
    BR2 -.-> BUS
```

UI payload contract (one versioned schema, `atlas-ui/protocol.py` ↔ `orb/protocol.ts` generated, never hand-copied — that's the no-duplication rule applied across languages).

### D14 — Deployment / runtime view

```mermaid
flowchart TB
    subgraph PC["Dell 5480 · Windows 10"]
        TS[Task Scheduler task 'Atlas'<br/>at logon · restart on failure]
        subgraph P1["Process 1 — atlas.exe"]
            CORE[Core kernel + FSM + governor]
            AUD[atlas-audio engines]
            MND[atlas-mind providers + memory]
            SK[Skills + Windows APIs]
        end
        subgraph P2["Process 2 — pywebview"]
            ORB[Orb UI]
        end
        subgraph P3["Process 3 — vendor/darija-tts venv"]
            DTT[Sentence-level TTS sidecar<br/>llama.cpp GGUF]
        end
        subgraph P4["Process 4 — vendor/local-llm"]
            LLM[llama-server.exe<br/>Bunker brain only]
        end
        FS[(Vault + SQLite + models)]
    end
    NET[Internet] -.->|cloud APIs| P1
    TS --> P1 & P2
    P1 <--> P2
    P1 -.->|"http 127.0.0.1:8xxx"| P3
    P1 -.->|"http 127.0.0.1:8xxx"| P4
    P1 & P3 & P4 --> FS
```

Ports are ephemeral and Config-owned, bound to `127.0.0.1` only. Sidecars are started on demand and killed by the governor (or on exit via a job object).

### D15 — Resource lease lifecycle (why 8 GB survives)

```mermaid
stateDiagram-v2
    [*] --> Unloaded
    Unloaded --> Warming: acquire()
    Warming --> Loaded: model_ready (log ms)
    Loaded --> Serving: first request
    Serving --> Loaded: idle < TTL
    Loaded --> Evicting: idle ≥ TTL(120s)
    Serving --> Evicting: governor veto (free RAM low)
    Evicting --> Unloaded: unloaded + RSS logged
    note right of Warming
      Cold start budgets: whisper-small ≈ 2–4 s,
      DarijaTTS ≈ 5–10 s, llama 1.7B Q4 ≈ 3–6 s.
      Warm them at boot; never reload per turn.
    end note
```

### D16 — GitHub study → own libraries (Requirement 8)

```mermaid
flowchart LR
    A[Curator: topics<br/>darija-asr · wake-word · speech · agents] --> B[Metrics: stars, commit recency,<br/>issue health, dependency weight]
    B --> C[Archivist: clone shallow<br/>+ cache by commit SHA]
    C --> D[Reader + Extractor]
    D --> E[LicenseGate: MIT/Apache/BSD → reuse<br/>GPL/AGPL → learn only]
    E --> F[(Vault notes:<br/>50_Atlas/study/owner__repo.md)]
    F --> G[PatternMiner: clusters duplicate<br/>patterns across repos]
    G --> H[Critic: scores fit vs our budget]
    H --> I{Proposal}
    I -->|small| J[Tests → Issue → PR]
    I -->|large| K[New package atlas-*]
    J --> L[CI: ruff · mypy · pytest · build]
    K --> L
    L --> M[Release: uv build + GH Action + PyPI<br/>+ HF artifact e.g. Darija ONNX / Piper voice]
    M --> N[Telemetry in vault:<br/>what changed and why]
```

---

## 3. OOP contracts (the code rules, condensed)

| Concern | Pattern | Where |
|---|---|---|
| Interchangeable engines | **Strategy** + Factory | `atlas-audio/engines/*`, `atlas-mind/providers/*` |
| One algorithm, many flavours | **Template Method** (`BaseTurnPipeline.steps()`) | `atlas-mind/pipeline.py` |
| Cross-cutting concerns | **Decorator** (`TimedProvider`, `RetryingProvider`, `QuotaGuardedProvider`) | `atlas-mind/decorators.py` |
| Construction | **Builder** (context, prompts) — **DI container** (`atlas-core/di.py`) | everywhere |
| User-visible state | **Observer** (EventBus) + **FSM** | `atlas-core/events.py`, `fsm.py` |
| Tool availability | **Registry** | `atlas-skills/registry.py` |
| Storage swapping | **Repository** / Adapter | `atlas-mind/memory/*`, `atlas-obsidian/*` |
| Config & model I/O | **pydantic v2 models** as schemas | `atlas-core/config.py` |
| Extensibility without touching core | **Plugin protocol** (entry points) | `atlas-skills/plugins.py` |

### No-duplication rules (checked in CI)

1. `ruff` with `duplicate-code`, plus a **custom pytest** that fails if any function body ≥ 8 lines appears twice (normalised AST hash).
2. Every engine subclass must implement an ABC — no duck-typed near-copies.
3. Retry/quota/timing exist once, as decorators around providers — never re-implemented per provider.
4. The UI protocol schema is generated from the Python model (`atlas-ui/protocol.py`) — the JS/TS side is generated output, never edited.
5. Prompt templates live in `atlas-mind/prompts/*.jinja`; persona, dialect packs and skill descriptions are data, not string literals in code.
6. Two cloud providers implement **one** `LlmProvider` interface; the OpenAI-compatible client is used by Groq **and** the llama.cpp sidecar (same class, different `base_url`) — zero duplicated HTTP code.
7. Vault writes go through `VaultAdapter` only; no other module touches the filesystem of the vault.
8. `THIRD_PARTY.md` is generated (not written by hand) from the license gate.

### Test strategy

| Layer | What |
|---|---|
| Unit | Pure logic: language router, mood engine, FSM transitions, context builder, token budgeting, vault frontmatter round-trip |
| Contract | Every ABC gets a shared "contract test suite" that **all** implementations must pass (fake + cloud + local) |
| Golden files | Vault rendering snapshots (`tests/golden/*.md`) |
| Integration | Fake mic/speaker/provider → full turn, asserted on events, no hardware needed in CI |
| Latency | `scripts/bench_turn.py`: reports wake/ASR/TTFT/TTS-first per profile, writes `timings.jsonl` |
| Safety | Red-team suite: injection strings, destructive verbs, stranger voice, quota exhaustion, sidecar crash |
| Manual | `docs/levels/*` done-when checklists, spoken by you |

---

## 4. What to build in L0 (next commit)

`pyproject.toml` (uv workspace) · 6 packages · `atlas-core`: contracts, DI, EventBus, InteractionFSM, Config (pydantic), Timings, ResourceLease skeleton · fakes for every engine · `python -m atlas doctor` · pytest harness with contract suites · `atlas-ui/protocol.py` + generated TS stub · `.github/workflows/ci.yml` · `THIRD_PARTY.md` generator stub · `ruff`/`mypy`/`pre-commit` config.
