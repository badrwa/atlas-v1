# atlas-mind

Atlas's thinking layer. Everything here is provider-agnostic: the router cannot
tell a cloud API from a local `llama.cpp` sidecar, and a fake from either.

| Module | What it does |
|---|---|
| `providers/openai_compatible.py` | **One** implementation serving Groq, OpenRouter, Together, HuggingFace router, Ollama and llama.cpp (different `base_url` only) |
| `providers/gemini.py` | Gemini native REST streaming (best Darija + audio, the free-tier workhorse) |
| `providers/factory.py` | Builds a provider from config — the only place that knows about `kind` |
| `router.py` | Fallback chain + decorators: quota guard → retry → timing; never emits text from two providers in one turn |
| `language.py` | `LanguageRouter`: Darija (`ar-MA`) default, British English (`en-GB`) secondary, Arabizi-aware |
| `darija.py` | Darija normalisation (script, diacritics, Arabizi mapping, user lexicon) |
| `dialects.py` | `DialectPack` data: greetings, fillers, humour register, banned words per dialect |
| `persona.py` | System prompt assembly from Jinja templates + dialect pack + preferences |
| `context.py` | `ContextBuilder` — token-budgeted context (TTFT is the thing you're protecting) |
| `chat.py` | `ChatSession` — one text turn, streamed, timed, evented |

Design rules: prompts are templates, dialects are data, and swapping a provider
is a config change (`config.toml` → `[[providers]]`).
