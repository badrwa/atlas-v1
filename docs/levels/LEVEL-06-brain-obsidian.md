# L6 — Second brain (Obsidian vault + retrieval)

**Goal:** Atlas's memory lives in an **Obsidian vault** as plain Markdown, indexed by SQLite FTS5 for instant recall, written through one adapter, git-committed so every change is reversible, and readable live by the Obsidian app via REST/MCP.
**Effort:** ~1.5 weeks. **Depends on:** L1 (persona), L4 (attribution). **Gate:** *"Atlas, dir note: …"* → note in the inbox → searchable in < 50 ms → `"undo that"` reverts it. Answer quality visibly improves with vault context.

---

## Deliverables

- `VaultAdapter` (the **only** writer), `VaultSchema` (folder contract), note templates, frontmatter handling with pydantic round-trip.
- `FtsIndexer` + `Watcher` (watchdog, incremental) → FTS5 tables for notes, facts, people, logs.
- `MemoryService`: remember / recall / forget / summarise / undo, with a token budget.
- `GitJournal`: auto-commit per write batch, rollback by voice, `.gitignore` for private files.
- `ObsidianRest` + `McpClient` (optional accelerators when the app is running).
- Copy of `vault-template/` deployed to the vault path by `atlas vault init`.

## Vault contract (from `vault-template/`)

| Folder | Atlas may | Rule |
|---|---|---|
| `00_Inbox/` | create | unsorted capture, never overwritten |
| `10_Daily/` | append | one file per day, voice journal + tasks, never rewrite the past |
| `20_Notes/` | create/update | permanent notes, one idea each, ≥ 1 link |
| `30_People/` | create/update | relationship memory, per enrolled speaker |
| `40_Projects/` | read; append only | your work, Atlas doesn't reorganise |
| `50_Atlas/` | full control | `memory.md`, `preferences.md`, `jokes.md`, `log/`, `study/` |
| `90_Archive/` | move-to only | nothing is ever deleted |

Frontmatter on everything Atlas writes: `type`, `tags`, `created`, `atlas: managed`, optional `atlas: private` (excluded from git). Backlinks (`[[…]]`) are mandatory in `20_Notes` — link hygiene is what makes it a *brain* instead of a pile.

## Retrieval strategy (why FTS5 and not a vector DB)

You have ~2 GB of headroom. An embedding model + vector store costs RAM and cold-start time for marginal gains on a personal vault. **FTS5 + backlink expansion + recency + people-filtering** gets you most of the way for ~0 MB, and the LLM does the synthesis. Design:

1. Query expansion: strip Darija diacritics, map Arabizi, expand with the lexicon file, add synonyms from `50_Atlas/preferences.md`.
2. Search FTS5 across notes/facts/logs → top N by BM25, boosted by recency (half-life ~30 days) and by a matching file in `30_People/` when the speaker is identified.
3. Backlink expansion: pull the 1-hop linked notes' titles (cheap context, big quality win).
4. Token-budgeted assembly: ≤ 400 tokens of recalled memory per turn, tagged with source paths (so Atlas can cite: *"f note dyal 12 mars…"*).
5. If a query is a *question about you*, prefer `memory.md` + people notes + recent daily notes.

## Steps

1. **`atlas vault init <path>`** — copy `vault-template/`, save the path in config, `git init`, initial commit, verify writability, add `.obsidian/` exclusions.
2. **`VaultSchema` + `Note` models.** Parse/render frontmatter with `python-frontmatter` + pydantic; round-trip tests must be lossless (this is where "Atlas mutated my note" bugs are born).
3. **`VaultAdapter`** — the single write path: `read`, `append`, `create`, `update_section` (by heading), `move_to_archive`, `link`, `search`. Atomic writes (temp + `os.replace`), UTF-8, preserve line endings, never touch files outside the contract.
4. **Note renderers** — one per kind (`DailyLogRenderer`, `PermanentNoteRenderer`, `PersonRenderer`, `MemoryRenderer`). Templates live in `atlas-obsidian/templates/*.jinja`; **no string-building of Markdown anywhere else** (R5).
5. **`GitJournal`** — before each write: record `sha`; after a batch: `git add -A && git commit -m "atlas(<skill>): <summary>"`. `"Atlas, undo that"` → `git revert --no-edit <sha>` (or `git checkout` for uncommitted), then reindex. Log every commit in `50_Atlas/log/`.
6. **`FtsIndexer`** — SQLite FTS5 tables (`notes`, `facts`, `people`, `logs`) with the vault path as the key; incremental updates from `Watcher` events; full rebuild command (`atlas vault reindex`); skip `atlas: private` files if configured.
7. **`Watcher`** — `watchdog` observer on the vault; debounce 400 ms; ignore `.obsidian/`, `.git/`, temp files; on external edit, reindex and (optionally) tell Atlas its memory changed.
8. **`MemoryService`** — the API the brain uses:
   `remember(text, kind, person)` · `recall(query, budget_tokens, person)` · `forget(path|query)` (→ archive, never delete) · `summarise_session(turns)` · `undo_last_write()`. All of it speaks Darija back: *"mzyan، dert note."*
9. **Voice capture flows** (implement across L6/L7, tested here):
   - quick capture → `00_Inbox/`
   - *"remember that…"* → append to `50_Atlas/memory.md` (person-attributed if someone else's fact)
   - *"note this idea"* → `20_Notes/` with generated title + one link
   - *"how was my day?"* → read today's daily note + reminders
   - *"what do I know about X?"* → recall + backlinks → spoken answer with source
10. **Live integration (optional).** `ObsidianRest`: check `127.0.0.1:27124`, use the API key; `search/simple`, `open/<path>` (open the note in the UI when the user asks — a genuinely delightful touch), `commands/<id>`. `McpClient` for the plugin's MCP endpoint (`/mcp/`), so Atlas can also *consume other MCP servers* later (L9). Both must degrade silently when Obsidian is closed.
11. **Privacy.** `atlas: private` notes excluded from git (via `.gitignore` or a pre-commit filter); a `private_mode` config that keeps a session's log out of the vault entirely; doctor check that no API keys ever land inside the vault.
12. **Quality feedback loop.** Whenever Atlas cites a note, log it. Weekly summary in `50_Atlas/log/` of what it remembered well and what it missed — the input for improving query expansion.

## Classes

`VaultAdapter` · `VaultSchema` · `Note` (pydantic) · `Frontmatter` · `NoteRenderer` (+4 subclasses) · `TemplateEngine` · `GitJournal` · `FtsIndexer` · `Watcher` · `MemoryService` · `RecallQuery` · `RecallResult` · `ContextBudget` · `ObsidianRest` · `McpClient` · `VaultDoctor`.

## Tests

- Frontmatter round-trip: 30 real-world files (yours, messy) → parse → render → byte-identical.
- Contract: `VaultStore` and `SqliteStore` both pass the `Store` contract suite.
- Atomicity: interrupt a write (injected failure) → file unchanged, no partial Markdown.
- Git journal: write → commit exists → `undo_last_write()` → content restored, index consistent.
- FTS: recall ranking tests on a fixture vault (Darija + English queries, Arabizi input).
- Watcher: external edit → reindexed within 1 s; no infinite loop from Atlas's own writes.
- Budget: assembly never exceeds the token cap; sources included.
- Privacy: `private` notes never appear in git status; embeddings/keys never in vault.
- MCP/REST: mocked server → tool calls issued correctly; server absent → silent no-op.

## Done when

- `"Atlas, dir note: l'idée dyal l'app l'jadida"` → note in `00_Inbox/` with frontmatter, committed, and `"undo that"` reverts it.
- `"Atlas, chno 3reft 3la Projet X?"` → a spoken answer drawn from your own notes, citing paths.
- Recall p50 < 50 ms for a 2,000-note vault; context assembly < 400 tokens.
- Observations from an experiment: answer quality with vault context is measurably better than without (keep 10 questions, compare with/without recall — write the result in `NOTES-L06.md`).

## Pitfalls

- ❌ Writing Markdown from more than one module → one adapter, one renderer per kind. Non-negotiable.
- ❌ Rewriting whole daily notes to append a line → append-only edits for logs; `update_section` only for sections Atlas owns.
- ❌ Deleting anything → archive with a date prefix, always.
- ❌ Indexing the whole vault into the prompt → budget it, or you'll pay for it in TTFT.
- ❌ Assuming Obsidian is running → it's an accelerator, not a dependency.
- ⚠️ Obsidian Sync/iCloud duplicating your vault → keep the Atlas vault in a plain folder (ideally under git only) to avoid duplicate-file chaos.
- ⚠️ Vault growth: monthly job that summarises old daily notes into `90_Archive/` summaries and trims `memory.md` to a hard line cap.
