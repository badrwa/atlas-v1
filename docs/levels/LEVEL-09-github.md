# L9 — Growth (study GitHub, publish your own libraries)

**Goal:** Atlas studies open source on GitHub and *improves itself through pull requests*, and the project publishes its own reusable libraries and models — so the ecosystem feeds Atlas, and Atlas feeds the ecosystem back.
**Effort:** ~1.5 weeks to build, then a weekly rhythm. **Depends on:** L0–L7. **Gate:** 5 repos studied with real notes and 1 merged improvement; 1 package published and installable from a clean machine.

---

## Why this level exists (requirement 8, honestly framed)

An LLM scanning repos unattended and committing code is how you get a broken assistant. So Atlas's growth loop is **human-gated**: it discovers, clones, reads, extracts patterns, writes study notes, and *proposes* — you approve with one click, tests decide the rest. That is also exactly how senior engineers work, and it produces a GitHub profile you can point at.

## Deliverables

- `atlas.study` package: `Curator` · `Archivist` · `Reader` · `PatternMiner` · `LicenseGate` · `Critic` · `Proposer`.
- Vault study notes (`50_Atlas/study/owner__repo.md`) cached by commit SHA — never re-summarise the same commit.
- Proposal pipeline: `Issue → branch → tests → PR (CI) → review → merge`, with Atlas writing the PR body.
- Six publishable packages (already structured this way since L0):
  `atlas-core` · `atlas-audio` · `atlas-mind` · `atlas-obsidian` · `atlas-skills-win` · `atlas-ui-bridge`
- **HF artifacts** (high-value, and genuinely novel): CTranslate2/ONNX conversion of a Darija Whisper checkpoint, and (stretch) a **Piper Darija voice** trained on open Darija data with a model card.
- A skills registry (`atlas-skills` marketplace stub): `skills.yaml` catalogs installable skill packs by permission level.

## Steps

1. **`Curator`** — discovery queries GitHub search API (topics: `darija-asr`, `moroccan-arabic`, `wake-word`, `keyword-spotting`, `speech-to-text cpu`, `local-llm`, `mcp-server`, `voice-assistant`), ranks by stars, recency, issue health, licence, and **dependency weight** (a repo requiring CUDA is useless to you — rank it down automatically).
2. **`Archivist`** — shallow clone (`depth=1`) into `%LOCALAPPDATA%\atlas\study\<owner>__<repo>\<sha>`, cache by SHA, size cap, auto-clean older than 90 days.
3. **`Reader`** — walks the repo with a token budget: README, `pyproject`/`requirements`, entry points, key modules (by import graph centrality), tests, CI config, and the top 20 open issues. Produces a structured digest.
4. **`PatternMiner`** — compares digests across repos and across *your own* code to find: proven patterns worth adopting, duplicated concepts in your codebase, and cost hazards (heavy deps). Cross-signal with your `timings.jsonl` (e.g. "this queue design would cut our ASR latency") — that's where real value comes from.
5. **`LicenseGate`** — allowlist MIT/Apache-2.0/BSD/ISC for *code reuse*; everything else = read-and-learn only. Writes `THIRD_PARTY.md` entries automatically (name, version, licence, URL, how used). **No code enters the repo without a gate pass.**
6. **`Critic`** — scores each proposal: expected benefit vs RAM/CPU/latency cost on an i5-6200U, maintenance burden, and dependency risk. Reject anything that adds a resident model, torch, or a GPU assumption — those are automatic vetoes.
7. **`Proposer`** — two outputs:
   - small (≤ 200 LOC, no new deps) → generates a failing test first, then the code, opens a PR with the reasoning and measurements;
   - large → opens an **issue** with a design sketch and waits for you.
   Never auto-merges. CI (ruff/mypy/pytest + measured latency budget) decides whether the PR is even reviewable.
8. **Weekly study rhythm.** A scheduled task (L10) runs the loop Friday evening and produces `50_Atlas/study/weekly-YYYY-WW.md`: what it read, what it learned, what it proposes. You review for 15 minutes with your tea. This is the whole point of the level.
9. **Publish the libraries.** For each package: README with a 10-line quickstart, typed public API, semantic version, `CHANGELOG.md`, `uv build`, GitHub Action publishing to PyPI on tag (trusted publishing, no tokens in the repo), and a `docs/` page. Start with `atlas-core` (contracts) and `atlas-obsidian` (vault adapter) — those are useful to other people independent of Atlas.
10. **HF artifacts.** `atlas-convert` CLI: HF Whisper → CTranslate2 INT8, and (stretch) Qwen3-ASR → ONNX for sherpa-onnx. Publish with a proper model card (source, licence, conversion params, benchmark WER on a held-out Darija sample, and honest limitations). Then the Piper Darija voice project: collect open Darija speech, train, evaluate with natives, publish voice + card.
11. **Contribute upstream.** The highest-signal outcome: bug reports and small PRs back to the projects Atlas depends on (sherpa-onnx, faster-whisper, openWakeWord, AtlasIA's MoulSot repo). A repo that studies others and gives back gets taken seriously.
12. **Guardrails.** Rate limits respected (GitHub API), no scraping of private repos, no secrets in study notes, and a hard cap of 3 open PRs/2 open issues at a time so Atlas isn't the spammer of the neighbourhood.

## Classes

`StudyScheduler` · `Curator` (queries + ranking) · `RepoMetrics` · `Archivist` · `RepoDigest` · `Reader` · `PatternMiner` · `PatternCandidate` · `LicenseGate` · `ThirdPartyRegistry` · `Critic` · `Proposal` · `Proposer` · `StudyNoteRenderer` · `ConversionTool` (HF→CT2/ONNX) · `ReleaseTool`.

## Tests

- Curator: fixture API responses → ranking deterministic; CUDA-only repos ranked down.
- LicenseGate: 12 licence fixtures → correct allow/deny; `THIRD_PARTY.md` regenerated deterministically.
- Archivist: SHA cache hit does not re-clone or re-summarise.
- Critic: proposals adding torch/GPU/7B models are vetoed with a reason.
- Proposer: generated patch has a test that fails before and passes after (run in a scratch repo).
- Publish dry-run: `uv build` + `twine check` on each package; install into a clean venv and import it.
- Note rendering: golden study notes.
- Guardrails: PR/issue caps enforced; rate-limit backoff works.

## Done when

- 5 repos studied with substantive vault notes (each with "what we could use / what we must avoid").
- At least **1 merged improvement** in Atlas that came from a study note, with measurements showing it helped (or honestly showing it didn't).
- `pip install atlas-obsidian` (or `atlas-core`) works on a clean machine from GitHub/PyPI, CI badge green.
- One HF artifact published with a model card and a measured baseline.
- Weekly study note produced automatically and reviewed by you once.

## Pitfalls

- ❌ Unattended auto-merge. Ever.
- ❌ "Adopt this framework" proposals that add 300 MB of dependencies → Critic vetoes on cost, not on novelty.
- ❌ Re-summarising the same repo at the same commit → SHA cache.
- ❌ Copying code with an incompatible licence → the gate is a build failure, not a suggestion.
- ⚠️ GitHub API rate limits (5,000/h authenticated) — batch and back off; use a token with minimal scopes.
- ⚠️ Publishing quality beats publishing quantity: one well-documented package someone actually installs is worth more than six empty ones.
