# L4 — Identity (it knows it's you)

**Goal:** Atlas recognises **your** voice, greets you like it knows you, and locks private behaviour when someone else is talking.
**Effort:** 3–4 evenings. **Depends on:** L2 (L3 recommended). **Gate:** your voice accepted ≥ 19/20 utterances; a stranger's voice blocked from memory/vault/PC control; no false accept in a 30-minute two-person test.

---

> **Implementation notes: [`NOTES-L04.md`](NOTES-L04.md)** — built and
> sandbox-verified (`pytest` 581 passed, ruff clean, mypy 94 files). Implemented:
> `atlas_core/identity.py` (the whole policy layer), `atlas_audio/speaker.py`
> (`SherpaSpeakerVerifier` + `EnrollmentSession`), the capability gate in
> `SkillRegistry`, person notes in the vault, the audience block in the persona,
> `atlas identity status|enrol|verify|forget|log`, identity rows in `doctor`, and
> `scripts/bench_speaker.py` (FAR/FRR sweep). Eight bugs the tests caught are in
> §4 of the notes — the two interesting ones: a guest could never actually write
> to their own note (a config-key/path mix-up), and a stale `owner=True` flag
> could outvote the verdict in the audit trail.
> **Still the laptop's job (§5):** download the speaker ONNX, enrol your voice,
> run the sweep on your own clips, and do the 30-minute two-person test. The
> threshold in `config.toml` stays a guess until then — the plan says FAR ≈ 0
> even if FRR rises, and that number can only come from your microphone.

## Built differently from the draft (and why)

| Draft | Shipped | Why |
|---|---|---|
| `guard.effective_permissions(match, confidence, mood)` | `guard.verify(embedding, utterance_ms=…)` → one `Permissions`, plus `effective_permissions()` for an already-computed match | the guard owns the *comparison* too, so there is one place where a score becomes a capability set |
| cosine vs 3 embeddings + centroid | best match over a rolling window of 6 | a tired voice should still match the sample from a good day |
| embeddings + centroid in SQLite | embeddings only, float32 blobs, one owner | a centroid is derivable; storing it is a second thing that can leak |
| vault `30_People/*.md` per person | same, human-readable only, and `ProfileRepository` **refuses** to live inside the vault | the vault is git-journaled and synced |
| restricted mode = unknown voice | restricted + `utterance_too_short` + `verifier_unavailable` + `no_profiles` | every path that is not a clean owner match must be *named* in the log |
| `speaker_log.jsonl` with scores | scores + threshold + reason + mood + ms, never a vector | the sweep needs the threshold that was in force at the time |

---

## Deliverables

- `SpeakerVerifier` (sherpa-onnx speaker embedding model, e.g. an ERes2Net/3D-Speaker ONNX) with enrolment, verification and score logging.
- `SpeakerProfiles` store (SQLite + vault `30_People/`).
- Enrolment flow by voice ("Atlas, hadi hyati / this is my voice" → 3 × 10 s samples).
- **Restricted mode** for unknown voices, enforced in one place (`ContextGuard`).
- Speaker-aware memory: facts are attributed to a person (`30_People/*.md`).

## Design contract

Verification is a **gate on capability**, not a per-word check:

| Capability | Owner (you) | Known other | Stranger (restricted) |
|---|---|---|---|
| General chat, weather, web search | ✅ | ✅ | ✅ |
| Read personal memory / daily notes | ✅ | ✅ (if enrolled) | ❌ |
| Write to vault | ✅ | ✅ (own `30_People` + inbox only) | ❌ |
| PC control (apps, volume, files) | ✅ | ⚠️ SAFE-only, never CONFIRM-free | ❌ |
| Destructive / irreversible | ✅ with confirmation | ❌ | ❌ |

Everything else stays behind the `SkillRegistry` permission model (L7) — the speaker gate shortens the allowlist, it doesn't replace it.

---

## Steps

1. **Pick the embedding model.** sherpa-onnx ships speaker-ID/verification with ONNX embedding extractors; download one, verify it loads in ~50–100 MB RSS and embeds a 2 s clip in < 60 ms.
2. **Enrolment UX.**
   - Trigger: `"Atlas, sjel sowti"` / `"Atlas, enrol my voice"`, or the doctor CLI.
   - 3 sessions × 10 s, read prompts varied in loudness/pace; trim silence; require ≥ 2 s of speech per clip.
   - Store 3 embeddings per person + their centroid in SQLite (`SPEAKER_PROFILE`) and export a human-readable note `30_People/<name>.md` (embedding stays in SQLite — never in the vault, never in git).
   - Quality gate: pairwise cosine between the 3 samples must be ≥ a floor, else ask for a quieter room and repeat.
3. **Verification at runtime.** Embed the utterance (from the pre-roll + utterance, ~2–3 s), cosine vs profiles, accept the best above `threshold` (start 0.65, tune on your own data). Store per-utterance scores → `speaker_log.jsonl` for honest tuning.
4. **Tune with data, not vibes.** Record 30 genuine utterances and 30 non-genuine (TV, friend, phone call, music). Compute FAR/FRR across thresholds; pick the threshold where **FAR ≈ 0** even if FRR rises — a repeat request is cheaper than a leak.
5. **`ContextGuard`** — one class, one decision point:
   ```python
   guard.effective_permissions(speaker_match, confidence, mood) -> Permissions
   ```
   `ContextGuard` is queried by `ToolLoop` and `MemoryService`; no other component makes access decisions (R5: one implementation of the rule).
6. **Restricted mode behaviour.** Unknown voice: Atlas answers general questions in the right language, and if the request is private it says so plainly — *"سمح ليا، هادشي خاص بصاحبي."* / *"Sorry, that's personal — only for my owner."* Polite, brief, no data leakage, no hint of what exists.
7. **Personalised greeting + memory attribution.** On a verified first wake of the day: greet by name, mention one relevant thing (today's first calendar/reminder, or a running gag from `jokes.md`). Facts learned while person X spoke go to `30_People/X.md`.
8. **Drift handling.** Voices change (cold, tired, microphone moved). Keep a rolling window: if a genuine speaker gets rejected twice in one session, offer a 5 s re-enrolment ("sowtek tbdel chwiya, 3awd sjel?").
9. **Privacy rules.** Embeddings are biometric data: store locally only, never send to a provider, exclude from git (`.gitignore` for `*.db`), offer `"Atlas, msa7 les données dyal sowti"` → wipe profiles and confirm.
10. **Verification cost check.** Embedding + cosine must add < 100 ms and < 120 MB; measure and record.

## Classes

`SpeakerVerifier` (ABC) · `SherpaSpeakerVerifier` · `EnrollmentSession` · `SpeakerProfile` · `SpeakerMatch` · `ProfileRepository` · `ContextGuard` · `Permissions` · `SpeakerLog`.

## Tests

- Contract suite: `embed()` shape/determinism; `verify()` ordering on fixture clips.
- Threshold sweep test with the recorded fixtures → asserts FAR ≤ target at the chosen threshold.
- `ContextGuard` truth table: every (speaker × capability) cell asserted — this is the safety test, keep it exhaustive.
- Restricted mode integration: stranger utterance asking for memory → refused, and **no personal data appears in any emitted event or log line**.
- Enrolment: bad audio (silence, noise, 1 clip only) → rejected with the right spoken message.
- Privacy: assert embeddings never appear in vault files, git diffs, or provider payloads (snapshot test).

## Done when

- You: ≥ 19/20 utterances accepted; greetings feel personal.
- 30-minute two-person test: stranger fully blocked from private capabilities, zero false accepts, and you can still talk over them without Atlas switching identity.
- `NOTES-L04.md` documents the threshold, FAR/FRR numbers and your enrolment clips' quality.

## Pitfalls

- ❌ Speaker ID as the *only* security boundary → it's a strong gate, not a biometric lock; destructive actions still need confirmation.
- ❌ Promising "voiceprint security" in the README → be precise: "local voice profile that reduces accidental access; not a cryptographic identity".
- ❌ Storing embeddings in the vault (Obsidian + git = leakage) → SQLite only, gitignored.
- ❌ Re-enrolling from the same 3 clips forever → re-enrol periodically, and after a mic/room change.
- ⚠️ Background TV can partly match you → require a minimum utterance length before trusting a match, and always log the score.
