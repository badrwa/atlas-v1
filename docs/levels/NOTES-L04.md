# NOTES-L04 — Identity (built, verified, and what the laptop still has to prove)

**Status:** implemented, tested, committed; sandbox-verified. The two things that
need *your* machine and *your* voice are marked **[laptop]** and listed in §5.

**Gate (LEVEL-04):** your voice accepted ≥ 19/20 utterances; a stranger blocked
from memory/vault/PC control; no false accept in a 30-minute two-person test.

---

## 1. What was built

| File | What it is | Left there already |
|---|---|---|
| `packages/atlas-core/src/atlas_core/identity.py` | the whole policy layer: `Permissions`, `Audience`, `SpeakerProfile`, `ProfileRepository`, `SpeakerLog`, `ContextGuard`, `DailyGreeter` | — |
| `packages/atlas-audio/src/atlas_audio/speaker.py` | `SherpaSpeakerVerifier` (ERes2Net ONNX), `EnrollmentSession`, `build_verifier/build_identity/speaker_status` | — |
| `packages/atlas-core/src/atlas_core/contracts.py` | `Capability` enum; `SkillContext.subject`/`permissions`; `Skill.capability` | it now fills `speaker/subject/owner` **from the verdict** |
| `packages/atlas-skills/src/atlas_skills/registry.py` | the capability gate + a verdict-based refusal | denied calls are audited with the refusal the person heard |
| `packages/atlas-skills/src/atlas_skills/packs/system.py` | capabilities on all four skills; `remember` writes to *their* note | — |
| `packages/atlas-obsidian/src/atlas_obsidian/vault.py` | `people/ensure_person/append_person_fact/read_person` | fixed a key-vs-folder bug (see §4, #3) |
| `packages/atlas-mind/.../persona.jinja` + `persona.py` + `chat.py` | the audience block, `ChatSession.send(..., audience=)` | — |
| `apps/atlas/src/atlas/cli.py` | `atlas identity status/enrol/verify/forget/log`; the live loop's verdict; the greeting | — |
| `apps/atlas/src/atlas/doctor.py` | `identity`, `verifier`, `profiles`, `identity · trust` rows | `atlas-audio/engines.py` (the L4 stub file) was **deleted** |
| `scripts/bench_speaker.py` | FAR/FRR sweep over your own clips; `--from-log` | — |

**Permissions, the one rule.** `Capability` is the vocabulary; `Permissions.allows`
is the only place that decides. Owner = all seven capabilities; known other =
`{general, own_notes}` (+ `read_memory`, + `pc_control` only if you opt in with
`guest_pc_control`); stranger = `{general}`. `WRITE_VAULT` is allowed exactly when
`OWN_NOTES` is and a `subject` exists — so "a guest can write" always means "a
guest can write to *their own* note".

## 2. Decisions worth remembering

1. **Identity is one gate per utterance, not a per-word guess.** The loop embeds
   once, compares, and hands one `Permissions` to everyone else (turn record, tool
   loop, memory reader, prompt). Nothing re-derives identity downstream.
2. **A short utterance never carries an owner-only decision.** A 0.4 s match is
   still *you* for conversation, but `read_memory`/`pc_control`/`destructive` are
   held back (`reason = utterance_too_short`). The television says one word too.
3. **A verifier that cannot run is logged, never silent.** `guard.unavailable()`
   returns owner capabilities *and* writes `verifier_unavailable` /
   `embed_too_short` to `data/speaker_log.jsonl`. A machine without the model is
   the L3 machine — but the log says so, every turn.
4. **The refusal follows the verdict, not a flag.** `SkillContext.__post_init__`
   overwrites a stale `owner=True` with `permissions.owner`, and the registry's
   refusal sentence is chosen from `permissions.restricted`. Two tests exist only
   because a half-wired caller is the realistic bug.
5. **Two guards around memory, on purpose.** The prompt tells the model it is
   talking to a guest *and* the caller withholds the memory string (`_memory_for`
   returns `""`). A prompt-only guard is one hallucination away from a leak.
6. **Embeddings live in SQLite, and nowhere else** — not in the vault, not in the
   log, not in `as_dict()`, not in the fingerprint. `ProfileRepository` refuses a
   path inside the vault so this cannot be undone by a config change.
7. **Memory travels in the system prompt**, not as a second user message
   (`ChatSession.stream` passes `memory=""` to `ContextBuilder` and says why).
   One home for it, one budget charge.
8. **A rejection is logged under the *closest profile's name* with
   `accepted=false`.** The log never claims someone spoke; it records what the
   score was near. `atlas identity log` prints `≈badr` to keep that visible.
9. **Darija first, everywhere.** Refusal, enrolment prompts, low-quality warning,
   greeting and drift re-enrol offer all exist in Darija and en-GB; the enrolment
   prompts are Latin-letter Darija because that is how it is read aloud.

## 3. Sandbox verification (no model, no microphone — which is the point)

| Check | Result |
|---|---|
| `pytest` | **581 passed** (477 → 581: +104 L4 tests) |
| `ruff check .` | clean |
| `mypy packages apps` | **94 source files**, no issues |
| `check_import_rules` / `check_no_torch` / `check_duplicates` | ✓ (no torch in the core venv) |
| `atlas identity status` | 4 rows + `nobody enrolled` + the "convenience gate, not a cryptographic identity" note |
| `atlas identity enrol` (no model) | exit 1, "no speaker model · pip install 'atlas-audio[local]'", hint names `models/speaker/…` |
| `atlas identity forget --all` | exit 1, "confirmation", nothing deleted; `--yes` erases and says the vault notes stay |
| `atlas listen --status` | ear report + identity rows (`! nobody enrolled`, `! verifier`) |
| `bench_speaker.py --clips` (empty/missing) | exit 1 with the clip-layout instructions; `--from-log` reads the log and names the limit |
| sweep maths (synthetic) | genuine 0.91–0.64 / impostor 0.55–0.62 → recommends 0.65, FAR 0.00, FRR 0.20 |

Truth table tests are exhaustive by construction:
`test_every_capability_for_every_kind_of_speaker` walks the whole matrix from one
`EXPECTED` dict, so a widened permission fails a test instead of shipping.

## 4. Bugs the tests caught (each one would have shipped silently)

1. **`ContextGuard.unavailable()` did not exist.** The loop plan called it; without
   it, a missing model would have raised at the first utterance — or worse, been
   caught somewhere and treated as "no identity".
2. **`SkillContext.owner` could lie.** The audit trail recorded `owner=True` for a
   restricted voice, and the refusal came out in English. Now the verdict wins
   (`__post_init__`) and the wording follows `permissions.restricted`.
3. **`VaultAdapter.people_folder` returned a folder name where a *config key* was
   expected** → `AttributeError: 'ObsidianSection' object has no attribute
   'folder_30_People'`. Guest facts had *never* worked end to end. Keys and paths
   are now two properties (`people_folder` / `people_dir`) with a comment.
4. **The persona audience block never rendered in Darija**: it tested
   `language_label == "Darija (ar-MA)"` — the real label is "Moroccan Darija", and
   `code` was not in the render context. Now the template switches on `code`.
5. **`IdentityConfig` had no `model_path`**, so `[identity] model_path` had no
   reader and `build_verifier` read the raw TOML section instead. One reader now.
6. **The doctor's identity rows overlapped** (two rows both titled `identity`);
   the threshold row is `identity · trust`.
7. **`atlas identity enrol` demanded a model before noticing a missing name** —
   the user's typo is cheaper to check than a model load. Name first.
8. **`.sqlite3-wal` / `-shm` side files were not ignored**; a WAL file can contain
   an embedding. Added to `.gitignore` next to the existing `*.db-wal` rules.

## 5. What the laptop must prove **[laptop]**

Do these in order; nothing here needs the cloud.

```powershell
# 1. model (~50–100 MB). Any 3D-Speaker/ERes2Net speaker-embedding ONNX works;
#    the config default names this one:
#    https://github.com/k2-fsa/sherpa-onnx/releases  →  speaker-recognition models
#    put it at: models\speaker\3dspeaker_speech_eres2net_base.onnx
pip install -e "packages/atlas-audio[local]"      # sherpa-onnx + numpy

# 2. does the machine see it?
atlas identity status        # verifier row must be OK, profiles row "0 people"

# 3. enrol yourself (3 samples × 10 s, quiet room, same mic, normal voice)
atlas identity enrol --name badr --owner
#    → "sample 1 · 3.4s of speech · agreement 0.93 …" then "enrolled · quality 0.9x"
#    bad audio is refused with the Darija line, and nothing is stored

# 4. does it recognise you?  record 5 clips, verify each
atlas identity verify data\speaker_clips\badr\01.wav     # ✓ badr  0.9x
#    then 5 clips of someone else (or the TV) → must print · (below threshold)

# 5. tune the threshold on your own data, not on vibes
#    data\speaker_clips\badr\*.wav   (4–5 clips)
#    data\speaker_clips\said\*.wav   (2 clips is enough)
#    data\speaker_clips\tv\*.wav     (record the TV: the honest impostor)
python scripts\bench_speaker.py --out data\speaker_sweep.json --markdown data\speaker_sweep.md
#    → paste the table + recommended threshold into §6 below, then set it:
#      [identity] threshold = 0.xx

# 6. the real test: 30 minutes, two people
atlas listen
#    - you: ≥ 19/20 utterances accepted (gate)
#    - stranger: ask for "شنو كنت كتدير البارح؟" / "what did I do yesterday?"
#      → refusal in Darija, and NOTHING personal in the reply
#    - stranger: "shutdown the pc" / "open my notes" → refused
#    - zero false accepts: the log shows no accepted utterance that was not yours
atlas identity log                 # every score, with ≈ for near-misses
```

Paste your numbers below (this is the part of the gate only a human can satisfy).

## 6. [laptop] results

```
$ atlas identity status
(paste)

$ atlas identity enrol --name badr --owner
(paste the sample lines + the enrolled line)

$ python scripts/bench_speaker.py
(paste the table; FAR/FRR columns matter more than the threshold)

$ atlas identity log     (after the 30-minute two-person test)
(paste the summary: n, p50, max, accept rate)

accepted:      __ / 20    (gate: ≥ 19)
false accepts: __         (gate: 0)
stranger requests refused: __ / __
verdict: (your words)
```

## 7. Known limits — say these out loud, do not paper over them

- **A voice profile is a convenience gate, not a cryptographic identity.** A
  recording of you can pass it. Anything destructive still needs confirmation
  (`Permission.CONFIRM`), and `shutdown_pc` is owner-only *and* confirmed.
- **Background TV can partly match.** That is what `trust_min_ms` is for: a short
  match keeps general capabilities only. Watch the log for `utterance_too_short`.
- **`--from-log` cannot compute FAR.** `data/speaker_log.jsonl` records the
  *decision*, not the label; only clips of a known second voice can measure a
  false accept. `bench_speaker.py --clips` is the honest measurement.
- **The verifier is embed-only in v1**: no ageing model, no per-mic calibration.
  Voices drift — `drift_rejections = 2` triggers the re-enrolment offer, and a
  later re-enrol replaces the profile (the window keeps the last 6 samples).
- **One owner only.** Enrolling a second person as owner demotes the first; that
  is deliberate, and `ProfileRepository` enforces it in one place.
- **Embedding cost on the target laptop is unmeasured [laptop].** Budget is
  < 100 ms and < 120 MB RSS; `identity_ms` is recorded per turn (`atlas listen`
  snapshot) so §6 can report the real number.
