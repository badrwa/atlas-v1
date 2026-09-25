# atlas-skills

Atlas's hands — and the gate that keeps them safe.

| Permission | Meaning | Examples |
|---|---|---|
| `SAFE` | runs immediately | weather, system stats, read your notes, open an allowlisted URL |
| `CONFIRM` | Atlas asks aloud, waits for a **fresh** yes from the **verified owner** | delete/move files, shutdown, write outside the vault |
| `BLOCKED` | never callable, no confirmation path | shell, registry edits, credential access |

Rules that are enforced in code, not in prose:

1. The model **proposes**; the registry **decides** (`SkillRegistry.call`).
2. Unknown skills and invalid arguments never execute.
3. Confirmation is fresh and single-use — a stray "yes" from a web page or a tool
   result is not consent (`SkillContext.confirmed` can only be set by the
   confirmation flow, inside the turn that asked the question).
4. `dry_run` logs instead of acting (used by tests and by first-time users).
5. Everything is written to the audit log: who, what, permission, result.
