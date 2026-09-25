# atlas-obsidian

Atlas's second brain, as plain Markdown you can read in Obsidian.

**The rule that makes this safe:** there is exactly **one** writer
(`VaultAdapter`). Nothing else in Atlas touches vault files. Every write is
renderered from a template, obeying a folder contract, and committed to git so
`"Atlas, undo that"` actually works.

| Module | What it does |
|---|---|
| `vault.py` | `VaultAdapter` — init from template, read, append-only daily logs, create notes, archive (never delete), folder contract enforcement |
| `journal.py` | `GitJournal` — one commit per write batch, `undo_last_write()` via revert |
| `index.py` | FTS5 index over the vault (recall in <50 ms) — L6 |
| `rest.py` | Optional Local REST API / MCP client when Obsidian is running — L6 |

Folder contract: `00_Inbox` (capture) · `10_Daily` (append only) · `20_Notes`
(permanent) · `30_People` · `40_Projects` (read + append) · `50_Atlas` (Atlas's
own memory, logs, study notes) · `90_Archive` (nothing is ever deleted).
