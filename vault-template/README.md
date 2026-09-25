# AtlasVault — the template

This is the **structure** of Atlas's second brain. Copy it to where the vault will live, e.g.:

```
robocopy vault-template D:\AtlasVault /E
cd D:\AtlasVault
git init && git add -A && git commit -m "Atlas vault: initial structure"
```

Then open `D:\AtlasVault` in Obsidian as a vault, and point Atlas at it in `config.toml` → `[obsidian] vault_path`.

## Why git?
Every note Atlas writes is committed automatically (`GitJournal`). That makes Atlas's memory **auditable and reversible**:
*"Atlas, undo your last note change"* → `git revert`. No other assistant feature is worth more than this one.

## Optional (recommended) Obsidian plugin
`Local REST API with MCP` (coddingtonbear) — v5+ serves an MCP server at `https://127.0.0.1:27124/mcp/`.
It lets Atlas search the live vault, run Obsidian commands and open notes in your UI while Obsidian is running.
Atlas works off the filesystem when Obsidian is closed — the plugin is an enhancement, never a dependency.

## Folder contract (Atlas obeys this)
| Folder | Atlas may | Notes |
|---|---|---|
| `00_Inbox/` | create | quick capture, unsorted |
| `10_Daily/` | append | voice journal, tasks as `- [ ]`, never rewrite past entries |
| `20_Notes/` | create/update | permanent notes (Zettelkasten style), one idea per note |
| `30_People/` | create/update | relationship memory (only for enrolled speakers) |
| `40_Projects/` | read; append only | your work, Atlas doesn't reorganise it |
| `50_Atlas/` | full control | Atlas's own memory, logs, jokes, study notes |
| `90_Archive/` | move-to only | nothing is ever deleted, only archived with a date prefix |
