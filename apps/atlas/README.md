# atlas-app

The command line you actually type. Everything here is a thin shell over the
packages — if a command needs logic, that logic belongs in a package where tests
can reach it.

```bash
python -m atlas doctor          # what works on this machine, honestly
python -m atlas providers       # configured providers, keys, fallback order
python -m atlas chat            # talk to Atlas (streaming, Darija by default)
python -m atlas vault init ~/AtlasVault
python -m atlas skills          # what Atlas can do, and what needs confirmation
python -m atlas ui-protocol     # regenerate the UI TypeScript contract
```

Global flags: `--config PATH` (or `ATLAS_CONFIG`), `--verbose`.

`doctor` exits 1 only when something is actually broken. Warnings are things you
can run without (no microphone yet, no vault configured, a key that looks wrong).
