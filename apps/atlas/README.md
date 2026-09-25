# atlas (app)

The composition root: config loading, dependency wiring, and the CLI you actually
type.

```bash
python -m atlas doctor                 # hardware, keys, vault, audio, import rules
python -m atlas providers              # what is configured (add --live to call it)
python -m atlas chat                   # text conversation (L1)
python -m atlas vault init D:\AtlasVault
python -m atlas vault status
python -m atlas vault undo
python -m atlas skills
python -m atlas ui-protocol
python -m atlas listen                 # says "arrives in L2" — no fake audio yet
```

Everything here is wiring. If logic starts accumulating in this package it
belongs in one of the packages under `packages/` instead.
