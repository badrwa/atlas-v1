# atlas-audio

Atlas's ears and mouth. **Status: device plumbing only — engines land in L2/L3.**

| Module | State | What it will hold |
|---|---|---|
| `devices.py` | ✅ working | WASAPI device listing, 16 kHz mono negotiation, self-test report |
| `engines.py` | 🚧 L2/L3 | `SherpaKws`, `SileroVad`, `LocalWhisperRecognizer`, `CloudRecognizer`, `PiperSynthesizer`, `DarijaTtsSidecar` |

The engine classes exist now but raise `Unsupported` on `load()`, so `atlas doctor`
reports the real state instead of pretending audio works. Nothing here is
imported by the resident core until L2 wires it up.

Hardware rules this package must respect (see the plan's anti-pattern list):
16 kHz mono at the edge, no PyTorch in the core venv, heavy engines are *leased*
never resident, and the microphone is muted while Atlas speaks.
