"""Atlas kernel — the only package every other package is allowed to depend on.

Nothing in here may import a sibling Atlas package (enforced by
``scripts/check_import_rules.py``).
"""

from atlas_core.config import AppConfig, load_config
from atlas_core.contracts import (
    Engine,
    LlmProvider,
    Sense,
    Skill,
    SpeakerVerifier,
    SpeechRecognizer,
    SpeechSynthesizer,
    Store,
    WakeWordEngine,
)
from atlas_core.di import Container
from atlas_core.errors import AtlasError, ConfigError, ProviderError, QuotaExhausted
from atlas_core.events import Event, EventBus
from atlas_core.fsm import InteractionFSM, State
from atlas_core.resources import ResourceGovernor, ResourceLease
from atlas_core.timings import TimingRecord, TimingRecorder

__version__ = "0.1.0"

__all__ = [
    "AppConfig",
    "AtlasError",
    "ConfigError",
    "Container",
    "Engine",
    "Event",
    "EventBus",
    "InteractionFSM",
    "LlmProvider",
    "ProviderError",
    "QuotaExhausted",
    "ResourceGovernor",
    "ResourceLease",
    "Sense",
    "Skill",
    "SpeakerVerifier",
    "SpeechRecognizer",
    "SpeechSynthesizer",
    "State",
    "Store",
    "TimingRecord",
    "TimingRecorder",
    "WakeWordEngine",
    "__version__",
    "load_config",
]
