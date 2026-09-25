"""Atlas mind — providers, routing, language, persona, context, mood."""

from atlas_mind.chat import ChatSession, TurnResult, reply_envelope_from_text
from atlas_mind.context import ContextBuilder
from atlas_mind.darija import DarijaNormalizer, Lexicon
from atlas_mind.envelope import (
    EMOTIONS,
    REPLY_SCHEMA,
    EnvelopeError,
    ReplyEnvelope,
    envelope_instructions,
    parse_reply,
)
from atlas_mind.language import DialectPack, LanguageRouter
from atlas_mind.mood import MoodEngine, MoodView
from atlas_mind.persona import Persona
from atlas_mind.providers.factory import build_provider, build_providers
from atlas_mind.providers.llama_cpp import LlamaCppProvider, SidecarSpec, find_default_paths
from atlas_mind.router import ProviderRouter

__all__ = [
    "EMOTIONS",
    "REPLY_SCHEMA",
    "ChatSession",
    "ContextBuilder",
    "DarijaNormalizer",
    "DialectPack",
    "EnvelopeError",
    "LanguageRouter",
    "Lexicon",
    "LlamaCppProvider",
    "MoodEngine",
    "MoodView",
    "Persona",
    "ProviderRouter",
    "ReplyEnvelope",
    "SidecarSpec",
    "TurnResult",
    "build_provider",
    "build_providers",
    "envelope_instructions",
    "find_default_paths",
    "parse_reply",
    "reply_envelope_from_text",
]
