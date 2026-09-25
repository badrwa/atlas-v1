"""L4 — the permission truth table, the guard, the store, and the log.

Two tests in this file are load-bearing and must stay exhaustive:

* `test_every_capability_for_every_kind_of_speaker` — the whole (speaker ×
  capability) matrix, so a refactor cannot quietly widen anybody's access.
* `test_no_embedding_leaks_into_a_log_or_a_note` — embeddings are biometric data,
  and the vault is git-journaled and synced (plan pitfall #3).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from atlas_core.contracts import Capability, SpeakerMatch
from atlas_core.identity import (
    KNOWN_CAPABILITIES,
    OWNER_CAPABILITIES,
    RESTRICTED_CAPABILITIES,
    Audience,
    ContextGuard,
    DailyGreeter,
    IdentityConfig,
    Permissions,
    ProfileRepository,
    SpeakerEvent,
    SpeakerLog,
    SpeakerProfile,
    best_match,
    centroid,
    cosine,
    fingerprint,
)

DIM = 8


def vector(*values: float) -> list[float]:
    padded = list(values) + [0.0] * (DIM - len(values))
    return padded[:DIM]


def near(base: list[float], *, delta: float = 0.02) -> list[float]:
    """A slightly different recording of the same voice."""
    return [value + (delta if index % 2 else -delta) for index, value in enumerate(base)]


OWNER_VOICE = vector(1.0, 0.4, 0.2, 0.1)
GUEST_VOICE = vector(-0.3, 0.9, 0.5, -0.2)
STRANGER_VOICE = vector(0.1, -0.8, 0.3, 0.7)


def owner_profile(**kwargs) -> SpeakerProfile:
    return SpeakerProfile(
        name="badr",
        embeddings=[OWNER_VOICE, near(OWNER_VOICE)],
        quality=0.92,
        samples=5,
        owner=True,
        **kwargs,
    )


def guest_profile() -> SpeakerProfile:
    return SpeakerProfile(name="said", embeddings=[GUEST_VOICE], quality=0.80, samples=3)


def guard_with(**kwargs) -> ContextGuard:
    profiles = kwargs.pop("profiles", None)
    if profiles is None:
        profiles = {"badr": owner_profile(), "said": guest_profile()}
    return ContextGuard(
        IdentityConfig(**kwargs),
        profiles=profiles,
        log=SpeakerLog(":memory:", load=False),
    )


# ── the vocabulary ───────────────────────────────────────────────────
def test_the_capability_sets_are_the_plan_table() -> None:
    assert frozenset(Capability) == OWNER_CAPABILITIES
    assert frozenset({Capability.GENERAL}) == RESTRICTED_CAPABILITIES
    assert Capability.READ_MEMORY not in RESTRICTED_CAPABILITIES
    assert Capability.PC_CONTROL not in KNOWN_CAPABILITIES
    assert Capability.DESTRUCTIVE not in KNOWN_CAPABILITIES
    assert Capability.ENROL not in KNOWN_CAPABILITIES


# ── the safety test: every cell ──────────────────────────────────────
OWNER = Permissions.owner_of("badr")
GUEST = Permissions.guest_of("said")
STRANGER = Permissions.stranger()

#: The plan's design contract, transcribed once.  `True` = may reach it.
EXPECTED: dict[str, dict[Capability, bool]] = {
    "owner": dict.fromkeys(Capability, True),
    "guest": {
        Capability.GENERAL: True,
        Capability.READ_MEMORY: False,  # the guard grants it per-config
        Capability.WRITE_VAULT: True,  # …their own note, because subject is set
        Capability.OWN_NOTES: True,
        Capability.PC_CONTROL: False,
        Capability.DESTRUCTIVE: False,
        Capability.ENROL: False,
    },
    "stranger": {
        Capability.GENERAL: True,
        Capability.READ_MEMORY: False,
        Capability.WRITE_VAULT: False,
        Capability.OWN_NOTES: False,
        Capability.PC_CONTROL: False,
        Capability.DESTRUCTIVE: False,
        Capability.ENROL: False,
    },
}


@pytest.mark.parametrize("kind", ["owner", "guest", "stranger"])
def test_every_capability_for_every_kind_of_speaker(kind: str) -> None:
    permissions = {"owner": OWNER, "guest": GUEST, "stranger": STRANGER}[kind]
    for capability, allowed in EXPECTED[kind].items():
        assert permissions.allows(capability) is allowed, f"{kind} × {capability}"


def test_a_guest_without_a_subject_cannot_write_anywhere() -> None:
    subjectless = Permissions(
        capabilities=frozenset({Capability.GENERAL, Capability.OWN_NOTES}), speaker="ghost"
    )
    assert subjectless.allows(Capability.WRITE_VAULT) is False


def test_a_stranger_is_restricted_and_an_owner_is_not() -> None:
    assert STRANGER.restricted is True
    assert OWNER.restricted is False
    assert GUEST.restricted is False


def test_the_refusal_is_polite_in_both_languages() -> None:
    assert STRANGER.spoken_refusal("ar-MA") == "سمح ليا، هادشي خاص بصاحبي."
    assert "personal" in STRANGER.spoken_refusal("en-GB")


# ── the guard ────────────────────────────────────────────────────────
def test_the_owner_is_recognised_and_gets_everything() -> None:
    guard = guard_with()
    permissions = guard.verify(near(OWNER_VOICE), utterance_ms=3000)
    assert permissions.owner is True
    assert permissions.speaker == "badr"
    assert permissions.score > 0.9
    assert permissions.allows(Capability.DESTRUCTIVE) is True
    assert permissions.reason == "verified_owner"


def test_an_enrolled_other_person_is_known_but_not_the_owner() -> None:
    guard = guard_with(guest_pc_control=True)
    permissions = guard.verify(near(GUEST_VOICE), utterance_ms=3000)
    assert permissions.known is True
    assert permissions.owner is False
    assert permissions.subject == "said"
    assert permissions.allows(Capability.OWN_NOTES) is True
    assert permissions.allows(Capability.WRITE_VAULT) is True  # their own note
    assert permissions.allows(Capability.PC_CONTROL) is True  # SAFE only, opted in
    assert permissions.allows(Capability.DESTRUCTIVE) is False
    assert permissions.allows(Capability.ENROL) is False


def test_guests_do_not_get_pc_control_unless_the_owner_opted_in() -> None:
    permissions = guard_with(guest_pc_control=False).verify(near(GUEST_VOICE), utterance_ms=3000)
    assert permissions.allows(Capability.PC_CONTROL) is False


def test_a_stranger_gets_general_conversation_only() -> None:
    guard = guard_with()
    permissions = guard.verify(STRANGER_VOICE, utterance_ms=3000)
    assert permissions.accepted is False
    assert permissions.restricted is True
    assert permissions.allows(Capability.GENERAL) is True
    assert permissions.allows(Capability.READ_MEMORY) is False
    assert permissions.reason == "below_threshold"
    assert permissions.spoken_refusal("ar-MA") == "سمح ليا، هادشي خاص بصاحبي."


def test_a_short_utterance_never_carries_an_owner_only_decision() -> None:
    """The television says one word too — so one word is not a licence."""
    guard = guard_with()
    permissions = guard.verify(near(OWNER_VOICE), utterance_ms=400)
    assert permissions.owner is True  # still you
    assert permissions.reason == "utterance_too_short"
    assert permissions.allows(Capability.GENERAL) is True
    assert permissions.allows(Capability.DESTRUCTIVE) is False
    assert permissions.allows(Capability.READ_MEMORY) is False


def test_no_profiles_means_nobody_is_verified() -> None:
    guard = guard_with(profiles={})
    permissions = guard.verify(near(OWNER_VOICE), utterance_ms=3000)
    assert permissions.restricted is True
    assert permissions.reason == "no_profiles"


def test_identity_off_is_single_user_behaviour() -> None:
    guard = guard_with(enabled=False)
    permissions = guard.verify(near(OWNER_VOICE), utterance_ms=3000)
    assert permissions.owner is True
    assert permissions.allows(Capability.DESTRUCTIVE) is True
    assert permissions.reason == "identity_disabled"


def test_a_verifier_that_cannot_run_falls_back_to_the_owner_and_logs_why() -> None:
    guard = guard_with()
    permissions = guard.unavailable(reason="verifier_unavailable", utterance_ms=3000)
    assert permissions.owner is True
    assert permissions.reason == "verifier_unavailable"
    assert guard.last_event is not None
    assert guard.last_event.reason == "verifier_unavailable"


def test_mood_is_logged_and_never_widens_access() -> None:
    class Mood:
        label = "happy"

    guard = guard_with()
    restricted = guard.verify(STRANGER_VOICE, utterance_ms=3000, mood=Mood())
    assert restricted.allows(Capability.READ_MEMORY) is False
    assert guard.last_event is not None
    assert guard.last_event.mood == "happy"


def test_effective_permissions_uses_the_same_rule_for_a_ready_match() -> None:
    guard = guard_with()
    as_owner = guard.effective_permissions(SpeakerMatch(name="badr", score=0.9, owner=True))
    as_stranger = guard.effective_permissions(SpeakerMatch(name="unknown", score=0.4))
    as_guest = guard.effective_permissions(SpeakerMatch(name="said", score=0.8))
    assert as_owner.allows(Capability.DESTRUCTIVE) is True
    assert as_stranger.restricted is True
    assert as_guest.subject == "said"


# ── drift ────────────────────────────────────────────────────────────
def test_two_rejections_offer_re_enrolment_and_a_match_clears_it() -> None:
    guard = guard_with(threshold=0.9)
    first = guard.verify(STRANGER_VOICE, utterance_ms=3000)
    assert guard.should_reoffer_enrolment(first.speaker) is False
    guard.verify(STRANGER_VOICE, utterance_ms=3000)
    assert guard.should_reoffer_enrolment(first.speaker) is True
    assert "3awd" in guard.reoffer_line("ar-MA")
    assert "re-enrol" in guard.reoffer_line("en-GB")

    guard.verify(near(OWNER_VOICE), utterance_ms=3000)
    assert guard.should_reoffer_enrolment("badr") is False


def test_the_greeting_names_the_person_in_both_languages() -> None:
    guard = guard_with()
    assert guard.greeting("badr").startswith("Sbah lkhir badr")
    assert guard.greeting("badr", language="en-GB").startswith("Morning, badr")
    assert guard.greeting("", ) == ""


# ── the repository ───────────────────────────────────────────────────
def test_a_profile_round_trips_through_sqlite(tmp_path: Path) -> None:
    repository = ProfileRepository(tmp_path / "speakers.sqlite3")
    saved = repository.save(owner_profile())

    loaded = repository.load("badr")
    assert loaded is not None
    assert loaded.owner is True
    assert len(loaded.embeddings) == 2
    assert loaded.dimension == DIM
    assert loaded.quality == pytest.approx(0.92)
    assert cosine(loaded.embeddings[0], OWNER_VOICE) == pytest.approx(1.0, abs=1e-6)
    assert saved.name == "badr"


def test_enrolling_a_second_owner_demotes_the_first(tmp_path: Path) -> None:
    repository = ProfileRepository(tmp_path / "speakers.sqlite3")
    repository.save(owner_profile(), owner=True)
    repository.save(
        SpeakerProfile(name="said", embeddings=[GUEST_VOICE], quality=0.8, samples=3), owner=True
    )
    assert repository.owner_name() == "said"
    assert repository.load("badr").owner is False  # type: ignore[union-attr]


def test_the_embedding_window_rolls_forward(tmp_path: Path) -> None:
    repository = ProfileRepository(tmp_path / "speakers.sqlite3", window=3)
    repository.save(SpeakerProfile(name="badr", embeddings=[OWNER_VOICE] * 5, samples=5))
    assert len(repository.load("badr").embeddings) == 3  # type: ignore[union-attr]

    grown = repository.add_embedding("badr", near(GUEST_VOICE))
    assert grown is not None and len(grown.embeddings) == 3
    assert repository.add_embedding("nobody", STRANGER_VOICE) is None


def test_forget_removes_one_person_and_wipe_removes_everyone(tmp_path: Path) -> None:
    repository = ProfileRepository(tmp_path / "speakers.sqlite3")
    repository.save(owner_profile(), owner=True)
    repository.save(guest_profile())
    assert repository.names() == ["badr", "said"]

    assert repository.delete("said") is True
    assert repository.delete("said") is False
    assert repository.wipe() == 1
    assert repository.names() == []
    assert repository.owner_name() == ""


def test_the_repository_refuses_to_live_in_the_vault(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="vault"):
        ProfileRepository(tmp_path / "ObsidianVault" / "speakers.sqlite3")
    with pytest.raises(ValueError):
        ProfileRepository(tmp_path / "my-vault" / "speakers.sqlite3")


def test_stats_describe_the_store_without_leaking_a_vector(tmp_path: Path) -> None:
    repository = ProfileRepository(tmp_path / "speakers.sqlite3")
    repository.save(owner_profile(), owner=True)
    stats = repository.stats()
    assert stats["people"] == 1
    assert stats["owner"] == "badr"
    assert stats["samples"] == 2
    assert stats["dimension"] == DIM
    assert "1.0" not in json.dumps(stats)


def test_the_guard_enrols_through_the_repository(tmp_path: Path) -> None:
    repository = ProfileRepository(tmp_path / "speakers.sqlite3")
    guard = ContextGuard(IdentityConfig(), repository=repository)
    guard.enroll(SpeakerProfile(name="badr", embeddings=[OWNER_VOICE], samples=1))
    assert guard.owner_name == "badr"
    assert repository.owner_name() == "badr"
    assert guard.known_names == ["badr"]

    guard.enroll(SpeakerProfile(name="said", embeddings=[GUEST_VOICE], samples=1), owner=False)
    assert repository.load("said").owner is False  # type: ignore[union-attr]


# ── the log ──────────────────────────────────────────────────────────
def test_the_log_keeps_scores_and_never_a_vector(tmp_path: Path) -> None:
    path = tmp_path / "speaker_log.jsonl"
    log = SpeakerLog(path)
    guard = ContextGuard(
        IdentityConfig(), profiles={"badr": owner_profile(), "said": guest_profile()}, log=log
    )
    guard.verify(near(OWNER_VOICE), utterance_ms=2500, elapsed_ms=41.0)
    guard.verify(STRANGER_VOICE, utterance_ms=1800, elapsed_ms=38.0)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    payload = json.loads(lines[0])
    assert payload["speaker"] == "badr"
    assert payload["accepted"] is True
    assert "embeddings" not in payload and "vector" not in payload
    assert guard.events[0].speaker == "badr"
    assert "0.02" not in json.dumps(payload), "no component of the embedding may survive"

    summary = log.summary()
    assert summary["events"] == 2
    assert summary["accept_rate"] == 0.5
    # The rejected utterance is logged as "badr, 0.31, not accepted": the name is
    # the *closest* profile, `accepted` is the decision, `reason` says why.  That
    # pair is exactly what the threshold sweep needs — the log never claims
    # someone spoke, only that their profile was near.
    per_speaker: dict[str, dict[str, float]] = summary["speakers"]  # type: ignore[assignment]
    assert set(per_speaker) == {"badr"}
    assert per_speaker["badr"]["n"] == 2
    assert per_speaker["badr"]["max"] == pytest.approx(1.0, abs=0.01)
    rejected = guard.last_event
    assert rejected is not None
    assert rejected.accepted is False
    assert rejected.reason == "below_threshold"


def test_a_rejected_match_keeps_the_candidate_name_but_no_trust(tmp_path: Path) -> None:
    log = SpeakerLog(tmp_path / "speaker_log.jsonl")
    guard = ContextGuard(IdentityConfig(threshold=0.9), profiles={"badr": owner_profile()}, log=log)
    permissions = guard.verify(STRANGER_VOICE, utterance_ms=2000)
    assert permissions.accepted is False
    assert permissions.speaker == "badr", "the candidate is named, so the CLI can show the score"
    assert permissions.owner is False and permissions.known is False
    assert permissions.allows(Capability.READ_MEMORY) is False
    assert permissions.allows(Capability.GENERAL) is True


def test_the_log_reloads_previous_sessions(tmp_path: Path) -> None:
    path = tmp_path / "speaker_log.jsonl"
    SpeakerLog(path).add(SpeakerEvent(speaker="badr", score=0.81, accepted=True))
    reopened = SpeakerLog(path, load=True)
    assert len(reopened.events) == 1
    assert reopened.events[0].speaker == "badr"
    assert reopened.scores("badr") == [0.81]


def test_a_disabled_log_still_answers_in_memory(tmp_path: Path) -> None:
    log = SpeakerLog(tmp_path / "speaker_log.jsonl", enabled=False)
    log.add(SpeakerEvent(speaker="badr", score=0.7, accepted=True))
    assert log.persisted is False
    assert len(log.events) == 1
    assert not (tmp_path / "speaker_log.jsonl").exists()


# ── privacy ──────────────────────────────────────────────────────────
def test_no_embedding_leaks_into_a_log_or_a_note(tmp_path: Path) -> None:
    """The vault is synced and git-journaled; a voice never goes in it."""
    repository = ProfileRepository(tmp_path / "speakers.sqlite3")
    profile = repository.save(owner_profile(), owner=True)

    safe = profile.as_dict()
    assert "embeddings" not in safe and "centroid" not in safe
    assert "embeddings" in profile.as_dict(include_vectors=True)

    log = SpeakerLog(tmp_path / "speaker_log.jsonl")
    log.add(SpeakerEvent(speaker="badr", score=0.91, accepted=True))
    for text in (safe, log.path.read_text(encoding="utf-8")):
        assert "embeddings" not in text
        assert str(round(OWNER_VOICE[0], 6)) not in text


def test_the_fingerprint_identifies_a_profile_without_exposing_it(tmp_path: Path) -> None:
    profile = owner_profile()
    mark = fingerprint(profile)
    assert len(mark) == 12
    assert mark == fingerprint(owner_profile())
    assert mark != fingerprint(guest_profile())
    assert str(OWNER_VOICE[0]) not in mark


def test_the_database_file_is_the_only_place_vectors_live(tmp_path: Path) -> None:
    repository = ProfileRepository(tmp_path / "speakers.sqlite3")
    repository.save(owner_profile())
    with sqlite3.connect(tmp_path / "speakers.sqlite3") as connection:
        blobs = connection.execute("SELECT COUNT(*) FROM speaker_embedding").fetchone()[0]
    assert blobs == 2
    # …and nothing else in the folder pretends to hold them.  WAL/SHM side files
    # belong to SQLite itself; everything else would be a leak waiting to happen.
    others = [
        path.name for path in tmp_path.iterdir() if ".sqlite3" not in path.name
    ]
    assert others == [], f"unexpected side files: {others}"


# ── the maths ────────────────────────────────────────────────────────
def test_cosine_is_symmetric_and_bounded() -> None:
    assert cosine(OWNER_VOICE, OWNER_VOICE) == pytest.approx(1.0)
    assert cosine(OWNER_VOICE, GUEST_VOICE) == cosine(GUEST_VOICE, OWNER_VOICE)
    assert -1.0 <= cosine(OWNER_VOICE, STRANGER_VOICE) <= 1.0
    assert cosine([], OWNER_VOICE) == 0.0
    assert cosine([0.0] * 4, OWNER_VOICE) == 0.0
    assert cosine([1.0, 2.0], [1.0]) == 0.0  # different models, no comparison


def test_the_centroid_is_the_average_direction() -> None:
    mean = centroid([OWNER_VOICE, OWNER_VOICE])
    assert cosine(mean, OWNER_VOICE) == pytest.approx(1.0, abs=1e-6)
    assert centroid([]) == []
    assert centroid([[1.0, 0.0], [1.0, 0.0, 0.0]]) != []  # ragged input is tolerated


def test_a_match_beats_the_centroid_when_only_one_sample_fits() -> None:
    """A tired voice should still match the sample from a good day."""
    profile = SpeakerProfile(name="badr", embeddings=[OWNER_VOICE, vector(0.2, 0.2, 0.9)])
    assert best_match(OWNER_VOICE, profile) == pytest.approx(1.0)


def test_the_profile_keeps_a_rolling_window() -> None:
    profile = SpeakerProfile(name="badr")
    for index in range(10):
        profile.add(vector(1.0, index / 10))
    assert len(profile.embeddings) == 6
    assert profile.samples == 10
    profile.add([])
    assert profile.samples == 10, "an empty embedding is not a sample"


# ── audience ─────────────────────────────────────────────────────────
def test_the_audience_view_is_smaller_than_the_permissions() -> None:
    guarded = Permissions.stranger(score=0.42)
    audience = Audience.from_permissions(guarded, language="en-GB")
    assert audience.restricted is True
    assert audience.owner is False
    assert audience.as_dict() == {
        "name": "",
        "owner": False,
        "known": False,
        "restricted": True,
        "subject": "",
    }
    assert "unrecognised" in audience.display

    owner_view = Audience.from_permissions(Permissions.owner_of("badr"))
    assert owner_view.display == "badr"
    assert Audience.from_permissions(Permissions.guest_of("said")).display == "said (not the owner)"


# ── the daily greeting ───────────────────────────────────────────────
def test_the_greeting_is_due_once_a_day_and_survives_a_restart(tmp_path: Path) -> None:
    greeter = DailyGreeter(tmp_path / "last_greeting.txt")
    assert greeter.due("") is False  # no name, nothing to say
    assert greeter.due("badr", now=1_700_000_000) is True  # first time today

    greeter.mark("badr", now=1_700_000_000)
    assert greeter.due("badr", now=1_700_000_000) is False
    assert DailyGreeter(tmp_path / "last_greeting.txt").due("badr", now=1_700_000_000) is False
    assert greeter.due("badr", now=1_700_000_000 + 86_400) is True
    assert greeter.due("said", now=1_700_000_000) is True


def test_a_greeting_file_that_cannot_be_written_does_not_break_the_turn(tmp_path: Path) -> None:
    greeter = DailyGreeter(tmp_path / "nested" / "deep" / "state.txt")
    greeter.mark("badr")
    assert greeter.due("badr") is False


# ── config ───────────────────────────────────────────────────────────
def test_identity_config_reads_the_section_and_reads_safe_defaults() -> None:
    class Section:
        enabled = True
        owner_name = "badr"
        threshold = 0.71
        trust_min_ms = 1500
        guest_pc_control = True
        profiles_path = "data/other.sqlite3"
        log_path = "data/other.jsonl"
        window = 4
        enrol_samples = 5
        enrol_seconds = 8.0
        enrol_min_speech_ms = 2000
        quality_floor = 0.6
        drift_rejections = 3
        greet_once_per_day = False

    class Config:
        identity = Section()

    settings = IdentityConfig.from_config(Config())
    assert settings.threshold == 0.71
    assert settings.window == 4
    assert settings.guest_pc_control is True
    assert settings.greet_once_per_day is False

    default = IdentityConfig.from_config(None)
    assert default.enabled is True
    assert default.threshold == 0.65
    assert default.guest_pc_control is False, "guests do not get PC control by default"
    assert default.as_dict()["owner"] == ""


def test_an_in_memory_store_works_and_leaves_nothing_behind(tmp_path: Path, monkeypatch) -> None:
    """`:memory:` is the spelling tests use — it must mean *nowhere on disk*."""
    monkeypatch.chdir(tmp_path)
    repository = ProfileRepository(":memory:")
    repository.save(owner_profile(), owner=True)
    assert repository.owner_name() == "badr"
    assert len(repository.load("badr").embeddings) == 2  # type: ignore[union-attr]
    assert repository.delete("badr") is True
    assert list(tmp_path.iterdir()) == [], "an in-memory store writes no file at all"


def test_an_in_memory_log_writes_no_file_with_that_name(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    log = SpeakerLog(":memory:", load=False)
    log.add(SpeakerEvent(speaker="badr", score=0.9, accepted=True))
    assert log.memory_only is True
    assert log.persisted is False
    assert len(log.events) == 1
    assert list(tmp_path.iterdir()) == []
