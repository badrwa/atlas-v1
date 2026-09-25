"""`atlas identity` — the L4 command, run for real.

CI has no sherpa-onnx and no microphone.  Everything here proves the command is
*honest* in that world: it reports the missing model instead of a fake profile, a
bad enrolment is refused instead of stored, and a wipe asks first.
"""

from __future__ import annotations

import io
import wave
from array import array
from collections.abc import Iterator
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from atlas.cli import main
from atlas_core.contracts import Capability
from atlas_core.identity import IdentityConfig, ProfileRepository, SpeakerProfile

CONFIG = """
[app]
profile = "lean"
language = "ar-MA"

[obsidian]
vault_path = ""

[identity]
enabled = true
owner_name = "badr"
threshold = 0.65
profiles_path = "data/speakers.sqlite3"
log_path = "data/speaker_log.jsonl"
"""


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A real config file and a real cwd, so the relative `data/` paths resolve."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text(CONFIG, encoding="utf-8")
    (tmp_path / "data").mkdir()
    yield tmp_path


def run(*args: str) -> tuple[int, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            code = main(["--config", "config.toml", *args])
        except SystemExit as exc:  # argparse rejects an unknown action itself
            code = int(exc.code or 0)
    return code, out.getvalue() + err.getvalue()


def write_wav(path: Path, seconds: float = 3.0, *, hz: float = 220.0) -> Path:
    import math

    rate = 16000
    samples = array(
        "h",
        [
            int(6000 * math.sin(2 * math.pi * hz * index / rate))
            for index in range(int(rate * seconds))
        ],
    )
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())
    return path


# ── status ───────────────────────────────────────────────────────────
def test_status_reports_the_truth_on_a_machine_without_the_model(workspace: Path) -> None:
    code, text = run("identity")
    assert code == 0
    assert "Identity" in text
    assert "nobody enrolled" in text
    assert "atlas identity enrol --name" in text
    # The model is missing here, and the row says so rather than pretending.
    assert "sherpa-onnx is not installed" in text or "not found" in text


def test_status_lists_enrolled_profiles_without_a_single_vector(workspace: Path) -> None:
    repository = ProfileRepository("data/speakers.sqlite3")
    repository.save(
        SpeakerProfile(
            name="badr", embeddings=[[0.5, 0.25, 0.125, 0.0625]], quality=0.91, samples=3
        ),
        owner=True,
    )
    code, text = run("identity")
    assert code == 0
    assert "badr" in text and "0.91" in text
    assert "0.5" not in text, "a component of the embedding must never be printed"


def test_status_summarises_the_speaker_log(workspace: Path) -> None:
    from atlas_core.identity import SpeakerEvent, SpeakerLog

    log = SpeakerLog("data/speaker_log.jsonl")
    log.add(SpeakerEvent(speaker="badr", score=0.88, accepted=True, reason="verified_owner"))
    log.add(SpeakerEvent(speaker="badr", score=0.41, accepted=False, reason="below_threshold"))

    code, text = run("identity")
    assert code == 0
    assert "2 utterances" in text
    assert "50% accepted" in text


def test_log_prints_the_scores_and_marks_the_near_misses(workspace: Path) -> None:
    from atlas_core.identity import SpeakerEvent, SpeakerLog

    log = SpeakerLog("data/speaker_log.jsonl")
    log.add(SpeakerEvent(speaker="badr", score=0.88, accepted=True, reason="verified_owner"))
    log.add(SpeakerEvent(speaker="badr", score=0.41, accepted=False, reason="below_threshold"))

    code, text = run("identity", "log")
    assert code == 0
    assert "0.88" in text and "0.41" in text
    assert "below_threshold" in text
    assert "≈badr" in text, "a rejected match is shown as a near-miss, not as a speaker"


def test_identity_off_says_single_user_rather_than_pretending(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (workspace / "config.toml").write_text(
        CONFIG.replace("enabled = true", "enabled = false"), encoding="utf-8"
    )
    code, text = run("identity")
    assert code == 0
    assert "single user" in text or "off" in text


# ── enrol ────────────────────────────────────────────────────────────
def test_enrol_refuses_without_the_model_and_a_way_forward(workspace: Path) -> None:
    code, text = run("identity", "enrol", "--name", "badr", "--from", "a.wav")
    assert code == 1
    assert "no speaker model" in text
    assert "models/speaker" in text, "the hint names where the model belongs"


def test_enrol_needs_a_name(workspace: Path) -> None:
    (workspace / "anonymous.toml").write_text(
        CONFIG.replace('owner_name = "badr"', 'owner_name = ""'), encoding="utf-8"
    )
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(["--config", "anonymous.toml", "identity", "enrol"])
    assert code == 1
    assert "no name" in out.getvalue() + err.getvalue()


def test_enrol_from_files_reports_a_missing_clip(workspace: Path) -> None:
    code, text = run("identity", "enrol", "--name", "badr", "--from", "nope.wav")
    assert code == 1
    assert "no speaker model" in text or "missing file" in text


def test_enrol_from_files_builds_a_profile_and_a_vault_note(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole path with a stand-in verifier — the real one needs the model."""
    monkeypatch.setattr("atlas_audio.build_identity", _fake_identity)
    clips = [write_wav(workspace / f"clip{index}.wav", hz=220 + index * 0.05) for index in range(3)]
    args = ["identity", "enrol", "--name", "badr", "--owner"]
    for clip in clips:
        args += ["--from", str(clip)]

    code, text = run(*args)
    assert code == 0, text
    assert "enrolled" in text
    assert "badr" in text and "1.00" in text

    repository = ProfileRepository("data/speakers.sqlite3")
    profile = repository.load("badr")
    assert profile is not None
    assert profile.owner is True
    assert len(profile.embeddings) == 3


def test_a_bad_enrolment_is_refused_and_nothing_is_stored(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("atlas_audio.build_identity", _fake_identity)
    quiet = write_wav(workspace / "quiet.wav", seconds=3.0)
    # Overwrite with silence: enough duration, no speech.
    with wave.open(str(quiet), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(array("h", [0] * 48000).tobytes())

    code, text = run(
        "identity", "enrol", "--name", "badr", "--from", str(quiet), "--from", str(quiet),
        "--from", str(quiet),
    )
    assert code == 1
    assert "not enrolled" in text
    assert ProfileRepository("data/speakers.sqlite3").load("badr") is None


# ── verify ───────────────────────────────────────────────────────────
def test_verify_scores_a_clip_against_every_profile(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("atlas_audio.build_identity", _fake_identity)
    repository = ProfileRepository("data/speakers.sqlite3")
    repository.save(
        SpeakerProfile(name="badr", embeddings=[_vector(220.0)], quality=0.9, samples=3),
        owner=True,
    )
    repository.save(
        SpeakerProfile(name="said", embeddings=[_vector(520.0)], quality=0.8, samples=3)
    )

    code, text = run("identity", "verify", str(write_wav(workspace / "me.wav", hz=220.0)))
    assert code == 0
    assert "badr" in text and "said" in text
    assert "threshold 0.65" in text
    assert text.index("badr") < text.index("said") or "✓" in text


def test_verify_without_the_model_tells_you_what_is_missing(workspace: Path) -> None:
    code, text = run("identity", "verify", "me.wav")
    assert code == 1
    assert "no speaker model" in text


def test_verify_without_a_file_asks_for_one(workspace: Path) -> None:
    code, text = run("identity", "verify")
    assert code == 1
    assert "no file" in text or "no speaker model" in text


# ── forget ───────────────────────────────────────────────────────────
def test_forget_all_asks_first_and_keeps_the_vault_notes(workspace: Path) -> None:
    repository = ProfileRepository("data/speakers.sqlite3")
    repository.save(
        SpeakerProfile(name="badr", embeddings=[[1.0, 0.0, 0.0, 0.0]], samples=1), owner=True
    )
    code, text = run("identity", "forget", "--all")
    assert code == 1
    assert "confirmation" in text
    assert repository.load("badr") is not None, "nothing is deleted without --yes"

    code, text = run("identity", "forget", "--all", "--yes")
    assert code == 0
    assert "1 voice profile" in text
    assert "person notes in the vault are untouched" in text
    assert repository.load("badr") is None


def test_forget_one_person_by_name(workspace: Path) -> None:
    repository = ProfileRepository("data/speakers.sqlite3")
    repository.save(SpeakerProfile(name="said", embeddings=[[0.0, 1.0, 0.0, 0.0]], samples=1))
    code, text = run("identity", "forget", "--name", "said")
    assert code == 0 and "erased" in text
    assert repository.load("said") is None

    code, text = run("identity", "forget", "--name", "said")
    assert code == 1 and "nothing to forget" in text


def test_forget_needs_a_name_or_an_all(workspace: Path) -> None:
    code, text = run("identity", "forget")
    assert code == 1
    assert "--name" in text


def test_an_unknown_action_is_refused(workspace: Path) -> None:
    code, text = run("identity", "teleport")
    assert code == 2  # argparse rejects it before we get a chance to
    assert "teleport" in text or "invalid choice" in text


# ── the loop wiring ──────────────────────────────────────────────────
def test_listen_warns_when_every_voice_would_be_the_owner(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dangerous state is quiet, so `listen` says it out loud at startup."""
    monkeypatch.setattr("atlas_audio.build_identity", _fake_identity)
    code, text = run("listen", "--status")
    assert code in (0, 1)
    assert "identity" in text


# ── helpers ──────────────────────────────────────────────────────────
def _vector(hz: float) -> list[float]:
    """The same crude pitch feature the L4 audio tests use."""
    import math

    rate = 16000
    samples = array(
        "h",
        [int(6000 * math.sin(2 * math.pi * hz * index / rate)) for index in range(rate * 3)],
    )
    energy = sum(value * value for value in samples) or 1.0
    return [
        sum(samples[index] * samples[index + lag] for index in range(len(samples) - lag)) / energy
        for lag in (1, 3, 9, 27, 81)
    ]


def _fake_identity(config, *args: object, **kwargs: object) -> dict[str, object]:
    """`build_identity` with a deterministic extractor instead of the model."""
    from atlas_audio.speaker import SherpaSpeakerVerifier
    from atlas_core.identity import ContextGuard, SpeakerLog

    settings = IdentityConfig.from_config(config)
    verifier = SherpaSpeakerVerifier(
        settings.model_path, extractor=lambda samples, rate: _extract(samples)
    )
    repository = ProfileRepository(settings.profiles_path, window=settings.window)
    log = SpeakerLog(settings.log_path)
    guard = ContextGuard(
        settings, profiles=repository.load_all(), repository=repository, log=log
    )
    return {
        "config": settings,
        "verifier": verifier,
        "repository": repository,
        "log": log,
        "guard": guard,
        "owner": repository.owner_name(),
    }


def _extract(samples) -> list[float]:
    energy = sum(value * value for value in samples) or 1.0
    return [
        sum(samples[index] * samples[index + lag] for index in range(len(samples) - lag)) / energy
        for lag in (1, 3, 9, 27, 81)
    ]


# ── the whole path: a stranger's turn, end to end ────────────────────
def test_a_stranger_turn_carries_no_personal_data_anywhere(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The L4 acceptance test: restricted mode must not leak at *any* seam.

    This runs the pieces the CLI wires together — guard → permissions → memory
    reader → audience → prompt — with a real vault holding a real private note,
    and asserts that the private line appears in none of them.
    """
    from atlas.cli import _memory_for
    from atlas_core.config import load_config
    from atlas_core.identity import Audience, ContextGuard, Permissions
    from atlas_obsidian.vault import VaultAdapter

    secret = "kankhdem f projet sirri 3la l 3ayla"
    (workspace / "config.toml").write_text(
        CONFIG.replace('vault_path = ""', 'vault_path = "vault"'), encoding="utf-8"
    )
    vault = VaultAdapter.init_from_template(
        workspace / "vault", Path(__file__).resolve().parents[3] / "vault-template", git=False
    )
    vault.remember(secret)

    config = load_config("config.toml")
    guard = ContextGuard(
        IdentityConfig.from_config(config), profiles={"badr": _owner_profile()}
    )

    for permissions in (
        guard.verify([0.0, 0.0, 0.0, 1.0], utterance_ms=2500),  # a stranger
        Permissions.stranger(score=0.2),
    ):
        assert permissions.allows(Capability.READ_MEMORY) is False
        memory = _memory_for(config, permissions)
        audience = Audience.from_permissions(permissions, language="ar-MA")
        assert secret not in memory
        assert memory == ""
        assert audience.restricted is True
        assert permissions.spoken_refusal("ar-MA") == "سمح ليا، هادشي خاص بصاحبي."

    # …and the owner still gets it, so this is a gate and not a wall.
    owner = guard.verify(_owner_vector(), utterance_ms=2500)
    assert owner.owner is True
    assert secret in _memory_for(config, owner)


def _owner_vector() -> list[float]:
    return _vector(220.0)


def _owner_profile():
    from atlas_core.identity import SpeakerProfile

    return SpeakerProfile(
        name="badr", embeddings=[_owner_vector()], quality=0.9, samples=3, owner=True
    )
