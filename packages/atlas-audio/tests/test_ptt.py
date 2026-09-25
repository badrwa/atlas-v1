"""Push-to-talk: the key press is the trigger, and nothing else is.

Two behaviours matter here, and both are easy to get wrong:

* while `recording`, the loop must **not** also consume frames — otherwise a
  wake-word detection in the middle of a key press starts a second turn;
* a stuck key must not eat the machine: the buffer is capped.
"""

from __future__ import annotations

from doubles import FakeRecognizer, utterance_audio

from atlas_audio.capture import silence, speech
from atlas_audio.loop import LoopConfig, VoiceLoop
from atlas_audio.ptt import HOTKEYS, HotkeyListener, PushToTalk
from atlas_audio.vad import SegmenterConfig, VadSegmenter
from atlas_audio.wake import EnergyWakeEngine


def make_loop() -> tuple[VoiceLoop, FakeRecognizer]:
    recognizer = FakeRecognizer()

    async def respond(text: str, language: str):
        yield "safi"

    loop = VoiceLoop(
        wake=EnergyWakeEngine(confirm_frames=1),
        segmenter=VadSegmenter(SegmenterConfig()),
        recognizer=recognizer,
        respond=respond,
        config=LoopConfig(),
    )
    loop.arm()
    return loop, recognizer


def test_frames_are_buffered_only_while_recording():
    loop, _ = make_loop()
    ptt = PushToTalk(loop)
    for frame in speech(5):
        ptt.push(frame)  # not recording: dropped, as the loop is handling them
    assert ptt.frames == []

    ptt.begin()
    for frame in speech(12):
        ptt.push(frame)
    assert len(ptt.frames) == 12
    assert ptt.seconds == 12 * 80 / 1000
    assert ptt.recording is True

    frames = ptt.end()
    assert len(frames) == 12
    assert ptt.recording is False
    assert ptt.presses == 1


async def test_finish_transcribes_the_press_as_one_turn():
    loop, recognizer = make_loop()
    ptt = PushToTalk(loop)
    ptt.begin()
    for frame in speech(14) + silence(8):
        ptt.push(frame)
    turn = await ptt.finish()

    assert turn is not None
    assert turn.text == "chno ljaw"
    assert turn.wake is None  # a key started this turn, not a keyword
    assert loop.wake_hits == 0
    assert recognizer.calls  # exactly one round of ASR


async def test_an_empty_press_is_not_a_turn():
    loop, recognizer = make_loop()
    ptt = PushToTalk(loop)
    ptt.begin()
    ptt.push(silence(1)[0])  # just a click
    assert await ptt.finish() is None
    assert recognizer.calls == []
    assert loop.turns == []


def test_a_stuck_key_is_capped_instead_of_eating_the_machine():
    loop, _ = make_loop()
    ptt = PushToTalk(loop, max_seconds=2.0)
    ptt.begin()
    for _ in range(500):  # 40 seconds of audio pushed into a 2 s buffer
        ptt.push(speech(1)[0])
    assert len(ptt.frames) == ptt.max_frames == 25


async def test_the_loop_still_works_after_a_push_to_talk_turn():
    """PTT must not leave the loop armed against a half-consumed segmenter."""
    loop, _ = make_loop()
    ptt = PushToTalk(loop)
    ptt.begin()
    for frame in speech(14) + silence(8):
        ptt.push(frame)
    assert await ptt.finish() is not None

    # The wake-word path still works: fresh loop, same recogniser.
    loop2, _ = make_loop()
    turns = await loop2.feed_frames(utterance_audio())
    assert len(turns) == 1


# ── hotkeys ──────────────────────────────────────────────────────────
def test_hotkeys_match_the_plan():
    assert HOTKEYS["ask"] == "ctrl+alt+space"
    assert HOTKEYS["mute"] == "ctrl+alt+m"
    assert HOTKEYS["stop"] == "ctrl+alt+s"


def test_hotkey_listener_without_a_backend_reports_absence():
    listener = HotkeyListener()
    if listener.available():
        import pytest

        pytest.skip("a keyboard hook is installed in this environment")
    assert listener.start(on_ask=lambda: None) is False
    assert listener.bindings == []


def test_hotkey_listener_with_an_injected_backend_binds_every_action():
    calls: list[tuple[str, str]] = []

    def backend(keys: str, callback) -> None:  # pragma: no cover - trivial
        calls.append((keys, callback.__name__))

    started = []
    listener = HotkeyListener(backend=backend)
    assert listener.start(
        on_ask=lambda: started.append("ask"),
        on_mute=lambda: started.append("mute"),
        on_stop=lambda: started.append("stop"),
    )
    assert listener.bindings == [HOTKEYS["ask"], HOTKEYS["mute"], HOTKEYS["stop"]]
    assert [keys for keys, _ in calls] == listener.bindings
    assert "ask" in listener.describe()
