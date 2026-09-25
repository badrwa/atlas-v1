"""The FSM is the contract between mic, brain, mouth and UI — test every edge."""

from __future__ import annotations

import pytest

from atlas_core.fsm import (
    MUTED_STATES,
    TRANSITIONS,
    IllegalTransition,
    InteractionFSM,
    Signal,
    State,
)


def test_every_transition_target_is_a_state() -> None:
    for (source, signal), target in TRANSITIONS.items():
        assert isinstance(source, State)
        assert isinstance(signal, Signal)
        assert isinstance(target, State)


def test_happy_path_wake_to_done() -> None:
    fsm = InteractionFSM()
    assert fsm.send(Signal.WAKE) is State.WAKING
    assert fsm.send(Signal.SPEECH_STARTED) is State.LISTENING
    assert fsm.send(Signal.SPEECH_ENDED) is State.THINKING
    assert fsm.send(Signal.FIRST_SENTENCE_READY) is State.SPEAKING
    assert fsm.send(Signal.DONE_SPEAKING) is State.DORMANT
    assert [step[1] for step in fsm.history] == [
        Signal.WAKE,
        Signal.SPEECH_STARTED,
        Signal.SPEECH_ENDED,
        Signal.FIRST_SENTENCE_READY,
        Signal.DONE_SPEAKING,
    ]


def test_illegal_signal_raises() -> None:
    fsm = InteractionFSM()
    with pytest.raises(IllegalTransition):
        fsm.send(Signal.SPEECH_ENDED)


def test_try_send_swallows_noise() -> None:
    fsm = InteractionFSM()
    assert fsm.try_send(Signal.SPEECH_ENDED) is None
    assert fsm.state is State.DORMANT
    assert fsm.try_send(Signal.WAKE) is State.WAKING


def test_wake_timeout_returns_to_dormant() -> None:
    fsm = InteractionFSM()
    fsm.send(Signal.WAKE)
    assert fsm.send(Signal.WAKE_TIMEOUT) is State.DORMANT


def test_confirm_timeout_defaults_to_no() -> None:
    fsm = InteractionFSM(state=State.THINKING)
    fsm.send(Signal.NEEDS_CONFIRMATION)
    assert fsm.state is State.CONFIRMING
    assert fsm.send(Signal.CONFIRM_TIMEOUT) is State.SPEAKING


def test_provider_failure_reaches_error_and_can_recover() -> None:
    fsm = InteractionFSM(state=State.THINKING)
    fsm.send(Signal.PROVIDER_FAILED)
    assert fsm.state is State.ERROR
    assert fsm.send(Signal.RECOVERED) is State.SPEAKING


def test_stop_works_from_every_speaking_or_listening_state() -> None:
    for state in (State.LISTENING, State.THINKING, State.SPEAKING, State.ERROR):
        fsm = InteractionFSM(state=state)
        assert fsm.send(Signal.STOP_REQUESTED) is State.DORMANT, state


def test_mic_is_muted_only_while_speaking() -> None:
    assert {State.SPEAKING} == MUTED_STATES
    fsm = InteractionFSM(state=State.SPEAKING)
    assert fsm.mic_should_be_muted is True
    fsm.send(Signal.DONE_SPEAKING)
    assert fsm.mic_should_be_muted is False


def test_hot_states_track_listening_window() -> None:
    fsm = InteractionFSM()
    assert fsm.is_hot is False
    fsm.send(Signal.WAKE)
    assert fsm.is_hot is True
    fsm.send(Signal.WAKE_TIMEOUT)
    assert fsm.is_hot is False


def test_hooks_fire_in_order() -> None:
    seen: list[tuple[str, str]] = []
    fsm = InteractionFSM()
    fsm.on_exit(State.DORMANT, lambda prev, sig, nxt: seen.append(("exit", prev.value)))
    fsm.on_enter(State.WAKING, lambda prev, sig, nxt: seen.append(("enter", nxt.value)))
    fsm.send(Signal.WAKE)
    assert seen == [("exit", "dormant"), ("enter", "waking")]


def test_double_wake_stays_awake() -> None:
    fsm = InteractionFSM()
    fsm.send(Signal.WAKE)
    assert fsm.send(Signal.WAKE) is State.WAKING


def test_reset_and_snapshot() -> None:
    fsm = InteractionFSM()
    fsm.send(Signal.WAKE)
    snapshot = fsm.snapshot()
    assert snapshot["state"] == "waking"
    assert snapshot["transitions"] == 1
    fsm.reset()
    assert fsm.state is State.DORMANT
    assert fsm.history == []
