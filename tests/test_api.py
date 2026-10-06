"""Core protocol tests for SchellenbergUsbApi.

Covers the transmit lock (regular, non-movement commands must still fully
serialize through it), the direct unlocked write used for blind moves
(open/close/stop - see control_blind), the busy/retry cycle (including the
eager-task-start regression described below), the connection-loss/disconnect
cleanup of that same retry state, and the three multi-phase RF sequences
(teach_motor, pair_device_and_wait, send_raw_transmit) - these were
previously only exercised indirectly through tests/test_config_flow.py's
config-flow-level tests (real API + FakeTransport, but driven through the
flow rather than called directly), so failure branches specific to each
function's own retry/ACK/block handling had no direct coverage here.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect

import custom_components.schellenberg_usb.api as api_module
from custom_components.schellenberg_usb.api import (
    TRANSMIT_MAX_RETRIES,
    SchellenbergUsbApi,
)
from custom_components.schellenberg_usb.const import (
    CMD_ALLOW_PAIRING,
    CMD_GET_PARAM_P,
    CMD_PAIR,
    CMD_STOP,
    CMD_TRANSMIT,
    CMD_UP,
    PAIRING_DEVICE_ENUM_START,
    SIGNAL_DEVICE_EVENT,
    SIGNAL_DEVICE_EVENT_CAPTURE,
)
from tests.conftest import FakeTransport, written

# ---------------------------------------------------------------------------
# connect(): opens the serial port, verifies the stick, enters listening
# mode, then fetches the hub's device ID. Previously exercised only via the
# connected_api fixture (which sets the already-connected state directly,
# bypassing connect() entirely) or indirectly through a real failed handshake
# against a nonexistent port (Developer Tools' "Reset stick", which only
# reaches the outer connection exception, never verify_device()'s or
# _enter_listening_mode()'s own failure handling). verify_device(),
# _enter_listening_mode(), and get_device_id() are mocked here - their own
# wire-level behavior is a separate concern - so these tests are purely
# about connect() correctly sequencing and reacting to them.
# ---------------------------------------------------------------------------


async def test_connect_success_initializes_state_and_fetches_hub_id(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = SchellenbergUsbApi(hass, "/dev/fake-schellenberg")
    fake_transport = FakeTransport()
    monkeypatch.setattr(
        api_module.serialx,  # type: ignore[attr-defined]
        "create_serial_connection",
        AsyncMock(return_value=(fake_transport, Mock())),
    )
    monkeypatch.setattr(api, "verify_device", AsyncMock(return_value=True))
    monkeypatch.setattr(api, "_enter_listening_mode", AsyncMock(return_value=True))
    monkeypatch.setattr(api, "get_device_id", AsyncMock(return_value="A1B2C3"))

    result = await api.connect()

    assert result is True
    assert api._is_connected is True
    assert api._hub_id == "A1B2C3"
    assert api._transport is fake_transport  # type: ignore[comparison-overlap]


async def test_connect_resets_state_when_verification_fails(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stick that answers but fails verify_device() - not a genuine
    Schellenberg stick - must not be left half-connected.
    """
    api = SchellenbergUsbApi(hass, "/dev/fake-schellenberg")
    fake_transport = FakeTransport()
    monkeypatch.setattr(
        api_module.serialx,  # type: ignore[attr-defined]
        "create_serial_connection",
        AsyncMock(return_value=(fake_transport, Mock())),
    )
    monkeypatch.setattr(api, "verify_device", AsyncMock(return_value=False))

    result = await api.connect()

    assert result is False
    assert api._is_connected is False
    assert api._transport is None
    assert fake_transport.is_closing() is True
    # A real failed handshake schedules a retry via async_call_later, which
    # would otherwise outlive this test as a "lingering timer" - clean it up
    # the same way test_developer_tools_reset_stick already does.
    await api.disconnect()


async def test_connect_resets_state_when_listening_mode_fails(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same recovery as the verification-failure case above, for the other
    way a freshly opened port can fail to become usable: the device
    answers and verifies, but never actually reaches listening mode.
    """
    api = SchellenbergUsbApi(hass, "/dev/fake-schellenberg")
    fake_transport = FakeTransport()
    monkeypatch.setattr(
        api_module.serialx,  # type: ignore[attr-defined]
        "create_serial_connection",
        AsyncMock(return_value=(fake_transport, Mock())),
    )
    monkeypatch.setattr(api, "verify_device", AsyncMock(return_value=True))
    monkeypatch.setattr(api, "_enter_listening_mode", AsyncMock(return_value=False))

    result = await api.connect()

    assert result is False
    assert api._is_connected is False
    assert api._transport is None
    assert fake_transport.is_closing() is True
    await api.disconnect()


async def test_send_command_writes_ascii_with_crlf_terminator(
    connected_api: SchellenbergUsbApi,
) -> None:
    assert await connected_api.send_command("!?") is True
    assert written(connected_api) == [b"!?\r\n"]


async def test_control_blind_rejects_an_invalid_action(
    connected_api: SchellenbergUsbApi,
) -> None:
    assert await connected_api.control_blind("10", "99", device_id="ABCDEF") is False
    assert written(connected_api) == []


async def test_control_blind_is_blocked_while_disconnected(
    connected_api: SchellenbergUsbApi,
) -> None:
    connected_api._is_connected = False

    assert (
        await connected_api.control_blind("10", CMD_STOP, device_id="ABCDEF") is False
    )
    assert written(connected_api) == []


async def test_regular_commands_still_fully_serialize_through_the_transmit_lock(
    connected_api: SchellenbergUsbApi,
) -> None:
    """The normal lock logic must stay completely intact for regular commands."""
    await connected_api._transmit_lock.acquire()
    task = asyncio.ensure_future(connected_api.send_command("!?"))
    try:
        await asyncio.sleep(0)  # let the task run up to the lock and block
        assert not task.done()
        assert written(connected_api) == []
    finally:
        connected_api._transmit_lock.release()

    assert await task is True
    assert written(connected_api) == [b"!?\r\n"]


async def test_control_blind_sends_one_direct_unlocked_write(
    connected_api: SchellenbergUsbApi,
) -> None:
    """control_blind() must reach the wire immediately, never queued behind the lock.

    Regression: confirmed against real hardware that routing a blind move
    through the transmit lock and busy-retry cycle, while stop went through
    a separate lock-bypassing path, let the two fight over the stick's one
    RF transmitter - a move's own busy-retry could still be resending its
    payload seconds after a stop had already gone out, and on exhausting its
    retries would latch a busy fault that had nothing to do with the stop.
    control_blind() now writes directly, holding the transmit lock for
    nothing and never becoming eligible for that retry cycle in the first
    place - proven here by a held lock not blocking it at all.
    """
    await connected_api._transmit_lock.acquire()
    try:
        result = await asyncio.wait_for(
            connected_api.control_blind("10", CMD_STOP, device_id="ABCDEF"),
            timeout=1,
        )
    finally:
        connected_api._transmit_lock.release()

    assert result is True
    expected = f"{CMD_TRANSMIT}109{CMD_STOP}0000\r\n".encode("ascii")
    assert written(connected_api) == [expected]


async def test_control_blind_does_not_enter_the_busy_retry_cycle(
    connected_api: SchellenbergUsbApi,
) -> None:
    """A blind move must never be retried or latch a busy fault on tE.

    This link gives no delivery confirmation for a move to begin with (the
    USB stick's t1/t0/tE responses report only its own transmitter state,
    never what the motor did), so there is nothing worth retrying for open,
    close, or stop: control_blind() writes once, straight to the transport,
    and never sets _pending_retry_command - a tE that follows it must be a
    no-op, not the start of a multi-second retry-then-latch sequence.
    """
    assert await connected_api.control_blind("10", CMD_UP, device_id="ABCDEF") is True
    assert len(written(connected_api)) == 1

    connected_api._handle_message("tE")

    assert connected_api._retry_task is None
    assert connected_api.busy_latched is False
    assert len(written(connected_api)) == 1


async def test_control_blind_write_without_a_pending_command_stays_unflagged(
    connected_api: SchellenbergUsbApi,
) -> None:
    """The ordinary case - nothing pending via send_command() - must not flag.

    Companion to test_control_blind_does_not_enter_the_busy_retry_cycle
    above: that test covers the resulting tE behavior when nothing is
    pending; this checks the new _control_blind_write_pending_conflict
    flag itself stays False, matching control_blind()'s own
    `if self._pending_retry_command is not None` guard around setting it.
    """
    assert await connected_api.control_blind("10", CMD_UP, device_id="ABCDEF") is True

    assert connected_api._control_blind_write_pending_conflict is False


async def test_control_blind_write_while_command_pending_marks_next_tE_ambiguous(
    connected_api: SchellenbergUsbApi,
) -> None:
    """A tE arriving after an overlapping control_blind() write is ambiguous.

    control_blind() (open/close/stop) writes straight to the transport with
    no lock and no bookkeeping of its own - see its own docstring - so if it
    writes while a send_command()-tracked transmission (teach, pairing,
    raw-transmit, verify) is still waiting on its own t0, the next tE could
    be about either write; the protocol gives no id saying which. Blindly
    retrying the pending command in that situation risks resending it for
    no reason, which for teach specifically can change the motor's rotation
    direction (see teach_motor()'s own docstring). This tE must be skipped
    instead of guessed at, and it must not touch the pending command's own
    retry count or the sticky busy-latch fault flag - both of which stay
    reserved for a stick that is genuinely, unambiguously stuck.
    """
    assert await connected_api.send_command(f"{CMD_TRANSMIT}109{CMD_UP}0000") is True
    assert len(written(connected_api)) == 1

    assert await connected_api.control_blind("20", CMD_STOP, device_id="112233") is True
    assert len(written(connected_api)) == 2

    connected_api._handle_message("tE")

    # No resend of the pending command - the second write above was the
    # control_blind() one, not a retry of it.
    assert len(written(connected_api)) == 2
    assert connected_api._retry_task is None
    assert connected_api._retry_count == 0
    assert connected_api.busy_latched is False
    # The pending command itself is untouched - a later, unambiguous tE
    # (no intervening control_blind() write) must still retry it normally;
    # see the next test.
    assert connected_api._pending_retry_command == f"{CMD_TRANSMIT}109{CMD_UP}0000"
    assert connected_api._control_blind_write_pending_conflict is False


async def test_ambiguous_tE_does_not_block_a_later_unambiguous_retry(
    connected_api: SchellenbergUsbApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Skipping one ambiguous tE must not disable retrying altogether.

    Only the one reply that arrived while a control_blind() write was also
    in flight is ambiguous. If the stick is genuinely still busy afterwards,
    with no further control_blind() write in between, that next tE is
    unambiguous again and must retry the still-pending command normally.
    """

    async def instant_sleep(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", instant_sleep)

    assert await connected_api.send_command(f"{CMD_TRANSMIT}109{CMD_UP}0000") is True
    await connected_api.control_blind("20", CMD_STOP, device_id="112233")
    connected_api._handle_message("tE")  # ambiguous, skipped (previous test)
    assert len(written(connected_api)) == 2

    connected_api._handle_message("tE")  # no further control_blind() write since

    assert connected_api._retry_count == 1
    # The initial send_command() write, the control_blind() write, and now
    # one resend of the still-pending command.
    assert len(written(connected_api)) == 3


async def test_a_fresh_pending_command_clears_a_stale_conflict_flag(
    connected_api: SchellenbergUsbApi,
) -> None:
    """A new pending command's own window must start clean.

    _control_blind_write_pending_conflict must not leak from a previous,
    already-finished pending command into a completely new one that has not
    raced with control_blind() at all.
    """
    connected_api._control_blind_write_pending_conflict = True

    assert await connected_api.send_command(f"{CMD_TRANSMIT}109{CMD_UP}0000") is True

    assert connected_api._control_blind_write_pending_conflict is False


async def test_busy_retry_stops_after_max_attempts(
    connected_api: SchellenbergUsbApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stick that stays busy is retried up to the limit, then latched.

    Uses a raw transmit sent through send_command() rather than
    control_blind() - a blind move never enters this cycle at all (see
    test_control_blind_does_not_enter_the_busy_retry_cycle below); this
    test is about the generic machinery that teach/pairing/raw-transmit
    still rely on.

    Regression: Home Assistant's task factories start a created task
    "eagerly" - it runs synchronously up to its first real suspension point
    before async_create_task() even returns the Task object. When nothing
    inside _retry_command_after_delay actually suspends (as here, where
    asyncio.sleep is mocked to resolve instantly and the transmit lock and
    transmitter-idle event are both immediately available), the whole retry
    - including its own `finally: self._retry_task = None` - runs to
    completion before that assignment line. Overwriting self._retry_task
    unconditionally afterwards would then stomp that already-finished
    cleanup and leave a stale, already-done Task sitting in self._retry_task
    forever instead of None.
    """

    async def instant_sleep(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", instant_sleep)

    assert await connected_api.send_command(f"{CMD_TRANSMIT}109{CMD_UP}0000") is True
    assert len(written(connected_api)) == 1

    for _ in range(TRANSMIT_MAX_RETRIES + 1):
        connected_api._handle_message("tE")

    assert connected_api.busy_latched is True
    assert connected_api._retry_count == 0
    assert connected_api._retry_task is None
    # The initial write, plus one resend per retry - the final "tE" (the
    # +1 above) finds the limit already reached and abandons without
    # resending again.
    assert len(written(connected_api)) == 1 + TRANSMIT_MAX_RETRIES


async def test_a_stick_accepting_a_transmission_resets_the_retry_count(
    connected_api: SchellenbergUsbApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A "t1" (transmit accepted) marks the stick as no longer stuck.

    Only the retry count is this test's concern - busy_latched is a
    separate flag that t1 never touches (see _handle_message: only t0,
    the matching completion ACK, clears it). A trailing
    `assert busy_latched is False` used to sit here too, but busy_latched
    was never True in this scenario to begin with (only one "tE", far
    under the retry limit), so that assertion was trivially true and
    wrongly implied t1 was responsible for clearing it. The real
    self-healing behaviour - busy_latched flipping from True back to
    False - is covered on its own by
    test_t0_self_heals_busy_latched_without_a_disconnect below.
    """

    async def instant_sleep(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", instant_sleep)

    await connected_api.send_command(f"{CMD_TRANSMIT}109{CMD_UP}0000")
    connected_api._handle_message("tE")
    assert connected_api._retry_count == 1

    connected_api._handle_message("t1")

    assert connected_api._retry_count == 0


async def test_t0_resets_the_last_transmit_source_to_internal(
    connected_api: SchellenbergUsbApi,
) -> None:
    """A completed ACK cycle ("t0") must not leave a stale source behind.

    Otherwise later, unrelated t1/t0/tE traffic would keep being logged
    (or not logged) under a source that has nothing to do with it - see
    the comment above this branch in _handle_message.
    """
    assert (
        await connected_api.send_command(
            f"{CMD_TRANSMIT}109{CMD_UP}0000", source="developer_tools"
        )
        is True
    )
    assert connected_api._last_transmit_source == "developer_tools"

    connected_api._handle_message("t0")

    assert connected_api._last_transmit_source == "internal"


async def test_t0_self_heals_busy_latched_without_a_disconnect(
    connected_api: SchellenbergUsbApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stick that later answers cleanly must clear a stale busy latch.

    test_handle_connection_lost_resets_retry_and_busy_state already covers
    the reset-on-disconnect path; this covers the other way busy_latched
    comes back down - the stick simply recovering on its own and
    completing a transmission normally, with no disconnect in between.
    """

    async def instant_sleep(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", instant_sleep)

    assert await connected_api.send_command(f"{CMD_TRANSMIT}109{CMD_UP}0000") is True
    for _ in range(TRANSMIT_MAX_RETRIES + 1):
        connected_api._handle_message("tE")
    assert connected_api.busy_latched is True

    connected_api._handle_message("t0")

    assert connected_api.busy_latched is False


def test_handle_connection_lost_resets_retry_and_busy_state(
    connected_api: SchellenbergUsbApi,
) -> None:
    connected_api._pending_retry_command = f"{CMD_TRANSMIT}10900000"
    connected_api._busy_latched = True
    connected_api._retry_count = 2
    connected_api._transmitter_active = True
    connected_api._transmitter_idle.clear()
    connected_api._control_blind_write_pending_conflict = True

    connected_api.handle_connection_lost(connected_api._protocol, Exception("gone"))  # type: ignore[arg-type]

    assert connected_api.is_connected is False
    assert connected_api.busy_latched is False
    assert connected_api._retry_count == 0
    assert connected_api._pending_retry_command is None
    assert connected_api.transmitter_active is False
    assert connected_api._transmitter_idle.is_set() is True
    assert connected_api._control_blind_write_pending_conflict is False
    # A reconnect was scheduled as a result; don't leave it pending past
    # the end of the test.
    connected_api._cancel_scheduled_reconnect()


async def test_disconnect_cancels_a_pending_retry_task(
    connected_api: SchellenbergUsbApi,
) -> None:
    retry_task = asyncio.ensure_future(asyncio.sleep(100))
    connected_api._retry_task = retry_task

    await connected_api.disconnect()

    assert connected_api._retry_task is None
    with pytest.raises(asyncio.CancelledError):
        await retry_task


# ---------------------------------------------------------------------------
# teach_motor(): the two-phase (teach_60, finish_40) remote-assisted teach
# sequence used by Developer Tools' own teach_motor step - config_flow.py's
# own tests only ever exercise this through that call site, never
# teach_motor() itself in isolation.
# ---------------------------------------------------------------------------


async def test_teach_motor_sends_teach_then_finish_and_returns_true(
    connected_api: SchellenbergUsbApi,
) -> None:
    assert (
        await connected_api.teach_motor(
            "10", device_id="ABCDEF", source="developer_tools"
        )
        is True
    )
    assert written(connected_api) == [
        f"{CMD_TRANSMIT}109{CMD_PAIR}0000\r\n".encode("ascii"),
        f"{CMD_TRANSMIT}109{CMD_ALLOW_PAIRING}0000\r\n".encode("ascii"),
    ]


async def test_teach_motor_blocked_while_disconnected(
    connected_api: SchellenbergUsbApi,
) -> None:
    connected_api._is_connected = False

    assert await connected_api.teach_motor("10", device_id="ABCDEF") is False
    assert written(connected_api) == []


async def test_teach_motor_returns_false_when_first_phase_write_fails(
    connected_api: SchellenbergUsbApi,
) -> None:
    """A transport failure on teach_60 itself must stop the sequence there.

    finish_40 must never be attempted once the first phase's own write
    failed - sending it alone, without the matching teach_60, would not
    match the documented 60-then-40 pairing sequence.
    """
    connected_api._transport.write = Mock(  # type: ignore[method-assign,union-attr]
        side_effect=OSError("gone")
    )

    assert await connected_api.teach_motor("10", device_id="ABCDEF") is False
    assert written(connected_api) == []


async def test_teach_motor_latches_busy_when_ack_wait_times_out(
    connected_api: SchellenbergUsbApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Teach must latch busy the same way send_command()'s own retry cycle
    does when the stick never reports its transmitter idle - here that is
    teach_motor()'s own _wait_for_transmitter_idle() call between phases,
    not the busy/retry cycle, so it needs its own coverage.
    """

    async def _time_out_without_awaiting(
        coro: Coroutine[Any, Any, Any], *_args: Any, **_kwargs: Any
    ) -> None:
        # asyncio.wait_for is replaced wholesale below, so the coroutine the
        # real wait_for would normally await (self._transmitter_idle.wait())
        # is instead just handed to this side_effect and never awaited by
        # anything. Closing it here disposes of it cleanly, avoiding a
        # "coroutine was never awaited" warning, while still raising
        # TimeoutError to simulate an immediate timeout.
        coro.close()
        raise TimeoutError

    connected_api._transmitter_idle.clear()
    monkeypatch.setattr(
        asyncio, "wait_for", AsyncMock(side_effect=_time_out_without_awaiting)
    )

    assert connected_api.busy_latched is False
    assert await connected_api.teach_motor("10", device_id="ABCDEF") is False
    assert connected_api.busy_latched is True
    # finish_40 must never be sent once teach_60's own ACK wait timed out.
    assert written(connected_api) == [
        f"{CMD_TRANSMIT}109{CMD_PAIR}0000\r\n".encode("ascii")
    ]


# ---------------------------------------------------------------------------
# pair_device_and_wait(): listens for an unknown transmitter's frame, then
# runs the same teach_60/finish_40 sequence on a freshly assigned enum.
# Every config_flow.py pairing test monkeypatches this method outright
# (a controlled (device_id, enum) return value for branching), so its own
# wait/timeout/teach logic had no coverage anywhere before this.
# ---------------------------------------------------------------------------


async def test_pair_device_and_wait_success_sends_teach_then_finish(
    connected_api: SchellenbergUsbApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = asyncio.ensure_future(connected_api.pair_device_and_wait())
    await asyncio.sleep(0)  # real yield: let it send "sp" and park on the future
    assert written(connected_api) == [f"{CMD_GET_PARAM_P}\r\n".encode("ascii")]

    # Pairing succeeding makes the "finally" cleanup sleep for real (see
    # _stop_pairing_mode(delay=True)); patch only now, after the task is
    # already parked on the still-unresolved pairing future, so this does
    # not swallow the real yield above.
    async def instant_sleep(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", instant_sleep)

    # Simulates an "sl" pairing/list response frame from a new transmitter.
    connected_api._handle_message("sl00BEABCDEF")

    result = await task

    assert result == ("ABCDEF", f"{PAIRING_DEVICE_ENUM_START:02X}")
    enum = result[1]
    assert written(connected_api) == [
        f"{CMD_GET_PARAM_P}\r\n".encode("ascii"),
        f"{CMD_TRANSMIT}{enum}9{CMD_PAIR}0000\r\n".encode("ascii"),
        f"{CMD_TRANSMIT}{enum}9{CMD_ALLOW_PAIRING}0000\r\n".encode("ascii"),
        f"{CMD_GET_PARAM_P}\r\n".encode("ascii"),  # "sp" again: stop-pairing cleanup
    ]
    assert connected_api.pairing_active is False


async def test_pair_device_and_wait_returns_none_when_not_transmit_capable(
    connected_api: SchellenbergUsbApi,
) -> None:
    connected_api._is_connected = False

    assert await connected_api.pair_device_and_wait() is None
    assert written(connected_api) == []


async def test_pair_device_and_wait_refuses_a_second_concurrent_pairing(
    connected_api: SchellenbergUsbApi,
) -> None:
    connected_api._pairing_future = connected_api.hass.loop.create_future()

    assert await connected_api.pair_device_and_wait() is None
    assert written(connected_api) == []


async def test_pair_device_and_wait_times_out_when_nothing_responds(
    connected_api: SchellenbergUsbApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_module, "PAIRING_TIMEOUT", 0.01)

    result = await connected_api.pair_device_and_wait()

    assert result is None
    assert connected_api.pairing_active is False


# ---------------------------------------------------------------------------
# send_raw_transmit(): Developer Tools' raw-payload escape hatch. Also only
# ever exercised through config_flow.py's send_raw_command step before.
# ---------------------------------------------------------------------------


async def test_send_raw_transmit_rejects_an_invalid_payload(
    connected_api: SchellenbergUsbApi,
) -> None:
    with pytest.raises(ValueError):
        await connected_api.send_raw_transmit("not-a-valid-payload")
    assert written(connected_api) == []


async def test_send_raw_transmit_blocked_while_disconnected(
    connected_api: SchellenbergUsbApi,
) -> None:
    connected_api._is_connected = False

    payload = f"{CMD_TRANSMIT}109{CMD_UP}0000"
    assert await connected_api.send_raw_transmit(payload) is False
    assert written(connected_api) == []


async def test_send_raw_transmit_success_normalizes_case_and_sends_payload(
    connected_api: SchellenbergUsbApi,
) -> None:
    payload = f"{CMD_TRANSMIT}109{CMD_UP}0000"
    mixed_case = "Ss" + payload[2:].lower()

    assert await connected_api.send_raw_transmit(mixed_case) is True
    assert written(connected_api) == [f"{payload}\r\n".encode("ascii")]


# ---------------------------------------------------------------------------
# _handle_message(): the "ss" status frame and the "RFTU_" verification
# reply. Every other test above feeds _handle_message() an ACK ("t1"/"t0"/
# "tE"), a device-ID reply ("sr..."), or a pairing-list reply ("sl...") -
# the "ss" frame (a real remote press or motor echo overheard on the
# channel) and "RFTU_" (the device-verification handshake's own reply) were
# never fed to it by any test. test_cover.py's own activity-event-handler
# tests fire the SIGNAL_DEVICE_EVENT dispatcher signal directly instead of
# through this parser, so the parsing itself - the only path by which a
# real remote press ever reaches a cover entity - had no coverage at all.
# ---------------------------------------------------------------------------


async def test_handle_message_ss_frame_updates_tracking_and_dispatches_signal(
    connected_api: SchellenbergUsbApi,
) -> None:
    connected_api.register_entity("ABCDEF", "10", "Living Room Blind")
    received: list[str] = []

    # A plain received.append would be seen as a non-@callback listener, so
    # async_dispatcher_send() hands it to the executor instead of calling it
    # inline - making the assertion below a race against that executor
    # thread. @callback marks it event-loop-safe so it runs synchronously.
    @callback
    def _on_event(command: str) -> None:
        received.append(command)

    async_dispatcher_connect(
        connected_api.hass, f"{SIGNAL_DEVICE_EVENT}_ABCDEF_10", _on_event
    )

    # ss + enum(10) + device_id(ABCDEF) + incrementor(0000, ignored) +
    # command(CMD_UP) + padding(00, ignored) + signal(00, ignored)
    connected_api._handle_message("ss" + "10" + "ABCDEF" + "0000" + CMD_UP + "0000")

    frame = connected_api.get_last_received("ABCDEF", "10")
    assert frame is not None
    assert frame["matched"] is True
    assert frame["identity_role"] == "primary"
    assert frame["interpreted_command"] == "open"
    assert frame["position_tracking"] is True
    tracking_frame = connected_api.get_last_primary_tracking_frame("ABCDEF", "10")
    assert tracking_frame is not None
    assert tracking_frame["command"] == CMD_UP
    assert received == [CMD_UP]


async def test_handle_message_ss_frame_unmatched_device_skips_the_per_entity_signal(
    connected_api: SchellenbergUsbApi,
) -> None:
    """An "ss" frame from a device nobody registered still gets recorded
    (matched=False, for diagnostics) and still dispatches the ID-only
    signal - used for calibration before any entity exists - but the
    second, entity-specific signal only fires once something is actually
    registered for that exact device_id/enum (see the test above).
    """
    id_only: list[str] = []
    id_and_enum: list[str] = []

    # See the @callback note in the test above - without it these listeners
    # would run on the executor, racing the assertions below instead of
    # being guaranteed to have run by the time _handle_message() returns.
    @callback
    def _on_id_only(command: str) -> None:
        id_only.append(command)

    @callback
    def _on_id_and_enum(command: str) -> None:
        id_and_enum.append(command)

    async_dispatcher_connect(
        connected_api.hass, f"{SIGNAL_DEVICE_EVENT}_FEDCBA", _on_id_only
    )
    async_dispatcher_connect(
        connected_api.hass, f"{SIGNAL_DEVICE_EVENT}_FEDCBA_20", _on_id_and_enum
    )

    connected_api._handle_message("ss" + "20" + "FEDCBA" + "0000" + CMD_STOP + "0000")

    frame = connected_api.get_last_received("FEDCBA", "20")
    assert frame is not None
    assert frame["matched"] is False
    assert frame["identity_role"] == "unmatched"
    assert id_only == [CMD_STOP]
    assert id_and_enum == []


async def test_handle_message_ss_frame_resolves_pairing_future_for_unknown_device(
    connected_api: SchellenbergUsbApi,
) -> None:
    """While pair_device_and_wait() is listening, an "ss" frame from a
    device nobody has registered yet is treated as the pairing candidate:
    the pairing future resolves, and the function returns immediately
    afterwards - the diagnostic frame above is still recorded, but the
    dispatcher signal further down is never reached for a device that
    isn't configured as a cover yet.
    """
    received: list[str] = []
    async_dispatcher_connect(
        connected_api.hass, f"{SIGNAL_DEVICE_EVENT}_112233", received.append
    )
    connected_api._pairing_future = connected_api.hass.loop.create_future()

    connected_api._handle_message("ss" + "30" + "112233" + "0000" + CMD_STOP + "0000")

    assert connected_api._pairing_future.done()
    assert connected_api._pairing_future.result() == "112233"
    assert received == []


async def test_handle_message_ss_frame_skips_capture_signal_when_no_capture_active(
    connected_api: SchellenbergUsbApi,
) -> None:
    """No status-frame capture window is open, so SIGNAL_DEVICE_EVENT_CAPTURE
    must not fire - only the two existing, identity-specific signals may.
    """
    received: list[tuple[str, str]] = []

    @callback
    def _on_capture_event(device_id: str, command: str) -> None:
        received.append((device_id, command))

    async_dispatcher_connect(
        connected_api.hass, SIGNAL_DEVICE_EVENT_CAPTURE, _on_capture_event
    )

    connected_api._handle_message("ss" + "10" + "ABCDEF" + "0000" + CMD_UP + "0000")

    assert received == []


async def test_handle_message_ss_frame_broadcasts_capture_signal_during_capture(
    connected_api: SchellenbergUsbApi,
) -> None:
    """Regression test for the calibration root-cause fix: while a status-
    frame capture window is open (calibration, remote-status discovery,
    teach_motor), every "ss" frame must also go out on the capture-wide
    SIGNAL_DEVICE_EVENT_CAPTURE signal - independent of normalized_device_id
    and of whether anything is registered for it - so
    options_flow_calibration.py's movement-wait listeners can still see a
    device whose status identity differs from the command/pairing identity
    they are keyed to (see SIGNAL_DEVICE_EVENT_CAPTURE's own comment in
    const.py).
    """
    received: list[tuple[str, str]] = []

    @callback
    def _on_capture_event(device_id: str, command: str) -> None:
        received.append((device_id, command))

    async_dispatcher_connect(
        connected_api.hass, SIGNAL_DEVICE_EVENT_CAPTURE, _on_capture_event
    )
    connected_api.start_status_frame_capture(phase="opening")

    connected_api._handle_message("ss" + "10" + "FEDCBA" + "0000" + CMD_UP + "0000")

    assert received == [("FEDCBA", CMD_UP)]


async def test_handle_message_rftu_resolves_verify_future_and_records_mode(
    connected_api: SchellenbergUsbApi,
) -> None:
    """The device-verification handshake's own reply line. connect()'s own
    tests above mock verify_device() itself rather than feeding it this
    wire-level reply, so this parsing had no coverage anywhere.
    """
    connected_api._verify_future = connected_api.hass.loop.create_future()

    connected_api._handle_message("RFTU_V20 F:20180510_DFBD B:1")

    assert connected_api._device_version == "RFTU_V20"
    assert connected_api._device_mode == "initial"  # B:1
    assert connected_api._verify_future.done()
    assert connected_api._verify_future.result() is True
