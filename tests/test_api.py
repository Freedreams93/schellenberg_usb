"""Core protocol tests for SchellenbergUsbApi.

Covers the transmit lock (regular, non-movement commands must still fully
serialize through it), the direct unlocked write used for blind moves
(open/close/stop - see control_blind), the busy/retry cycle (including the
eager-task-start regression described below), and the
connection-loss/disconnect cleanup of that same retry state.
"""

from __future__ import annotations

import asyncio

import pytest
from conftest import written

from custom_components.schellenberg_usb.api import (
    TRANSMIT_MAX_RETRIES,
    SchellenbergUsbApi,
)
from custom_components.schellenberg_usb.const import CMD_STOP, CMD_TRANSMIT, CMD_UP


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
    """A "t1" (transmit accepted) marks the stick as no longer stuck."""

    async def instant_sleep(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", instant_sleep)

    await connected_api.send_command(f"{CMD_TRANSMIT}109{CMD_UP}0000")
    connected_api._handle_message("tE")
    assert connected_api._retry_count == 1

    connected_api._handle_message("t1")

    assert connected_api._retry_count == 0
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
