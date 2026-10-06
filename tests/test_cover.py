"""Tests for the cover entity's command dispatch and stop-at-target logic.

SchellenbergCover instances here are built directly (not through
async_setup_entry) with a MagicMock api, since these tests are only about
the entity's own decision logic: which commands it sends and when.
async_write_ha_state() is stubbed out because these tests assert on command
dispatch, not on the entity's reported state, and a bare entity built this
way was never handed a real entity platform to write state through.

The final section (test_async_setup_entry_...) is the one exception: it
drives the real async_setup_entry() against a MockConfigEntry/ConfigSubentry
pair, because that function's own job - turning a saved blind subentry into
a device-registry device plus one SchellenbergCover, and migrating a
pre-stable-ID entity's unique_id when it finds one - cannot be exercised by
building the entity directly.
"""

from __future__ import annotations

import asyncio
import time
from types import MappingProxyType
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from homeassistant.components.cover import ATTR_CURRENT_POSITION, ATTR_POSITION
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant, State
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.dispatcher import async_dispatcher_send
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.schellenberg_usb.const import (
    CMD_DOWN,
    CMD_STOP,
    CMD_UP,
    CONF_CLOSE_TIME,
    CONF_DEVICE_ENUM,
    CONF_DEVICE_ID,
    CONF_OPEN_TIME,
    CONF_SERIAL_PORT,
    CONF_STATUS_DEVICE_ID,
    CONF_STATUS_ENUM,
    DOMAIN,
    EVENT_STARTED_MOVING_DOWN,
    EVENT_STARTED_MOVING_UP,
    EVENT_STOPPED,
    SIGNAL_DEVICE_EVENT,
    SUBENTRY_TYPE_BLIND,
)
from custom_components.schellenberg_usb.cover import (
    SchellenbergCover,
    async_setup_entry,
)
from custom_components.schellenberg_usb.device_registry_compat import (
    async_get_device_by_identifier_compat,
)


class _FakeClock:
    """A controllable stand-in for time.monotonic().

    _update_position() and the position-tracking loop only ever care about
    *elapsed* time, so a fake clock that advances by an explicit amount each
    time the loop's own asyncio.sleep() is awaited reproduces real elapsed-
    time behavior deterministically and instantly, without any real
    waiting.
    """

    def __init__(self, start: float = 1_000.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def _build_cover(hass: HomeAssistant, api: Any, **overrides: Any) -> SchellenbergCover:
    kwargs: dict[str, Any] = {
        "api": api,
        "device_id": "ABCDEF",
        "device_enum": "10",
        "device_name": "Living Room Blind",
    }
    kwargs.update(overrides)
    cover = SchellenbergCover(**kwargs)
    cover.hass = hass
    cover.entity_id = "cover.living_room_blind"
    cover.async_write_ha_state = MagicMock()  # type: ignore[method-assign, misc]
    return cover


async def test_async_open_cover_sends_up_via_control_blind(
    hass: HomeAssistant,
) -> None:
    api = MagicMock()
    api.control_blind = AsyncMock(return_value=True)
    cover = _build_cover(hass, api)
    cover._start_position_tracking = MagicMock()  # type: ignore[method-assign]

    await cover.async_open_cover()

    api.control_blind.assert_awaited_once_with(
        cover._command_enum, CMD_UP, device_id=cover._command_device_id
    )


async def test_async_close_cover_sends_down_via_control_blind(
    hass: HomeAssistant,
) -> None:
    api = MagicMock()
    api.control_blind = AsyncMock(return_value=True)
    cover = _build_cover(hass, api)
    cover._start_position_tracking = MagicMock()  # type: ignore[method-assign]

    await cover.async_close_cover()

    api.control_blind.assert_awaited_once_with(
        cover._command_enum, CMD_DOWN, device_id=cover._command_device_id
    )


async def test_async_open_cover_raises_when_control_blind_fails(
    hass: HomeAssistant,
) -> None:
    """Mirrors test_async_stop_cover_raises_when_control_blind_fails for open.

    Regression guard: a failed open used to leave _attr_is_opening stuck
    True, reporting a movement that never physically happened.
    """
    api = MagicMock()
    api.control_blind = AsyncMock(return_value=False)
    api.transmit_block_reason = "disconnected"  # stable code, see api.py
    cover = _build_cover(hass, api)
    cover._start_position_tracking = MagicMock()  # type: ignore[method-assign]

    with pytest.raises(HomeAssistantError):
        await cover.async_open_cover()

    assert cover._attr_is_opening is False
    assert cover._move_start_time is None


async def test_async_close_cover_raises_when_control_blind_fails(
    hass: HomeAssistant,
) -> None:
    """Mirrors test_async_stop_cover_raises_when_control_blind_fails for close.

    Regression guard: a failed close used to leave _attr_is_closing stuck
    True, reporting a movement that never physically happened.
    """
    api = MagicMock()
    api.control_blind = AsyncMock(return_value=False)
    api.transmit_block_reason = "disconnected"  # stable code, see api.py
    cover = _build_cover(hass, api)
    cover._start_position_tracking = MagicMock()  # type: ignore[method-assign]

    with pytest.raises(HomeAssistantError):
        await cover.async_close_cover()

    assert cover._attr_is_closing is False
    assert cover._move_start_time is None


async def test_async_stop_cover_sends_stop_via_control_blind(
    hass: HomeAssistant,
) -> None:
    """Stop is sent exactly like open/close: one direct control_blind() call.

    async_stop_cover()'s own initiating stop always went through this same
    control_blind() path, same as open/close - it was only the two automatic
    follow-up stop frames sent after it that used to bypass the transmit
    lock via send_priority_stop(). Confirmed real-hardware testing showed one
    of those follow-ups could still collide with an in-flight open/close
    command's own busy-retry, fighting over the stick's one RF transmitter.
    With open/close no longer using that busy-retry cycle either, the
    follow-ups (and send_priority_stop() itself) are gone, and there is
    nothing left to bypass here.
    """
    api = MagicMock()
    api.control_blind = AsyncMock(return_value=True)
    cover = _build_cover(hass, api)

    await cover.async_stop_cover()

    api.control_blind.assert_awaited_once_with(
        cover._command_enum, CMD_STOP, device_id=cover._command_device_id
    )


async def test_async_stop_cover_raises_when_control_blind_fails(
    hass: HomeAssistant,
) -> None:
    api = MagicMock()
    api.control_blind = AsyncMock(return_value=False)
    api.transmit_block_reason = "disconnected"  # stable code, see api.py
    cover = _build_cover(hass, api)

    with pytest.raises(HomeAssistantError):
        await cover.async_stop_cover()


async def test_async_stop_cover_restarts_tracking_when_stop_fails_while_moving(
    hass: HomeAssistant,
) -> None:
    """Regression: a failed stop used to leave a moving cover's position frozen.

    async_stop_cover() restores the pre-stop moving state when control_blind()
    fails, on the theory that a blind whose stop never reached the stick is
    probably still moving - but restoring _move_start_time etc. alone does not
    resurrect the position-tracking task, which was already cancelled earlier
    in the same call. Without an explicit restart, the entity would report
    is_closing/is_opening forever with a position frozen at the instant of the
    failed stop, until some unrelated later command happened to start tracking
    again.
    """
    api = MagicMock()
    api.control_blind = AsyncMock(return_value=False)
    api.transmit_block_reason = "disconnected"  # stable code, see api.py
    cover = _build_cover(hass, api)
    cover._attr_is_closing = True
    cover._move_start_time = 123.0
    cover._move_start_position = 80
    cover._target_position = 0
    cover._start_position_tracking = MagicMock()  # type: ignore[method-assign]

    with pytest.raises(HomeAssistantError):
        await cover.async_stop_cover()

    assert cover._attr_is_closing is True
    assert cover._attr_is_opening is False
    assert cover._move_start_time == 123.0
    assert cover._move_start_position == 80
    assert cover._target_position == 0
    cover._start_position_tracking.assert_called_once_with()


async def test_async_stop_cover_does_not_restart_tracking_when_already_idle(
    hass: HomeAssistant,
) -> None:
    """A failed stop on an already-idle cover must not spin up tracking.

    Companion to the regression test above: guards the `if previous_is_opening
    or previous_is_closing` check so a fix for the moving case can't
    regress into unconditionally restarting tracking.
    """
    api = MagicMock()
    api.control_blind = AsyncMock(return_value=False)
    api.transmit_block_reason = "disconnected"  # stable code, see api.py
    cover = _build_cover(hass, api)
    cover._start_position_tracking = MagicMock()  # type: ignore[method-assign]

    with pytest.raises(HomeAssistantError):
        await cover.async_stop_cover()

    cover._start_position_tracking.assert_not_called()


async def test_position_loop_stops_at_a_partial_target(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def instant_sleep(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", instant_sleep)
    api = MagicMock()
    api.control_blind = AsyncMock(return_value=True)
    cover = _build_cover(hass, api)
    monkeypatch.setattr(
        cover,
        "_update_position",
        lambda: setattr(cover, "_attr_current_cover_position", 40),
    )
    cover._attr_is_closing = True
    cover._attr_current_cover_position = 50
    cover._target_position = 40

    await cover._async_position_update_loop()

    api.control_blind.assert_awaited_once_with(
        cover._command_enum, CMD_STOP, device_id=cover._command_device_id
    )
    assert cover._target_position is None


async def test_position_loop_reaching_an_endstop_target_sends_no_active_stop(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reaching an explicit target of exactly 0 percent (full close) sends no stop.

    The motor's own physical end-stop handles full travel, unlike stopping
    partway which needs an actual stop command.
    """

    async def instant_sleep(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", instant_sleep)
    api = MagicMock()
    api.control_blind = AsyncMock(return_value=True)
    cover = _build_cover(hass, api)
    monkeypatch.setattr(
        cover,
        "_update_position",
        lambda: setattr(cover, "_attr_current_cover_position", 0),
    )
    cover._attr_is_closing = True
    cover._attr_current_cover_position = 5
    cover._target_position = 0

    await cover._async_position_update_loop()

    api.control_blind.assert_not_awaited()


# ---------------------------------------------------------------------------
# _update_position(): the elapsed-time position estimate itself.
#
# The stick has no return channel from the motor - it never learns the
# shutter's real position while moving. Every position shown in Home
# Assistant while a move is in progress is this calculation: elapsed time
# since the move started, divided by the configured (or default) travel
# time for that direction, applied to the position the move started from.
# This is the actual core logic the UI's "live" position during a move is
# built on, so it gets its own direct tests with controlled elapsed time,
# not just indirectly through the loop tests above (which replace this
# method with a stub).
# ---------------------------------------------------------------------------


async def test_update_position_estimates_opening_progress_from_elapsed_time(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _FakeClock(1_000.0)
    monkeypatch.setattr(time, "monotonic", clock)
    api = MagicMock()
    cover = _build_cover(hass, api, device_data={CONF_OPEN_TIME: 20.0})
    cover._attr_is_opening = True
    cover._move_start_time = clock()
    cover._move_start_position = 0

    clock.advance(5.0)  # a quarter of the 20s open travel time
    cover._update_position()

    assert cover._attr_current_cover_position == 25
    assert cover._attr_is_closed is False


async def test_update_position_estimates_closing_progress_from_elapsed_time(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _FakeClock(1_000.0)
    monkeypatch.setattr(time, "monotonic", clock)
    api = MagicMock()
    cover = _build_cover(hass, api, device_data={CONF_CLOSE_TIME: 10.0})
    cover._attr_is_closing = True
    cover._move_start_time = clock()
    cover._move_start_position = 100

    clock.advance(5.0)  # half of the 10s close travel time
    cover._update_position()

    assert cover._attr_current_cover_position == 50


async def test_update_position_clamps_overshoot_to_the_travel_limits(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Elapsed time past the travel time must clamp to 0/100, never overshoot.

    This is what makes it safe for the tracking loop to rely on this method
    alone to detect "fully closed"/"fully open" (see the loop tests below) -
    an estimate of -8% or 107% would break that comparison.
    """
    clock = _FakeClock(1_000.0)
    monkeypatch.setattr(time, "monotonic", clock)
    api = MagicMock()
    cover = _build_cover(hass, api, device_data={CONF_CLOSE_TIME: 10.0})
    cover._attr_is_closing = True
    cover._move_start_time = clock()
    cover._move_start_position = 20

    clock.advance(60.0)  # far past the 10s close travel time
    cover._update_position()

    assert cover._attr_current_cover_position == 0
    assert cover._attr_is_closed is True


async def test_update_position_is_a_noop_without_a_move_in_progress(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No _move_start_time means no move is tracked - nothing to estimate."""
    clock = _FakeClock(1_000.0)
    monkeypatch.setattr(time, "monotonic", clock)
    api = MagicMock()
    cover = _build_cover(hass, api)
    cover._attr_current_cover_position = 42
    cover._move_start_time = None
    cover._move_start_position = None

    cover._update_position()

    assert cover._attr_current_cover_position == 42


# ---------------------------------------------------------------------------
# _async_position_update_loop(): the two branches reached purely by elapsed
# time with no explicit _target_position - a plain async_open_cover() or
# async_close_cover() all the way to an end stop, as opposed to a partial
# set_cover_position() target (already covered above) or an explicit 0/100
# target (also already covered above). These use the loop's own real
# _update_position() (not stubbed), advancing a fake clock each time the
# loop's asyncio.sleep() is awaited, so the loop actually runs its own
# elapsed-time math to termination exactly as it would in production.
# ---------------------------------------------------------------------------


async def test_position_loop_detects_fully_closed_by_elapsed_time_alone(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _FakeClock(1_000.0)
    monkeypatch.setattr(time, "monotonic", clock)

    async def advancing_sleep(seconds: float, *_a: object, **_kw: object) -> None:
        clock.advance(seconds)

    monkeypatch.setattr(asyncio, "sleep", advancing_sleep)
    api = MagicMock()
    api.control_blind = AsyncMock(return_value=True)
    cover = _build_cover(hass, api, device_data={CONF_CLOSE_TIME: 1.0})
    cover._attr_is_closing = True
    cover._attr_current_cover_position = 100
    cover._move_start_time = clock()
    cover._move_start_position = 100
    cover._target_position = None

    await cover._async_position_update_loop()

    assert cover._attr_current_cover_position == 0
    assert cover._attr_is_closed is True
    assert cover._attr_is_closing is False
    # Reaching 0/100 by elapsed time alone relies on the motor's own
    # end-stop, same as an explicit target of exactly 0/100 - no active
    # stop command is sent for either.
    api.control_blind.assert_not_awaited()


async def test_position_loop_detects_fully_open_by_elapsed_time_alone(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _FakeClock(1_000.0)
    monkeypatch.setattr(time, "monotonic", clock)

    async def advancing_sleep(seconds: float, *_a: object, **_kw: object) -> None:
        clock.advance(seconds)

    monkeypatch.setattr(asyncio, "sleep", advancing_sleep)
    api = MagicMock()
    api.control_blind = AsyncMock(return_value=True)
    cover = _build_cover(hass, api, device_data={CONF_OPEN_TIME: 1.0})
    cover._attr_is_opening = True
    cover._attr_current_cover_position = 0
    cover._move_start_time = clock()
    cover._move_start_position = 0
    cover._target_position = None

    await cover._async_position_update_loop()

    assert cover._attr_current_cover_position == 100
    assert cover._attr_is_closed is False
    assert cover._attr_is_opening is False
    api.control_blind.assert_not_awaited()


async def test_waiting_for_full_travel_resync_holds_until_the_full_travel_time_elapses(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_waiting_for_full_travel_resync() is the loop's gate against declaring
    "fully closed/open" too early after a restart.

    Set right after Home Assistant restarts (see async_open_cover's/
    async_close_cover's own _full_travel_resync_direction logic), it means
    this move's elapsed time must cover the *entire* configured travel time
    before an end stop reached by elapsed time alone is trusted - guarding
    against a stale/wrong _move_start_position restored from a previous run
    making the loop declare "fully closed" after only a fraction of the
    real travel. Tested directly against this pure gate rather than by
    running the full loop, which would make the same point far less
    directly and at the mercy of the mocked loop's own scheduling.
    """
    clock = _FakeClock(1_000.0)
    monkeypatch.setattr(time, "monotonic", clock)
    api = MagicMock()
    cover = _build_cover(hass, api, device_data={CONF_CLOSE_TIME: 2.0})
    cover._move_start_time = clock()
    cover._full_travel_resync_direction = "closing"

    clock.advance(1.0)  # half of the 2s close travel time
    assert cover._waiting_for_full_travel_resync("closing") is True

    clock.advance(1.5)  # 2.5s total, past the 2s close travel time
    assert cover._waiting_for_full_travel_resync("closing") is False


async def test_waiting_for_full_travel_resync_is_false_for_the_other_direction(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resync pending for one direction must not gate the other.

    E.g. a close that was interrupted and immediately reopened must not
    have its still-pending close-resync flag block the open side's own
    "fully open" detection.
    """
    clock = _FakeClock(1_000.0)
    monkeypatch.setattr(time, "monotonic", clock)
    api = MagicMock()
    cover = _build_cover(hass, api, device_data={CONF_OPEN_TIME: 2.0})
    cover._move_start_time = clock()
    cover._full_travel_resync_direction = "closing"

    assert cover._waiting_for_full_travel_resync("opening") is False


# ---------------------------------------------------------------------------
# _handle_event(): reacting to a device-activity signal picked up from the
# original remote's own RF identity (status discovery/calibration), which
# is how this integration learns a move is happening at all - the stick
# cannot ask the motor directly. This is the other half of the "no
# bidirectional feedback" story: _update_position() estimates *while*
# moving, _handle_event() is what starts and stops that estimate in the
# first place in response to that overheard activity.
# ---------------------------------------------------------------------------


async def test_handle_event_started_moving_up_sets_opening_and_starts_tracking(
    hass: HomeAssistant,
) -> None:
    api = MagicMock()
    cover = _build_cover(hass, api)
    cover._start_position_tracking = MagicMock()  # type: ignore[method-assign]
    cover._attr_current_cover_position = 30

    cover._handle_event(EVENT_STARTED_MOVING_UP)

    assert cover._attr_is_opening is True
    assert cover._attr_is_closing is False
    assert cover._move_start_position == 30
    cover._start_position_tracking.assert_called_once_with()


async def test_handle_event_started_moving_down_sets_closing_and_starts_tracking(
    hass: HomeAssistant,
) -> None:
    api = MagicMock()
    cover = _build_cover(hass, api)
    cover._start_position_tracking = MagicMock()  # type: ignore[method-assign]
    cover._attr_current_cover_position = 70

    cover._handle_event(EVENT_STARTED_MOVING_DOWN)

    assert cover._attr_is_closing is True
    assert cover._attr_is_opening is False
    cover._start_position_tracking.assert_called_once_with()


async def test_handle_event_respects_invert_direction_for_the_physical_signal(
    hass: HomeAssistant,
) -> None:
    """A physical "moving up" signal on an inverted device is a logical close.

    Some Schellenberg motors are wired/oriented so the stick's "up" command
    physically closes rather than opens - invert_direction flips the
    logical meaning of commands sent (see async_open_cover/
    async_close_cover) and must equally flip the logical meaning of
    activity overheard from the original remote, or the UI would show the
    cover opening while it is actually closing.
    """
    api = MagicMock()
    cover = _build_cover(hass, api, invert_direction=True)
    cover._start_position_tracking = MagicMock()  # type: ignore[method-assign]

    cover._handle_event(EVENT_STARTED_MOVING_UP)

    assert cover._attr_is_closing is True
    assert cover._attr_is_opening is False


async def test_handle_event_stopped_freezes_the_estimated_position(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _FakeClock(1_000.0)
    monkeypatch.setattr(time, "monotonic", clock)
    api = MagicMock()
    cover = _build_cover(hass, api, device_data={CONF_OPEN_TIME: 20.0})
    cover._stop_position_tracking = MagicMock()  # type: ignore[method-assign]
    cover._attr_is_opening = True
    cover._attr_current_cover_position = 20
    cover._move_start_time = clock()
    cover._move_start_position = 20
    cover._target_position = 55  # an externally-interrupted set_position

    clock.advance(5.0)  # a quarter of the 20s open travel time: +25
    cover._handle_event(EVENT_STOPPED)

    assert cover._attr_current_cover_position == 45
    assert cover._attr_is_opening is False
    assert cover._attr_is_closing is False
    # A STOPPED echo must not snap to the stale target - see _handle_event's
    # own comment: the blind was interrupted externally (e.g. the physical
    # remote), so the commanded target was never actually reached.
    assert cover._target_position is None
    cover._stop_position_tracking.assert_called_once_with()


@pytest.mark.parametrize(
    ("start_position", "elapsed", "expected"),
    [
        (2, 10.0, 0),  # within rounding distance of the bottom end stop
        (97, 10.0, 100),  # within rounding distance of the top end stop
    ],
)
async def test_handle_event_stopped_clamps_a_near_endstop_position(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    start_position: int,
    elapsed: float,
    expected: int,
) -> None:
    """A STOPPED right at an end stop must clamp to a clean 0/100.

    Without this, floating-point elapsed-time math could leave the cover
    reporting 99% as "open" or 1% as "closed" indefinitely after physically
    reaching an end stop, since nothing else would ever nudge it the rest
    of the way without a further command.
    """
    clock = _FakeClock(1_000.0)
    monkeypatch.setattr(time, "monotonic", clock)
    api = MagicMock()
    opening = expected == 100
    device_data = {CONF_OPEN_TIME: 1.0} if opening else {CONF_CLOSE_TIME: 1.0}
    cover = _build_cover(hass, api, device_data=device_data)
    cover._stop_position_tracking = MagicMock()  # type: ignore[method-assign]
    cover._attr_is_opening = opening
    cover._attr_is_closing = not opening
    cover._attr_current_cover_position = start_position
    cover._move_start_time = clock()
    cover._move_start_position = start_position

    clock.advance(elapsed)  # far past the 1s travel time either way
    cover._handle_event(EVENT_STOPPED)

    assert cover._attr_current_cover_position == expected
    assert cover._attr_is_closed == (expected == 0)


async def test_handle_event_ignores_an_unrecognized_event(
    hass: HomeAssistant,
) -> None:
    api = MagicMock()
    cover = _build_cover(hass, api)
    cover._attr_current_cover_position = 33
    cover._attr_is_opening = False
    cover._attr_is_closing = False

    cover._handle_event("some_future_event_type")

    assert cover._attr_current_cover_position == 33
    assert cover._attr_is_opening is False
    assert cover._attr_is_closing is False


# ---------------------------------------------------------------------------
# async_set_cover_position(): the position slider in the UI. Chains into
# async_open_cover()/async_close_cover() with _preserve_target=True so the
# tracking loop's existing partial-target logic (already covered above)
# does the actual driving - this only covers the routing/no-op decision
# itself.
# ---------------------------------------------------------------------------


async def test_set_cover_position_above_current_opens_toward_the_target(
    hass: HomeAssistant,
) -> None:
    api = MagicMock()
    api.control_blind = AsyncMock(return_value=True)
    cover = _build_cover(hass, api)
    cover._start_position_tracking = MagicMock()  # type: ignore[method-assign]
    cover._attr_current_cover_position = 30

    await cover.async_set_cover_position(**{ATTR_POSITION: 80})

    assert cover._target_position == 80
    assert cover._attr_is_opening is True
    api.control_blind.assert_awaited_once_with(
        cover._command_enum, CMD_UP, device_id=cover._command_device_id
    )


async def test_set_cover_position_below_current_closes_toward_the_target(
    hass: HomeAssistant,
) -> None:
    api = MagicMock()
    api.control_blind = AsyncMock(return_value=True)
    cover = _build_cover(hass, api)
    cover._start_position_tracking = MagicMock()  # type: ignore[method-assign]
    cover._attr_current_cover_position = 80

    await cover.async_set_cover_position(**{ATTR_POSITION: 30})

    assert cover._target_position == 30
    assert cover._attr_is_closing is True
    api.control_blind.assert_awaited_once_with(
        cover._command_enum, CMD_DOWN, device_id=cover._command_device_id
    )


async def test_set_cover_position_equal_to_current_while_moving_stops_instead(
    hass: HomeAssistant,
) -> None:
    """Reaching exactly the already-in-flight target by a new command of the
    same value must stop, not let the loop drive past it."""
    api = MagicMock()
    api.control_blind = AsyncMock(return_value=True)
    cover = _build_cover(hass, api)
    cover._attr_is_opening = True
    cover._attr_current_cover_position = 50

    await cover.async_set_cover_position(**{ATTR_POSITION: 50})

    api.control_blind.assert_awaited_once_with(
        cover._command_enum, CMD_STOP, device_id=cover._command_device_id
    )


async def test_set_cover_position_equal_to_current_while_idle_is_a_noop(
    hass: HomeAssistant,
) -> None:
    api = MagicMock()
    api.control_blind = AsyncMock(return_value=True)
    cover = _build_cover(hass, api)
    cover._attr_current_cover_position = 50

    await cover.async_set_cover_position(**{ATTR_POSITION: 50})

    api.control_blind.assert_not_awaited()


# ---------------------------------------------------------------------------
# _handle_manual_position_sync() / _handle_calibration_completed(): the two
# other ways a position becomes authoritative without a live command -
# confirmed by a person via Developer Tools, or by finishing a calibration
# run.
# ---------------------------------------------------------------------------


async def test_handle_manual_position_sync_applies_the_confirmed_position(
    hass: HomeAssistant,
) -> None:
    api = MagicMock()
    cover = _build_cover(hass, api)
    cover._stop_position_tracking = MagicMock()  # type: ignore[method-assign]
    cover._attr_is_opening = True
    cover._attr_current_cover_position = 61
    cover._target_position = 90

    cover._handle_manual_position_sync(73)

    assert cover._attr_current_cover_position == 73
    assert cover._attr_is_closed is False
    assert cover._attr_is_opening is False
    assert cover._target_position is None
    assert cover._position_confirmed_since_restart is True
    cover._stop_position_tracking.assert_called_once_with()


async def test_handle_calibration_completed_ignores_a_different_device(
    hass: HomeAssistant,
) -> None:
    """Every SchellenbergCover instance listens on the shared
    SIGNAL_CALIBRATION_COMPLETED signal (see async_added_to_hass), so this
    guard is what keeps one blind's calibration from resetting every other
    configured blind to 0% as well.
    """
    api = MagicMock()
    cover = _build_cover(hass, api, device_id="ABCDEF")
    cover._attr_current_cover_position = 55

    cover._handle_calibration_completed("A_DIFFERENT_DEVICE", 12.0, 9.0)

    assert cover._attr_current_cover_position == 55


async def test_handle_calibration_completed_updates_travel_times_and_closes(
    hass: HomeAssistant,
) -> None:
    api = MagicMock()
    cover = _build_cover(hass, api, device_id="ABCDEF")
    cover._attr_current_cover_position = 55

    cover._handle_calibration_completed("ABCDEF", 18.5, 16.5)

    assert cover._travel_time_open == 18.5
    assert cover._travel_time_close == 16.5
    assert cover._attr_current_cover_position == 0
    assert cover._attr_is_closed is True
    assert cover._position_confirmed_since_restart is True


# ---------------------------------------------------------------------------
# async_added_to_hass(): restoring the last known position after a restart
# (there is no hardware to ask), and wiring up the signal that lets
# _handle_event() react to activity overheard on this blind's status
# identity.
# ---------------------------------------------------------------------------


async def test_added_to_hass_restores_position_from_the_current_position_attribute(
    hass: HomeAssistant,
) -> None:
    api = MagicMock()
    cover = _build_cover(hass, api)
    cover.async_get_last_state = AsyncMock(  # type: ignore[method-assign]
        return_value=State(
            "cover.living_room_blind", "open", {ATTR_CURRENT_POSITION: 73}
        )
    )

    await cover.async_added_to_hass()

    assert cover._attr_current_cover_position == 73
    assert cover._attr_is_closed is False


async def test_added_to_hass_restores_position_from_a_plain_closed_state(
    hass: HomeAssistant,
) -> None:
    """A restored state with no current_position attribute (e.g. from an
    older version that never recorded one) falls back to the plain
    open/closed state string.
    """
    api = MagicMock()
    cover = _build_cover(hass, api)
    cover.async_get_last_state = AsyncMock(  # type: ignore[method-assign]
        return_value=State(cover.entity_id, "closed")
    )

    await cover.async_added_to_hass()

    assert cover._attr_current_cover_position == 0
    assert cover._attr_is_closed is True


async def test_added_to_hass_defaults_to_closed_when_nothing_was_restored(
    hass: HomeAssistant,
) -> None:
    """A brand-new entity with no prior Home Assistant state at all."""
    api = MagicMock()
    cover = _build_cover(hass, api)
    cover.async_get_last_state = AsyncMock(return_value=None)  # type: ignore[method-assign]

    await cover.async_added_to_hass()

    assert cover._attr_current_cover_position == 0
    assert cover._attr_is_closed is True


async def test_added_to_hass_wires_up_the_status_identity_signal(
    hass: HomeAssistant,
) -> None:
    """The dispatcher signal subscribed to must exactly match the one
    _handle_event's own status identity would be broadcast on - this is an
    end-to-end check of that string assembly, not just that *some* callback
    got registered.
    """
    api = MagicMock()
    cover = _build_cover(hass, api, status_device_id="FEDCBA", status_enum="05")
    cover.async_get_last_state = AsyncMock(return_value=None)  # type: ignore[method-assign]
    cover._start_position_tracking = MagicMock()  # type: ignore[method-assign]

    await cover.async_added_to_hass()
    async_dispatcher_send(
        hass, f"{SIGNAL_DEVICE_EVENT}_FEDCBA_05", EVENT_STARTED_MOVING_UP
    )
    await hass.async_block_till_done()

    assert cover._attr_is_opening is True


# ---------------------------------------------------------------------------
# async_setup_entry(): turns each saved blind subentry into a device-registry
# device plus one SchellenbergCover, keyed on the subentry's own unique_id -
# a stable per-blind ID that survives a re-pair onto a different RF channel
# - rather than the raw RF command identity that is only ever used to
# transmit. It also migrates a pre-stable-ID entity's unique_id onto the
# current format when it finds one. See the module docstring for why this
# is the one place here that drives the real function instead of building
# the entity by hand.
# ---------------------------------------------------------------------------


def _build_hub_entry(hass: HomeAssistant, api: Any) -> MockConfigEntry:
    entry = MockConfigEntry(domain=DOMAIN, data={CONF_SERIAL_PORT: "/dev/fake"})
    entry.add_to_hass(hass)
    entry.runtime_data = api
    return entry


def _add_blind_subentry(
    hass: HomeAssistant,
    entry: MockConfigEntry,
    *,
    unique_id: str | None = "ABCDEF",
    title: str = "Living Room Blind",
    data: dict[str, Any] | None = None,
) -> ConfigSubentry:
    if data is None:
        data = {
            CONF_DEVICE_ID: "ABCDEF",
            CONF_DEVICE_ENUM: "10",
            CONF_STATUS_DEVICE_ID: "ABCDEF",
            CONF_STATUS_ENUM: "10",
            CONF_OPEN_TIME: 20.0,
            CONF_CLOSE_TIME: 18.0,
        }
    subentry = ConfigSubentry(
        data=MappingProxyType(data),
        subentry_type=SUBENTRY_TYPE_BLIND,
        title=title,
        unique_id=unique_id,
    )
    hass.config_entries.async_add_subentry(entry, subentry)
    return subentry


async def test_async_setup_entry_creates_device_and_cover_with_stable_id(
    hass: HomeAssistant,
) -> None:
    """The device registry's identity - and the entity's own device_id,
    which its DeviceInfo is keyed on - must come from the subentry's own
    unique_id, never from the raw command device_id that is only used to
    actually transmit. A second, incomplete subentry (no command identity
    at all) must be silently skipped rather than crashing setup for every
    other blind.
    """
    api = MagicMock()
    entry = _build_hub_entry(hass, api)
    _add_blind_subentry(hass, entry, unique_id="blind-uuid-1")
    _add_blind_subentry(
        hass,
        entry,
        unique_id=None,
        title="Incomplete Blind",
        data={CONF_OPEN_TIME: 20.0, CONF_CLOSE_TIME: 18.0},
    )
    added_entities: list[Any] = []
    async_add_entities = Mock(
        side_effect=lambda entities, **kwargs: added_entities.extend(entities)
    )

    await async_setup_entry(hass, entry, async_add_entities)

    assert len(added_entities) == 1
    cover = added_entities[0]
    assert isinstance(cover, SchellenbergCover)
    # The stable, device-registry-facing ID is the subentry's own unique_id...
    assert cover._device_id == "blind-uuid-1"
    # ...never the raw RF identity, which is only ever used to transmit.
    assert cover._command_device_id == "ABCDEF"
    assert cover._command_enum == "10"
    api.register_entity.assert_called_once_with(
        "ABCDEF",
        "10",
        "Living Room Blind",
        command_device_id="ABCDEF",
        command_enum="10",
        secondary_status_identities=(),
    )
    device_registry = dr.async_get(hass)
    device = async_get_device_by_identifier_compat(
        device_registry, (DOMAIN, "blind-uuid-1"), entry.entry_id
    )
    assert device is not None
    assert device.name == "Living Room Blind"
    assert (
        async_get_device_by_identifier_compat(
            device_registry, (DOMAIN, "ABCDEF"), entry.entry_id
        )
        is None
    )


async def test_async_setup_entry_migrates_a_legacy_entity_unique_id(
    hass: HomeAssistant,
) -> None:
    """Before blinds had their own stable per-blind unique_id, a cover's
    entity-registry unique_id was "schellenberg_<raw device_id>". An
    installation upgrading from that era must not end up with a duplicate,
    disconnected entity - the existing registry entry is migrated onto the
    new f"{DOMAIN}_blind_<blind_id>" unique_id in place instead.
    """
    api = MagicMock()
    entry = _build_hub_entry(hass, api)
    blind = _add_blind_subentry(hass, entry, unique_id="ABCDEF")
    entity_registry = er.async_get(hass)
    legacy_entry = entity_registry.async_get_or_create(
        "cover", DOMAIN, "schellenberg_ABCDEF", config_entry=entry
    )
    expected_unique_id = f"{DOMAIN}_blind_{blind.subentry_id}"

    await async_setup_entry(hass, entry, Mock())

    assert (
        entity_registry.async_get_entity_id("cover", DOMAIN, "schellenberg_ABCDEF")
        is None
    )
    migrated_entity_id = entity_registry.async_get_entity_id(
        "cover", DOMAIN, expected_unique_id
    )
    assert migrated_entity_id == legacy_entry.entity_id
    assert (
        entity_registry.entities[legacy_entry.entity_id].config_subentry_id
        == blind.subentry_id
    )


async def test_async_setup_entry_ignores_a_non_hub_entry(hass: HomeAssistant) -> None:
    """A config entry with no serial port belongs to some other platform -
    cover setup must leave the registries and the api alone for it.
    """
    api = MagicMock()
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    entry.runtime_data = api
    async_add_entities = Mock()

    await async_setup_entry(hass, entry, async_add_entities)

    async_add_entities.assert_not_called()
    api.register_entity.assert_not_called()


async def test_async_setup_entry_does_nothing_with_no_saved_blinds(
    hass: HomeAssistant,
) -> None:
    """A freshly added hub with no blinds taught yet must not crash or
    create anything - there is simply nothing to set up.
    """
    api = MagicMock()
    entry = _build_hub_entry(hass, api)
    async_add_entities = Mock()

    await async_setup_entry(hass, entry, async_add_entities)

    async_add_entities.assert_not_called()
    api.register_entity.assert_not_called()


# ---------------------------------------------------------------------------
# Position-tracking shutdown/cleanup: _async_handle_hass_stop,
# async_will_remove_from_hass, _async_shutdown_position_tracking, and
# _async_cancel_position_tracking. These run whenever Home Assistant stops
# or the entity is removed/reloaded while a blind is mid-move, and must
# never leave the background tracking task running, or let a failure in
# it escape and break shutdown for every other entity.
# ---------------------------------------------------------------------------


async def test_shutdown_position_tracking_is_a_no_op_with_nothing_running(
    hass: HomeAssistant,
) -> None:
    api = MagicMock()
    cover = _build_cover(hass, api)

    await cover._async_shutdown_position_tracking("no tracking active")

    cover.async_write_ha_state.assert_not_called()  # type: ignore[attr-defined]
    assert cover._position_update_task is None


async def test_shutdown_position_tracking_snapshots_and_cancels_the_running_task(
    hass: HomeAssistant,
) -> None:
    """A live tracking task must be snapshotted - a final position update
    and a state write, so the UI reflects where the blind actually stopped
    - and then actually cancelled, not just forgotten, so it cannot keep
    running or writing state after shutdown.
    """
    api = MagicMock()
    cover = _build_cover(hass, api)
    task = asyncio.ensure_future(asyncio.sleep(100))
    cover._position_update_task = task

    await cover._async_shutdown_position_tracking("test reason")

    cover.async_write_ha_state.assert_called_once()  # type: ignore[attr-defined]
    assert task.cancelled()
    assert cover._position_update_task is None


async def test_shutdown_position_tracking_skips_the_state_write_without_an_entity_id(
    hass: HomeAssistant,
) -> None:
    """A cover with no entity_id yet was never actually added to hass;
    writing state for it would raise, so the snapshot must skip that
    write while still cancelling the task.
    """
    api = MagicMock()
    cover = _build_cover(hass, api)
    cover.entity_id = None  # type: ignore[assignment]
    task = asyncio.ensure_future(asyncio.sleep(100))
    cover._position_update_task = task

    await cover._async_shutdown_position_tracking("test reason")

    cover.async_write_ha_state.assert_not_called()  # type: ignore[attr-defined]
    assert task.cancelled()


async def test_cancel_position_tracking_swallows_a_non_cancellation_exception(
    hass: HomeAssistant,
) -> None:
    """A tracking task that converts its own cancellation into a different
    exception must not be allowed to escape and break shutdown for every
    other entity - it is logged and swallowed instead.
    """
    api = MagicMock()
    cover = _build_cover(hass, api)

    async def _stubborn_loop() -> None:
        try:
            await asyncio.sleep(100)
        except asyncio.CancelledError:
            raise RuntimeError("boom") from None

    task = asyncio.ensure_future(_stubborn_loop())
    await asyncio.sleep(0)  # let it reach the sleep before cancelling it
    cover._position_update_task = task

    await cover._async_cancel_position_tracking("test reason")

    assert task.done()
    assert cover._position_update_task is None


async def test_handle_hass_stop_delegates_to_shutdown_with_its_own_reason(
    hass: HomeAssistant,
) -> None:
    api = MagicMock()
    cover = _build_cover(hass, api)
    cover._async_shutdown_position_tracking = AsyncMock()  # type: ignore[method-assign]

    await cover._async_handle_hass_stop(Mock())

    cover._async_shutdown_position_tracking.assert_awaited_once_with(
        "Home Assistant stopping"
    )


async def test_will_remove_from_hass_shuts_down_tracking_then_calls_super(
    hass: HomeAssistant,
) -> None:
    """async_will_remove_from_hass must shut down tracking under its own
    reason and still chain up to the base class afterwards (a no-op here,
    but skipping it would silently drop any cleanup a future Home
    Assistant release adds to it).
    """
    api = MagicMock()
    cover = _build_cover(hass, api)
    cover._async_shutdown_position_tracking = AsyncMock()  # type: ignore[method-assign]

    await cover.async_will_remove_from_hass()

    cover._async_shutdown_position_tracking.assert_awaited_once_with(
        "entity removal or entry unload"
    )
