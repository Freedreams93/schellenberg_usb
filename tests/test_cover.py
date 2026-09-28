"""Tests for the cover entity's command dispatch and stop-at-target logic.

SchellenbergCover instances here are built directly (not through
async_setup_entry) with a MagicMock api, since these tests are only about
the entity's own decision logic: which commands it sends and when.
async_write_ha_state() is stubbed out because these tests assert on command
dispatch, not on the entity's reported state, and a bare entity built this
way was never handed a real entity platform to write state through.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

from custom_components.schellenberg_usb.const import CMD_STOP
from custom_components.schellenberg_usb.cover import SchellenbergCover


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
    api.transmit_block_reason = "serial stick is disconnected"
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
    api.transmit_block_reason = "serial stick is disconnected"
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
    api.transmit_block_reason = "serial stick is disconnected"
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
