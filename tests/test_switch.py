"""Tests for the LED switch's reconnect-triggered hardware-state restore."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.core import HomeAssistant

from custom_components.schellenberg_usb.switch import SchellenbergLedSwitch


def _build_switch(hass: HomeAssistant, api: Any) -> SchellenbergLedSwitch:
    entry: Any = MagicMock()
    entry.entry_id = "test-entry"
    switch = SchellenbergLedSwitch(api, entry)
    switch.hass = hass
    switch.entity_id = "switch.schellenberg_usb_stick_led"
    switch.async_write_ha_state = MagicMock()  # type: ignore[method-assign]
    return switch


async def test_reconnect_schedules_hardware_restore_as_a_background_task(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: the restore must run as a background task, not a tracked one.

    A plain async_create_task() task is awaited by Home Assistant during
    shutdown; this one is meant to run detached (see
    _restore_hardware_state's own docstring), so it must be scheduled with
    async_create_background_task() instead, which Home Assistant cancels
    outright at shutdown rather than waiting on.
    """
    api = MagicMock()
    api.is_connected = True
    api.led_on = AsyncMock(return_value=True)
    switch = _build_switch(hass, api)
    switch._is_on = True
    switch._was_available = False

    background_task_spy = MagicMock(wraps=hass.async_create_background_task)
    monkeypatch.setattr(hass, "async_create_background_task", background_task_spy)
    tracked_task_spy = MagicMock(wraps=hass.async_create_task)
    monkeypatch.setattr(hass, "async_create_task", tracked_task_spy)

    switch._handle_status_update()
    await hass.async_block_till_done()

    background_task_spy.assert_called_once()
    tracked_task_spy.assert_not_called()
    api.led_on.assert_awaited_once()


async def test_no_restore_scheduled_when_already_available(hass: HomeAssistant) -> None:
    """No availability transition must not re-trigger a hardware restore."""
    api = MagicMock()
    api.is_connected = True
    api.led_on = AsyncMock(return_value=True)
    switch = _build_switch(hass, api)
    switch._was_available = True

    switch._handle_status_update()
    await hass.async_block_till_done()

    api.led_on.assert_not_awaited()
