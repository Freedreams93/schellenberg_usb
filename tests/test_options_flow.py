"""Tests for the hub options flow (editing the USB serial port).

Driven through the real options flow manager (hass.config_entries.options)
rather than constructing SchellenbergOptionsFlowHandler by hand, so these
also exercise SchellenbergUsbConfigFlow.async_get_options_flow() and Home
Assistant's own options-flow wiring, not just this integration's own code.
check_serial_port is monkeypatched at its options_flow.py import site to
avoid touching a real serial port.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import serial
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from pytest_homeassistant_custom_component.common import MockConfigEntry

import custom_components.schellenberg_usb.options_flow as options_flow_module
from custom_components.schellenberg_usb.const import CONF_SERIAL_PORT, DOMAIN


def _build_entry(hass: HomeAssistant, port: str = "/dev/ttyUSB0") -> MockConfigEntry:
    entry = MockConfigEntry(domain=DOMAIN, data={CONF_SERIAL_PORT: port})
    entry.add_to_hass(hass)
    return entry


async def test_init_shows_a_form_prefilled_with_the_current_port(
    hass: HomeAssistant,
) -> None:
    entry = _build_entry(hass, port="/dev/ttyUSB3")

    result = await hass.config_entries.options.async_init(entry.entry_id)

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"
    assert result["data_schema"] is not None
    assert result["data_schema"]({})[CONF_SERIAL_PORT] == "/dev/ttyUSB3"


async def test_submitting_the_same_port_creates_entry_without_checking_it(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = _build_entry(hass, port="/dev/ttyUSB0")
    check_spy = Mock()
    monkeypatch.setattr(options_flow_module, "check_serial_port", check_spy)
    reload_spy = Mock()
    monkeypatch.setattr(hass.config_entries, "async_schedule_reload", reload_spy)

    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], user_input={CONF_SERIAL_PORT: "/dev/ttyUSB0"}
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    check_spy.assert_not_called()
    reload_spy.assert_not_called()
    assert entry.data[CONF_SERIAL_PORT] == "/dev/ttyUSB0"


async def test_submitting_a_new_valid_port_updates_entry_and_schedules_reload(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = _build_entry(hass, port="/dev/ttyUSB0")
    monkeypatch.setattr(
        options_flow_module, "check_serial_port", Mock(return_value=None)
    )
    reload_spy = Mock()
    monkeypatch.setattr(hass.config_entries, "async_schedule_reload", reload_spy)

    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], user_input={CONF_SERIAL_PORT: "/dev/ttyUSB1"}
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert entry.data[CONF_SERIAL_PORT] == "/dev/ttyUSB1"
    reload_spy.assert_called_once_with(entry.entry_id)


async def test_submitting_an_unreachable_new_port_shows_cannot_connect(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_check(port: str) -> None:
        raise serial.SerialException("no such device")

    monkeypatch.setattr(options_flow_module, "check_serial_port", failing_check)
    entry = _build_entry(hass, port="/dev/ttyUSB0")

    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], user_input={CONF_SERIAL_PORT: "/dev/ttyUSB9"}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "cannot_connect"}
    # The rejected port must not have been persisted.
    assert entry.data[CONF_SERIAL_PORT] == "/dev/ttyUSB0"


async def test_submitting_a_new_port_with_an_unexpected_error_shows_unknown(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    def exploding_check(port: str) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(options_flow_module, "check_serial_port", exploding_check)
    entry = _build_entry(hass, port="/dev/ttyUSB0")

    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], user_input={CONF_SERIAL_PORT: "/dev/ttyUSB9"}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "unknown"}
