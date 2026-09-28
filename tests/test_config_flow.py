"""Tests for the complete configuration UI: main config flow and the full
blind-pairing/reconfigure/developer-tools subentry flow.

Driven through the real flow managers (hass.config_entries.flow /
.subentries) rather than calling step methods directly, matching Home
Assistant's own idiomatic flow-testing style. Simple command-dispatch
actions (test_open/close/stop, teach_motor's open+stop) run against the real
SchellenbergUsbApi wired to conftest.py's FakeTransport and are checked via
written(), the same way tests/test_api.py and tests/test_cover.py already
do - api.py's exact wire format is that module's own job to guard, not
this file's. Only calls needing a controlled return value for branching
(pairing, discovery, teach failure) are monkeypatched directly.
"""

from __future__ import annotations

import asyncio
from types import MappingProxyType
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
import serial
from conftest import written
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers.service_info.usb import UsbServiceInfo
from pytest_homeassistant_custom_component.common import MockConfigEntry

import custom_components.schellenberg_usb.config_flow as config_flow_module
from custom_components.schellenberg_usb.api import SchellenbergUsbApi
from custom_components.schellenberg_usb.const import (
    CONF_CLOSE_TIME,
    CONF_CLOSE_TIME_SECONDS,
    CONF_DEVICE_ENUM,
    CONF_DEVICE_ID,
    CONF_DEVICE_NAME,
    CONF_INVERT_DIRECTION,
    CONF_OPEN_TIME,
    CONF_OPEN_TIME_SECONDS,
    CONF_SERIAL_PORT,
    CONF_STATUS_DEVICE_ID,
    CONF_STATUS_ENUM,
    DOMAIN,
    SUBENTRY_TYPE_BLIND,
)

DEVICE_ID = "ABCDEF"
DEVICE_ENUM = "10"


# ---------------------------------------------------------------------------
# Main config flow: user / usb / usb_confirm
# ---------------------------------------------------------------------------


async def test_user_step_creates_entry_on_successful_connection(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        config_flow_module, "check_serial_port", Mock(return_value=None)
    )

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], user_input={CONF_SERIAL_PORT: "/dev/ttyUSB0"}
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_SERIAL_PORT] == "/dev/ttyUSB0"


async def test_user_step_shows_cannot_connect_on_serial_exception(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_check(port: str) -> None:
        raise serial.SerialException("no such device")

    monkeypatch.setattr(config_flow_module, "check_serial_port", failing_check)

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], user_input={CONF_SERIAL_PORT: "/dev/ttyUSB0"}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "cannot_connect"}


async def test_user_step_shows_unknown_on_unexpected_exception(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    def exploding_check(port: str) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(config_flow_module, "check_serial_port", exploding_check)

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], user_input={CONF_SERIAL_PORT: "/dev/ttyUSB0"}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "unknown"}


async def test_user_step_aborts_if_the_port_is_already_configured(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        config_flow_module, "check_serial_port", Mock(return_value=None)
    )
    existing = MockConfigEntry(
        domain=DOMAIN,
        unique_id="/dev/ttyUSB0",
        data={CONF_SERIAL_PORT: "/dev/ttyUSB0"},
    )
    existing.add_to_hass(hass)

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], user_input={CONF_SERIAL_PORT: "/dev/ttyUSB0"}
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"


async def test_usb_discovery_shows_a_confirm_form_prefilled_with_the_device_path(
    hass: HomeAssistant,
) -> None:
    discovery_info = UsbServiceInfo(
        device="/dev/ttyUSB7",
        vid="16C0",
        pid="05E1",
        serial_number="SN123",
        manufacturer="van ooijen",
        description="FunkStick",
    )

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "usb"}, data=discovery_info
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "usb_confirm"
    assert result["data_schema"]({})[CONF_SERIAL_PORT] == "/dev/ttyUSB7"


async def test_usb_confirm_creates_entry_on_successful_connection(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        config_flow_module, "check_serial_port", Mock(return_value=None)
    )
    discovery_info = UsbServiceInfo(
        device="/dev/ttyUSB7",
        vid="16C0",
        pid="05E1",
        serial_number="SN123",
        manufacturer="van ooijen",
        description="FunkStick",
    )
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "usb"}, data=discovery_info
    )

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], user_input={CONF_SERIAL_PORT: "/dev/ttyUSB7"}
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "van ooijen FunkStick"


async def test_usb_confirm_shows_cannot_connect_on_serial_exception(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_check(port: str) -> None:
        raise serial.SerialException("gone")

    monkeypatch.setattr(config_flow_module, "check_serial_port", failing_check)
    discovery_info = UsbServiceInfo(
        device="/dev/ttyUSB7",
        vid="16C0",
        pid="05E1",
        serial_number="SN123",
        manufacturer="van ooijen",
        description="FunkStick",
    )
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "usb"}, data=discovery_info
    )

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], user_input={CONF_SERIAL_PORT: "/dev/ttyUSB7"}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "cannot_connect"}


# ---------------------------------------------------------------------------
# Shared subentry-flow fixtures
# ---------------------------------------------------------------------------


def _build_hub_entry(
    hass: HomeAssistant, connected_api: SchellenbergUsbApi
) -> MockConfigEntry:
    entry = MockConfigEntry(domain=DOMAIN, data={CONF_SERIAL_PORT: "/dev/fake"})
    entry.add_to_hass(hass)
    entry.runtime_data = connected_api
    return entry


def _add_existing_blind(hass: HomeAssistant, entry: MockConfigEntry) -> ConfigSubentry:
    blind = ConfigSubentry(
        data=MappingProxyType(
            {
                CONF_DEVICE_ID: DEVICE_ID,
                CONF_DEVICE_ENUM: DEVICE_ENUM,
                CONF_STATUS_DEVICE_ID: DEVICE_ID,
                CONF_STATUS_ENUM: DEVICE_ENUM,
                CONF_OPEN_TIME: 20.0,
                CONF_CLOSE_TIME: 18.0,
                CONF_INVERT_DIRECTION: False,
            }
        ),
        subentry_type=SUBENTRY_TYPE_BLIND,
        title="Living Room Blind",
        unique_id=DEVICE_ID,
    )
    hass.config_entries.async_add_subentry(entry, blind)
    return blind


async def _start_pairing_flow(
    hass: HomeAssistant, entry: MockConfigEntry
) -> dict[str, Any]:
    """Reach the blind subentry's initial pairing-choice menu.

    Real Home Assistant starts a *new* subentry flow with
    context={"source": "user"} (see this integration's strings.json:
    config_subentries.blind.initiate_flow.user is the "Add blind" button;
    there is no initiate_flow.blind). Subentry-type dispatch happens via the
    (entry_id, subentry_type) handler tuple, not via the context source, so
    async_step_user() is the correct, real entry point regardless of type.
    """
    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_BLIND),
        context={"source": "user"},
    )
    assert result["type"] is FlowResultType.MENU
    assert result["step_id"] == "user"
    assert set(result["menu_options"]) == {"pair_test", "pair_device", "manual"}
    return result


async def _start_reconfigure_flow(
    hass: HomeAssistant, entry: MockConfigEntry, blind: ConfigSubentry
) -> dict[str, Any]:
    """Reach an existing blind's reconfigure menu."""
    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_BLIND),
        context={"source": "reconfigure", "subentry_id": blind.subentry_id},
    )
    assert result["type"] is FlowResultType.MENU
    assert result["step_id"] == "reconfigure"
    assert set(result["menu_options"]) == {
        "edit",
        "test_existing",
        "developer_tools",
        "calibrate",
    }
    return result


# ---------------------------------------------------------------------------
# Subentry flow: pairing a new blind
# ---------------------------------------------------------------------------


async def test_pair_device_legacy_workflow_names_device_then_starts_calibration(
    hass: HomeAssistant,
    connected_api: SchellenbergUsbApi,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = _build_hub_entry(hass, connected_api)
    monkeypatch.setattr(
        connected_api,
        "pair_device_and_wait",
        AsyncMock(return_value=(DEVICE_ID, DEVICE_ENUM)),
    )
    result = await _start_pairing_flow(hass, entry)

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"next_step_id": "pair_device"}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "pair_device"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "name_device"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={CONF_DEVICE_NAME: "Kitchen Blind"}
    )

    # Legacy workflow goes straight into calibration after naming.
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "calibration_close"


async def test_pair_device_timeout_aborts_with_pairing_timeout(
    hass: HomeAssistant,
    connected_api: SchellenbergUsbApi,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = _build_hub_entry(hass, connected_api)
    monkeypatch.setattr(
        connected_api, "pair_device_and_wait", AsyncMock(return_value=None)
    )
    result = await _start_pairing_flow(hass, entry)

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"next_step_id": "pair_device"}
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={}
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "pairing_timeout"


async def test_pair_test_hybrid_workflow_runs_test_motor_before_calibration_choice(
    hass: HomeAssistant,
    connected_api: SchellenbergUsbApi,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = _build_hub_entry(hass, connected_api)
    monkeypatch.setattr(
        connected_api,
        "pair_device_and_wait",
        AsyncMock(return_value=(DEVICE_ID, DEVICE_ENUM)),
    )
    monkeypatch.setattr(asyncio, "sleep", AsyncMock(return_value=None))
    result = await _start_pairing_flow(hass, entry)

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"next_step_id": "pair_test"}
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={}
    )
    assert result["step_id"] == "name_device"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={CONF_DEVICE_NAME: "Hybrid Blind"}
    )
    # Hybrid workflow tests the motor before offering the calibration choice.
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "test_motor"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "did_motor_move"
    assert len(written(connected_api)) == 2  # open, then stop

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"motor_moved": True}
    )
    assert result["type"] is FlowResultType.MENU
    assert result["step_id"] == "test_success"
    assert set(result["menu_options"]) == {"calibration_close", "manual_times"}


async def test_manual_workflow_validates_and_creates_entry_via_save_manual(
    hass: HomeAssistant, connected_api: SchellenbergUsbApi
) -> None:
    entry = _build_hub_entry(hass, connected_api)
    result = await _start_pairing_flow(hass, entry)

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"next_step_id": "manual"}
    )
    assert result["step_id"] == "manual"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        user_input={
            CONF_DEVICE_NAME: "Manual Blind",
            CONF_DEVICE_ID: DEVICE_ID,
            CONF_DEVICE_ENUM: DEVICE_ENUM,
            CONF_OPEN_TIME_SECONDS: 25.0,
            CONF_CLOSE_TIME_SECONDS: 22.0,
        },
    )
    assert result["step_id"] == "manual_next"
    assert set(result["menu_options"]) == {"test_motor", "save_manual"}

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"next_step_id": "save_manual"}
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "Manual Blind"
    assert result["data"][CONF_OPEN_TIME] == 25.0
    assert result["data"][CONF_CLOSE_TIME] == 22.0


async def test_manual_workflow_rejects_an_invalid_device_id(
    hass: HomeAssistant, connected_api: SchellenbergUsbApi
) -> None:
    entry = _build_hub_entry(hass, connected_api)
    result = await _start_pairing_flow(hass, entry)
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"next_step_id": "manual"}
    )

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        user_input={
            CONF_DEVICE_NAME: "Manual Blind",
            CONF_DEVICE_ID: "NOTHEX",
            CONF_DEVICE_ENUM: DEVICE_ENUM,
            CONF_OPEN_TIME_SECONDS: 25.0,
            CONF_CLOSE_TIME_SECONDS: 22.0,
        },
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "manual"
    assert result["errors"] == {CONF_DEVICE_ID: "invalid_device_id"}


async def test_manual_workflow_rejects_a_duplicate_device_id(
    hass: HomeAssistant, connected_api: SchellenbergUsbApi
) -> None:
    entry = _build_hub_entry(hass, connected_api)
    _add_existing_blind(hass, entry)
    result = await _start_pairing_flow(hass, entry)
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"next_step_id": "manual"}
    )

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        user_input={
            CONF_DEVICE_NAME: "Duplicate Blind",
            CONF_DEVICE_ID: DEVICE_ID,
            CONF_DEVICE_ENUM: "20",
            CONF_OPEN_TIME_SECONDS: 25.0,
            CONF_CLOSE_TIME_SECONDS: 22.0,
        },
    )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {CONF_DEVICE_ID: "already_configured"}


async def test_did_motor_move_false_returns_to_the_manual_form_prefilled(
    hass: HomeAssistant,
    connected_api: SchellenbergUsbApi,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = _build_hub_entry(hass, connected_api)
    monkeypatch.setattr(
        connected_api,
        "pair_device_and_wait",
        AsyncMock(return_value=(DEVICE_ID, DEVICE_ENUM)),
    )
    monkeypatch.setattr(asyncio, "sleep", AsyncMock(return_value=None))
    result = await _start_pairing_flow(hass, entry)
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"next_step_id": "pair_test"}
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={}
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={CONF_DEVICE_NAME: "Hybrid Blind"}
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={}
    )
    assert result["step_id"] == "did_motor_move"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"motor_moved": False}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "manual"
    # Open/close time have no default here (never collected on this hybrid
    # path), so vol.Required() with no default means probing the schema with
    # an empty dict would raise MultipleInvalid; supply both to inspect the
    # unrelated device_id default.
    probe = result["data_schema"](
        {CONF_OPEN_TIME_SECONDS: 25.0, CONF_CLOSE_TIME_SECONDS: 22.0}
    )
    assert probe[CONF_DEVICE_ID] == DEVICE_ID


# ---------------------------------------------------------------------------
# Subentry flow: status discovery (guided remote capture)
# ---------------------------------------------------------------------------


async def test_manual_times_leads_into_discover_status(
    hass: HomeAssistant,
    connected_api: SchellenbergUsbApi,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = _build_hub_entry(hass, connected_api)
    monkeypatch.setattr(
        connected_api,
        "pair_device_and_wait",
        AsyncMock(return_value=(DEVICE_ID, DEVICE_ENUM)),
    )
    monkeypatch.setattr(asyncio, "sleep", AsyncMock(return_value=None))
    result = await _start_pairing_flow(hass, entry)
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"next_step_id": "pair_test"}
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={}
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={CONF_DEVICE_NAME: "Hybrid Blind"}
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={}
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"motor_moved": True}
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"next_step_id": "manual_times"}
    )
    assert result["step_id"] == "manual_times"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        user_input={CONF_OPEN_TIME_SECONDS: 30.0, CONF_CLOSE_TIME_SECONDS: 28.0},
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "discover_status"


async def test_discover_status_unavailable_shows_an_error(
    hass: HomeAssistant,
    connected_api: SchellenbergUsbApi,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = _build_hub_entry(hass, connected_api)
    blind = _add_existing_blind(hass, entry)
    monkeypatch.setattr(
        connected_api,
        "async_discover_status_identities",
        AsyncMock(side_effect=ConnectionError),
    )
    result = await _start_reconfigure_flow(hass, entry, blind)
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"next_step_id": "developer_tools"}
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"next_step_id": "discover_status"}
    )
    assert result["step_id"] == "discover_status"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "status_discovery_unavailable"}


async def test_discover_status_busy_shows_an_error(
    hass: HomeAssistant,
    connected_api: SchellenbergUsbApi,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = _build_hub_entry(hass, connected_api)
    blind = _add_existing_blind(hass, entry)
    monkeypatch.setattr(
        connected_api,
        "async_discover_status_identities",
        AsyncMock(side_effect=RuntimeError),
    )
    result = await _start_reconfigure_flow(hass, entry, blind)
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"next_step_id": "developer_tools"}
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"next_step_id": "discover_status"}
    )

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "status_discovery_busy"}


async def test_discover_status_success_confirms_and_updates_existing_blind(
    hass: HomeAssistant,
    connected_api: SchellenbergUsbApi,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = _build_hub_entry(hass, connected_api)
    blind = _add_existing_blind(hass, entry)
    discovery_result = {
        "primary": {
            "device_id": "112233",
            "enum": "01",
            "commands": ["00", "01"],
            "timestamps": ["10:00:00"],
        },
        "secondary": [],
        "unknown_commands": [],
        "frames": [{}, {}],
    }
    monkeypatch.setattr(
        connected_api,
        "async_discover_status_identities",
        AsyncMock(return_value=discovery_result),
    )
    result = await _start_reconfigure_flow(hass, entry, blind)
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"next_step_id": "developer_tools"}
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"next_step_id": "discover_status"}
    )

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={}
    )
    assert result["step_id"] == "confirm_status_discovery"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={}
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    updated = entry.subentries[blind.subentry_id]
    assert updated.data[CONF_STATUS_DEVICE_ID] == "112233"
    assert updated.data[CONF_STATUS_ENUM] == "01"


# ---------------------------------------------------------------------------
# Subentry flow: reconfigure an existing blind (edit / test_existing)
# ---------------------------------------------------------------------------


async def test_edit_updates_an_existing_blind(
    hass: HomeAssistant, connected_api: SchellenbergUsbApi
) -> None:
    entry = _build_hub_entry(hass, connected_api)
    blind = _add_existing_blind(hass, entry)
    result = await _start_reconfigure_flow(hass, entry, blind)

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"next_step_id": "edit"}
    )
    assert result["step_id"] == "edit"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        user_input={
            CONF_DEVICE_NAME: "Renamed Blind",
            CONF_DEVICE_ID: DEVICE_ID,
            CONF_DEVICE_ENUM: DEVICE_ENUM,
            CONF_OPEN_TIME_SECONDS: 33.0,
            CONF_CLOSE_TIME_SECONDS: 31.0,
        },
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    updated = entry.subentries[blind.subentry_id]
    assert updated.title == "Renamed Blind"
    assert updated.data[CONF_OPEN_TIME] == 33.0


async def test_edit_rejects_a_device_id_already_used_by_another_blind(
    hass: HomeAssistant, connected_api: SchellenbergUsbApi
) -> None:
    entry = _build_hub_entry(hass, connected_api)
    blind = _add_existing_blind(hass, entry)
    other = ConfigSubentry(
        data=MappingProxyType({CONF_DEVICE_ID: "112233", CONF_DEVICE_ENUM: "20"}),
        subentry_type=SUBENTRY_TYPE_BLIND,
        title="Other Blind",
        unique_id="112233",
    )
    hass.config_entries.async_add_subentry(entry, other)
    result = await _start_reconfigure_flow(hass, entry, blind)
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"next_step_id": "edit"}
    )

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        user_input={
            CONF_DEVICE_NAME: "Living Room Blind",
            CONF_DEVICE_ID: "112233",
            CONF_DEVICE_ENUM: DEVICE_ENUM,
            CONF_OPEN_TIME_SECONDS: 20.0,
            CONF_CLOSE_TIME_SECONDS: 18.0,
        },
    )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {CONF_DEVICE_ID: "already_configured"}


async def test_test_existing_leads_into_the_short_command_test(
    hass: HomeAssistant,
    connected_api: SchellenbergUsbApi,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asyncio, "sleep", AsyncMock(return_value=None))
    entry = _build_hub_entry(hass, connected_api)
    blind = _add_existing_blind(hass, entry)
    result = await _start_reconfigure_flow(hass, entry, blind)

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"next_step_id": "test_existing"}
    )
    assert result["step_id"] == "test_motor"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={}
    )

    assert result["step_id"] == "did_motor_move"
    assert len(written(connected_api)) == 2


async def test_did_motor_move_true_for_an_existing_blind_aborts_successfully(
    hass: HomeAssistant,
    connected_api: SchellenbergUsbApi,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asyncio, "sleep", AsyncMock(return_value=None))
    entry = _build_hub_entry(hass, connected_api)
    blind = _add_existing_blind(hass, entry)
    result = await _start_reconfigure_flow(hass, entry, blind)
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"next_step_id": "test_existing"}
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={}
    )

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"motor_moved": True}
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "command_test_successful"


# ---------------------------------------------------------------------------
# Subentry flow: developer tools
# ---------------------------------------------------------------------------


async def test_developer_tools_shows_the_full_diagnostics_menu(
    hass: HomeAssistant, connected_api: SchellenbergUsbApi
) -> None:
    entry = _build_hub_entry(hass, connected_api)
    blind = _add_existing_blind(hass, entry)
    result = await _start_reconfigure_flow(hass, entry, blind)

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"next_step_id": "developer_tools"}
    )

    assert result["type"] is FlowResultType.MENU
    assert result["step_id"] == "developer_tools"
    assert set(result["menu_options"]) == {
        "test_open",
        "test_close",
        "test_stop",
        "discover_status",
        "set_position_open",
        "set_position_closed",
        "set_position_manual",
        "reset_stick",
        "copy_diagnostics",
        "teach_motor",
        "send_raw_command",
    }
    assert result["description_placeholders"]["selected_blind"] == "Living Room Blind"
    assert result["description_placeholders"]["command_device_id"] == DEVICE_ID


async def test_developer_tools_test_open_close_stop_each_send_one_command(
    hass: HomeAssistant, connected_api: SchellenbergUsbApi
) -> None:
    entry = _build_hub_entry(hass, connected_api)
    blind = _add_existing_blind(hass, entry)

    for action in ("test_open", "test_close", "test_stop"):
        result = await _start_reconfigure_flow(hass, entry, blind)
        result = await hass.config_entries.subentries.async_configure(
            result["flow_id"], user_input={"next_step_id": "developer_tools"}
        )
        before = len(written(connected_api))

        result = await hass.config_entries.subentries.async_configure(
            result["flow_id"], user_input={"next_step_id": action}
        )

        assert result["step_id"] == "developer_tools"
        assert len(written(connected_api)) == before + 1
        assert "written successfully" in result["description_placeholders"]["result"]
        # On real hardware the stick's own t0 ACK clears the pending-transmit
        # latch shortly after each write; FakeTransport never talks back, so
        # the next loop iteration's command would otherwise be refused with
        # "transmit is pending" by config_flow's own transmit_block_reason
        # guard. Simulate the ACK the same way SchellenbergProtocol would
        # feed it in from a real connection.
        connected_api._handle_message("t0")


async def test_developer_tools_set_position_open_and_closed_sync_without_sending_rf(
    hass: HomeAssistant, connected_api: SchellenbergUsbApi
) -> None:
    entry = _build_hub_entry(hass, connected_api)
    blind = _add_existing_blind(hass, entry)
    # manual_sync_position() only succeeds for a "live" cover, i.e. a device
    # id the cover platform has registered on the api; these tests drive the
    # subentry flow directly without setting up that platform, so register
    # it by hand the way cover.py's async_added_to_hass() normally would.
    connected_api.register_existing_devices(
        [{"id": DEVICE_ID, "enum": DEVICE_ENUM}]
    )

    for action, expected_position in (
        ("set_position_open", 100),
        ("set_position_closed", 0),
    ):
        result = await _start_reconfigure_flow(hass, entry, blind)
        result = await hass.config_entries.subentries.async_configure(
            result["flow_id"], user_input={"next_step_id": "developer_tools"}
        )
        before = len(written(connected_api))

        result = await hass.config_entries.subentries.async_configure(
            result["flow_id"], user_input={"next_step_id": action}
        )

        assert result["step_id"] == "developer_tools"
        # Manual position confirmation never transmits an RF command.
        assert len(written(connected_api)) == before
        assert str(expected_position) in result["description_placeholders"]["result"]


async def test_developer_tools_set_position_manual_accepts_an_exact_value(
    hass: HomeAssistant, connected_api: SchellenbergUsbApi
) -> None:
    entry = _build_hub_entry(hass, connected_api)
    blind = _add_existing_blind(hass, entry)
    # See test_developer_tools_set_position_open_and_closed_sync_without_
    # sending_rf: manual_sync_position() requires the device to be
    # registered as a "live" cover, which this direct flow-driven test
    # never does on its own.
    connected_api.register_existing_devices(
        [{"id": DEVICE_ID, "enum": DEVICE_ENUM}]
    )
    result = await _start_reconfigure_flow(hass, entry, blind)
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"next_step_id": "developer_tools"}
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"next_step_id": "set_position_manual"}
    )
    assert result["step_id"] == "set_position_manual"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"position": 42}
    )

    assert result["step_id"] == "developer_tools"
    assert "42" in result["description_placeholders"]["result"]


async def test_developer_tools_teach_motor_success_sends_teach_open_and_stop(
    hass: HomeAssistant,
    connected_api: SchellenbergUsbApi,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asyncio, "sleep", AsyncMock(return_value=None))
    entry = _build_hub_entry(hass, connected_api)
    blind = _add_existing_blind(hass, entry)
    result = await _start_reconfigure_flow(hass, entry, blind)
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"next_step_id": "developer_tools"}
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"next_step_id": "teach_motor"}
    )
    assert result["step_id"] == "teach_motor"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={}
    )

    assert result["step_id"] == "developer_tools"
    assert "Teach, Open, and Stop were transmitted" in (
        result["description_placeholders"]["result"]
    )
    # teach_motor() itself is a two-phase transmit (teach_60, finish_40),
    # followed by open and stop: 4 writes in total.
    assert len(written(connected_api)) == 4  # teach_60, finish_40, open, stop


async def test_developer_tools_teach_motor_failure_shows_failure_notice(
    hass: HomeAssistant,
    connected_api: SchellenbergUsbApi,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        connected_api, "teach_motor", AsyncMock(return_value=False)
    )
    entry = _build_hub_entry(hass, connected_api)
    blind = _add_existing_blind(hass, entry)
    result = await _start_reconfigure_flow(hass, entry, blind)
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"next_step_id": "developer_tools"}
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"next_step_id": "teach_motor"}
    )

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={}
    )

    assert result["step_id"] == "developer_tools"
    assert "failed" in result["description_placeholders"]["result"]


async def test_developer_tools_send_raw_command_success(
    hass: HomeAssistant, connected_api: SchellenbergUsbApi
) -> None:
    entry = _build_hub_entry(hass, connected_api)
    blind = _add_existing_blind(hass, entry)
    result = await _start_reconfigure_flow(hass, entry, blind)
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"next_step_id": "developer_tools"}
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"next_step_id": "send_raw_command"}
    )
    assert result["step_id"] == "send_raw_command"
    payload = result["data_schema"]({})["payload"]

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"payload": payload}
    )

    assert result["step_id"] == "developer_tools"
    assert len(written(connected_api)) == 1


async def test_developer_tools_send_raw_command_rejects_an_invalid_payload(
    hass: HomeAssistant, connected_api: SchellenbergUsbApi
) -> None:
    entry = _build_hub_entry(hass, connected_api)
    blind = _add_existing_blind(hass, entry)
    result = await _start_reconfigure_flow(hass, entry, blind)
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"next_step_id": "developer_tools"}
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"next_step_id": "send_raw_command"}
    )

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"payload": "not a valid payload!"}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "send_raw_command"
    assert result["errors"] == {"payload": "invalid_raw_payload"}


async def test_developer_tools_reset_stick(
    hass: HomeAssistant, connected_api: SchellenbergUsbApi
) -> None:
    entry = _build_hub_entry(hass, connected_api)
    blind = _add_existing_blind(hass, entry)
    result = await _start_reconfigure_flow(hass, entry, blind)
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"next_step_id": "developer_tools"}
    )

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"next_step_id": "reset_stick"}
    )

    assert result["step_id"] == "developer_tools"
    assert "reset" in result["description_placeholders"]["result"].lower()
    # reset_and_reconnect() tried a real connect() against the fake port
    # above, which fails (no such device) and schedules a 5s retry via
    # async_call_later - exactly what a real failed reconnect should do, but
    # it would otherwise outlive this test as a "lingering timer". Clean it
    # up the same way a normal integration unload would.
    await connected_api.disconnect()


async def test_developer_tools_copy_diagnostics_shows_a_text_snapshot_then_returns(
    hass: HomeAssistant, connected_api: SchellenbergUsbApi
) -> None:
    entry = _build_hub_entry(hass, connected_api)
    blind = _add_existing_blind(hass, entry)
    result = await _start_reconfigure_flow(hass, entry, blind)
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"next_step_id": "developer_tools"}
    )

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"next_step_id": "copy_diagnostics"}
    )
    assert result["step_id"] == "copy_diagnostics"
    diagnostics_text = result["data_schema"]({})["diagnostics"]
    assert "Living Room Blind" in diagnostics_text

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"diagnostics": diagnostics_text}
    )

    assert result["step_id"] == "developer_tools"
