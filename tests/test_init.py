"""Tests for integration setup/unload and the test_command service handler."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any
from unittest.mock import AsyncMock

import pytest
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.schellenberg_usb import (
    _async_backfill_blind_ids,
    async_setup,
    async_setup_entry,
    async_unload_entry,
)
from custom_components.schellenberg_usb.api import SchellenbergUsbApi
from custom_components.schellenberg_usb.blind_id import normalize_blind_id
from custom_components.schellenberg_usb.const import (
    CMD_STOP,
    CMD_TRANSMIT,
    CONF_BLIND_ID,
    CONF_CLOSE_TIME,
    CONF_COMMAND,
    CONF_CONFIG_ENTRY_ID,
    CONF_DEVICE_ENUM,
    CONF_DEVICE_ID,
    CONF_ENUM,
    CONF_LAST_CALIBRATION,
    CONF_OPEN_TIME,
    CONF_SERIAL_PORT,
    DOMAIN,
    SERVICE_TEST_COMMAND,
    SUBENTRY_TYPE_BLIND,
)
from tests.conftest import written

EXPECTED_STOP_PAYLOAD = f"{CMD_TRANSMIT}109{CMD_STOP}0000\r\n".encode("ascii")


async def test_service_rejects_an_entry_that_is_not_loaded(hass: HomeAssistant) -> None:
    """A configured-but-unloaded entry must not crash the service handler.

    Regression: the handler used to read `candidate.runtime_data` directly
    while looping over every configured entry, including ones that were
    never set up - which have no such attribute at all yet, so this raised
    AttributeError instead of the intended ServiceValidationError.
    """
    not_loaded_entry = MockConfigEntry(domain=DOMAIN, data={})
    not_loaded_entry.add_to_hass(hass)

    await async_setup(hass, {})

    with pytest.raises(ServiceValidationError):
        await hass.services.async_call(
            DOMAIN,
            SERVICE_TEST_COMMAND,
            {
                CONF_DEVICE_ID: "ABCDEF",
                CONF_ENUM: "10",
                CONF_COMMAND: "stop",
                CONF_CONFIG_ENTRY_ID: not_loaded_entry.entry_id,
            },
            blocking=True,
        )


async def test_service_uses_the_single_loaded_entry_when_others_are_unloaded(
    hass: HomeAssistant, connected_api_factory: Any
) -> None:
    """The loop must survive an unloaded entry and still find the loaded one."""
    not_loaded_entry = MockConfigEntry(domain=DOMAIN, data={})
    not_loaded_entry.add_to_hass(hass)

    loaded_entry = MockConfigEntry(domain=DOMAIN, data={})
    loaded_entry.add_to_hass(hass)
    api = connected_api_factory()
    loaded_entry.runtime_data = api

    await async_setup(hass, {})

    await hass.services.async_call(
        DOMAIN,
        SERVICE_TEST_COMMAND,
        {CONF_DEVICE_ID: "ABCDEF", CONF_ENUM: "10", CONF_COMMAND: "stop"},
        blocking=True,
    )

    assert written(api) == [EXPECTED_STOP_PAYLOAD]


async def test_service_requires_config_entry_id_when_multiple_are_loaded(
    hass: HomeAssistant, connected_api_factory: Any
) -> None:
    entry_a = MockConfigEntry(domain=DOMAIN, data={})
    entry_a.add_to_hass(hass)
    entry_a.runtime_data = connected_api_factory("/dev/fake-a")

    entry_b = MockConfigEntry(domain=DOMAIN, data={})
    entry_b.add_to_hass(hass)
    entry_b.runtime_data = connected_api_factory("/dev/fake-b")

    await async_setup(hass, {})

    with pytest.raises(ServiceValidationError):
        await hass.services.async_call(
            DOMAIN,
            SERVICE_TEST_COMMAND,
            {CONF_DEVICE_ID: "ABCDEF", CONF_ENUM: "10", CONF_COMMAND: "stop"},
            blocking=True,
        )


async def test_async_unload_entry_disconnects_the_api(
    hass: HomeAssistant, connected_api_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = MockConfigEntry(domain=DOMAIN, data={CONF_SERIAL_PORT: "/dev/fake"})
    entry.add_to_hass(hass)
    api = connected_api_factory()
    entry.runtime_data = api

    # async_unload_platforms() itself belongs to Home Assistant's own
    # forwarded-platform bookkeeping, not to this integration's logic; it is
    # stubbed out so this test is only about async_unload_entry's own two
    # lines - unload, then disconnect the right api using entry.runtime_data.
    monkeypatch.setattr(
        hass.config_entries, "async_unload_platforms", AsyncMock(return_value=True)
    )

    result = await async_unload_entry(hass, entry)

    assert result is True
    assert api.is_connected is False


async def test_calibration_only_subentry_update_does_not_trigger_reload(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recalibrating an existing blind must not force a full hub reload.

    Regression: _on_entry_updated used to compare every subentry data field,
    including open_time/close_time/last_calibration - exactly the fields a
    recalibration persists via async_update_and_abort in
    options_flow_calibration.py. _notify_calibration_completed already
    updates the live cover entity for those three fields with no reload
    needed (see SchellenbergCover._handle_calibration_completed), so the
    reload this used to trigger right afterwards undid that live update and
    also disconnected/rebuilt every other blind on the same hub. Changing
    any other subentry field (e.g. a re-paired device_enum) must still
    reload, since cover.py only reads those once at entity construction.
    """
    monkeypatch.setattr(SchellenbergUsbApi, "connect", AsyncMock())
    monkeypatch.setattr(
        hass.config_entries, "async_forward_entry_setups", AsyncMock(return_value=True)
    )
    mock_reload = AsyncMock()
    monkeypatch.setattr(hass.config_entries, "async_reload", mock_reload)

    entry = MockConfigEntry(domain=DOMAIN, data={CONF_SERIAL_PORT: "/dev/fake"})
    entry.add_to_hass(hass)
    blind = ConfigSubentry(
        data=MappingProxyType(
            {
                CONF_DEVICE_ID: "ABCDEF",
                CONF_DEVICE_ENUM: "10",
                CONF_OPEN_TIME: 20.0,
                CONF_CLOSE_TIME: 18.0,
            }
        ),
        subentry_type=SUBENTRY_TYPE_BLIND,
        title="Living Room Blind",
        unique_id="ABCDEF",
    )
    hass.config_entries.async_add_subentry(entry, blind)
    subentry_id = blind.subentry_id

    assert await async_setup_entry(hass, entry)
    await hass.async_block_till_done()
    mock_reload.reset_mock()  # ignore anything triggered by setup itself

    # ConfigSubentry is immutable - async_update_subentry() swaps in a new
    # instance under the same subentry_id rather than mutating `blind` in
    # place, so every read/write below goes through entry.subentries[...]
    # again instead of reusing the original `blind`/dict reference.
    calibration_only_update = dict(entry.subentries[subentry_id].data)
    calibration_only_update[CONF_OPEN_TIME] = 21.5
    calibration_only_update[CONF_CLOSE_TIME] = 19.0
    calibration_only_update[CONF_LAST_CALIBRATION] = "2026-09-27T10:00:00+00:00"
    hass.config_entries.async_update_subentry(
        entry, entry.subentries[subentry_id], data=calibration_only_update
    )
    await hass.async_block_till_done()

    mock_reload.assert_not_called()

    rebind_update = dict(entry.subentries[subentry_id].data)
    rebind_update[CONF_DEVICE_ENUM] = "11"
    hass.config_entries.async_update_subentry(
        entry, entry.subentries[subentry_id], data=rebind_update
    )
    await hass.async_block_till_done()

    mock_reload.assert_awaited_once_with(entry.entry_id)


def test_backfill_blind_ids_assigns_missing_ids_to_blind_subentries(
    hass: HomeAssistant,
) -> None:
    entry = MockConfigEntry(domain=DOMAIN, data={CONF_SERIAL_PORT: "/dev/fake"})
    entry.add_to_hass(hass)
    subentry = ConfigSubentry(
        data=MappingProxyType({}),
        subentry_type=SUBENTRY_TYPE_BLIND,
        title="Blind 1",
        unique_id=None,
    )
    hass.config_entries.async_add_subentry(entry, subentry)

    changed = _async_backfill_blind_ids(hass, entry)

    assert changed is True
    updated_data = entry.subentries[subentry.subentry_id].data
    assert normalize_blind_id(updated_data.get(CONF_BLIND_ID)) is not None


def test_backfill_blind_ids_is_a_noop_once_ids_are_already_valid(
    hass: HomeAssistant,
) -> None:
    valid_id = "12345678-1234-5678-1234-567812345678"
    entry = MockConfigEntry(domain=DOMAIN, data={CONF_SERIAL_PORT: "/dev/fake"})
    entry.add_to_hass(hass)
    subentry = ConfigSubentry(
        data=MappingProxyType({CONF_BLIND_ID: valid_id}),
        subentry_type=SUBENTRY_TYPE_BLIND,
        title="Blind 1",
        unique_id=None,
    )
    hass.config_entries.async_add_subentry(entry, subentry)

    changed = _async_backfill_blind_ids(hass, entry)

    assert changed is False
    assert entry.subentries[subentry.subentry_id].data[CONF_BLIND_ID] == valid_id
