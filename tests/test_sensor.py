"""Tests for the three USB-stick status sensors (connection/version/mode).

Entities are built directly with a MagicMock api, following the same house
style as test_switch.py/test_cover.py, since these tests are only about each
sensor's own native_value/icon/availability logic and its dispatcher-driven
state refresh - not about platform setup plumbing (covered separately below
for async_setup_entry's own subentry-grouping logic).
"""

from __future__ import annotations

from collections.abc import Iterable
from types import MappingProxyType
from typing import Any
from unittest.mock import MagicMock

from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.entity import Entity
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.schellenberg_usb.const import (
    DOMAIN,
    SIGNAL_STICK_STATUS_UPDATED,
    SUBENTRY_TYPE_HUB,
)
from custom_components.schellenberg_usb.sensor import (
    SchellenbergConnectionSensor,
    SchellenbergModeSensor,
    SchellenbergVersionSensor,
    async_setup_entry,
)


def _build_entry(hass: HomeAssistant) -> Any:
    entry: Any = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    return entry


def test_connection_sensor_reflects_connected_state(hass: HomeAssistant) -> None:
    api = MagicMock()
    api.is_connected = True
    sensor = SchellenbergConnectionSensor(api, _build_entry(hass))

    assert sensor.native_value == "connected"
    assert sensor.icon == "mdi:usb"
    assert sensor.available is True


def test_connection_sensor_reflects_disconnected_state(hass: HomeAssistant) -> None:
    api = MagicMock()
    api.is_connected = False
    sensor = SchellenbergConnectionSensor(api, _build_entry(hass))

    assert sensor.native_value == "disconnected"
    assert sensor.icon == "mdi:usb-off"
    assert sensor.available is False


def test_version_sensor_reports_the_apis_device_version(hass: HomeAssistant) -> None:
    api = MagicMock()
    api.device_version = "20180510_DFBD"
    sensor = SchellenbergVersionSensor(api, _build_entry(hass))

    assert sensor.native_value == "20180510_DFBD"
    assert sensor.icon == "mdi:chip"


def test_version_sensor_reports_none_before_verification(hass: HomeAssistant) -> None:
    api = MagicMock()
    api.device_version = None
    sensor = SchellenbergVersionSensor(api, _build_entry(hass))

    assert sensor.native_value is None


def test_mode_sensor_maps_each_known_mode_to_its_own_icon(hass: HomeAssistant) -> None:
    api = MagicMock()
    sensor = SchellenbergModeSensor(api, _build_entry(hass))

    for mode, expected_icon in (
        ("listening", "mdi:ear-hearing"),
        ("bootloader", "mdi:restart"),
        ("initial", "mdi:power"),
        ("pairing", "mdi:help-circle"),
        ("unknown", "mdi:help-circle"),
    ):
        api.device_mode = mode
        assert sensor.native_value == mode
        assert sensor.icon == expected_icon


async def test_sensor_updates_on_dispatcher_signal(hass: HomeAssistant) -> None:
    """A sensor must refresh its state when the stick status signal fires."""
    api = MagicMock()
    api.is_connected = True
    entry = _build_entry(hass)
    sensor = SchellenbergConnectionSensor(api, entry)
    sensor.hass = hass
    sensor.entity_id = "sensor.schellenberg_usb_stick_connection"
    sensor.async_write_ha_state = MagicMock()  # type: ignore[method-assign]

    await sensor.async_added_to_hass()
    async_dispatcher_send(hass, SIGNAL_STICK_STATUS_UPDATED)
    await hass.async_block_till_done()

    sensor.async_write_ha_state.assert_called_once()


async def test_async_setup_entry_creates_three_sensors_grouped_under_hub(
    hass: HomeAssistant,
) -> None:
    """All three sensors must be added under the hub subentry, not ungrouped."""
    api = MagicMock()
    api.device_version = "20180510_DFBD"
    entry = _build_entry(hass)
    entry.runtime_data = api
    hub_subentry = ConfigSubentry(
        data=MappingProxyType({}),
        subentry_type=SUBENTRY_TYPE_HUB,
        title="Hub",
        unique_id="hub",
    )
    hass.config_entries.async_add_subentry(entry, hub_subentry)

    added: list[Any] = []
    subentry_ids: list[str | None] = []

    def fake_add_entities(
        new_entities: Iterable[Entity],
        update_before_add: bool = False,
        *,
        config_subentry_id: str | None = None,
    ) -> None:
        added.extend(new_entities)
        subentry_ids.append(config_subentry_id)

    await async_setup_entry(hass, entry, fake_add_entities)

    assert len(added) == 3
    assert {type(entity).__name__ for entity in added} == {
        "SchellenbergConnectionSensor",
        "SchellenbergVersionSensor",
        "SchellenbergModeSensor",
    }
    assert subentry_ids == [hub_subentry.subentry_id]
