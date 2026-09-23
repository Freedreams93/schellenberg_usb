"""Compatibility helpers for Home Assistant's device registry API.

``dr.async_get_device_id_by_identifier`` never existed in Home Assistant -
calling it raised AttributeError, which used to crash ``async_setup_entry``
on every attempt (see the NOTE in __init__.py and cover.py). These helpers
centralize the correct calls instead of leaving each call site to get the
device-registry API right on its own.

A future Home Assistant release may add a config-entry-scoped device
lookup (some in-development branches already expose one); if so,
``async_get_device_by_identifier_compat`` picks it up automatically via
feature detection, without needing a code change here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.helpers.device_registry import DeviceEntry, DeviceRegistry


def async_get_device_by_identifier_compat(
    device_registry: DeviceRegistry,
    identifier: tuple[str, str],
    config_entry_id: str,
) -> DeviceEntry | None:
    """Look up a device by identifier, scoped to a config entry when possible.

    Uses a config-entry-scoped lookup when the installed Home Assistant
    release exposes one; falls back to the standard, un-scoped
    ``async_get_device(identifiers=...)`` otherwise (identifiers are
    globally unique on every currently released Home Assistant version).
    """
    scoped_lookup = getattr(device_registry, "async_get_device_by_identifier", None)
    if scoped_lookup is not None:
        return scoped_lookup(identifier, config_entry_id)
    return device_registry.async_get_device(identifiers={identifier})


def async_reassign_device_subentry_compat(
    device_registry: DeviceRegistry,
    device: DeviceEntry,
    config_entry_id: str,
    new_subentry_id: str,
) -> None:
    """Move ``device`` onto ``new_subentry_id`` of ``config_entry_id``.

    Home Assistant models a device's config-subentry membership as a set
    per config entry (``DeviceEntry.config_entries_subentries``), not a
    single value, and has no single-call "move" primitive. The new
    subentry is therefore added before the old one is removed: removing
    it first would - for a device with only one subentry on this config
    entry, the normal case here - momentarily leave the config entry with
    no subentries for this device. ``DeviceRegistry`` treats that as the
    device no longer belonging to the config entry at all, and if it was
    the device's only config entry, ``async_update_device`` deletes the
    device outright.
    """
    current_subentry_ids = device.config_entries_subentries.get(config_entry_id, set())
    if current_subentry_ids == {new_subentry_id}:
        return  # already on the right subentry - nothing to do

    device_registry.async_update_device(
        device.id,
        add_config_entry_id=config_entry_id,
        add_config_subentry_id=new_subentry_id,
    )
    for old_subentry_id in current_subentry_ids - {new_subentry_id}:
        device_registry.async_update_device(
            device.id,
            remove_config_entry_id=config_entry_id,
            remove_config_subentry_id=old_subentry_id,
        )
