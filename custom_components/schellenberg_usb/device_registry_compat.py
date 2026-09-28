"""Compatibility helpers for Home Assistant's device registry API.

``dr.async_get_device_id_by_identifier`` never existed in Home Assistant -
calling it raised AttributeError, which used to crash ``async_setup_entry``
on every attempt (see the NOTE in __init__.py and cover.py). These helpers
centralize the correct calls instead of leaving each call site to get the
device-registry API right on its own.

As of Home Assistant Core 2026.9, ``async_get_device_by_identifier`` exists
as a stable, released config-entry-scoped device lookup.
``async_get_device_by_identifier_compat`` picks it up automatically via
feature detection when present, and falls back to the older, unscoped
``async_get_device(identifiers=...)`` on any release without it - which
still matters here since this integration's declared minimum supported
version (see hacs.json) predates 2026.9.

Home Assistant Core 2026.10 also deprecated ``DeviceEntry.config_entries``,
``.config_entries_subentries``, and ``.primary_config_entry``: a device can
now only ever belong to one config entry (and one subentry of it), so those
historical multi-entry properties are being replaced by singular
``config_entry_id`` / ``config_subentry_id`` attributes. Reading the old
properties only logs a warning on a custom integration (it is not a hard
error there until Home Assistant Core 2027.10), but
``async_device_on_subentry_compat`` below avoids even that by reading the
new attributes first, via the same feature-detection approach.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, cast

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
        # The feature-detected method's return type is unknown to mypy -
        # cast rather than let `Any` leak into this function's own return
        # type, since we know its contract matches async_get_device's.
        return cast("DeviceEntry | None", scoped_lookup(identifier, config_entry_id))
    return device_registry.async_get_device(identifiers={identifier})


def async_device_on_subentry_compat(
    device: DeviceEntry,
    config_entry_id: str,
    subentry_id: str,
) -> bool:
    """Return whether ``device`` is already on ``subentry_id`` of ``config_entry_id``.

    Reads the singular ``config_entry_id`` / ``config_subentry_id``
    attributes when the installed Home Assistant release already exposes
    them (see the module docstring), so this never touches the deprecated
    ``config_entries_subentries`` mapping on a release new enough to warn
    about it. Falls back to that mapping, unchanged, on older releases that
    do not have the singular attributes yet.
    """
    new_entry_id = getattr(device, "config_entry_id", None)
    if new_entry_id is not None:
        return (
            new_entry_id == config_entry_id
            and getattr(device, "config_subentry_id", None) == subentry_id
        )
    return subentry_id in device.config_entries_subentries.get(config_entry_id, set())


def async_reassign_device_subentry_compat(
    device_registry: DeviceRegistry,
    device: DeviceEntry,
    config_entry_id: str,
    new_subentry_id: str,
) -> None:
    """Move ``device`` onto ``new_subentry_id`` of ``config_entry_id``.

    Home Assistant 2026.8 added ``async_update_device(new_config_entry_id=,
    new_config_subentry_id=)`` as a single-call replacement that also keeps
    the device's disabled state consistent with its (possibly new) config
    entry; that call is used when the installed release supports it.

    On older releases, without that parameter, moving a device is done by
    adding the new config-subentry membership before removing the old one:
    removing it first would - for a device with only one subentry on this
    config entry, the normal case here - momentarily leave the config
    entry with no subentries for this device. ``DeviceRegistry`` treats
    that as the device no longer belonging to the config entry at all,
    and if it was the device's only config entry, ``async_update_device``
    deletes the device outright. ``add_config_entry_id``/
    ``remove_config_entry_id`` and their ``*_subentry_id`` counterparts
    are deprecated for this reason - they cannot express the move in one
    call - but still work on every currently supported Home Assistant
    release.
    """
    if async_device_on_subentry_compat(device, config_entry_id, new_subentry_id):
        return  # already on the right subentry - nothing to do

    update_params = inspect.signature(device_registry.async_update_device).parameters
    if "new_config_entry_id" in update_params:
        # mypy type-checks this call against the *installed* Home Assistant
        # release's concrete async_update_device signature. On any release
        # older than 2026.8 that signature has no new_config_entry_id/
        # new_config_subentry_id parameters at all, which mypy would then
        # correctly report against such a release - it cannot see that the
        # inspect.signature() check above already guarantees this call only
        # runs on a release new enough to actually have them. The pinned
        # test toolchain's installed release is new enough that the call
        # matches directly, so no type: ignore is currently needed here -
        # but reintroducing one (with the call-arg code) is expected and
        # correct if a future toolchain bump lands on an older release again.
        device_registry.async_update_device(
            device.id,
            new_config_entry_id=config_entry_id,
            new_config_subentry_id=new_subentry_id,
        )
        return

    # This release's async_update_device predates new_config_entry_id, and
    # both that parameter and the config_entries_subentries deprecation are
    # part of the same single-config-entry-per-device migration, so a
    # release old enough to lack the former necessarily predates the
    # latter too - reading the plural property below does not warn here.
    current_subentry_ids = device.config_entries_subentries.get(config_entry_id, set())
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
