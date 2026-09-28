"""Tests for the device-registry compatibility helpers.

These exercise the feature-detection logic itself (new singular
config_entry_id/config_subentry_id attributes vs. the legacy
config_entries_subentries mapping, and the new single-call
async_update_device(new_config_entry_id=...) vs. the old add/remove pair)
with small stand-in objects, since the real DeviceEntry/DeviceRegistry
classes only matter here for their shape, not their behavior.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from custom_components.schellenberg_usb.device_registry_compat import (
    async_device_on_subentry_compat,
    async_get_device_by_identifier_compat,
    async_reassign_device_subentry_compat,
)

DOMAIN = "schellenberg_usb"


def test_get_device_by_identifier_uses_scoped_lookup_when_available() -> None:
    calls: list[tuple[Any, Any]] = []

    def scoped_lookup(identifier: Any, config_entry_id: Any) -> str:
        calls.append((identifier, config_entry_id))
        return "the-device"

    registry: Any = SimpleNamespace(async_get_device_by_identifier=scoped_lookup)

    result: Any = async_get_device_by_identifier_compat(
        registry,
        (DOMAIN, "abc"),
        "entry1",
    )

    assert result == "the-device"
    assert calls == [((DOMAIN, "abc"), "entry1")]


def test_get_device_by_identifier_falls_back_to_unscoped_lookup() -> None:
    calls: list[Any] = []

    def async_get_device(*, identifiers: Any) -> str:
        calls.append(identifiers)
        return "the-device"

    # Deliberately has no async_get_device_by_identifier attribute at all,
    # simulating every currently released Home Assistant version.
    registry: Any = SimpleNamespace(async_get_device=async_get_device)

    result: Any = async_get_device_by_identifier_compat(
        registry,
        (DOMAIN, "abc"),
        "entry1",
    )

    assert result == "the-device"
    assert calls == [{(DOMAIN, "abc")}]


def test_on_subentry_compat_prefers_new_singular_attributes() -> None:
    device = SimpleNamespace(
        config_entry_id="entry1",
        config_subentry_id="sub1",
        # A deliberately wrong legacy mapping proves it is never consulted
        # once the new singular attributes are present.
        config_entries_subentries={"entry1": {"sub-wrong"}},
    )

    assert async_device_on_subentry_compat(device, "entry1", "sub1") is True  # type: ignore[arg-type]
    assert async_device_on_subentry_compat(device, "entry1", "sub-wrong") is False  # type: ignore[arg-type]
    assert async_device_on_subentry_compat(device, "other-entry", "sub1") is False  # type: ignore[arg-type]


def test_on_subentry_compat_falls_back_to_legacy_mapping() -> None:
    # No config_entry_id attribute at all - simulates a Home Assistant
    # release older than the singular-attribute migration.
    device = SimpleNamespace(config_entries_subentries={"entry1": {"sub1", "sub2"}})

    assert async_device_on_subentry_compat(device, "entry1", "sub1") is True  # type: ignore[arg-type]
    assert async_device_on_subentry_compat(device, "entry1", "sub9") is False  # type: ignore[arg-type]
    assert async_device_on_subentry_compat(device, "other-entry", "sub1") is False  # type: ignore[arg-type]


def test_reassign_is_a_noop_when_already_on_the_target_subentry() -> None:
    device = SimpleNamespace(
        id="dev1", config_entry_id="entry1", config_subentry_id="sub1"
    )
    calls: list[Any] = []
    registry = SimpleNamespace(
        async_update_device=lambda *a, **kw: calls.append((a, kw))
    )

    async_reassign_device_subentry_compat(registry, device, "entry1", "sub1")  # type: ignore[arg-type]

    assert calls == []


def test_reassign_uses_the_new_single_call_signature_when_available() -> None:
    device = SimpleNamespace(
        id="dev1", config_entry_id="entry1", config_subentry_id="old-sub"
    )
    calls: list[tuple[Any, Any, Any]] = []

    def async_update_device(
        device_id: Any,
        *,
        new_config_entry_id: Any = None,
        new_config_subentry_id: Any = None,
    ) -> None:
        calls.append((device_id, new_config_entry_id, new_config_subentry_id))

    registry = SimpleNamespace(async_update_device=async_update_device)

    async_reassign_device_subentry_compat(registry, device, "entry1", "new-sub")  # type: ignore[arg-type]

    assert calls == [("dev1", "entry1", "new-sub")]


def test_reassign_falls_back_to_add_then_remove_on_old_signature() -> None:
    # No config_entry_id attribute -> also exercises the pre-migration
    # branch of the no-op check above, alongside the old update signature.
    device = SimpleNamespace(
        id="dev1", config_entries_subentries={"entry1": {"old-sub"}}
    )
    calls: list[tuple[Any, Any, Any, Any, Any]] = []

    def async_update_device(
        device_id: Any,
        *,
        add_config_entry_id: Any = None,
        add_config_subentry_id: Any = None,
        remove_config_entry_id: Any = None,
        remove_config_subentry_id: Any = None,
    ) -> None:
        calls.append(
            (
                device_id,
                add_config_entry_id,
                add_config_subentry_id,
                remove_config_entry_id,
                remove_config_subentry_id,
            )
        )

    registry = SimpleNamespace(async_update_device=async_update_device)

    async_reassign_device_subentry_compat(registry, device, "entry1", "new-sub")  # type: ignore[arg-type]

    # Adding the new membership must happen before removing the old one -
    # removing first would momentarily leave the device with none at all.
    assert calls == [
        ("dev1", "entry1", "new-sub", None, None),
        ("dev1", None, None, "entry1", "old-sub"),
    ]
