"""Tests for the stable per-blind UUID helpers in blind_id.py."""

from __future__ import annotations

from uuid import UUID

from custom_components.schellenberg_usb.blind_id import (
    claim_blind_id,
    normalize_blind_id,
)


def test_normalize_blind_id_accepts_a_valid_uuid() -> None:
    valid = "12345678-1234-5678-1234-567812345678"
    assert normalize_blind_id(valid) == valid


def test_normalize_blind_id_rejects_legacy_and_garbage_values() -> None:
    # Legacy configs stored a device ID (six hex chars) here, not a UUID.
    assert normalize_blind_id("5D3E7C") is None
    assert normalize_blind_id(None) is None
    assert normalize_blind_id("") is None


def test_claim_blind_id_keeps_a_valid_unused_id_unchanged() -> None:
    existing = "12345678-1234-5678-1234-567812345678"
    used: set[str] = set()

    claimed, needs_update = claim_blind_id(existing, used)

    assert claimed == existing
    assert needs_update is False
    assert used == {existing}


def test_claim_blind_id_generates_one_for_a_legacy_or_missing_value() -> None:
    used: set[str] = set()

    claimed, needs_update = claim_blind_id("5D3E7C", used)

    assert UUID(claimed)  # a fresh, well-formed UUID was generated
    assert needs_update is True
    assert used == {claimed}


def test_claim_blind_id_regenerates_on_collision_with_an_already_used_id() -> None:
    taken = "12345678-1234-5678-1234-567812345678"
    used = {taken}

    claimed, needs_update = claim_blind_id(taken, used)

    assert claimed != taken
    assert UUID(claimed)
    assert needs_update is True
    assert used == {taken, claimed}
