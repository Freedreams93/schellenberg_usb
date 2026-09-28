"""Tests for the status-identity parsing/normalization helpers."""

from __future__ import annotations

import pytest

from custom_components.schellenberg_usb.identities import (
    normalize_status_identities,
    normalize_status_identity,
    parse_status_identities_text,
    summarize_status_discovery_frames,
)


def test_normalize_status_identity_accepts_and_rejects() -> None:
    assert normalize_status_identity("5d3e7c", "1") == ("5D3E7C", "01")
    assert normalize_status_identity("5D3E7C", "10") == ("5D3E7C", "10")
    # Too short / not hex -> malformed, must not raise.
    assert normalize_status_identity("5D3E7", "01") is None
    assert normalize_status_identity("5D3E7G", "01") is None


def test_parse_status_identities_text_accepts_documented_separators() -> None:
    text = "5D3E7C/01, ABCDEF-02\n112233:3"
    assert parse_status_identities_text(text) == (
        ("5D3E7C", "01"),
        ("ABCDEF", "02"),
        ("112233", "03"),
    )


def test_parse_status_identities_text_rejects_malformed_tokens() -> None:
    with pytest.raises(ValueError, match="invalid status identity"):
        parse_status_identities_text("not-an-identity")


def test_normalize_status_identities_accepts_multiple_input_shapes() -> None:
    expected = (("5D3E7C", "01"),)

    assert normalize_status_identities("5D3E7C/01") == expected
    assert normalize_status_identities([("5D3E7C", "01")]) == expected
    assert (
        normalize_status_identities([{"device_id": "5D3E7C", "enum": "01"}]) == expected
    )
    assert normalize_status_identities(None) == ()
    assert normalize_status_identities("") == ()
    # A malformed free-text value must degrade to "no identities", not raise.
    assert normalize_status_identities("garbage") == ()


def test_summarize_status_discovery_frames_prefers_most_recognized_group() -> None:
    frames = [
        {"device_id": "5D3E7C", "enum": "01", "command": "01", "time": "10:00:00"},
        {"device_id": "5D3E7C", "enum": "01", "command": "00", "time": "10:00:05"},
        # A second, less-frequently-seen transmitter caught on the same
        # channel must not be mistaken for the primary remote.
        {"device_id": "AAAAAA", "enum": "02", "command": "01", "time": "10:00:02"},
    ]

    result = summarize_status_discovery_frames(frames)

    assert result["primary"]["device_id"] == "5D3E7C"
    assert result["primary"]["recognized_frame_count"] == 2
    assert result["position_tracking_available"] is True
    assert len(result["secondary"]) == 1
    assert result["secondary"][0]["device_id"] == "AAAAAA"


def test_summarize_status_discovery_frames_with_no_recognized_commands() -> None:
    frames = [
        {"device_id": "5D3E7C", "enum": "01", "command": "40", "time": "10:00:00"}
    ]

    result = summarize_status_discovery_frames(frames)

    assert result["primary"] is None
    assert result["position_tracking_available"] is False
    assert result["unknown_commands"] == [
        {"device_id": "5D3E7C", "enum": "01", "commands": ["40"]}
    ]
