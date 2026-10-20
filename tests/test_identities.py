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


def test_normalize_status_identity_rejects_an_empty_enum_instead_of_padding_it() -> (
    None
):
    """Regression: "".zfill(2) is "00", a real enum some other device could use.

    The zfill guard used to run unconditionally, so a device_enum that was
    empty (rather than merely a short, genuine value like "1") silently
    became the specific, valid-looking enum "00" instead of being rejected
    as the malformed input it actually was.
    """
    assert normalize_status_identity("5D3E7C", "") is None


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


def test_summarize_status_discovery_frames_prefers_phase_confirmed_group() -> None:
    """A calibration run's capture window labels each frame with the leg
    it was taken during (see api.py's phase tagging). A frame whose
    command matches the direction actually being driven in that phase is
    strong evidence of being this blind's own motor; weighting by that
    match lets a single ordinary calibration cycle resolve the right
    status identity even when a noisier, unrelated device on the same
    stick (e.g. one of several Gurtwickler sharing it) happens to
    transmit *more* raw frames during the same window - without asking
    the person to cycle the blind extra times to get a reliable reading.

    Without the phase-aware tiebreaker, this exact scenario picks the
    wrong device: both candidates tie on 3 distinct recognized commands,
    and AABBCC/03's 5 recognized frames would outrank E3D0A5/08's 4.
    """
    frames = [
        # --- the real blind, correctly phased ---
        {"device_id": "E3D0A5", "enum": "08", "command": "01", "phase": "opening"},
        {
            "device_id": "E3D0A5",
            "enum": "08",
            "command": "00",
            "phase": "opening_endstop",
        },
        {"device_id": "E3D0A5", "enum": "08", "command": "02", "phase": "closing"},
        {
            "device_id": "E3D0A5",
            "enum": "08",
            "command": "00",
            "phase": "closing_endstop",
        },
        # --- an unrelated device, coincidental noise, wrong phase every
        # time, but *more* raw frames than the real device above ---
        {"device_id": "AABBCC", "enum": "03", "command": "01", "phase": "closing"},
        {"device_id": "AABBCC", "enum": "03", "command": "02", "phase": "opening"},
        {
            "device_id": "AABBCC",
            "enum": "03",
            "command": "00",
            "phase": "idle_between_legs",
        },
        {
            "device_id": "AABBCC",
            "enum": "03",
            "command": "01",
            "phase": "idle_between_legs",
        },
        {
            "device_id": "AABBCC",
            "enum": "03",
            "command": "01",
            "phase": "idle_between_legs",
        },
    ]

    result = summarize_status_discovery_frames(frames)

    primary = result["primary"]
    assert primary is not None
    assert (primary["device_id"], primary["enum"]) == ("E3D0A5", "08")
    assert primary["phase_confirmed_frame_count"] == 4
    assert primary["recognized_frame_count"] == 4


def test_summarize_status_discovery_frames_phase_match_breaks_a_frame_count_tie() -> (
    None
):
    """Minimal direct tie-break: two devices with the exact same
    recognized-command set and frame count (so the pre-existing
    tiebreakers alone would pick whichever is listed first - here,
    AAAAAA), where only one of them actually matches the phase it was
    heard in.
    """
    frames = [
        {"device_id": "AAAAAA", "enum": "02", "command": "01", "phase": "closing"},
        {"device_id": "AAAAAA", "enum": "02", "command": "00", "phase": "idle"},
        {"device_id": "5D3E7C", "enum": "01", "command": "01", "phase": "opening"},
        {"device_id": "5D3E7C", "enum": "01", "command": "00", "phase": "idle"},
    ]

    result = summarize_status_discovery_frames(frames)

    assert result["primary"]["device_id"] == "5D3E7C"
    assert result["primary"]["phase_confirmed_frame_count"] == 1


def test_summarize_status_discovery_frames_prefers_instructed_press_order() -> None:
    """The "Discover status from original remote" step tells the person to
    press OPEN, STOP, CLOSE, STOP on their existing remote (see
    strings.json's discover_status description) - and that capture window
    labels every frame with the same constant phase throughout (api.py's
    "remote_discovery"), so phase_confirmed_frame_count can never help
    disambiguate here (see the unphased test below). A device whose first
    distinct recognized commands follow that exact order is strong
    evidence of being the remote actually being pressed; an unrelated
    device transmitting coincidentally during the same 45-second window
    has no reason to follow it.

    Without that signal, this exact scenario picks the wrong device: both
    candidates tie on 3 distinct recognized commands, and AABBCC's 5
    recognized frames would outrank 5D3E7C's 4.
    """
    frames = [
        # --- the real remote, pressed in the instructed order ---
        {
            "device_id": "5D3E7C",
            "enum": "01",
            "command": "01",
            "phase": "remote_discovery",
        },
        {
            "device_id": "5D3E7C",
            "enum": "01",
            "command": "00",
            "phase": "remote_discovery",
        },
        {
            "device_id": "5D3E7C",
            "enum": "01",
            "command": "02",
            "phase": "remote_discovery",
        },
        {
            "device_id": "5D3E7C",
            "enum": "01",
            "command": "00",
            "phase": "remote_discovery",
        },
        # --- an unrelated device, coincidental noise during the same
        # window: same 3 distinct commands but in the wrong order, and
        # *more* raw frames than the real remote above ---
        {
            "device_id": "AABBCC",
            "enum": "03",
            "command": "02",
            "phase": "remote_discovery",
        },
        {
            "device_id": "AABBCC",
            "enum": "03",
            "command": "00",
            "phase": "remote_discovery",
        },
        {
            "device_id": "AABBCC",
            "enum": "03",
            "command": "01",
            "phase": "remote_discovery",
        },
        {
            "device_id": "AABBCC",
            "enum": "03",
            "command": "01",
            "phase": "remote_discovery",
        },
        {
            "device_id": "AABBCC",
            "enum": "03",
            "command": "00",
            "phase": "remote_discovery",
        },
    ]

    result = summarize_status_discovery_frames(frames)

    primary = result["primary"]
    assert primary is not None
    assert (primary["device_id"], primary["enum"]) == ("5D3E7C", "01")
    assert primary["remote_discovery_sequence_match_length"] == 3
    assert primary["recognized_frame_count"] == 4


def test_summarize_status_discovery_frames_sequence_match_breaks_a_tie() -> None:
    """Minimal direct tie-break: two devices with the exact same
    recognized-command set and frame count (so the pre-existing
    tiebreakers alone would pick whichever is listed first - here,
    AAAAAA), both in the unphased "remote_discovery" window where
    phase_confirmed_frame_count can never help, where only one of them
    was actually pressed in the instructed OPEN-before-STOP order.
    """
    frames = [
        {
            "device_id": "AAAAAA",
            "enum": "02",
            "command": "02",
            "phase": "remote_discovery",
        },
        {
            "device_id": "AAAAAA",
            "enum": "02",
            "command": "00",
            "phase": "remote_discovery",
        },
        {
            "device_id": "5D3E7C",
            "enum": "01",
            "command": "01",
            "phase": "remote_discovery",
        },
        {
            "device_id": "5D3E7C",
            "enum": "01",
            "command": "00",
            "phase": "remote_discovery",
        },
    ]

    result = summarize_status_discovery_frames(frames)

    assert result["primary"]["device_id"] == "5D3E7C"
    assert result["primary"]["remote_discovery_sequence_match_length"] == 2


def test_summarize_status_discovery_frames_unphased_capture_is_unaffected() -> None:
    """The manual "discover status" remote-listening window labels every
    frame with the same constant phase throughout (e.g. "remote_
    discovery"), which never matches a direction in
    _PHASE_CONFIRMING_COMMANDS - so every candidate ties at
    phase_confirmed_frame_count=0 there. Neither candidate here starts
    with the instructed first press ("01"/open) either, so they also tie
    at zero on remote_discovery_sequence_match_length, and the comparison
    falls through to the original distinct-commands/frame-count tiebreaker
    exactly as it did before either signal existed.
    """
    frames = [
        {
            "device_id": "5D3E7C",
            "enum": "01",
            "command": "00",
            "phase": "remote_discovery",
        },
        {
            "device_id": "5D3E7C",
            "enum": "01",
            "command": "02",
            "phase": "remote_discovery",
        },
        {
            "device_id": "AAAAAA",
            "enum": "02",
            "command": "02",
            "phase": "remote_discovery",
        },
    ]

    result = summarize_status_discovery_frames(frames)

    assert result["primary"]["device_id"] == "5D3E7C"
    assert result["primary"]["phase_confirmed_frame_count"] == 0
    assert result["primary"]["remote_discovery_sequence_match_length"] == 0


def test_summarize_status_discovery_frames_without_a_phase_key_is_unaffected() -> None:
    """Frames with no "phase" key at all (e.g. from before this field
    existed, or a caller that never set one) must not raise, and must
    behave exactly as if the phase-aware scoring did not exist.
    """
    frames = [
        {"device_id": "5D3E7C", "enum": "01", "command": "01"},
        {"device_id": "5D3E7C", "enum": "01", "command": "00"},
    ]

    result = summarize_status_discovery_frames(frames)

    assert result["primary"]["device_id"] == "5D3E7C"
    assert result["primary"]["phase_confirmed_frame_count"] == 0
