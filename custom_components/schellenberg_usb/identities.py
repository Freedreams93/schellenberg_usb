"""Helpers for persisted Schellenberg status identities."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

type StatusIdentity = tuple[str, str]

_STATUS_IDENTITY_PATTERN = re.compile(
    r"^([0-9A-Fa-f]{6})\s*[/:-]\s*([0-9A-Fa-f]{1,2})$"
)


def normalize_status_identity(
    device_id: object, device_enum: object
) -> StatusIdentity | None:
    """Normalize one status identity, returning None when it is malformed."""
    normalized_id = str(device_id).strip().upper()
    normalized_enum = str(device_enum).strip().upper()
    # Guarded on non-empty first: "".zfill(2) is "00", a real, specific
    # enum some other device could legitimately use - padding a genuinely
    # missing enum into it would silently accept malformed input as that
    # enum instead of correctly falling through to the rejection below.
    if normalized_enum and len(normalized_enum) <= 2:
        normalized_enum = normalized_enum.zfill(2)
    if (
        len(normalized_id) != 6
        or len(normalized_enum) != 2
        or any(character not in "0123456789ABCDEF" for character in normalized_id)
        or any(character not in "0123456789ABCDEF" for character in normalized_enum)
    ):
        return None
    return normalized_id, normalized_enum


def normalize_status_identities(value: object) -> tuple[StatusIdentity, ...]:
    """Normalize persisted list or text representations, ignoring invalid items."""
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        try:
            return parse_status_identities_text(value)
        except ValueError:
            return ()
    if not isinstance(value, Iterable) or isinstance(value, (bytes, Mapping)):
        return ()

    identities: list[StatusIdentity] = []
    for item in value:
        identity: StatusIdentity | None = None
        if isinstance(item, Mapping):
            identity = normalize_status_identity(
                item.get("device_id", ""),
                item.get("enum", item.get("device_enum", "")),
            )
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            identity = normalize_status_identity(item[0], item[1])
        if identity is not None and identity not in identities:
            identities.append(identity)
    return tuple(identities)


def parse_status_identities_text(value: object) -> tuple[StatusIdentity, ...]:
    """Parse comma/newline-separated DEVICE_ID/ENUM identities strictly."""
    text = str(value or "").strip()
    if not text:
        return ()
    identities: list[StatusIdentity] = []
    for raw_token in re.split(r"[,;\n]+", text):
        token = raw_token.strip()
        if not token:
            continue
        match = _STATUS_IDENTITY_PATTERN.fullmatch(token)
        if match is None:
            raise ValueError(f"invalid status identity: {token}")
        identity = normalize_status_identity(match.group(1), match.group(2))
        if identity is None:
            raise ValueError(f"invalid status identity: {token}")
        if identity not in identities:
            identities.append(identity)
    return tuple(identities)


def serialize_status_identities(
    identities: Iterable[StatusIdentity],
) -> list[dict[str, str]]:
    """Return stable JSON-compatible status identity dictionaries."""
    return [
        {"device_id": device_id, "enum": device_enum}
        for device_id, device_enum in identities
    ]


def format_status_identities(value: object) -> str:
    """Format persisted identities for a multiline config-flow text field."""
    return "\n".join(
        f"{device_id}/{device_enum}"
        for device_id, device_enum in normalize_status_identities(value)
    )


#: Commands api.py's frame handler accepts as confirming the direction *it
#: itself just commanded* during that capture phase (see its "{phase}_endstop"
#: relabelling of a stop frame received mid-leg). "00"/"01"/"02" are
#: CMD_STOP/CMD_UP/CMD_DOWN from const.py, duplicated here as literals
#: rather than imported, matching how `recognized` below already does it.
_PHASE_CONFIRMING_COMMANDS = {
    "opening": {"01"},
    "opening_endstop": {"00", "01"},
    "closing": {"02"},
    "closing_endstop": {"00", "02"},
}

#: The exact order "Discover status from original remote" instructs the
#: person to press on their *existing* remote (strings.json's
#: discover_status description: "press OPEN, STOP, CLOSE, STOP"), reduced
#: to first-seen distinct commands: "01" (open) before "00" (the first
#: stop) before "02" (close). Unlike _PHASE_CONFIRMING_COMMANDS above,
#: nothing commands the blind during this capture - api.py labels every
#: frame the same constant "remote_discovery" phase throughout (there is no
#: leg to tag), so that mechanism cannot help here at all. This instructed
#: press order is the one ground-truth fact this flow still has to go on.
_REMOTE_DISCOVERY_EXPECTED_SEQUENCE = ("01", "00", "02")


def _remote_discovery_sequence_match_length(commands: list[str]) -> int:
    """Return how many leading commands match the instructed press order.

    A device actually being operated per the on-screen instructions should
    see its first (up to) three distinct recognized commands in this order -
    group["commands"] already collapses to first-occurrence order (see
    summarize_status_discovery_frames below), so this does not care how many
    raw frames each press produced or how many times "00" was sent. An
    unrelated device transmitting coincidentally during the same window has
    no reason to follow the instructed order, so a longer match is real
    evidence this group is the one actually being pressed.
    """
    match_length = 0
    # strict=False: commands may be shorter than the 3-element expected
    # sequence (fewer distinct commands seen so far) or longer (more than
    # open/stop/close were recognized) - both are fine, zip() should just
    # stop at whichever is shorter rather than require equal length.
    for expected, actual in zip(
        _REMOTE_DISCOVERY_EXPECTED_SEQUENCE, commands, strict=False
    ):
        if expected != actual:
            break
        match_length += 1
    return match_length


def summarize_status_discovery_frames(
    frames: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Group one bounded remote-capture window and choose its tracking stream.

    A calibration run's capture window is not a neutral listening post: for
    each leg, *we* already know which direction we just commanded (the phase
    recorded alongside each frame - see api.py). A frame whose command
    matches the direction being driven in that phase (an "01" while
    "opening", a "02" while "closing", a "00" at either leg's end stop) is
    strong evidence of being this blind's own motor; one that does not (an
    "01" received mid-"closing", say) is more likely another device on the
    same stick coincidentally transmitting during the window - exactly the
    ambiguity a busy installation with several blinds/winders on one stick
    runs into with only a single up/down cycle to go on. Weighting by that
    match lets a single ordinary calibration run resolve this correctly far
    more often, without asking the person to cycle the blind extra times.

    Frames from a plain, unphased capture (the manual "Discover status from
    original remote" window, which labels everything the same constant
    phase throughout, since nothing is commanded there) never match the
    phase check above, so every candidate ties at zero on it and the
    comparison falls through to _remote_discovery_sequence_match_length
    instead: that flow's own on-screen instructions name a specific press
    order ("OPEN, STOP, CLOSE, STOP"), which is exactly the kind of
    ground-truth signal the phase check relies on for calibration, just
    sourced from the person's button presses instead of a commanded
    direction. Frames from *neither* flow (e.g. identities normalized from
    plain historical/manual data with no "phase" key at all) tie at zero on
    both and fall through to the original distinct-commands/frame-count
    comparison, exactly as before either signal existed.
    """
    recognized = {"00", "01", "02"}
    groups: dict[StatusIdentity, dict[str, Any]] = {}
    for frame in frames:
        identity = normalize_status_identity(
            frame.get("device_id", ""), frame.get("enum", "")
        )
        if identity is None:
            continue
        command = str(frame.get("command", "")).strip().upper()
        if len(command) != 2 or any(
            character not in "0123456789ABCDEF" for character in command
        ):
            continue
        group = groups.setdefault(
            identity,
            {
                "device_id": identity[0],
                "enum": identity[1],
                "commands": [],
                "timestamps": [],
                "frame_count": 0,
                "recognized_frame_count": 0,
                "phase_confirmed_frame_count": 0,
            },
        )
        group["frame_count"] += 1
        if command in recognized:
            group["recognized_frame_count"] += 1
        phase = str(frame.get("phase", "")).strip().lower()
        if command in _PHASE_CONFIRMING_COMMANDS.get(phase, ()):
            group["phase_confirmed_frame_count"] += 1
        if command not in group["commands"]:
            group["commands"].append(command)
        timestamp = str(frame.get("time", "")).strip()
        if timestamp:
            group["timestamps"].append(timestamp)

    # Computed once per group from its finished "commands" list (unlike
    # phase_confirmed_frame_count above, nothing here needs per-frame state),
    # and stored on the group - not just inlined in the max() key below - so
    # it is visible for diagnostics/tests the same way phase_confirmed_frame_
    # count already is.
    for group in groups.values():
        group["remote_discovery_sequence_match_length"] = (
            _remote_discovery_sequence_match_length(group["commands"])
        )

    candidates = [
        group for group in groups.values() if recognized.intersection(group["commands"])
    ]
    primary = (
        max(
            candidates,
            key=lambda group: (
                group["phase_confirmed_frame_count"],
                group["remote_discovery_sequence_match_length"],
                len(recognized.intersection(group["commands"])),
                group["recognized_frame_count"],
            ),
        )
        if candidates
        else None
    )
    secondary = [group for group in groups.values() if group is not primary]
    unknown_commands = [
        {
            "device_id": group["device_id"],
            "enum": group["enum"],
            "commands": [
                command for command in group["commands"] if command not in recognized
            ],
        }
        for group in groups.values()
        if any(command not in recognized for command in group["commands"])
    ]
    return {
        "primary": primary,
        "secondary": secondary,
        "groups": list(groups.values()),
        "unknown_commands": unknown_commands,
        "position_tracking_available": primary is not None,
    }
