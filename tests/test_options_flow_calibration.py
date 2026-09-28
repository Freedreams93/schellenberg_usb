"""Tests for the shared calibration step machinery (options_flow_calibration.py).

Driven through the real subentry flow manager (hass.config_entries.subentries)
via the "reconfigure" -> "calibrate" route on an *existing* blind subentry,
since CalibrationFlowHandler is only ever constructed by
SchellenbergPairingSubentryFlow (see its own class docstring) - this also
exercises that delegation in config_flow.py, not just this module in
isolation. The "new pairing -> calibration -> create subentry" path is
covered separately in test_config_flow.py, since setting that path up
requires first driving the pairing wizard itself.

Movement-start/stop waits use a real asyncio.Event under the hood (not
asyncio.sleep), so firing the SIGNAL_DEVICE_EVENT dispatcher signal while a
configure() call is in flight resolves them, with no need to mock asyncio -
the same asyncio.ensure_future() pattern tests/test_cover.py and
tests/test_api.py already use to let a task run up to a blocking point.

The one thing every test below still has to get right by hand is *when* to
fire that signal: too early - before the flow task has actually registered
its listener for it - and the signal is silently missed, since
async_dispatcher_send() does not queue a signal for a listener that
subscribes later. Two earlier approaches tried to bridge that gap with a
fixed amount of "settle time" between creating the flow task and sending
its signal - first a fixed count of bare asyncio.sleep(0) calls, then a flat
real asyncio.sleep(0.5) - and neither reliably worked, because both are
still guesses: the exact number of event-loop iterations (or the exact
wall-clock time) the flow task needs to reach its next
async_dispatcher_connect() call depends on internal implementation details
(such as asyncio.wait_for()'s own task-wrapping) this test has no business
knowing, and that can vary across machines and Python versions.

_spy_on_dispatcher_connect() below removes the guesswork entirely: it wraps
async_dispatcher_connect() (as imported into options_flow_calibration.py) so
that every real call to it - which is exactly what
_wait_for_movement_start()/_wait_for_stop_event() each do, synchronously,
right before they start actually waiting - also flips an asyncio.Event the
test can await. Waiting for *that* event resolves the instant the flow task
is actually listening, however many loop iterations or however much
wall-clock time that took, with its own bounded, clearly labelled
pytest.fail() if a registration never happens at all, rather than relying on
pytest-timeout's blunter, whole-test signal timeout to eventually notice.

None of the above was ever the real problem, as it turns out: the actual bug
was in options_flow_calibration.py itself.
_wait_for_movement_start()/_wait_for_stop_event()'s own dispatcher listener
set its asyncio.Event directly, and Home Assistant's dispatcher can - and,
reproduced live via a thread-based pytest-timeout stack dump, did - run that
listener on a worker thread rather than the event loop even though it is
decorated @callback; @callback is good style for a fast listener but is not
a guarantee of inline execution on this integration's tested Home Assistant
version. asyncio.Event.set() is not thread-safe (only reachable safely
cross-thread via loop.call_soon_threadsafe(), not the plain call_soon() a
Future's own set_result() uses internally), so that cross-thread call
raised RuntimeError - which Home Assistant's background job runner only
logs, never surfaces here - meaning the event was never actually set and
the wait below ran to its own full timeout every time. Now fixed at the
source: the listener schedules the .set() call via
hass.loop.call_soon_threadsafe() instead of calling it directly (see that
module). The deterministic wait above remains, since waiting for the real
event beats guessing regardless - it just isn't covering for a live bug
anymore.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from types import MappingProxyType
from typing import Any
from unittest.mock import MagicMock

import pytest
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers.dispatcher import (
    async_dispatcher_connect,
    async_dispatcher_send,
)
from pytest_homeassistant_custom_component.common import MockConfigEntry

import custom_components.schellenberg_usb.options_flow_calibration as calibration_module
from custom_components.schellenberg_usb.api import SchellenbergUsbApi
from custom_components.schellenberg_usb.const import (
    CONF_CLOSE_TIME,
    CONF_DEVICE_ENUM,
    CONF_DEVICE_ID,
    CONF_INVERT_DIRECTION,
    CONF_OPEN_TIME,
    CONF_SECONDARY_STATUS_IDENTITIES,
    CONF_SERIAL_PORT,
    CONF_STATUS_DEVICE_ID,
    CONF_STATUS_ENUM,
    CONF_STATUS_IDENTITY_SOURCE,
    DOMAIN,
    EVENT_STARTED_MOVING_DOWN,
    EVENT_STARTED_MOVING_UP,
    EVENT_STOPPED,
    SIGNAL_CALIBRATION_COMPLETED,
    SIGNAL_DEVICE_EVENT,
    STATUS_IDENTITY_SOURCE_CALIBRATION,
    STATUS_IDENTITY_SOURCE_UNKNOWN,
    SUBENTRY_TYPE_BLIND,
)
from custom_components.schellenberg_usb.options_flow_calibration import (
    CalibrationFlowHandler,
)

DEVICE_ID = "ABCDEF"
DEVICE_ENUM = "10"


def _build_hub_with_blind(
    hass: HomeAssistant, connected_api: SchellenbergUsbApi
) -> tuple[MockConfigEntry, ConfigSubentry]:
    entry = MockConfigEntry(domain=DOMAIN, data={CONF_SERIAL_PORT: "/dev/fake"})
    entry.add_to_hass(hass)
    entry.runtime_data = connected_api
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
    return entry, blind


async def _start_calibrate_flow(
    hass: HomeAssistant, entry: MockConfigEntry, blind: ConfigSubentry
) -> dict[str, Any]:
    """Reach the calibration_close form via reconfigure -> calibrate."""
    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_BLIND),
        context={"source": "reconfigure", "subentry_id": blind.subentry_id},
    )
    assert result["type"] is FlowResultType.MENU
    assert result["step_id"] == "reconfigure"
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={"next_step_id": "calibrate"}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "calibration_close"
    return result


def _spy_on_dispatcher_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[], Awaitable[None]]:
    """Make async_dispatcher_connect() calls awaitable, one at a time.

    _wait_for_movement_start() and _wait_for_stop_event() each create their
    own asyncio.Event and then, with no `await` in between, immediately
    register a dispatcher listener for it - so the moment
    async_dispatcher_connect() is actually called is the precise,
    deterministic point at which the flow task is ready to receive the
    corresponding signal. Nothing about that moment depends on how many
    internal await points the surrounding machinery (asyncio.wait_for()'s
    own task-wrapping, Home Assistant's own flow-manager dispatch, ...) took
    to get there, unlike a fixed sleep(0) count or a flat real sleep - the
    two approaches this module tried before this one that both, in turn,
    still ended up racing that unknown number of hops instead of actually
    waiting for the thing that matters.

    This patches calibration_module.async_dispatcher_connect - the name
    _wait_for_movement_start()/_wait_for_stop_event() actually call, since
    Python resolves it from that module's own namespace at call time - with
    a wrapper that still performs the real registration (so the flow under
    test behaves exactly as it would unpatched) and additionally sets an
    asyncio.Event every time it's called. The coroutine function this
    returns waits for that event and then clears it again, so calling it
    once waits for the *next* registration - however many of them a given
    test needs to step through in turn (one for movement-start, one for
    stop, repeated across the open and close legs).
    """
    real_connect = calibration_module.async_dispatcher_connect
    registered = asyncio.Event()

    def spy(*args: Any, **kwargs: Any) -> Any:
        connection = real_connect(*args, **kwargs)
        registered.set()
        return connection

    monkeypatch.setattr(calibration_module, "async_dispatcher_connect", spy)

    async def wait_for_next_registration() -> None:
        try:
            await asyncio.wait_for(registered.wait(), timeout=5)
        except TimeoutError:
            pytest.fail(
                "async_dispatcher_connect() was not called within 5s - the "
                "calibration flow task never reached its next "
                "_wait_for_movement_start()/_wait_for_stop_event() listener "
                "registration."
            )
        registered.clear()

    return wait_for_next_registration


async def _advance_through_open_leg(
    hass: HomeAssistant,
    flow_id: str,
    wait_for_registration: Callable[[], Awaitable[None]],
) -> dict[str, Any]:
    """From calibration_close's form, drive through the whole open leg."""
    result = await hass.config_entries.subentries.async_configure(
        flow_id, user_input={}
    )
    assert result["step_id"] == "calibration_open_instruction"

    task = asyncio.ensure_future(
        hass.config_entries.subentries.async_configure(flow_id, user_input={})
    )
    await wait_for_registration()  # _wait_for_movement_start is now listening
    signal = f"{SIGNAL_DEVICE_EVENT}_{DEVICE_ID}"
    async_dispatcher_send(hass, signal, EVENT_STARTED_MOVING_UP)
    await wait_for_registration()  # _wait_for_stop_event is now listening
    async_dispatcher_send(hass, signal, EVENT_STOPPED)
    return await task


async def test_calibrate_menu_choice_shows_the_close_instruction_form(
    hass: HomeAssistant, connected_api: SchellenbergUsbApi
) -> None:
    entry, blind = _build_hub_with_blind(hass, connected_api)

    result = await _start_calibrate_flow(hass, entry, blind)

    assert result["description_placeholders"]["device_name"] == "Living Room Blind"


async def test_confirming_close_advances_to_the_open_instruction_form(
    hass: HomeAssistant, connected_api: SchellenbergUsbApi
) -> None:
    entry, blind = _build_hub_with_blind(hass, connected_api)
    result = await _start_calibrate_flow(hass, entry, blind)

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "calibration_open_instruction"


async def test_full_open_and_close_legs_reach_the_summary_with_measured_times(
    hass: HomeAssistant,
    connected_api: SchellenbergUsbApi,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry, blind = _build_hub_with_blind(hass, connected_api)
    result = await _start_calibrate_flow(hass, entry, blind)
    wait_for_registration = _spy_on_dispatcher_connect(monkeypatch)

    result = await _advance_through_open_leg(
        hass, result["flow_id"], wait_for_registration
    )
    assert result["step_id"] == "calibration_close_instruction"

    task = asyncio.ensure_future(
        hass.config_entries.subentries.async_configure(result["flow_id"], user_input={})
    )
    await wait_for_registration()  # _wait_for_movement_start is now listening
    async_dispatcher_send(
        hass, f"{SIGNAL_DEVICE_EVENT}_{DEVICE_ID}", EVENT_STARTED_MOVING_DOWN
    )
    await wait_for_registration()  # _wait_for_stop_event is now listening
    async_dispatcher_send(hass, f"{SIGNAL_DEVICE_EVENT}_{DEVICE_ID}", EVENT_STOPPED)
    result = await task

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "calibration_complete"
    # Both times are real (small but non-zero) durations, floored at 0.1s.
    assert float(result["description_placeholders"]["open_time"]) >= 0.1
    assert float(result["description_placeholders"]["close_time"]) >= 0.1


async def test_recalibrating_persists_new_times_and_notifies_the_live_entity(
    hass: HomeAssistant,
    connected_api: SchellenbergUsbApi,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry, blind = _build_hub_with_blind(hass, connected_api)
    result = await _start_calibrate_flow(hass, entry, blind)
    wait_for_registration = _spy_on_dispatcher_connect(monkeypatch)
    result = await _advance_through_open_leg(
        hass, result["flow_id"], wait_for_registration
    )

    task = asyncio.ensure_future(
        hass.config_entries.subentries.async_configure(result["flow_id"], user_input={})
    )
    await wait_for_registration()  # _wait_for_movement_start is now listening
    async_dispatcher_send(
        hass, f"{SIGNAL_DEVICE_EVENT}_{DEVICE_ID}", EVENT_STARTED_MOVING_DOWN
    )
    await wait_for_registration()  # _wait_for_stop_event is now listening
    async_dispatcher_send(hass, f"{SIGNAL_DEVICE_EVENT}_{DEVICE_ID}", EVENT_STOPPED)
    result = await task
    assert result["step_id"] == "calibration_complete"

    # This listener is a plain function - neither a coroutine nor decorated
    # @callback - so, like every other dispatcher listener this module has
    # run into, Home Assistant is free to invoke it via a worker thread
    # rather than inline; async_configure() below returning is no proof it
    # has actually run yet. A flaky local run caught this directly:
    # len(signals) read as 0 at assert time, while the failure message's own
    # repr of that same list showed the one entry that had, by then, just
    # landed from that other thread - a race, not a data bug. So don't infer
    # delivery from the flow call returning; wait for the listener to
    # actually fire, the same way wait_for_registration() above waits for a
    # registration instead of guessing when it happened.
    signals: list[tuple[Any, ...]] = []
    received = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _on_calibration_completed(*args: Any) -> None:
        signals.append(args)
        loop.call_soon_threadsafe(received.set)

    unsub = async_dispatcher_connect(
        hass, SIGNAL_CALIBRATION_COMPLETED, _on_calibration_completed
    )
    try:
        result = await hass.config_entries.subentries.async_configure(
            result["flow_id"], user_input={}
        )
        try:
            await asyncio.wait_for(received.wait(), timeout=5)
        except TimeoutError:
            pytest.fail(
                "SIGNAL_CALIBRATION_COMPLETED listener was not invoked "
                "within 5s after the recalibration flow finished."
            )
    finally:
        unsub()

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    updated = entry.subentries[blind.subentry_id]
    assert updated.data[CONF_OPEN_TIME] >= 0.1
    assert updated.data[CONF_CLOSE_TIME] >= 0.1
    assert len(signals) == 1
    assert signals[0][0] == DEVICE_ID  # entity_id fell back to the device id


def _build_bare_handler(
    hass: HomeAssistant, *, existing_subentry_data: dict[str, Any]
) -> tuple[CalibrationFlowHandler, MagicMock]:
    """Build a CalibrationFlowHandler with a mocked flow, skipping the wizard.

    The two tests below are only about async_step_calibration_complete()'s
    own recalibration branch - specifically, whether it keeps or overwrites
    an existing status identity - not about the movement-timing wizard that
    normally leads up to it. Driving that wizard for real (as the tests
    above do) would additionally require feeding fake raw serial frames
    through connected_api so finish_status_frame_capture() reports a
    "primary" identity, which exercises the frame-capture module, not the
    persistence decision under test here. Constructing the handler directly
    and setting its _pending_* fields is the same direct-construction
    approach tests/test_cover.py uses for SchellenbergCover, for the same
    reason: it isolates the decision logic from the machinery that normally
    feeds it.

    self.flow only needs to support what async_step_calibration_complete()
    and _notify_calibration_completed() actually call: a real `hass` (for
    the harmless SIGNAL_CALIBRATION_COMPLETED dispatcher send),
    _get_reconfigure_subentry().data returning the existing subentry data
    under test, _get_entry() (opaque, only passed through), and
    async_update_and_abort() (mocked so the test can assert on exactly what
    data_updates it was given).
    """
    flow = MagicMock()
    flow.hass = hass
    flow._get_reconfigure_subentry.return_value = MagicMock(data=existing_subentry_data)
    handler = CalibrationFlowHandler(flow)
    handler._selected_device = {
        "id": DEVICE_ID,
        "entity_id": DEVICE_ID,
        "name": "Living Room Blind",
    }
    handler._open_time = 21.5
    handler._close_time = 19.5
    return handler, flow


async def test_recalibration_keeps_existing_status_identity_when_discovered_differs(
    hass: HomeAssistant,
) -> None:
    """Regression for MEDIUM #2: don't let a recalibration clobber a good ID.

    A recalibration run is normally about remeasuring travel times, not
    re-discovering status identity - but a freshly captured "primary" can be
    wrong (RF noise, a neighboring blind's remote, another blind sharing the
    capture window) in ways the calibration code has no way to tell apart
    from a genuine re-discovery. Before this fix, whatever the new run
    happened to discover was applied unconditionally, silently overwriting
    an already-working status identity with a possibly-wrong guess. Only a
    *different* discovered identity needs protecting against - see the
    sibling test right below for the "confirms the same one" case.
    """
    handler, flow = _build_bare_handler(
        hass,
        existing_subentry_data={
            CONF_STATUS_DEVICE_ID: DEVICE_ID,
            CONF_STATUS_ENUM: DEVICE_ENUM,
            CONF_STATUS_IDENTITY_SOURCE: STATUS_IDENTITY_SOURCE_CALIBRATION,
        },
    )
    # This run "discovers" a different identity than the one already
    # configured above - the case that used to get silently applied.
    handler._pending_status_device_id = "112233"
    handler._pending_status_enum = "99"
    handler._pending_status_identity_source = STATUS_IDENTITY_SOURCE_CALIBRATION
    handler._pending_secondary_status_identities = [
        {"device_id": "445566", "enum": "77"}
    ]

    await handler.async_step_calibration_complete(user_input={})

    flow.async_update_and_abort.assert_called_once()
    data_updates = flow.async_update_and_abort.call_args.kwargs["data_updates"]
    assert data_updates[CONF_OPEN_TIME] == 21.5
    assert data_updates[CONF_CLOSE_TIME] == 19.5
    assert CONF_STATUS_DEVICE_ID not in data_updates
    assert CONF_STATUS_ENUM not in data_updates
    assert CONF_STATUS_IDENTITY_SOURCE not in data_updates
    assert CONF_SECONDARY_STATUS_IDENTITIES not in data_updates


async def test_recalibration_applies_status_identity_when_it_matches_existing(
    hass: HomeAssistant,
) -> None:
    """Confirming the same identity again is not "overwriting" - still applies.

    Refinement of the regression test above: the protection only needs to
    guard against a *different* discovered identity replacing a known-good
    one. When this run's frame capture confirms the exact identity that was
    already configured, applying the update is harmless - it is the same
    identity either way - and still lets CONF_SECONDARY_STATUS_IDENTITIES
    and CONF_STATUS_IDENTITY_SOURCE refresh from this run's own capture,
    instead of being frozen at whatever an earlier run happened to see.
    """
    handler, flow = _build_bare_handler(
        hass,
        existing_subentry_data={
            CONF_STATUS_DEVICE_ID: DEVICE_ID,
            CONF_STATUS_ENUM: DEVICE_ENUM,
            CONF_STATUS_IDENTITY_SOURCE: STATUS_IDENTITY_SOURCE_CALIBRATION,
        },
    )
    # Same identity as already configured, just spelled with different
    # case - normalize_status_identity() must equate the two, the same way
    # every other identity comparison in this integration does, so this
    # also guards against a naive raw-string comparison protecting a
    # "different" identity that is actually the same one.
    handler._pending_status_device_id = DEVICE_ID.lower()
    handler._pending_status_enum = DEVICE_ENUM
    handler._pending_status_identity_source = STATUS_IDENTITY_SOURCE_CALIBRATION
    handler._pending_secondary_status_identities = [
        {"device_id": "445566", "enum": "77"}
    ]

    await handler.async_step_calibration_complete(user_input={})

    flow.async_update_and_abort.assert_called_once()
    data_updates = flow.async_update_and_abort.call_args.kwargs["data_updates"]
    assert data_updates[CONF_STATUS_DEVICE_ID] == DEVICE_ID.lower()
    assert data_updates[CONF_STATUS_ENUM] == DEVICE_ENUM
    assert (
        data_updates[CONF_STATUS_IDENTITY_SOURCE] == STATUS_IDENTITY_SOURCE_CALIBRATION
    )
    assert data_updates[CONF_SECONDARY_STATUS_IDENTITIES] == [
        {"device_id": "445566", "enum": "77"}
    ]


async def test_recalibration_fills_status_identity_when_previously_unknown(
    hass: HomeAssistant,
) -> None:
    """Companion to the regression above: discovery still applies when there

    was nothing to protect. Guards the `if existing_status_known` branch so
    a fix for the "already known" case can't regress into never applying a
    freshly discovered identity at all.
    """
    handler, flow = _build_bare_handler(
        hass,
        existing_subentry_data={
            CONF_STATUS_DEVICE_ID: None,
            CONF_STATUS_ENUM: None,
            CONF_STATUS_IDENTITY_SOURCE: STATUS_IDENTITY_SOURCE_UNKNOWN,
        },
    )
    handler._pending_status_device_id = "112233"
    handler._pending_status_enum = "99"
    handler._pending_status_identity_source = STATUS_IDENTITY_SOURCE_CALIBRATION
    handler._pending_secondary_status_identities = [
        {"device_id": "445566", "enum": "77"}
    ]

    await handler.async_step_calibration_complete(user_input={})

    flow.async_update_and_abort.assert_called_once()
    data_updates = flow.async_update_and_abort.call_args.kwargs["data_updates"]
    assert data_updates[CONF_STATUS_DEVICE_ID] == "112233"
    assert data_updates[CONF_STATUS_ENUM] == "99"
    assert (
        data_updates[CONF_STATUS_IDENTITY_SOURCE] == STATUS_IDENTITY_SOURCE_CALIBRATION
    )
    assert data_updates[CONF_SECONDARY_STATUS_IDENTITIES] == [
        {"device_id": "445566", "enum": "77"}
    ]


async def test_open_instruction_movement_start_timeout_shows_error_and_reshows_form(
    hass: HomeAssistant,
    connected_api: SchellenbergUsbApi,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(calibration_module, "CALIBRATION_TIMEOUT", 0.01)
    entry, blind = _build_hub_with_blind(hass, connected_api)
    result = await _start_calibrate_flow(hass, entry, blind)
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={}
    )
    assert result["step_id"] == "calibration_open_instruction"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "calibration_open_instruction"
    assert result["errors"] == {"base": "calibration_start_timeout"}


async def test_open_instruction_stop_timeout_shows_error_and_reshows_form(
    hass: HomeAssistant,
    connected_api: SchellenbergUsbApi,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(calibration_module, "CALIBRATION_TIMEOUT", 0.05)
    entry, blind = _build_hub_with_blind(hass, connected_api)
    result = await _start_calibrate_flow(hass, entry, blind)
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input={}
    )

    wait_for_registration = _spy_on_dispatcher_connect(monkeypatch)
    task = asyncio.ensure_future(
        hass.config_entries.subentries.async_configure(result["flow_id"], user_input={})
    )
    # wait_for_registration() resolves the instant _wait_for_movement_start()
    # registers its listener, however long that actually takes - not after a
    # fixed delay - so it's safe to use here too even though
    # CALIBRATION_TIMEOUT is monkeypatched to 0.05s above. What exercises
    # that shrunk timeout is the *second* wait: this test deliberately never
    # sends EVENT_STOPPED below, so _wait_for_stop_event()'s own
    # asyncio.wait_for(..., timeout=CALIBRATION_TIMEOUT) genuinely expires
    # 0.05s after it registers its own listener in turn.
    await wait_for_registration()
    async_dispatcher_send(
        hass, f"{SIGNAL_DEVICE_EVENT}_{DEVICE_ID}", EVENT_STARTED_MOVING_UP
    )
    # Deliberately never send EVENT_STOPPED - the stop wait must time out.
    result = await task

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "calibration_open_instruction"
    assert result["errors"] == {"base": "calibration_timeout"}


async def test_calibration_close_step_aborts_without_a_selected_device() -> None:
    """Defensive guard: every step must abort cleanly if never wired up."""
    flow = MagicMock()
    handler = CalibrationFlowHandler(flow)

    await handler.async_step_calibration_close(None)

    flow.async_abort.assert_called_once_with(reason="device_not_found")


async def test_calibration_complete_step_aborts_without_recorded_times() -> None:
    flow = MagicMock()
    handler = CalibrationFlowHandler(flow)
    handler.set_selected_device({"id": DEVICE_ID, "entity_id": DEVICE_ID, "name": "X"})

    await handler.async_step_calibration_complete(None)

    flow.async_abort.assert_called_once_with(reason="device_not_found")
