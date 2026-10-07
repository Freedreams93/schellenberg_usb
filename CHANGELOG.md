## Unreleased

- Keep the v1.1.0 calibration flow and its newer status-frame discovery,
  dispatcher handling, monotonic timing, and subentry persistence.
- Fix calibration direction so it always measures physical Open first and
  physical Close second; `invert_direction` no longer reverses the calibration
  legs.
- Update all four translations and the English source strings: the pairing
  calibration option is labelled as device pairing/calibration, optional
Pairing instructions no longer depend on an additional motor signal.
  not told to wait for such a sound.
- Update **Motor anlernen / USB-Sender aktivieren** and its translations with
  the same correction.
- Update the README to document the corrected calibration sequence and the two
  pairing/calibration options.

"""Calibration options flow handlers for Schellenberg USB."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, cast

import voluptuous as vol
from homeassistant.config_entries import (
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.core import callback
from homeassistant.helpers.dispatcher import (
    # Re-exported (not just imported) on purpose: tests/test_options_flow_
    # calibration.py's _spy_on_dispatcher_connect() reaches into this module
    # as calibration_module.async_dispatcher_connect to monkeypatch the same
    # name this module calls below, so mypy's strict implicit-reexport check
    # needs to see this as a deliberate public re-export, not a private
    # import.
    async_dispatcher_connect as async_dispatcher_connect,
)
from homeassistant.helpers.dispatcher import (
    async_dispatcher_send,
)

from .api import SchellenbergUsbApi
from .blind_id import generate_blind_id, normalize_blind_id
from .const import (
    CALIBRATION_TIMEOUT,
    CONF_BLIND_ID,
    CONF_CLOSE_TIME,
    CONF_COMMAND_DEVICE_ID,
    CONF_COMMAND_ENUM,
    CONF_DEVICE_ENUM,
    CONF_DEVICE_ID,
    CONF_INVERT_DIRECTION,
    CONF_LAST_CALIBRATION,
    CONF_OPEN_TIME,
    CONF_SECONDARY_STATUS_IDENTITIES,
    CONF_STATUS_DEVICE_ID,
    CONF_STATUS_ENUM,
    CONF_STATUS_IDENTITY_SOURCE,
    EVENT_STARTED_MOVING_DOWN,
    EVENT_STARTED_MOVING_UP,
    EVENT_STOPPED,
    SIGNAL_CALIBRATION_COMPLETED,
    SIGNAL_DEVICE_EVENT,
    SIGNAL_DEVICE_EVENT_CAPTURE,
    STATUS_IDENTITY_SOURCE_CALIBRATION,
    STATUS_IDENTITY_SOURCE_UNKNOWN,
)
from .identities import normalize_status_identity
from .runtime_translation_text import (
    runtime_translation_text as _runtime_translation_text,
)
from .runtime_translation_text import translate_phrase as _translate_phrase

if TYPE_CHECKING:
    # Import cycle guard: config_flow.py imports CalibrationFlowHandler from
    # this module, so this back-reference is type-checking only, never
    # executed at import time. Used solely to cast self.flow below for
    # add_shutter_from_remote's repeat-pairing attributes, which live on the
    # concrete subentry flow class, not on the generic ConfigSubentryFlow
    # this handler is otherwise deliberately typed against (see the class
    # docstring).
    from .config_flow import SchellenbergPairingSubentryFlow

_LOGGER = logging.getLogger(__name__)

# Type alias for the subentry flow step results this handler returns.
type FlowResult = SubentryFlowResult


class CalibrationFlowHandler:
    """Handle calibration steps shared by the blind config-subentry flow.

    This is only ever constructed by SchellenbergPairingSubentryFlow
    (config_flow.py), never by SchellenbergOptionsFlowHandler
    (options_flow.py) - hub options only edit the serial port and do not
    calibrate. `flow` is therefore always a ConfigSubentryFlow, not a plain
    OptionsFlow, and code here can rely on ConfigSubentryFlow-only methods
    such as `_get_entry()`/`_get_reconfigure_subentry()`.
    """

    def __init__(self, flow: ConfigSubentryFlow) -> None:
        """Initialize the calibration run's timing and pending-data state.

        `_selected_device` through `_close_time` track the in-progress
        timed run itself (which device, when it started, the
        start/stop dispatcher events being awaited, the measured
        times). The `_pending_*` attributes mirror the owning pairing
        flow's own pending blind data so `async_step_calibration_complete`
        can build or update a subentry without reaching back into that
        flow for it.
        """
        self.flow = flow
        self._selected_device: dict[str, Any] | None = None
        self._calibration_start_time: float | None = None
        self._start_event: asyncio.Event | None = None
        self._stop_event: asyncio.Event | None = None
        # A list because _wait_for_movement_start()/_wait_for_stop_event()
        # each subscribe to two signals now (the device-id-specific one and
        # the capture-wide fallback) and must unsubscribe both.
        self._event_listener_unsubs: list[Any] = []
        self._open_time: float | None = None
        self._close_time: float | None = None
        self._create_subentry_after_calibration = False
        self._pending_blind_id: str | None = None
        self._pending_device_id: str | None = None
        self._pending_device_enum: str | None = None
        self._pending_device_name: str | None = None
        self._pending_status_device_id: str | None = None
        self._pending_status_enum: str | None = None
        self._pending_secondary_status_identities: list[dict[str, str]] = []
        self._pending_status_identity_source: str | None = None
        self._calibration_discovery_result: dict[str, Any] | None = None
        self._pending_invert_direction = False

    def _runtime_api(self) -> SchellenbergUsbApi | None:
        """Return the loaded hub API for this subentry flow's parent entry."""
        entry = self.flow._get_entry()
        api = entry.runtime_data
        return api if isinstance(api, SchellenbergUsbApi) else None

    def _start_calibration_capture(self) -> None:
        """Begin phase-labelled raw-frame capture for this calibration run."""
        api = self._runtime_api()
        if api is None:
            return
        try:
            api.start_status_frame_capture(phase="opening")
        except RuntimeError:
            # A stale capture should not break calibration; close it explicitly and
            # start the calibration-owned window.
            api.finish_status_frame_capture(end_reason="superseded_by_calibration")
            api.start_status_frame_capture(phase="opening")

    def _set_calibration_capture_phase(self, phase: str) -> None:
        """Label future frames with the current calibration leg."""
        api = self._runtime_api()
        if api is not None:
            api.set_status_frame_capture_phase(phase)

    def _finish_calibration_capture(self, end_reason: str) -> None:
        """Finish capture and retain candidates for persistence and summary."""
        api = self._runtime_api()
        if api is None:
            return
        self._calibration_discovery_result = api.finish_status_frame_capture(
            end_reason=end_reason
        )

    def _apply_calibration_status_candidates(self) -> None:
        """Apply captured identities to pending subentry fields without guessing."""
        result = self._calibration_discovery_result
        if not result:
            self._pending_status_device_id = None
            self._pending_status_enum = None
            self._pending_status_identity_source = STATUS_IDENTITY_SOURCE_UNKNOWN
            return
        primary = result.get("primary")
        if primary is None:
            self._pending_status_device_id = None
            self._pending_status_enum = None
            self._pending_status_identity_source = STATUS_IDENTITY_SOURCE_UNKNOWN
        else:
            self._pending_status_device_id = str(primary["device_id"])
            self._pending_status_enum = str(primary["enum"])
            self._pending_status_identity_source = STATUS_IDENTITY_SOURCE_CALIBRATION
        self._pending_secondary_status_identities = [
            {"device_id": str(group["device_id"]), "enum": str(group["enum"])}
            for group in result.get("secondary", [])
        ]

    def _calibration_record(self) -> dict[str, Any] | None:
        """Return JSON-compatible diagnostics for the completed calibration run."""
        if self._calibration_discovery_result is None:
            return None
        return {
            **self._calibration_discovery_result,
            "open_time": (
                round(self._open_time, 2) if self._open_time is not None else None
            ),
            "close_time": (
                round(self._close_time, 2) if self._close_time is not None else None
            ),
        }

    def _calibration_summary_placeholders(self) -> dict[str, str]:
        """Build a user-facing summary of measured times and received streams."""
        hass = self.flow.hass
        none_word = _runtime_translation_text(hass, "none_word")
        result = self._calibration_discovery_result or {}
        primary = result.get("primary")
        secondary = result.get("secondary", [])
        primary_text = (
            f"{primary['device_id']}/{primary['enum']}"
            if primary is not None
            else _runtime_translation_text(hass, "not_discovered")
        )
        primary_frames = (
            ", ".join(primary.get("commands", [])) if primary is not None else none_word
        )
        secondary_text = (
            ", ".join(
                f"{group['device_id']}/{group['enum']} "
                f"({','.join(group.get('commands', []))})"
                for group in secondary
            )
            or none_word
        )
        return {
            "primary_status_identity": primary_text,
            "primary_frames": primary_frames,
            "secondary_status_identities": secondary_text,
            "position_tracking": (
                _runtime_translation_text(hass, "position_tracking_available")
                if primary is not None
                else _runtime_translation_text(hass, "position_tracking_unavailable")
            ),
            "calibration_end_reason": _translate_phrase(
                hass, str(result.get("end_reason", "completed"))
            ),
            "observed_frame_count": str(len(result.get("frames", []))),
        }

    def set_selected_device(self, device: dict[str, Any]) -> None:
        """Public setter to assign selected device without storage lookup."""
        self._selected_device = device

    async def async_step_calibration_close(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Instruct user to close the blinds and press next."""
        if user_input is not None:
            # User has closed the blinds and is ready to proceed
            return await self.async_step_calibration_open_instruction()

        if self._selected_device is None:
            return self.flow.async_abort(reason="device_not_found")

        return self.flow.async_show_form(
            step_id="calibration_close",
            data_schema=vol.Schema({}),
            description_placeholders={
                "device_name": self._selected_device["name"],
            },
            last_step=False,
        )

    async def async_step_calibration_open_instruction(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Instruct user to open the blinds and wait for movement."""
        if self._selected_device is None:
            return self.flow.async_abort(reason="device_not_found")

        errors = {}

        # Show instruction form first time
        if user_input is None:
            return self.flow.async_show_form(
                step_id="calibration_open_instruction",
                data_schema=vol.Schema({}),
                description_placeholders={
                    "device_name": self._selected_device["name"],
                },
                last_step=False,
            )

        # User clicked Next - wait for movement start and measure timing
        self._start_calibration_capture()
        try:
            # Calibration always measures the physical open direction first.
            # Direction inversion applies to normal operation, not calibration.
            open_event = EVENT_STARTED_MOVING_UP
            start_ok = await self._wait_for_movement_start(open_event)
            if not start_ok:
                self._finish_calibration_capture("opening_start_timeout")
                errors["base"] = "calibration_start_timeout"
                return self.flow.async_show_form(
                    step_id="calibration_open_instruction",
                    data_schema=vol.Schema({}),
                    description_placeholders={
                        "device_name": self._selected_device["name"],
                    },
                    errors=errors,
                    last_step=False,
                )

            # Start timing the open movement. Monotonic, not wall-clock: an
            # NTP correction or manual clock change during the up-to-5-minute
            # calibration window must not corrupt the measured travel time.
            self._calibration_start_time = time.monotonic()

            # Wait for device to stop moving
            stop_ok = await self._wait_for_stop_event()
            if not stop_ok:
                self._finish_calibration_capture("opening_stop_timeout")
                errors["base"] = "calibration_timeout"
                return self.flow.async_show_form(
                    step_id="calibration_open_instruction",
                    data_schema=vol.Schema({}),
                    description_placeholders={
                        "device_name": self._selected_device["name"],
                    },
                    errors=errors,
                    last_step=False,
                )

            # Record the open time. Floor it well above zero: on a very fast
            # motor (or if the start/stop events land in the same clock
            # tick) this could otherwise measure as 0.0, which would later
            # divide-by-zero when cover.py uses it to compute position
            # during movement.
            self._open_time = max(time.monotonic() - self._calibration_start_time, 0.1)
            _LOGGER.debug("Calibration open_time: %s seconds", self._open_time)
            self._set_calibration_capture_phase("idle_between_legs")

            # Move to close instruction step
            return await self.async_step_calibration_close_instruction()

        except asyncio.CancelledError:
            # The flow itself was cancelled (e.g. HA shutting down, or the
            # user abandoning the wizard) while awaiting movement/stop
            # events. Still close out the capture window before propagating
            # the cancellation, otherwise it is left "active" forever and
            # the next status-discovery attempt fails with a busy error
            # until something else happens to supersede it.
            self._finish_calibration_capture("opening_cancelled")
            raise
        except Exception:
            self._finish_calibration_capture("opening_error")
            errors["base"] = "unknown"
            return self.flow.async_show_form(
                step_id="calibration_open_instruction",
                data_schema=vol.Schema({}),
                description_placeholders={
                    "device_name": self._selected_device["name"],
                },
                errors=errors,
                last_step=False,
            )

    async def async_step_calibration_close_instruction(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Instruct user to close the blinds and wait for movement."""
        if self._selected_device is None:
            return self.flow.async_abort(reason="device_not_found")

        errors = {}

        # Show instruction form first time
        if user_input is None:
            return self.flow.async_show_form(
                step_id="calibration_close_instruction",
                data_schema=vol.Schema({}),
                description_placeholders={
                    "device_name": self._selected_device["name"],
                },
                last_step=False,
            )

        # User clicked Next - wait for movement start and measure timing
        self._set_calibration_capture_phase("closing")
        try:
            # Calibration always measures the physical close direction second.
            # Direction inversion applies to normal operation, not calibration.
            close_event = EVENT_STARTED_MOVING_DOWN
            start_ok = await self._wait_for_movement_start(close_event)
            if not start_ok:
                self._finish_calibration_capture("closing_start_timeout")
                errors["base"] = "calibration_start_timeout"
                return self.flow.async_show_form(
                    step_id="calibration_close_instruction",
                    data_schema=vol.Schema({}),
                    description_placeholders={
                        "device_name": self._selected_device["name"],
                    },
                    errors=errors,
                    last_step=False,
                )

            # Start timing the close movement (see the open-instruction step
            # for why this uses monotonic rather than wall-clock time).
            self._calibration_start_time = time.monotonic()

            # Wait for device to stop moving
            stop_ok = await self._wait_for_stop_event()
            if not stop_ok:
                self._finish_calibration_capture("closing_stop_timeout")
                errors["base"] = "calibration_timeout"
                return self.flow.async_show_form(
                    step_id="calibration_close_instruction",
                    data_schema=vol.Schema({}),
                    description_placeholders={
                        "device_name": self._selected_device["name"],
                    },
                    errors=errors,
                    last_step=False,
                )

            # Record the close time (see the open_time floor above for why).
            self._close_time = max(time.monotonic() - self._calibration_start_time, 0.1)
            _LOGGER.debug("Calibration close_time: %s seconds", self._close_time)
            self._finish_calibration_capture("completed")
            self._apply_calibration_status_candidates()

            # Move to completion step
            return await self.async_step_calibration_complete()

        except asyncio.CancelledError:
            # See the matching comment in the open-instruction step: always
            # close out the capture window before the cancellation
            # propagates.
            self._finish_calibration_capture("closing_cancelled")
            raise
        except Exception:
            self._finish_calibration_capture("closing_error")
            errors["base"] = "unknown"
            return self.flow.async_show_form(
                step_id="calibration_close_instruction",
                data_schema=vol.Schema({}),
                description_placeholders={
                    "device_name": self._selected_device["name"],
                },
                errors=errors,
                last_step=False,
            )

    async def async_step_calibration_complete(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Confirm the recorded travel times, then persist or offer a repeat.

        With no `user_input` yet, shows a confirmation form summarizing
        the just-recorded open/close times. Once confirmed, notifies
        the live cover entity of the new times and either finishes a
        pairing flow - creating the new blind subentry, or letting it
        offer one more "pair another blind" round instead, see
        `_finish_pairing_or_offer_repeat` - or updates an already
        existing subentry's recorded times for a plain recalibration.
        A recalibration that re-discovers a *different* status identity
        than the one already confirmed discards the new guess rather
        than overwriting a working identity with what could just be RF
        noise or a neighboring remote; see the comment further below
        for why.
        """
        if (
            self._selected_device is None
            or self._open_time is None
            or self._close_time is None
        ):
            return self.flow.async_abort(reason="device_not_found")

        if user_input is not None:
            # User confirmed completion - notify the live entity. The actual
            # open/close times are persisted below via async_create_entry /
            # async_update_and_abort (the config subentry, not ad-hoc storage).
            await self._notify_calibration_completed(self._open_time, self._close_time)

            # If pairing flow requested creation after calibration,
            # create subentry entry now.
            if (
                self._create_subentry_after_calibration
                and self._pending_device_id
                and self._pending_device_enum
                and self._pending_device_name
            ):
                data: dict[str, Any] = {
                    CONF_BLIND_ID: self._pending_blind_id or generate_blind_id(),
                    CONF_DEVICE_ID: self._pending_device_id,
                    CONF_DEVICE_ENUM: self._pending_device_enum,
                    CONF_COMMAND_DEVICE_ID: self._pending_device_id,
                    CONF_COMMAND_ENUM: self._pending_device_enum,
                    CONF_SECONDARY_STATUS_IDENTITIES: list(
                        self._pending_secondary_status_identities
                    ),
                    CONF_OPEN_TIME: round(self._open_time, 2),
                    CONF_CLOSE_TIME: round(self._close_time, 2),
                    CONF_INVERT_DIRECTION: self._pending_invert_direction,
                }
                if (
                    self._pending_status_device_id is not None
                    and self._pending_status_enum is not None
                ):
                    data[CONF_STATUS_DEVICE_ID] = self._pending_status_device_id
                    data[CONF_STATUS_ENUM] = self._pending_status_enum
                if self._pending_status_identity_source is not None:
                    data[CONF_STATUS_IDENTITY_SOURCE] = (
                        self._pending_status_identity_source
                    )
                if calibration_record := self._calibration_record():
                    data[CONF_LAST_CALIBRATION] = calibration_record
                # Not a plain async_create_entry():
                # add_shutter_from_remote's "pair another blind on this
                # channel" option (see
                # _finish_pairing_or_offer_repeat() on the flow) can offer
                # one more round instead of creating the subentry here. The
                # cast is needed because _finish_pairing_or_offer_repeat()
                # lives on the concrete pairing flow, not on the generic
                # ConfigSubentryFlow this handler is typed against (see the
                # class docstring) - it is still always that concrete flow
                # at runtime, only ever constructed by it.
                pairing_flow = cast("SchellenbergPairingSubentryFlow", self.flow)
                return pairing_flow._finish_pairing_or_offer_repeat(
                    title=self._pending_device_name, data=data
                )

            # Otherwise this is a recalibration of an already-existing blind
            # subentry (self.flow is always a ConfigSubentryFlow; see the
            # class docstring).
            existing_data = self.flow._get_reconfigure_subentry().data
            existing_status_identity = normalize_status_identity(
                existing_data.get(CONF_STATUS_DEVICE_ID),
                existing_data.get(CONF_STATUS_ENUM),
            )
            existing_status_confirmed = (
                existing_data.get(CONF_STATUS_IDENTITY_SOURCE)
                != STATUS_IDENTITY_SOURCE_UNKNOWN
            )
            data_updates: dict[str, Any] = {
                CONF_OPEN_TIME: round(self._open_time, 2),
                CONF_CLOSE_TIME: round(self._close_time, 2),
            }
            if calibration_record := self._calibration_record():
                data_updates[CONF_LAST_CALIBRATION] = calibration_record
            if (
                self._pending_status_device_id is not None
                and self._pending_status_enum is not None
                and self._pending_status_identity_source
                == STATUS_IDENTITY_SOURCE_CALIBRATION
            ):
                discovered_status_identity = normalize_status_identity(
                    self._pending_status_device_id, self._pending_status_enum
                )
                # A recalibration run here is normally just about remeasuring
                # travel times, not re-discovering status identity - but a
                # freshly captured "primary" can be wrong in ways this code
                # cannot tell apart from a genuine re-discovery: RF noise, a
                # neighboring blind's remote, or another blind sharing the
                # capture window. When this run confirms the *same* identity
                # that was already configured, applying it is harmless (and
                # still refreshes secondary identities/source below); it is
                # only a *different* discovered identity, while a known-good
                # one is already configured, that gets silently discarded
                # instead of overwriting a working identity with a possibly
                # wrong guess. Someone who deliberately wants to replace an
                # existing identity with a genuinely different one can still
                # clear it via the manual edit step first, which goes through
                # this same STATUS_IDENTITY_SOURCE_UNKNOWN gate on the way
                # back in.
                if (
                    existing_status_identity is not None
                    and existing_status_confirmed
                    and discovered_status_identity != existing_status_identity
                ):
                    _LOGGER.info(
                        "Recalibration for %s discovered status identity %s/%s, "
                        "which differs from the already-configured %s/%s; "
                        "keeping the existing one and discarding the newly "
                        "discovered guess",
                        self._selected_device["name"]
                        if self._selected_device is not None
                        else "unknown device",
                        self._pending_status_device_id,
                        self._pending_status_enum,
                        existing_status_identity[0],
                        existing_status_identity[1],
                    )
                else:
                    data_updates.update(
                        {
                            CONF_STATUS_DEVICE_ID: self._pending_status_device_id,
                            CONF_STATUS_ENUM: self._pending_status_enum,
                            CONF_STATUS_IDENTITY_SOURCE: (
                                STATUS_IDENTITY_SOURCE_CALIBRATION
                            ),
                            CONF_SECONDARY_STATUS_IDENTITIES: list(
                                self._pending_secondary_status_identities
                            ),
                        }
                    )
            return self.flow.async_update_and_abort(
                self.flow._get_entry(),
                self.flow._get_reconfigure_subentry(),
                data_updates=data_updates,
            )

        return self.flow.async_show_form(
            step_id="calibration_complete",
            data_schema=vol.Schema({}),
            description_placeholders={
                "device_name": self._selected_device["name"],
                "open_time": f"{self._open_time:.2f}",
                "close_time": f"{self._close_time:.2f}",
                **self._calibration_summary_placeholders(),
            },
            last_step=True,
        )

    async def _wait_for_movement_start(self, event_type: str) -> bool:
        """Wait for the device to start moving.

        Args:
            event_type: The event type to wait for
                (EVENT_STARTED_MOVING_UP or EVENT_STARTED_MOVING_DOWN)

        Returns:
            True if movement start event received, False if timeout.
        """
        if self._selected_device is None:
            return False
        device_id = self._selected_device["id"]
        self._start_event = asyncio.Event()

        # Set up listener for movement start events. @callback is kept here
        # as correct HA style for a fast, non-blocking listener, but it does
        # NOT guarantee this runs on the event loop thread: a real reproduced
        # crash (thread-based pytest-timeout dump, not a theory) caught
        # Home Assistant's own dispatcher routing this exact @callback
        # target through hass.loop.run_in_executor(...) regardless -
        # homeassistant/core.py's _async_add_hass_job, at least on the
        # pinned HA version this integration tests against, does not treat
        # @callback as a hard guarantee of inline execution the way this
        # module previously assumed. So handle_device_event below CAN run on
        # a worker thread, and self._start_event.set() is therefore reaching
        # asyncio.Event.set() -> Future.set_result() -> loop.call_soon() from
        # off the loop thread. call_soon() is not thread-safe and raises
        # "Non-thread-safe operation invoked on an event loop other than the
        # current one" when that happens - which Home Assistant's background
        # job runner only logs ("Future exception was never retrieved"),
        # never surfaces to this method's own asyncio.wait_for() below. The
        # event is therefore never actually set, and the wait below runs all
        # the way to its own timeout every time, silently. This is also why
        # an earlier version of this method used
        # hass.loop.call_soon_threadsafe() here and a prior fix in this same
        # file's history replaced it with a plain .set() call, reasoning that
        # the *sender* (_handle_message) always runs on the loop thread; that
        # reasoning never accounted for the *listener* potentially running
        # off it regardless of its own decoration, which is exactly what
        # happens here. call_soon_threadsafe() is the only call that may
        # safely schedule work on the loop from any thread, so it is
        # restored below - now for the right, verified reason instead of
        # being removed for an incomplete one.
        @callback
        def handle_device_event(command: str) -> None:
            """Set `_start_event` once the awaited movement command arrives.

            Scheduled via `call_soon_threadsafe` rather than called
            directly, since this listener may run off the event loop
            thread - see the comment above this method for why.
            """
            if command == event_type and self._start_event:
                self.flow.hass.loop.call_soon_threadsafe(self._start_event.set)

        @callback
        def handle_capture_event(_captured_device_id: str, command: str) -> None:
            """Fallback for a status identity that differs from `device_id`.

            SIGNAL_DEVICE_EVENT above only fires for frames carrying
            exactly `device_id` (the command/pairing identity). The
            device actually being calibrated may broadcast its status
            frames under a different identity (see
            SIGNAL_DEVICE_EVENT_CAPTURE's comment in const.py) - without
            this second listener, such a device would never satisfy the
            wait below and calibration would always time out, even
            though the motor really did move. `_captured_device_id` is
            unused: a status-frame capture window is only ever open for
            one calibration/discovery run on this hub at a time, so any
            frame arriving here is assumed to belong to it.
            """
            if command == event_type and self._start_event:
                self.flow.hass.loop.call_soon_threadsafe(self._start_event.set)

        # Subscribe to device events: the exact device-id-specific signal,
        # plus the capture-wide fallback for a differing status identity.
        self._event_listener_unsubs = [
            async_dispatcher_connect(
                self.flow.hass,
                f"{SIGNAL_DEVICE_EVENT}_{device_id}",
                handle_device_event,
            ),
            async_dispatcher_connect(
                self.flow.hass,
                SIGNAL_DEVICE_EVENT_CAPTURE,
                handle_capture_event,
            ),
        ]

        try:
            # Wait for movement start event with timeout
            await asyncio.wait_for(
                self._start_event.wait(), timeout=CALIBRATION_TIMEOUT
            )
        except TimeoutError:
            return False
        else:
            return True
        finally:
            # Clean up both listeners
            for unsub in self._event_listener_unsubs:
                unsub()
            self._event_listener_unsubs = []
            self._start_event = None

    async def _wait_for_stop_event(self) -> bool:
        """Wait for the device to send a stop event.

        Returns:
            True if stop event received, False if timeout.
        """
        if self._selected_device is None:
            return False
        device_id = self._selected_device["id"]
        self._stop_event = asyncio.Event()

        # Set up listener for stop events. See _wait_for_movement_start above
        # for the full explanation, verified against a real reproduced
        # crash: Home Assistant's dispatcher can run this @callback target on
        # a worker thread regardless of the decoration, so the .set() call
        # must be scheduled back onto the loop thread via
        # call_soon_threadsafe() rather than called directly.
        @callback
        def handle_device_event(command: str) -> None:
            """Set `_stop_event` once the awaited stop command arrives.

            Scheduled via `call_soon_threadsafe` rather than called
            directly, since this listener may run off the event loop
            thread - see the comment above this method for why.
            """
            if command == EVENT_STOPPED and self._stop_event:
                self.flow.hass.loop.call_soon_threadsafe(self._stop_event.set)

        @callback
        def handle_capture_event(_captured_device_id: str, command: str) -> None:
            """Fallback for a status identity that differs from `device_id`.

            See the matching comment in _wait_for_movement_start() above -
            the same differing-identity gap applies to the stop event.
            """
            if command == EVENT_STOPPED and self._stop_event:
                self.flow.hass.loop.call_soon_threadsafe(self._stop_event.set)

        # Subscribe to device events: the exact device-id-specific signal,
        # plus the capture-wide fallback for a differing status identity.
        self._event_listener_unsubs = [
            async_dispatcher_connect(
                self.flow.hass,
                f"{SIGNAL_DEVICE_EVENT}_{device_id}",
                handle_device_event,
            ),
            async_dispatcher_connect(
                self.flow.hass,
                SIGNAL_DEVICE_EVENT_CAPTURE,
                handle_capture_event,
            ),
        ]

        try:
            # Wait for stop event with timeout
            await asyncio.wait_for(self._stop_event.wait(), timeout=CALIBRATION_TIMEOUT)
        except TimeoutError:
            return False
        else:
            return True
        finally:
            # Clean up both listeners
            for unsub in self._event_listener_unsubs:
                unsub()
            self._event_listener_unsubs = []
            self._stop_event = None

    async def _notify_calibration_completed(
        self, open_time: float, close_time: float
    ) -> None:
        """Notify the live cover entity that new calibration times are available.

        The open/close times themselves are persisted as part of the config
        subentry (see async_create_entry / async_update_and_abort in
        async_step_calibration_complete); this just signals the already-running
        cover entity so it can pick up the new times without waiting for a reload.
        """
        if self._selected_device is not None:
            async_dispatcher_send(
                self.flow.hass,
                SIGNAL_CALIBRATION_COMPLETED,
                self._selected_device.get("entity_id", self._selected_device["id"]),
                round(open_time, 2),
                round(close_time, 2),
            )

    def enable_subentry_creation(
        self,
        *,
        blind_id: str | None = None,
        device_id: str,
        device_enum: str,
        device_name: str,
        status_device_id: str | None = None,
        status_enum: str | None = None,
        secondary_status_identities: list[dict[str, str]] | None = None,
        status_identity_source: str | None = None,
        invert_direction: bool = False,
    ) -> None:
        """Enable creating a subentry after calibration completes."""
        self._create_subentry_after_calibration = True
        self._pending_blind_id = normalize_blind_id(blind_id) or generate_blind_id()
        self._pending_device_id = device_id
        self._pending_device_enum = device_enum
        self._pending_device_name = device_name
        self._pending_status_device_id = status_device_id
        self._pending_status_enum = status_enum
        self._pending_status_identity_source = status_identity_source
        self._pending_secondary_status_identities = list(
            secondary_status_identities or []
        )
        self._pending_invert_direction = invert_direction

    def disable_subentry_creation(self) -> None:
        """Disable subentry creation (used for reconfigure flows)."""
        self._create_subentry_after_calibration = False
        self._pending_blind_id = None
        self._pending_device_id = None
        self._pending_device_enum = None
        self._pending_device_name = None
        self._pending_status_device_id = None
        self._pending_status_enum = None
        self._pending_secondary_status_identities = []
        self._pending_status_identity_source = None
        self._calibration_discovery_result = None
        self._pending_invert_direction = False
