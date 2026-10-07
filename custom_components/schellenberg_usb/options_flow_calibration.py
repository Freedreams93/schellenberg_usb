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
    from .config_flow import SchellenbergPairingSubentryFlow

_LOGGER = logging.getLogger(__name__)

type FlowResult = SubentryFlowResult


class CalibrationFlowHandler:
    def __init__(self, flow: ConfigSubentryFlow) -> None:
        self.flow = flow
        self._selected_device: dict[str, Any] | None = None
        self._calibration_start_time: float | None = None
        self._start_event: asyncio.Event | None = None
        self._stop_event: asyncio.Event | None = None
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
        entry = self.flow._get_entry()
        api = entry.runtime_data
        return api if isinstance(api, SchellenbergUsbApi) else None

    def _start_calibration_capture(self) -> None:
        api = self._runtime_api()
        if api is None:
            return
        try:
            api.start_status_frame_capture(phase="opening")
        except RuntimeError:
            api.finish_status_frame_capture(end_reason="superseded_by_calibration")
            api.start_status_frame_capture(phase="opening")

    def _set_calibration_capture_phase(self, phase: str) -> None:
        api = self._runtime_api()
        if api is not None:
            api.set_status_frame_capture_phase(phase)

    def _finish_calibration_capture(self, end_reason: str) -> None:
        api = self._runtime_api()
        if api is None:
            return
        self._calibration_discovery_result = api.finish_status_frame_capture(
            end_reason=end_reason
        )

    def _apply_calibration_status_candidates(self) -> None:
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
        self._selected_device = device

    async def async_step_calibration_close(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        if user_input is not None:
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
        if self._selected_device is None:
            return self.flow.async_abort(reason="device_not_found")

        errors = {}

        if user_input is None:
            return self.flow.async_show_form(
                step_id="calibration_open_instruction",
                data_schema=vol.Schema({}),
                description_placeholders={
                    "device_name": self._selected_device["name"],
                },
                last_step=False,
            )

        self._start_calibration_capture()
        try:
            start_ok = await self._wait_for_movement_start(EVENT_STARTED_MOVING_UP)
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

            self._calibration_start_time = time.monotonic()

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

            self._open_time = max(time.monotonic() - self._calibration_start_time, 0.1)
            _LOGGER.debug("Calibration open_time: %s seconds", self._open_time)
            self._set_calibration_capture_phase("idle_between_legs")

            return await self.async_step_calibration_close_instruction()

        except asyncio.CancelledError:
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
        if self._selected_device is None:
            return self.flow.async_abort(reason="device_not_found")

        errors = {}

        if user_input is None:
            return self.flow.async_show_form(
                step_id="calibration_close_instruction",
                data_schema=vol.Schema({}),
                description_placeholders={
                    "device_name": self._selected_device["name"],
                },
                last_step=False,
            )

        self._set_calibration_capture_phase("closing")
        try:
            start_ok = await self._wait_for_movement_start(EVENT_STARTED_MOVING_DOWN)
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

            self._calibration_start_time = time.monotonic()

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

            self._close_time = max(time.monotonic() - self._calibration_start_time, 0.1)
            _LOGGER.debug("Calibration close_time: %s seconds", self._close_time)
            self._finish_calibration_capture("completed")
            self._apply_calibration_status_candidates()

            return await self.async_step_calibration_complete()

        except asyncio.CancelledError:
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
        if (
            self._selected_device is None
            or self._open_time is None
            or self._close_time is None
        ):
            return self.flow.async_abort(reason="device_not_found")

        if user_input is not None:
            await self._notify_calibration_completed(self._open_time, self._close_time)

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
                pairing_flow = cast("SchellenbergPairingSubentryFlow", self.flow)
                return pairing_flow._finish_pairing_or_offer_repeat(
                    title=self._pending_device_name, data=data
                )

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
        if self._selected_device is None:
            return False
        device_id = self._selected_device["id"]
        self._start_event = asyncio.Event()

        @callback
        def handle_device_event(command: str) -> None:
            if command == event_type and self._start_event:
                self.flow.hass.loop.call_soon_threadsafe(self._start_event.set)

        @callback
        def handle_capture_event(_captured_device_id: str, command: str) -> None:
            if command == event_type and self._start_event:
                self.flow.hass.loop.call_soon_threadsafe(self._start_event.set)

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
            await asyncio.wait_for(
                self._start_event.wait(), timeout=CALIBRATION_TIMEOUT
            )
        except TimeoutError:
            return False
        else:
            return True
        finally:
            for unsub in self._event_listener_unsubs:
                unsub()
            self._event_listener_unsubs = []
            self._start_event = None

    async def _wait_for_stop_event(self) -> bool:
        if self._selected_device is None:
            return False
        device_id = self._selected_device["id"]
        self._stop_event = asyncio.Event()

        @callback
        def handle_device_event(command: str) -> None:
            if command == EVENT_STOPPED and self._stop_event:
                self.flow.hass.loop.call_soon_threadsafe(self._stop_event.set)

        @callback
        def handle_capture_event(_captured_device_id: str, command: str) -> None:
            if command == EVENT_STOPPED and self._stop_event:
                self.flow.hass.loop.call_soon_threadsafe(self._stop_event.set)

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
            await asyncio.wait_for(self._stop_event.wait(), timeout=CALIBRATION_TIMEOUT)
        except TimeoutError:
            return False
        else:
            return True
        finally:
            for unsub in self._event_listener_unsubs:
                unsub()
            self._event_listener_unsubs = []
            self._stop_event = None

    async def _notify_calibration_completed(
        self, open_time: float, close_time: float
    ) -> None:
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
