"""Shared fixtures for the Schellenberg USB test suite.

These tests exercise the integration's core logic (protocol handling, the
transmit lock/retry/priority-stop bypass, device-registry compatibility
helpers, and the cover entity's position/stop logic) without opening a real
serial port: SchellenbergUsbApi is wired to a `_FakeTransport` instead of a
real `serialx` connection, which is what `connect()` would otherwise require
a live USB stick for.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from typing import Any, cast

import pytest

from custom_components.schellenberg_usb.api import SchellenbergUsbApi

pytest_plugins = "pytest_homeassistant_custom_component"


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(
    enable_custom_integrations: None,
) -> Generator[None]:
    """Make custom_components/schellenberg_usb loadable by every test."""
    yield


class FakeTransport:
    """Minimal stand-in for asyncio.Transport that just records writes.

    SchellenbergUsbApi only ever calls write()/is_closing()/close() on the
    transport; a real serial transport (or a bigger mock of asyncio's
    Transport interface) is unnecessary for exercising the API's own logic.
    """

    def __init__(self) -> None:
        self.written: list[bytes] = []
        self._closing = False

    def write(self, data: bytes) -> None:
        self.written.append(data)

    def is_closing(self) -> bool:
        return self._closing

    def close(self) -> None:
        self._closing = True


def make_connected_api(
    hass: Any, port: str = "/dev/fake-schellenberg"
) -> SchellenbergUsbApi:
    """Build a SchellenbergUsbApi already wired to a ready FakeTransport.

    Bypasses connect()/serialx entirely by setting the private
    connection-state attributes directly, exactly as a real connect() would
    have left them after a successful handshake (connected, in listening
    mode, transport ready to accept writes).
    """
    api = SchellenbergUsbApi(hass, port)
    # api._transport is typed as asyncio.Transport | None in production
    # code; FakeTransport deliberately only duck-types that interface
    # rather than subclassing it, so the assignment needs an explicit cast.
    api._transport = cast(Any, FakeTransport())
    api._is_connected = True
    api._device_mode = "listening"
    return api


def written(api: SchellenbergUsbApi) -> list[bytes]:
    """Type-safe access to a test API's recorded writes.

    api._transport is typed as asyncio.Transport | None in production code,
    so reading .written straight off it is a mypy union-attr error even
    though every fixture here actually wires it to a FakeTransport (see
    make_connected_api above). This narrows that once for every call site
    instead of a type: ignore comment at each assertion.
    """
    assert isinstance(api._transport, FakeTransport)
    return api._transport.written


@pytest.fixture
def connected_api(hass: Any) -> SchellenbergUsbApi:
    """A SchellenbergUsbApi ready to transmit, backed by a FakeTransport."""
    return make_connected_api(hass)


@pytest.fixture
def connected_api_factory(
    hass: Any,
) -> Callable[..., SchellenbergUsbApi]:
    """A factory for building extra ready-to-transmit API instances.

    Tests that need more than one independent SchellenbergUsbApi (e.g. one
    loaded config entry plus one that was never set up) use this instead of
    the single connected_api fixture.
    """

    def _factory(port: str = "/dev/fake-schellenberg") -> SchellenbergUsbApi:
        return make_connected_api(hass, port)

    return _factory
