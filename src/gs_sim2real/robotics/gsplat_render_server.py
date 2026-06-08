"""Headless splat render server for Dynamic Map Viewer sim2real topics.

This module loads a trained PLY/3DGS point cloud, projects it from a supplied
camera pose, and publishes RGB + depth frames to the standard Dynamic Map Viewer ROS2
topics. The first version uses a deterministic point-splat rasterizer so it can
run even when the optional ``gsplat`` package is unavailable.
"""

from __future__ import annotations

import argparse
import io
import importlib.util
import json
import math
import sys
import threading
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
from PIL import Image as PILImage

from gs_sim2real.core.render_backend_selection import (
    RenderBackendCapabilities,
    RenderBackendPreferences,
    RenderBackendRequest,
    RenderBackendSelection,
    select_render_backend,
)
from gs_sim2real.core.query_transport_selection import (
    DEFAULT_QUERY_TRANSPORT_ENDPOINTS,
    QueryTransportCapabilities,
    QueryTransportPreferences,
    QueryTransportRequest,
    QueryTransportSelection,
    resolve_query_transport_endpoint,
    select_query_transport,
)
from gs_sim2real.core.query_request_import import (
    LocalizationImageBenchmarkQuerySpec,
    QueryRequestImportRequest,
    RenderQueryDefaults,
    RenderQuerySpec,
    import_query_request,
)
from gs_sim2real.core.query_source_identity import (
    QuerySourceIdentityRequest,
    resolve_query_source_identity,
)
from gs_sim2real.core.query_error_mapping import (
    QueryErrorMappingRequest,
    resolve_query_error_mapping,
)
from gs_sim2real.core.query_cancellation_policy import (
    QueryCancellationRequest,
    resolve_query_cancellation_decision,
)
from gs_sim2real.core.query_coalescing_policy import (
    QueryCoalescingRequest,
    resolve_query_coalescing_decision,
)
from gs_sim2real.core.query_queue_policy import (
    QueryQueueState,
    build_queued_query_item_from_payload,
    admit_query_queue_item,
    dispatch_query_queue_item,
)
from gs_sim2real.core.query_timeout_policy import (
    build_query_timeout_policy_request,
    resolve_query_timeout_plan,
)
from gs_sim2real.core.query_response_build import (
    QUERY_RESPONSE_PROTOCOL_ID,
    QueryErrorResponseInput,
    QueryReadyDefaults,
    QueryReadyResponseInput,
    QueryRenderResultResponseInput,
    build_query_error_response_document,
    build_query_ready_response_document,
    build_render_result_response_document,
)
from gs_sim2real.viewer.web_viewer import load_ply

from .topic_map import build_ros_topic_map

sim2real_query_protocol_id = QUERY_RESPONSE_PROTOCOL_ID
default_query_endpoints = DEFAULT_QUERY_TRANSPORT_ENDPOINTS


@dataclass(frozen=True)
class CameraPose:
    """World-space camera pose."""

    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]


@dataclass(frozen=True)
class RenderFrameBundle:
    """Rendered RGB/depth frame plus the settings used to produce it."""

    pose: CameraPose
    width: int
    height: int
    fov_degrees: float
    near_clip: float
    far_clip: float
    point_radius: int
    rgb: np.ndarray
    depth: np.ndarray
    rgb_jpeg: bytes


@dataclass
class PendingRenderQuery:
    """Queued render query request waiting for the node thread."""

    payload: Any
    response_ready: threading.Event
    request_id: str = ""
    source_id: str = ""
    response: dict[str, Any] | None = None


@dataclass
class PendingRenderQueryEntry:
    """Queue item paired with its live pending render query state."""

    queue_item: Any
    pending: PendingRenderQuery


class PendingRenderQueryStore:
    """Policy-aware in-memory pending-query store for interactive transports."""

    def __init__(
        self,
        *,
        transport: str,
        queue_policy: str = "interactive_first",
        cancellation_policy: str = "cancel_source_backlog",
        coalescing_policy: str = "latest_render_per_source",
        max_pending: int = 4,
    ) -> None:
        self.transport = str(transport)
        self.queue_policy = str(queue_policy)
        self.cancellation_policy = str(cancellation_policy)
        self.coalescing_policy = str(coalescing_policy)
        self.max_pending = max(1, int(max_pending))
        self._entries: list[PendingRenderQueryEntry] = []
        self._lock = threading.Lock()
        self._counter = count(1)

    def enqueue(self, payload: Any, *, source_id: str = "") -> tuple[PendingRenderQuery | None, str]:
        """Queue one request, possibly evicting lower-priority pending work."""
        evicted_entries: list[PendingRenderQueryEntry] = []
        with self._lock:
            submitted_order = next(self._counter)
            request_id = f"query-{submitted_order}"
            pending = PendingRenderQuery(
                payload=payload,
                response_ready=threading.Event(),
                request_id=request_id,
                source_id=str(source_id or ""),
            )
            queue_item = build_queued_query_item_from_payload(
                payload,
                request_id=request_id,
                submitted_order=submitted_order,
                transport=self.transport,
                source_id=str(source_id or ""),
            )
            coalesce_decision = resolve_query_coalescing_decision(
                QueryCoalescingRequest(
                    pending_items=tuple(entry.queue_item for entry in self._entries),
                    incoming_item=queue_item,
                ),
                policy=self.coalescing_policy,
            )
            if not coalesce_decision.accepted:
                return None, coalesce_decision.reason or "query coalescing rejected request"

            existing_by_id = {entry.queue_item.request_id: entry for entry in self._entries}
            evicted_ids = set(coalesce_decision.evicted_request_ids)
            evicted_entries.extend(entry for entry in self._entries if entry.queue_item.request_id in evicted_ids)
            kept_entries = [
                existing_by_id[current_id]
                for current_id in coalesce_decision.pending_request_ids
                if current_id in existing_by_id
            ]
            state = QueryQueueState(
                pending_items=tuple(entry.queue_item for entry in kept_entries),
                max_pending=self.max_pending,
            )
            decision = admit_query_queue_item(state, queue_item, policy=self.queue_policy)
            if not decision.accepted:
                return None, decision.reason or "queue rejected the request"

            existing_by_id = {entry.queue_item.request_id: entry for entry in kept_entries}
            existing_by_id[request_id] = PendingRenderQueryEntry(queue_item=queue_item, pending=pending)
            evicted_ids = set(decision.evicted_request_ids)
            evicted_entries.extend(entry for entry in kept_entries if entry.queue_item.request_id in evicted_ids)
            self._entries = [existing_by_id[current_id] for current_id in decision.pending_request_ids]

        combined_reason = " / ".join(
            part
            for part in (
                coalesce_decision.reason,
                decision.reason,
            )
            if part
        )
        for entry in evicted_entries:
            entry.pending.response = build_query_event_error_response(
                "queue_dropped",
                reason=combined_reason or "evicted by queue policy",
                transport=self.transport,
                request_type=entry.queue_item.request_type,
            )
            entry.pending.response_ready.set()
        return pending, combined_reason or decision.reason

    def dispatch_next(self) -> PendingRenderQuery | None:
        """Pop the next pending query selected by the active queue policy."""
        with self._lock:
            state = QueryQueueState(
                pending_items=tuple(entry.queue_item for entry in self._entries),
                max_pending=self.max_pending,
            )
            decision = dispatch_query_queue_item(state, policy=self.queue_policy)
            if not decision.dispatch_request_id:
                return None
            existing_by_id = {entry.queue_item.request_id: entry for entry in self._entries}
            selected = existing_by_id.pop(decision.dispatch_request_id)
            self._entries = [existing_by_id[current_id] for current_id in decision.pending_request_ids]
            return selected.pending

    def cancel(self, request_id: str, *, event: str = "timeout", source_id: str = "") -> bool:
        """Remove one queued request before it is dispatched."""
        canceled_entries: list[PendingRenderQueryEntry] = []
        with self._lock:
            state_items = tuple(entry.queue_item for entry in self._entries)
            decision = resolve_query_cancellation_decision(
                QueryCancellationRequest(
                    pending_items=state_items,
                    event=event,
                    target_request_id=request_id,
                    source_id=str(source_id or ""),
                ),
                policy=self.cancellation_policy,
            )
            canceled_ids = set(decision.canceled_request_ids)
            if not canceled_ids:
                return False
            canceled_entries = [entry for entry in self._entries if entry.queue_item.request_id in canceled_ids]
            self._entries = [entry for entry in self._entries if entry.queue_item.request_id not in canceled_ids]
        for entry in canceled_entries:
            entry.pending.response = build_query_event_error_response(
                "query_canceled",
                reason=f"{event} / {decision.reason or 'canceled by policy'}",
                transport=self.transport,
                request_type=entry.queue_item.request_type,
            )
            entry.pending.response_ready.set()
        return True

    def cancel_source(self, source_id: str, *, event: str = "connection_closed") -> bool:
        """Cancel queued requests that belong to one source."""
        canceled_entries: list[PendingRenderQueryEntry] = []
        with self._lock:
            state_items = tuple(entry.queue_item for entry in self._entries)
            decision = resolve_query_cancellation_decision(
                QueryCancellationRequest(
                    pending_items=state_items,
                    event=event,
                    source_id=str(source_id or ""),
                ),
                policy=self.cancellation_policy,
            )
            canceled_ids = set(decision.canceled_request_ids)
            if not canceled_ids:
                return False
            canceled_entries = [entry for entry in self._entries if entry.queue_item.request_id in canceled_ids]
            self._entries = [entry for entry in self._entries if entry.queue_item.request_id not in canceled_ids]
        for entry in canceled_entries:
            entry.pending.response = build_query_event_error_response(
                "query_canceled",
                reason=f"{event} / {decision.reason or 'canceled by policy'}",
                transport=self.transport,
                request_type=entry.queue_item.request_type,
            )
            entry.pending.response_ready.set()
        return True

    def close(self, *, reason: str) -> None:
        """Drain queued requests with a transport-safe shutdown error."""
        drained_entries: list[PendingRenderQueryEntry] = []
        with self._lock:
            state_items = tuple(entry.queue_item for entry in self._entries)
            decision = resolve_query_cancellation_decision(
                QueryCancellationRequest(pending_items=state_items, event="shutdown"),
                policy=self.cancellation_policy,
            )
            canceled_ids = set(decision.canceled_request_ids)
            drained_entries = [entry for entry in self._entries if entry.queue_item.request_id in canceled_ids]
            self._entries = [entry for entry in self._entries if entry.queue_item.request_id not in canceled_ids]
        for entry in drained_entries:
            entry.pending.response = build_query_event_error_response(
                "server_shutdown",
                detail=reason,
                transport=self.transport,
                request_type=entry.queue_item.request_type,
            )
            entry.pending.response_ready.set()


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Compute the logistic sigmoid in float32."""
    return 1.0 / (1.0 + np.exp(-values, dtype=np.float32))


def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    """Normalize ``[x, y, z, w]`` quaternion and reject zero-norm input."""
    quaternion = np.asarray(quaternion, dtype=np.float32)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-8:
        raise ValueError("quaternion norm must be positive")
    return quaternion / norm


def quaternion_to_rotation_matrix(quaternion: tuple[float, float, float, float] | np.ndarray) -> np.ndarray:
    """Convert an ``[x, y, z, w]`` quaternion to a 3x3 rotation matrix."""
    x, y, z, w = normalize_quaternion(np.asarray(quaternion, dtype=np.float32))
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def yaw_to_quaternion(yaw_radians: float) -> tuple[float, float, float, float]:
    """Convert a planar yaw angle into an ``[x, y, z, w]`` quaternion."""
    half_yaw = float(yaw_radians) * 0.5
    return (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))


def build_view_matrix_world_to_camera(pose: CameraPose) -> np.ndarray:
    """Build a world-to-camera view matrix from a world-space pose.

    The pose orientation is interpreted as camera-to-world. The returned matrix
    matches the `[W | t]` world-to-camera form used by both the local simple
    rasterizer and gsplat.
    """
    rotation_camera_to_world = quaternion_to_rotation_matrix(pose.orientation)
    position = np.asarray(pose.position, dtype=np.float32)
    rotation_world_to_camera = rotation_camera_to_world.T
    translation = -rotation_world_to_camera @ position

    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = rotation_world_to_camera
    viewmat[:3, 3] = translation
    return viewmat


def resolve_render_backend(
    requested_backend: str,
    *,
    has_gaussian_splat: bool,
    gsplat_available: bool,
    cuda_available: bool,
) -> RenderBackendSelection:
    """Resolve the render backend under the current runtime constraints."""
    return select_render_backend(
        RenderBackendRequest(
            requested_backend=requested_backend,
            capabilities=RenderBackendCapabilities(
                has_gaussian_splat=has_gaussian_splat,
                gsplat_available=gsplat_available,
                cuda_available=cuda_available,
            ),
            preferences=RenderBackendPreferences(),
        )
    )


def compute_camera_intrinsics(width: int, height: int, fov_degrees: float) -> tuple[float, float, float, float]:
    """Compute pinhole intrinsics from vertical FOV."""
    if width <= 0 or height <= 0:
        raise ValueError("image width and height must be positive")
    if not math.isfinite(fov_degrees) or fov_degrees <= 0.0 or fov_degrees >= 179.0:
        raise ValueError("fov_degrees must be within (0, 179)")

    fov_radians = math.radians(fov_degrees)
    fy = height / (2.0 * math.tan(fov_radians * 0.5))
    fx = fy * (width / height)
    cx = width * 0.5
    cy = height * 0.5
    return (float(fx), float(fy), float(cx), float(cy))


def build_camera_info_payload(width: int, height: int, fov_degrees: float, frame_id: str) -> dict[str, Any]:
    """Build a serializable camera info payload for query responses."""
    fx, fy, cx, cy = compute_camera_intrinsics(width, height, fov_degrees)
    return {
        "frameId": frame_id,
        "width": int(width),
        "height": int(height),
        "distortionModel": "plumb_bob",
        "d": [0.0, 0.0, 0.0, 0.0, 0.0],
        "k": [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        "r": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "p": [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
    }


def parse_render_query_request(
    payload: Any,
    *,
    default_width: int,
    default_height: int,
    default_fov_degrees: float,
    default_near_clip: float,
    default_far_clip: float,
    default_point_radius: int,
) -> dict[str, Any]:
    """Parse a query render request into normalized render settings."""
    imported = import_query_request(
        QueryRequestImportRequest(
            payload=payload,
            defaults=RenderQueryDefaults(
                width=default_width,
                height=default_height,
                fov_degrees=default_fov_degrees,
                near_clip=default_near_clip,
                far_clip=default_far_clip,
                point_radius=default_point_radius,
            ),
        )
    )
    if imported.request_type != "render" or imported.render is None:
        raise ValueError("query payload type must be 'render'")
    render = imported.render
    pose = CameraPose(position=render.position, orientation=render.orientation)

    return {
        "pose": pose,
        "width": render.width,
        "height": render.height,
        "fov_degrees": render.fov_degrees,
        "near_clip": render.near_clip,
        "far_clip": render.far_clip,
        "point_radius": render.point_radius,
    }


def build_render_query_response(frame: RenderFrameBundle, *, frame_id: str) -> dict[str, Any]:
    """Build the serializable response for a render query request."""
    return build_render_result_response_document(
        QueryRenderResultResponseInput(
            frame_id=frame_id,
            width=frame.width,
            height=frame.height,
            fov_degrees=frame.fov_degrees,
            near_clip=frame.near_clip,
            far_clip=frame.far_clip,
            point_radius=frame.point_radius,
            position=frame.pose.position,
            orientation=frame.pose.orientation,
            camera_info=build_camera_info_payload(frame.width, frame.height, frame.fov_degrees, frame_id),
            color_jpeg_bytes=frame.rgb_jpeg,
            depth_float32_bytes=np.asarray(frame.depth, dtype="<f4").tobytes(),
        )
    )


def build_query_ready_payload(
    *,
    transport: str,
    endpoint: str,
    frame_id: str,
    renderer: str,
    renderer_reason: str,
    width: int,
    height: int,
    fov_degrees: float,
    near_clip: float,
    far_clip: float,
    point_radius: int,
) -> dict[str, Any]:
    """Build the handshake payload announced by interactive query transports."""
    return build_query_ready_response_document(
        QueryReadyResponseInput(
            transport=transport,
            endpoint=endpoint,
            frame_id=frame_id,
            renderer=renderer,
            renderer_reason=renderer_reason,
            request_types=("render", "localization-image-benchmark"),
            defaults=QueryReadyDefaults(
                width=int(width),
                height=int(height),
                fov_degrees=float(fov_degrees),
                near_clip=float(near_clip),
                far_clip=float(far_clip),
                point_radius=int(point_radius),
            ),
        )
    )


def build_query_error_response(error: Exception | str) -> dict[str, Any]:
    """Build a transport-safe error payload."""
    if isinstance(error, Exception):
        return build_query_event_error_response(
            "handler_exception",
            detail=str(error),
            exception=error,
        )
    return build_query_error_response_document(
        QueryErrorResponseInput(
            error=str(error),
            error_type="RuntimeError",
        )
    )


def build_query_event_error_input(
    event: str,
    *,
    reason: str = "",
    detail: str = "",
    transport: str = "",
    request_type: str = "",
    exception: Exception | None = None,
) -> QueryErrorResponseInput:
    """Build a canonical query error input from one transport/runtime event."""
    decision = resolve_query_error_mapping(
        QueryErrorMappingRequest(
            event=event,
            reason=reason,
            detail=detail,
            transport=transport,
            request_type=request_type,
            exception_type=type(exception).__name__ if isinstance(exception, Exception) else "",
        )
    )
    return QueryErrorResponseInput(
        error=decision.error,
        error_type=decision.error_type,
        error_code=decision.error_code,
    )


def build_query_event_error_response(
    event: str,
    *,
    reason: str = "",
    detail: str = "",
    transport: str = "",
    request_type: str = "",
    exception: Exception | None = None,
) -> dict[str, Any]:
    """Build one canonical error response for a transport/runtime event."""
    return build_query_error_response_document(
        build_query_event_error_input(
            event,
            reason=reason,
            detail=detail,
            transport=transport,
            request_type=request_type,
            exception=exception,
        )
    )


def _parse_remote_address(remote_address: Any) -> tuple[str, int | None]:
    if isinstance(remote_address, tuple) and len(remote_address) >= 2:
        host = str(remote_address[0] or "")
        try:
            return host, int(remote_address[1])
        except (TypeError, ValueError):
            return host, None
    if isinstance(remote_address, str):
        text = remote_address.strip()
        if ":" in text:
            host, _, raw_port = text.rpartition(":")
            try:
                return host, int(raw_port)
            except ValueError:
                return text, None
        return text, None
    return "", None


def resolve_connection_source_id(
    *,
    transport: str,
    endpoint: str,
    connection_serial: int,
    websocket: Any | None = None,
    client_hint: str = "",
) -> str:
    """Resolve one queue/coalescing source id for a live transport connection."""
    remote_host, remote_port = _parse_remote_address(getattr(websocket, "remote_address", None))
    identity = resolve_query_source_identity(
        QuerySourceIdentityRequest(
            transport=transport,
            connection_serial=connection_serial,
            endpoint=endpoint,
            remote_host=remote_host,
            remote_port=remote_port,
            client_hint=client_hint,
        )
    )
    return identity.source_id


def resolve_query_endpoint(query_transport: str, query_endpoint: str) -> str:
    """Resolve the default endpoint for the selected transport."""
    return resolve_query_transport_endpoint(query_transport, query_endpoint)


def resolve_query_transport_selection(
    requested_transport: str,
    *,
    pose_source: str,
    query_endpoint: str,
) -> QueryTransportSelection:
    """Resolve query transport and endpoint under the current runtime constraints."""
    return select_query_transport(
        QueryTransportRequest(
            requested_transport=requested_transport,
            pose_source=pose_source,
            endpoint=query_endpoint,
            capabilities=QueryTransportCapabilities(
                zmq_available=importlib.util.find_spec("zmq") is not None,
                ws_available=importlib.util.find_spec("websockets") is not None,
            ),
            preferences=QueryTransportPreferences(
                enable_query_transport=(pose_source == "query"),
                prefer_browser_clients=(pose_source == "query"),
                prefer_local_cli=False,
            ),
        )
    )


def encode_rgb_to_jpeg(rgb_image: np.ndarray, quality: int = 85) -> bytes:
    """Encode an ``H x W x 3`` RGB uint8 array into JPEG bytes."""
    image = PILImage.fromarray(np.asarray(rgb_image, dtype=np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality), optimize=False)
    return buffer.getvalue()


def resolve_query_response_timeout_seconds(
    payload: Any,
    *,
    transport: str = "ws",
    default: float = 30.0,
    maximum: float = 300.0,
) -> float:
    """Resolve a per-request response timeout while keeping the server bounded."""
    request = build_query_timeout_policy_request(
        payload,
        transport=transport,
        allow_retry=False,
        default_server_timeout_seconds=default,
        maximum_server_timeout_seconds=maximum,
    )
    return float(resolve_query_timeout_plan(request).server_timeout_seconds)


class RenderQueryHandler:
    """Handle request-response rendering independent of the transport layer."""

    def __init__(
        self,
        renderer: "HeadlessSplatRenderer",
        *,
        frame_id: str,
        default_width: int,
        default_height: int,
        default_fov_degrees: float,
        default_near_clip: float,
        default_far_clip: float,
        default_point_radius: int,
        jpeg_quality: int,
        query_endpoint: str = "",
        publish_callback: Any | None = None,
    ) -> None:
        self.renderer = renderer
        self.frame_id = frame_id
        self.default_width = int(default_width)
        self.default_height = int(default_height)
        self.default_fov_degrees = float(default_fov_degrees)
        self.default_near_clip = float(default_near_clip)
        self.default_far_clip = float(default_far_clip)
        self.default_point_radius = int(default_point_radius)
        self.jpeg_quality = int(jpeg_quality)
        self.query_endpoint = str(query_endpoint or "")
        self.publish_callback = publish_callback

    def _build_query_request_defaults(self) -> RenderQueryDefaults:
        """Build shared defaults for request import."""
        return RenderQueryDefaults(
            width=self.default_width,
            height=self.default_height,
            fov_degrees=self.default_fov_degrees,
            near_clip=self.default_near_clip,
            far_clip=self.default_far_clip,
            point_radius=self.default_point_radius,
        )

    def render_request(self, payload: Any | RenderQuerySpec, *, publish: bool = True) -> RenderFrameBundle:
        """Render one query payload into a frame bundle."""
        if isinstance(payload, RenderQuerySpec):
            request = {
                "pose": CameraPose(position=payload.position, orientation=payload.orientation),
                "width": payload.width,
                "height": payload.height,
                "fov_degrees": payload.fov_degrees,
                "near_clip": payload.near_clip,
                "far_clip": payload.far_clip,
                "point_radius": payload.point_radius,
            }
        else:
            request = parse_render_query_request(
                payload,
                default_width=self.default_width,
                default_height=self.default_height,
                default_fov_degrees=self.default_fov_degrees,
                default_near_clip=self.default_near_clip,
                default_far_clip=self.default_far_clip,
                default_point_radius=self.default_point_radius,
            )
        rgb, depth = self.renderer.render_rgbd(
            request["pose"],
            width=request["width"],
            height=request["height"],
            fov_degrees=request["fov_degrees"],
            near_clip=request["near_clip"],
            far_clip=request["far_clip"],
            point_radius=request["point_radius"],
        )
        frame = RenderFrameBundle(
            pose=request["pose"],
            width=request["width"],
            height=request["height"],
            fov_degrees=request["fov_degrees"],
            near_clip=request["near_clip"],
            far_clip=request["far_clip"],
            point_radius=request["point_radius"],
            rgb=rgb,
            depth=depth,
            rgb_jpeg=encode_rgb_to_jpeg(rgb, quality=self.jpeg_quality),
        )
        if publish and self.publish_callback is not None:
            self.publish_callback(frame)
        return frame

    def _handle_localization_image_benchmark_request(
        self,
        request: LocalizationImageBenchmarkQuerySpec,
    ) -> dict[str, Any]:
        """Benchmark a localization trajectory against a ground-truth capture bundle."""
        from .localization_image_benchmark import benchmark_localization_images

        def query_fn(_endpoint: str, render_payload: dict[str, Any], _timeout_ms: int) -> dict[str, Any]:
            return build_render_query_response(
                self.render_request(render_payload, publish=False),
                frame_id=self.frame_id,
            )

        return benchmark_localization_images(
            endpoint=self.query_endpoint or "inproc://render-query-handler",
            ground_truth_bundle=request.ground_truth_bundle,
            estimate_input=request.estimate,
            alignment=request.alignment,
            timeout_ms=request.timeout_ms,
            max_frames=request.max_frames,
            metrics=request.metrics,
            lpips_net=request.lpips_net,
            device=request.device,
            query_fn=query_fn,
        )

    def handle_request(self, payload: Any) -> dict[str, Any]:
        """Render a request and return the JSON-serializable response."""
        imported = import_query_request(
            QueryRequestImportRequest(payload=payload, defaults=self._build_query_request_defaults())
        )
        if imported.request_type == "render":
            assert imported.render is not None
            return build_render_query_response(self.render_request(imported.render), frame_id=self.frame_id)
        if imported.request_type == "localization-image-benchmark":
            assert imported.image_benchmark is not None
            return self._handle_localization_image_benchmark_request(imported.image_benchmark)
        raise ValueError(f"unsupported query payload type: {imported.request_type}")


class ZmqRenderQueryServer:
    """ZeroMQ request-response wrapper for ad-hoc render queries."""

    def __init__(self, endpoint: str, request_handler: Any) -> None:
        try:
            import zmq
        except ModuleNotFoundError as exc:
            raise RuntimeError("query-transport=zmq requires the optional `pyzmq` package") from exc

        self._zmq = zmq
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.REP)
        self._socket.linger = 0
        self._socket.bind(endpoint)
        self.endpoint = endpoint
        self.request_handler = request_handler

    def poll_once(self) -> bool:
        """Handle at most one pending request without blocking."""
        try:
            payload = self._socket.recv_json(flags=self._zmq.NOBLOCK)
        except self._zmq.Again:
            return False

        try:
            response = self.request_handler(payload)
        except Exception as error:
            response = build_query_event_error_response(
                "handler_exception",
                detail=str(error),
                exception=error,
                transport="zmq",
            )

        self._socket.send_json(response)
        return True

    def close(self) -> None:
        """Close the bound REP socket."""
        self._socket.close(linger=0)


class WebSocketRenderQueryServer:
    """WebSocket request-response wrapper for browser-facing render queries."""

    def __init__(self, endpoint: str, request_handler: Any, *, ready_payload: dict[str, Any]) -> None:
        try:
            from websockets.sync.server import serve
        except ModuleNotFoundError as exc:
            raise RuntimeError("query-transport=ws requires the optional `websockets` package") from exc

        parsed = urlparse(endpoint)
        if parsed.scheme != "ws":
            raise RuntimeError("query-transport=ws requires a ws:// endpoint")
        if not parsed.hostname or parsed.port is None:
            raise RuntimeError("query-transport=ws requires host and port in the endpoint")

        self.endpoint = endpoint
        self.host = parsed.hostname
        self.port = int(parsed.port)
        self.path = parsed.path or "/"
        self.request_handler = request_handler
        self.ready_payload = ready_payload
        self._serve = serve
        self._queue = PendingRenderQueryStore(
            transport="ws",
            queue_policy="interactive_first",
            cancellation_policy="cancel_source_backlog",
            coalescing_policy="latest_render_per_source",
            max_pending=4,
        )
        self._closed = threading.Event()
        self._started = threading.Event()
        self._connection_counter = count(1)
        self._server = None
        self._thread = threading.Thread(target=self._run_server, name="sim2real-ws-query-server", daemon=True)
        self._thread.start()
        if not self._started.wait(timeout=5.0):
            raise RuntimeError(f"timed out starting query-transport=ws server on {endpoint}")

    def _run_server(self) -> None:
        def process_request(connection: Any, request: Any) -> Any:
            if request.path != self.path:
                from websockets.http11 import Response

                return Response(404, "Not Found", headers=[], body=b"sim2real query path mismatch\n")
            return None

        server = self._serve(
            self._handle_connection,
            self.host,
            self.port,
            process_request=process_request,
            close_timeout=1,
            ping_interval=20,
            ping_timeout=20,
        )
        self._server = server
        self._started.set()

        try:
            server.serve_forever()
        finally:
            try:
                server.socket.close()
            except Exception:
                pass

    def _handle_connection(self, websocket: Any) -> None:
        connection_serial = next(self._connection_counter)
        connection_id = resolve_connection_source_id(
            transport="ws",
            endpoint=self.endpoint,
            connection_serial=connection_serial,
            websocket=websocket,
        )
        websocket.send(json.dumps(self.ready_payload))
        try:
            while not self._closed.is_set():
                try:
                    raw_message = websocket.recv(timeout=0.1)
                except TimeoutError:
                    continue
                except Exception:
                    return

                try:
                    payload = json.loads(raw_message if isinstance(raw_message, str) else raw_message.decode("utf-8"))
                except Exception as error:
                    websocket.send(
                        json.dumps(
                            build_query_event_error_response(
                                "invalid_json",
                                detail=str(error),
                                exception=error,
                                transport="ws",
                            )
                        )
                    )
                    continue

                response_timeout_seconds = resolve_query_response_timeout_seconds(payload, transport="ws")
                pending, enqueue_reason = self._queue.enqueue(payload, source_id=connection_id)
                if pending is None:
                    response = build_query_event_error_response(
                        "queue_rejected",
                        reason=enqueue_reason or "queue rejected the request",
                        transport="ws",
                    )
                    try:
                        websocket.send(json.dumps(response))
                    except Exception:
                        return
                    continue
                if not pending.response_ready.wait(timeout=response_timeout_seconds):
                    self._queue.cancel(pending.request_id, event="timeout", source_id=connection_id)
                    response = build_query_event_error_response("query_timeout", transport="ws")
                else:
                    response = pending.response or build_query_event_error_response("empty_response", transport="ws")

                try:
                    websocket.send(json.dumps(response))
                except Exception:
                    return
        finally:
            self._queue.cancel_source(connection_id, event="connection_closed")

    def poll_once(self) -> bool:
        """Process at most one pending WebSocket render query."""
        pending = self._queue.dispatch_next()
        if pending is None:
            return False

        try:
            pending.response = self.request_handler(pending.payload)
        except Exception as error:
            pending.response = build_query_event_error_response(
                "handler_exception",
                detail=str(error),
                exception=error,
                transport="ws",
            )
        finally:
            pending.response_ready.set()

        return True

    def close(self) -> None:
        """Shut down the WebSocket server and release pending requests."""
        self._closed.set()
        self._queue.close(reason="render query server is shutting down")

        if self._server is not None:
            self._server.shutdown()
            try:
                self._server.socket.close()
            except Exception:
                pass
            self._server = None

        if self._thread.is_alive():
            self._thread.join(timeout=5.0)


class HeadlessSplatRenderer:
    """Approximate headless splat renderer backed by projected point footprints."""

    def __init__(
        self,
        ply_path: Path | str,
        *,
        backend: str = "auto",
        max_points: int | None = None,
    ) -> None:
        ply = load_ply(ply_path)
        self.positions = np.asarray(ply.positions, dtype=np.float32)
        self.colors = np.asarray(np.clip(ply.colors, 0.0, 1.0), dtype=np.float32)
        self.scales = np.asarray(ply.scales, dtype=np.float32) if ply.scales is not None else None
        self.rotations = np.asarray(ply.rotations, dtype=np.float32) if ply.rotations is not None else None

        if ply.opacities is None:
            self.opacities = np.ones((self.positions.shape[0],), dtype=np.float32)
        else:
            self.opacities = np.asarray(sigmoid(np.asarray(ply.opacities, dtype=np.float32)), dtype=np.float32)

        if max_points is not None and max_points > 0 and self.positions.shape[0] > max_points:
            indices = np.linspace(0, self.positions.shape[0] - 1, num=max_points, dtype=np.int64)
            self.positions = self.positions[indices]
            self.colors = self.colors[indices]
            self.opacities = self.opacities[indices]

        self.num_points = int(self.positions.shape[0])
        if self.num_points == 0:
            raise ValueError("PLY contained no renderable points")

        self.has_gaussian_splat = bool(ply.is_gaussian_splat and self.scales is not None and self.rotations is not None)
        self._torch = None
        self._gsplat_rasterization = None
        self._means_torch = None
        self._scales_torch = None
        self._rotations_torch = None
        self._opacities_torch = None
        self._colors_torch = None
        self.backend_selection = resolve_render_backend(
            backend,
            has_gaussian_splat=self.has_gaussian_splat,
            gsplat_available=importlib.util.find_spec("gsplat") is not None,
            cuda_available=self._torch_cuda_available(),
        )
        self.backend = self.backend_selection.name
        self.backend_reason = self.backend_selection.reason
        if self.backend == "gsplat":
            self._initialize_gsplat_backend()

    def render_rgbd(
        self,
        pose: CameraPose,
        *,
        width: int,
        height: int,
        fov_degrees: float,
        near_clip: float,
        far_clip: float,
        point_radius: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render an RGB image and float32 depth image from ``pose``."""
        if self.backend == "gsplat":
            return self._render_rgbd_gsplat(
                pose,
                width=width,
                height=height,
                fov_degrees=fov_degrees,
                near_clip=near_clip,
                far_clip=far_clip,
            )
        return self._render_rgbd_simple(
            pose,
            width=width,
            height=height,
            fov_degrees=fov_degrees,
            near_clip=near_clip,
            far_clip=far_clip,
            point_radius=point_radius,
        )

    def _render_rgbd_simple(
        self,
        pose: CameraPose,
        *,
        width: int,
        height: int,
        fov_degrees: float,
        near_clip: float,
        far_clip: float,
        point_radius: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render RGBD with the deterministic numpy point-splat fallback."""
        fx, fy, cx, cy = compute_camera_intrinsics(width, height, fov_degrees)
        if near_clip <= 0.0 or far_clip <= near_clip:
            raise ValueError("clip planes must satisfy 0 < near_clip < far_clip")

        viewmat = build_view_matrix_world_to_camera(pose)
        rotation_world_to_camera = viewmat[:3, :3]
        translation = viewmat[:3, 3]
        camera_space = self.positions @ rotation_world_to_camera.T + translation
        depths = camera_space[:, 2]
        valid = (depths > near_clip) & (depths < far_clip)

        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        depth = np.full((height, width), float(far_clip), dtype=np.float32)
        if not np.any(valid):
            return rgb, depth

        camera_space = camera_space[valid]
        depths = depths[valid]
        colors = self.colors[valid]
        opacities = self.opacities[valid]

        px = (camera_space[:, 0] / depths) * fx + cx
        py = cy - (camera_space[:, 1] / depths) * fy
        px = np.rint(px).astype(np.int32)
        py = np.rint(py).astype(np.int32)

        footprint_offsets = self._build_footprint_offsets(point_radius)
        if footprint_offsets.shape[0] > 1:
            px = px[:, None] + footprint_offsets[None, :, 0]
            py = py[:, None] + footprint_offsets[None, :, 1]
            depths = np.broadcast_to(depths[:, None], px.shape).reshape(-1)
            colors = np.broadcast_to(colors[:, None, :], (colors.shape[0], footprint_offsets.shape[0], 3)).reshape(
                -1, 3
            )
            opacities = np.broadcast_to(opacities[:, None], px.shape).reshape(-1)
            px = px.reshape(-1)
            py = py.reshape(-1)

        in_bounds = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        if not np.any(in_bounds):
            return rgb, depth

        px = px[in_bounds]
        py = py[in_bounds]
        depths = depths[in_bounds]
        colors = colors[in_bounds]
        opacities = opacities[in_bounds]

        pixel_indices = py * width + px
        sort_order = np.lexsort((depths, pixel_indices))
        pixel_indices = pixel_indices[sort_order]
        depths = depths[sort_order]
        colors = colors[sort_order]
        opacities = opacities[sort_order]

        _, first_indices = np.unique(pixel_indices, return_index=True)
        pixel_indices = pixel_indices[first_indices]
        depths = depths[first_indices]
        colors = colors[first_indices]
        opacities = opacities[first_indices]

        rgb_flat = rgb.reshape(-1, 3)
        rgb_flat[pixel_indices] = np.clip(colors * opacities[:, None] * 255.0, 0.0, 255.0).astype(np.uint8)
        depth.reshape(-1)[pixel_indices] = depths.astype(np.float32)
        return rgb, depth

    @staticmethod
    def _torch_cuda_available() -> bool:
        """Return whether CUDA-backed torch is available."""
        try:
            import torch
        except ImportError:
            return False
        return bool(torch.cuda.is_available())

    def _initialize_gsplat_backend(self) -> None:
        """Initialize CUDA tensors for the gsplat rasterizer backend."""
        import torch
        from gsplat import rasterization

        device = torch.device("cuda")
        self._torch = torch
        self._gsplat_rasterization = rasterization
        self._means_torch = torch.tensor(self.positions, dtype=torch.float32, device=device)
        self._colors_torch = torch.tensor(self.colors, dtype=torch.float32, device=device)
        self._opacities_torch = torch.tensor(self.opacities, dtype=torch.float32, device=device)
        self._scales_torch = torch.tensor(np.exp(self.scales), dtype=torch.float32, device=device)
        self._rotations_torch = torch.nn.functional.normalize(
            torch.tensor(self.rotations, dtype=torch.float32, device=device),
            dim=-1,
        )

    def _render_rgbd_gsplat(
        self,
        pose: CameraPose,
        *,
        width: int,
        height: int,
        fov_degrees: float,
        near_clip: float,
        far_clip: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render RGBD with gsplat's CUDA rasterizer."""
        if near_clip <= 0.0 or far_clip <= near_clip:
            raise ValueError("clip planes must satisfy 0 < near_clip < far_clip")

        torch = self._torch
        assert torch is not None
        assert self._gsplat_rasterization is not None
        assert self._means_torch is not None
        assert self._colors_torch is not None
        assert self._opacities_torch is not None
        assert self._scales_torch is not None
        assert self._rotations_torch is not None

        fx, fy, cx, cy = compute_camera_intrinsics(width, height, fov_degrees)
        device = self._means_torch.device
        K = torch.tensor(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        viewmat = torch.tensor(build_view_matrix_world_to_camera(pose), dtype=torch.float32, device=device)
        rgbd, alphas, _ = self._gsplat_rasterization(
            means=self._means_torch,
            quats=self._rotations_torch,
            scales=self._scales_torch,
            opacities=self._opacities_torch,
            colors=self._colors_torch,
            viewmats=viewmat.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=width,
            height=height,
            near_plane=near_clip,
            far_plane=far_clip,
            render_mode="RGB+ED",
        )
        rgbd = rgbd[0].detach().cpu().numpy()
        alphas = alphas[0, ..., 0].detach().cpu().numpy()
        rgb = np.clip(rgbd[..., :3], 0.0, 1.0)
        depth = rgbd[..., 3].astype(np.float32, copy=True)
        depth[alphas <= 1e-4] = np.float32(far_clip)
        return (rgb * 255.0).astype(np.uint8), depth

    @staticmethod
    def _build_footprint_offsets(point_radius: int) -> np.ndarray:
        """Build integer pixel offsets for a circular point footprint."""
        radius = max(int(point_radius), 0)
        if radius == 0:
            return np.array([[0, 0]], dtype=np.int32)

        offsets: list[tuple[int, int]] = []
        radius_sq = radius * radius
        for y_offset in range(-radius, radius + 1):
            for x_offset in range(-radius, radius + 1):
                if x_offset * x_offset + y_offset * y_offset <= radius_sq:
                    offsets.append((x_offset, y_offset))
        return np.asarray(offsets, dtype=np.int32)


def _import_ros2() -> dict[str, Any]:
    try:
        import rclpy
        from geometry_msgs.msg import Pose2D, PoseStamped
        from rclpy.node import Node
        from sensor_msgs.msg import CameraInfo, CompressedImage, Image
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in real ROS2 env
        raise RuntimeError(
            "ROS2 runtime not available. Source your ROS2 environment so `rclpy` and message packages are importable."
        ) from exc

    return {
        "rclpy": rclpy,
        "Node": Node,
        "Pose2D": Pose2D,
        "PoseStamped": PoseStamped,
        "CameraInfo": CameraInfo,
        "CompressedImage": CompressedImage,
        "Image": Image,
    }


def _assign_stamp(header: Any, stamp: Any, frame_id: str) -> None:
    header.stamp = stamp
    header.frame_id = frame_id


def _populate_camera_info(message: Any, stamp: Any, frame_id: str, width: int, height: int, fov_degrees: float) -> None:
    fx, fy, cx, cy = compute_camera_intrinsics(width, height, fov_degrees)
    _assign_stamp(message.header, stamp, frame_id)
    message.width = int(width)
    message.height = int(height)
    message.distortion_model = "plumb_bob"
    message.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    message.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    message.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    message.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]


def _build_node_class(ros2: dict[str, Any]) -> type:
    Node = ros2["Node"]
    Pose2D = ros2["Pose2D"]
    PoseStamped = ros2["PoseStamped"]
    CameraInfo = ros2["CameraInfo"]
    CompressedImage = ros2["CompressedImage"]
    Image = ros2["Image"]

    class HeadlessRenderServerNode(Node):
        """ROS2 node that renders a PLY from the latest camera pose."""

        def __init__(self, args: argparse.Namespace) -> None:
            super().__init__(args.node_name)
            self.args = args
            self.topic_map = build_ros_topic_map(args.namespace)
            self.renderer = HeadlessSplatRenderer(args.ply, backend=args.renderer, max_points=args.max_points)
            self.camera_info_pub = self.create_publisher(CameraInfo, self.topic_map.camera_info, 10)
            self.camera_compressed_pub = self.create_publisher(CompressedImage, self.topic_map.camera_compressed, 10)
            self.depth_image_pub = self.create_publisher(Image, self.topic_map.depth_image, 10)
            self.latest_pose: CameraPose | None = None
            self.frames_published = 0
            self.completed = False
            self._waiting_for_pose_logged = False
            self._log_interval = max(1, int(round(max(args.fps, 1.0))))
            self.query_server = None
            self.render_timer = None
            self.query_timer = None
            self.query_handler = RenderQueryHandler(
                self.renderer,
                frame_id=args.frame_id,
                default_width=args.width,
                default_height=args.height,
                default_fov_degrees=args.fov_degrees,
                default_near_clip=args.near_clip,
                default_far_clip=args.far_clip,
                default_point_radius=args.point_radius,
                jpeg_quality=args.jpeg_quality,
                query_endpoint=args.query_endpoint,
                publish_callback=self._publish_query_frame,
            )

            if args.pose_source == "static":
                self.latest_pose = CameraPose(
                    position=tuple(float(value) for value in args.static_position),
                    orientation=tuple(float(value) for value in args.static_orientation),
                )
            elif args.pose_source == "robot_pose_stamped":
                self.create_subscription(PoseStamped, self.topic_map.robot_pose_stamped, self._on_pose_stamped, 10)
            elif args.pose_source == "robot_pose2d":
                self.create_subscription(Pose2D, self.topic_map.robot_pose2d, self._on_pose2d, 10)
            elif args.pose_source == "query":
                self.latest_pose = None
            else:  # pragma: no cover - parser restricts values
                raise ValueError(f"Unsupported pose source: {args.pose_source}")

            if args.pose_source != "query":
                self.render_timer = self.create_timer(1.0 / max(args.fps, 0.1), self._on_render_timer)

            if args.query_transport == "zmq":
                self.query_server = ZmqRenderQueryServer(args.query_endpoint, self.query_handler.handle_request)
            elif args.query_transport == "ws":
                self.query_server = WebSocketRenderQueryServer(
                    args.query_endpoint,
                    self.query_handler.handle_request,
                    ready_payload=build_query_ready_payload(
                        transport="ws",
                        endpoint=args.query_endpoint,
                        frame_id=args.frame_id,
                        renderer=self.renderer.backend,
                        renderer_reason=self.renderer.backend_reason,
                        width=args.width,
                        height=args.height,
                        fov_degrees=args.fov_degrees,
                        near_clip=args.near_clip,
                        far_clip=args.far_clip,
                        point_radius=args.point_radius,
                    ),
                )
                self.query_timer = self.create_timer(max(args.query_poll_period, 0.001), self._poll_query_server)

            self.get_logger().info(
                f"Headless render server ready ply={args.ply} points={self.renderer.num_points} "
                f"pose_source={args.pose_source} resolution={args.width}x{args.height} fps={args.fps:.2f} "
                f"backend={self.renderer.backend} reason={self.renderer.backend_reason}"
            )
            if self.query_server is not None:
                self.get_logger().info(
                    f"query transport={args.query_transport} endpoint={args.query_endpoint} "
                    f"reason={getattr(args, 'query_transport_reason', '')}"
                )

        def _on_pose_stamped(self, message: Any) -> None:
            self.latest_pose = CameraPose(
                position=(
                    float(message.pose.position.x),
                    float(message.pose.position.y),
                    float(message.pose.position.z),
                ),
                orientation=(
                    float(message.pose.orientation.x),
                    float(message.pose.orientation.y),
                    float(message.pose.orientation.z),
                    float(message.pose.orientation.w),
                ),
            )

        def _on_pose2d(self, message: Any) -> None:
            self.latest_pose = CameraPose(
                position=(float(message.x), float(message.y), float(self.args.pose2d_z)),
                orientation=yaw_to_quaternion(float(message.theta)),
            )

        def _on_render_timer(self) -> None:
            if self.latest_pose is None:
                if not self._waiting_for_pose_logged:
                    self.get_logger().info(f"Waiting for pose on source={self.args.pose_source}")
                    self._waiting_for_pose_logged = True
                return

            self._waiting_for_pose_logged = False
            frame = self._render_frame(
                self.latest_pose,
                width=self.args.width,
                height=self.args.height,
                fov_degrees=self.args.fov_degrees,
                near_clip=self.args.near_clip,
                far_clip=self.args.far_clip,
                point_radius=self.args.point_radius,
            )
            self._publish_frame(frame)

        def _render_frame(
            self,
            pose: CameraPose,
            *,
            width: int,
            height: int,
            fov_degrees: float,
            near_clip: float,
            far_clip: float,
            point_radius: int,
        ) -> RenderFrameBundle:
            rgb, depth = self.renderer.render_rgbd(
                pose,
                width=width,
                height=height,
                fov_degrees=fov_degrees,
                near_clip=near_clip,
                far_clip=far_clip,
                point_radius=point_radius,
            )
            return RenderFrameBundle(
                pose=pose,
                width=width,
                height=height,
                fov_degrees=fov_degrees,
                near_clip=near_clip,
                far_clip=far_clip,
                point_radius=point_radius,
                rgb=rgb,
                depth=depth,
                rgb_jpeg=encode_rgb_to_jpeg(rgb, quality=self.args.jpeg_quality),
            )

        def _publish_frame(self, frame: RenderFrameBundle) -> None:
            stamp = self.get_clock().now().to_msg()

            camera_info = CameraInfo()
            _populate_camera_info(
                camera_info,
                stamp,
                self.args.frame_id,
                frame.width,
                frame.height,
                frame.fov_degrees,
            )
            self.camera_info_pub.publish(camera_info)

            camera_image = CompressedImage()
            _assign_stamp(camera_image.header, stamp, self.args.frame_id)
            camera_image.format = "jpeg"
            camera_image.data = frame.rgb_jpeg
            self.camera_compressed_pub.publish(camera_image)

            depth_image = Image()
            _assign_stamp(depth_image.header, stamp, self.args.frame_id)
            depth_image.width = int(frame.width)
            depth_image.height = int(frame.height)
            depth_image.encoding = "32FC1"
            depth_image.is_bigendian = 0
            depth_image.step = int(frame.width * 4)
            depth_image.data = np.asarray(frame.depth, dtype="<f4").tobytes()
            self.depth_image_pub.publish(depth_image)

            self.frames_published += 1
            if self.frames_published == 1 or self.frames_published % self._log_interval == 0:
                self.get_logger().info(
                    f"published frame={self.frames_published} "
                    f"camera_bytes={len(camera_image.data)} depth_bytes={len(depth_image.data)}"
                )

            if self.args.run_once:
                self.completed = True

        def _publish_query_frame(self, frame: RenderFrameBundle) -> None:
            self.latest_pose = frame.pose
            self._publish_frame(frame)

        def _poll_query_server(self) -> None:
            if self.query_server is None:
                return
            while self.query_server.poll_once():
                pass

        def destroy_node(self) -> bool:
            if self.query_server is not None:
                self.query_server.close()
                self.query_server = None
            return super().destroy_node()

    return HeadlessRenderServerNode


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse CLI for the headless render server."""
    parser = argparse.ArgumentParser(
        prog="gs-sim2real sim2real-server",
        description="Headless PLY renderer that publishes RGB + depth to Dynamic Map Viewer ROS2 topics",
    )
    parser.add_argument("--ply", required=True, help="Path to the trained PLY point cloud")
    parser.add_argument("--namespace", default="/dreamwalker", help="ROS topic namespace")
    parser.add_argument("--node-name", default="dreamwalker_sim2real_server", help="ROS2 node name")
    parser.add_argument("--frame-id", default="dreamwalker_map", help="Camera frame id for published messages")
    parser.add_argument("--width", type=int, default=640, help="Render width in pixels")
    parser.add_argument("--height", type=int, default=480, help="Render height in pixels")
    parser.add_argument("--fps", type=float, default=5.0, help="Publish rate in Hz")
    parser.add_argument("--fov-degrees", type=float, default=60.0, help="Vertical field of view in degrees")
    parser.add_argument("--near-clip", type=float, default=0.05, help="Near clip plane in meters")
    parser.add_argument("--far-clip", type=float, default=50.0, help="Far clip plane in meters")
    parser.add_argument("--point-radius", type=int, default=1, help="Projected point footprint radius in pixels")
    parser.add_argument("--jpeg-quality", type=int, default=85, help="JPEG quality for camera output")
    parser.add_argument(
        "--renderer",
        choices=["auto", "simple", "gsplat"],
        default="auto",
        help="Rasterization backend. auto uses gsplat only when CUDA and Gaussian PLY parameters are available",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=200000,
        help="Maximum number of points to load from the PLY for rendering",
    )
    parser.add_argument(
        "--pose-source",
        choices=["static", "robot_pose_stamped", "robot_pose2d", "query"],
        default="robot_pose_stamped",
        help="Source of camera poses used for rendering",
    )
    parser.add_argument(
        "--static-position",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
        help="Static camera position used when --pose-source static",
    )
    parser.add_argument(
        "--static-orientation",
        nargs=4,
        type=float,
        metavar=("QX", "QY", "QZ", "QW"),
        default=(0.0, 0.0, 0.0, 1.0),
        help="Static camera orientation quaternion used when --pose-source static",
    )
    parser.add_argument(
        "--pose2d-z",
        type=float,
        default=0.0,
        help="Z position to use when pose source is robot_pose2d",
    )
    parser.add_argument(
        "--query-transport",
        choices=["auto", "none", "zmq", "ws"],
        default="none",
        help="Optional request-response transport for ad-hoc render queries",
    )
    parser.add_argument(
        "--query-endpoint",
        default=default_query_endpoints["zmq"],
        help=(
            "Bind endpoint for the query transport when enabled. "
            "Defaults: tcp://127.0.0.1:5588 for zmq, ws://127.0.0.1:8781/sim2real for ws"
        ),
    )
    parser.add_argument(
        "--query-poll-period",
        type=float,
        default=0.01,
        help="Polling period in seconds for the query transport",
    )
    parser.add_argument("--run-once", action="store_true", help="Publish one frame and exit")
    return parser


def run_cli(args: argparse.Namespace) -> None:
    """Run the headless render server from parsed CLI args."""
    query_transport_selection = resolve_query_transport_selection(
        args.query_transport,
        pose_source=args.pose_source,
        query_endpoint=args.query_endpoint,
    )
    args.query_transport = query_transport_selection.transport
    args.query_endpoint = query_transport_selection.endpoint
    args.query_transport_reason = query_transport_selection.reason

    ros2 = _import_ros2()
    rclpy = ros2["rclpy"]
    node_class = _build_node_class(ros2)

    rclpy.init(args=None)
    node = node_class(args)
    try:
        if args.run_once:
            while rclpy.ok() and not node.completed:
                rclpy.spin_once(node, timeout_sec=0.1)
        else:
            rclpy.spin(node)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the headless render server."""
    parser = build_parser()
    args = parser.parse_args(argv)
    run_cli(args)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
