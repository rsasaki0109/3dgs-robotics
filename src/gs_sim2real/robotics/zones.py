"""Semantic zone parsing and costmap rasterization for Dynamic Map Viewer robotics."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SemanticZone:
    """A semantic zone in world X/Z coordinates."""

    zone_id: str
    label: str
    shape: str
    center_x: float
    center_z: float
    size_x: float
    size_z: float
    radius: float
    cost: int
    tags: tuple[str, ...]


@dataclass(frozen=True)
class SemanticZoneMap:
    """A semantic zone collection with costmap raster settings."""

    frame_id: str
    resolution: float
    min_x: float
    max_x: float
    min_z: float
    max_z: float
    default_cost: int
    zones: tuple[SemanticZone, ...]


@dataclass(frozen=True)
class RasterizedCostMap:
    """A flattened occupancy/cost map."""

    frame_id: str
    resolution: float
    width: int
    height: int
    origin_x: float
    origin_z: float
    data: tuple[int, ...]


def clamp_cost(value: int | float) -> int:
    """Clamp cost into ROS occupancy range 0..100."""
    return max(0, min(100, int(round(value))))


def parse_world_position(value: Any) -> tuple[float, float]:
    """Parse `[x, z]` or `[x, y, z]` into world X/Z coordinates."""
    if not isinstance(value, (list, tuple)):
        raise ValueError("Position must be a list or tuple")

    if len(value) == 2:
        x, z = value
    elif len(value) >= 3:
        x = value[0]
        z = value[2]
    else:
        raise ValueError("Position must have at least 2 elements")

    return float(x), float(z)


def build_semantic_zone_map(payload: dict[str, Any]) -> SemanticZoneMap:
    """Build a zone map from JSON-like payload."""
    bounds = payload.get("bounds") or {}
    resolution = float(payload.get("resolution", 0.5))
    if resolution <= 0:
        raise ValueError("resolution must be > 0")

    min_x = float(bounds.get("minX"))
    max_x = float(bounds.get("maxX"))
    min_z = float(bounds.get("minZ"))
    max_z = float(bounds.get("maxZ"))
    if max_x <= min_x or max_z <= min_z:
        raise ValueError("bounds must define a positive X/Z area")

    zones: list[SemanticZone] = []
    for index, zone_payload in enumerate(payload.get("zones") or []):
        zone_id = str(zone_payload.get("id") or f"zone-{index}")
        label = str(zone_payload.get("label") or zone_id)
        shape = str(zone_payload.get("shape") or "rect").strip().lower()
        center_x, center_z = parse_world_position(zone_payload.get("center") or [0, 0, 0])
        cost = clamp_cost(zone_payload.get("cost", 100))
        tags = tuple(str(tag) for tag in zone_payload.get("tags") or [])

        if shape == "rect":
            size = zone_payload.get("size") or [1, 1]
            if len(size) != 2:
                raise ValueError(f"rect zone '{zone_id}' size must have 2 elements")
            size_x = float(size[0])
            size_z = float(size[1])
            if size_x <= 0 or size_z <= 0:
                raise ValueError(f"rect zone '{zone_id}' size must be positive")
            zone = SemanticZone(
                zone_id=zone_id,
                label=label,
                shape=shape,
                center_x=center_x,
                center_z=center_z,
                size_x=size_x,
                size_z=size_z,
                radius=0.0,
                cost=cost,
                tags=tags,
            )
        elif shape == "circle":
            radius = float(zone_payload.get("radius", 0))
            if radius <= 0:
                raise ValueError(f"circle zone '{zone_id}' radius must be positive")
            zone = SemanticZone(
                zone_id=zone_id,
                label=label,
                shape=shape,
                center_x=center_x,
                center_z=center_z,
                size_x=0.0,
                size_z=0.0,
                radius=radius,
                cost=cost,
                tags=tags,
            )
        else:
            raise ValueError(f"unsupported zone shape: {shape}")

        zones.append(zone)

    return SemanticZoneMap(
        frame_id=str(payload.get("frameId") or "dreamwalker_map"),
        resolution=resolution,
        min_x=min_x,
        max_x=max_x,
        min_z=min_z,
        max_z=max_z,
        default_cost=clamp_cost(payload.get("defaultCost", 0)),
        zones=tuple(zones),
    )


def load_semantic_zone_map(path: str | Path) -> SemanticZoneMap:
    """Load semantic zones from a JSON file."""
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return build_semantic_zone_map(payload)


def point_in_zone(zone: SemanticZone, x: float, z: float) -> bool:
    """Check whether a world point is inside a zone."""
    if zone.shape == "rect":
        return abs(x - zone.center_x) <= zone.size_x / 2 and abs(z - zone.center_z) <= zone.size_z / 2

    if zone.shape == "circle":
        return math.hypot(x - zone.center_x, z - zone.center_z) <= zone.radius

    raise ValueError(f"unsupported zone shape: {zone.shape}")


def find_zone_hits(zone_map: SemanticZoneMap, x: float, z: float) -> tuple[SemanticZone, ...]:
    """Find all zones containing a world point."""
    return tuple(zone for zone in zone_map.zones if point_in_zone(zone, x, z))


def build_zone_summary_payload(zone_map: SemanticZoneMap) -> dict[str, Any]:
    """Build a JSON-serializable summary for semantic zones."""
    return {
        "frameId": zone_map.frame_id,
        "resolution": zone_map.resolution,
        "bounds": {
            "minX": zone_map.min_x,
            "maxX": zone_map.max_x,
            "minZ": zone_map.min_z,
            "maxZ": zone_map.max_z,
        },
        "defaultCost": zone_map.default_cost,
        "zoneCount": len(zone_map.zones),
        "zones": [
            {
                "id": zone.zone_id,
                "label": zone.label,
                "shape": zone.shape,
                "center": [zone.center_x, zone.center_z],
                "size": [zone.size_x, zone.size_z] if zone.shape == "rect" else None,
                "radius": zone.radius if zone.shape == "circle" else None,
                "cost": zone.cost,
                "tags": list(zone.tags),
            }
            for zone in zone_map.zones
        ],
    }


def build_zone_hit_payload(zone_hits: tuple[SemanticZone, ...], x: float, z: float) -> dict[str, Any]:
    """Build a JSON-serializable payload for the current zone hits."""
    return {
        "position": {"x": x, "z": z},
        "zones": [
            {
                "id": zone.zone_id,
                "label": zone.label,
                "cost": zone.cost,
                "tags": list(zone.tags),
            }
            for zone in zone_hits
        ],
    }


def rasterize_costmap(zone_map: SemanticZoneMap) -> RasterizedCostMap:
    """Rasterize zones into a costmap."""
    width = max(1, math.ceil((zone_map.max_x - zone_map.min_x) / zone_map.resolution))
    height = max(1, math.ceil((zone_map.max_z - zone_map.min_z) / zone_map.resolution))
    data: list[int] = []

    for grid_y in range(height):
        world_z = zone_map.min_z + (grid_y + 0.5) * zone_map.resolution
        for grid_x in range(width):
            world_x = zone_map.min_x + (grid_x + 0.5) * zone_map.resolution
            cost = zone_map.default_cost
            for zone in zone_map.zones:
                if point_in_zone(zone, world_x, world_z):
                    cost = max(cost, zone.cost)
            data.append(clamp_cost(cost))

    return RasterizedCostMap(
        frame_id=zone_map.frame_id,
        resolution=zone_map.resolution,
        width=width,
        height=height,
        origin_x=zone_map.min_x,
        origin_z=zone_map.min_z,
        data=tuple(data),
    )
