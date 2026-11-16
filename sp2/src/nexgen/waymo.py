"""Small, dependency-free reader for Waymo Open Motion Scenario TFRecords.

The reader extracts only the scenario fields needed for mobility-derived V2V
topologies. Unknown protobuf fields, including map features, are skipped.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import google_crc32c

_TYPE_VEHICLE: int = 1


def _crc32c(data: bytes) -> int:
    return google_crc32c.value(data)


def _masked_crc32c(data: bytes) -> int:
    crc = _crc32c(data)
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    for shift in range(0, 70, 7):
        if offset >= len(data):
            raise ValueError("truncated protobuf varint")
        byte = data[offset]
        offset += 1
        if shift == 63 and byte > 1:
            raise ValueError("protobuf varint exceeds 64 bits")
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
    raise ValueError("protobuf varint exceeds 64 bits")


def _iter_protobuf_fields(data: bytes) -> Iterator[tuple[int, int, int | bytes]]:
    offset = 0
    while offset < len(data):
        tag, offset = _read_varint(data, offset)
        field_number, wire_type = tag >> 3, tag & 0x07
        if field_number == 0:
            raise ValueError("invalid protobuf field number 0")

        if wire_type == 0:
            value, offset = _read_varint(data, offset)
        elif wire_type == 1:
            end = offset + 8
            if end > len(data):
                raise ValueError("truncated protobuf fixed64 field")
            value = data[offset:end]
            offset = end
        elif wire_type == 2:
            length, offset = _read_varint(data, offset)
            end = offset + length
            if end > len(data):
                raise ValueError("truncated protobuf length-delimited field")
            value = data[offset:end]
            offset = end
        elif wire_type == 5:
            end = offset + 4
            if end > len(data):
                raise ValueError("truncated protobuf fixed32 field")
            value = data[offset:end]
            offset = end
        else:
            raise ValueError(f"unsupported protobuf wire type {wire_type}")
        yield field_number, wire_type, value


def read_tfrecord_records(path: str | Path) -> Iterator[bytes]:
    """Yield TFRecord payloads after validating both CRC32C checksums."""
    record_path = Path(path)
    file_size = record_path.stat().st_size
    with record_path.open("rb") as source:
        record_index = 0
        while True:
            header = source.read(12)
            if not header:
                return
            if len(header) != 12:
                raise ValueError(
                    f"{record_path.name}: truncated TFRecord header "
                    f"at record {record_index}"
                )

            length_bytes = header[:8]
            expected_length_crc = struct.unpack("<I", header[8:])[0]
            if expected_length_crc != _masked_crc32c(length_bytes):
                raise ValueError(
                    f"{record_path.name}: length CRC32C mismatch "
                    f"at record {record_index}"
                )

            length = struct.unpack("<Q", length_bytes)[0]
            remaining = file_size - source.tell()
            if length + 4 > remaining:
                raise ValueError(
                    f"{record_path.name}: truncated TFRecord payload "
                    f"at record {record_index}"
                )
            payload = source.read(length)
            data_crc_bytes = source.read(4)
            if len(payload) != length or len(data_crc_bytes) != 4:
                raise ValueError(
                    f"{record_path.name}: truncated TFRecord payload "
                    f"at record {record_index}"
                )
            expected_data_crc = struct.unpack("<I", data_crc_bytes)[0]
            if expected_data_crc != _masked_crc32c(payload):
                raise ValueError(
                    f"{record_path.name}: payload CRC32C mismatch "
                    f"at record {record_index}"
                )

            yield payload
            record_index += 1


@dataclass(frozen=True, slots=True)
class MotionState:
    """One object's state at a Waymo motion timestamp."""

    valid: bool
    x: float | None
    y: float | None
    velocity_x: float | None
    velocity_y: float | None

    @property
    def has_position(self) -> bool:
        return (
            self.valid
            and self.x is not None
            and self.y is not None
            and math.isfinite(self.x)
            and math.isfinite(self.y)
        )


@dataclass(frozen=True, slots=True)
class VehicleTrack:
    """A vehicle object and its states on the aligned scenario time axis."""

    track_id: int
    states: tuple[MotionState, ...]


@dataclass(frozen=True, slots=True)
class Scenario:
    """Mobility data needed to replay one complete Waymo scenario."""

    scenario_id: str
    timestamps: tuple[float, ...]
    vehicles: tuple[VehicleTrack, ...]
    total_track_count: int
    raw_timestamp_count: int
    track_state_count_min: int
    track_state_count_max: int
    current_time_index: int | None

    @property
    def aligned_frame_count(self) -> int:
        return len(self.timestamps)

    @property
    def frame_count_mismatch(self) -> bool:
        return (
            self.raw_timestamp_count != self.track_state_count_min
            or self.track_state_count_min != self.track_state_count_max
        )


def _parse_motion_state(data: bytes) -> MotionState:
    valid = False
    x: float | None = None
    y: float | None = None
    velocity_x: float | None = None
    velocity_y: float | None = None

    for field_number, wire_type, value in _iter_protobuf_fields(data):
        if field_number == 2 and wire_type == 1:
            x = struct.unpack("<d", value)[0]
        elif field_number == 3 and wire_type == 1:
            y = struct.unpack("<d", value)[0]
        elif field_number == 9 and wire_type == 5:
            velocity_x = struct.unpack("<f", value)[0]
        elif field_number == 10 and wire_type == 5:
            velocity_y = struct.unpack("<f", value)[0]
        elif field_number == 11 and wire_type == 0:
            valid = bool(value)

    return MotionState(
        valid=valid,
        x=x,
        y=y,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
    )


def _parse_track(data: bytes) -> tuple[int | None, int, tuple[MotionState, ...]]:
    track_id: int | None = None
    object_type = 0
    states: list[MotionState] = []

    for field_number, wire_type, value in _iter_protobuf_fields(data):
        if field_number == 1 and wire_type == 0:
            track_id = int(value)
            if track_id >= 2**31:
                track_id -= 2**32
        elif field_number == 2 and wire_type == 0:
            object_type = int(value)
        elif field_number == 3 and wire_type == 2:
            states.append(_parse_motion_state(value))

    return track_id, object_type, tuple(states)


def parse_scenario(data: bytes) -> Scenario:
    """Parse a Waymo Scenario protobuf, aligning every track to common frames.

    When the dataset has a timestamp/state-count mismatch, all timestamps and
    tracks are truncated to their common prefix. Raw lengths remain available
    on the returned object for reporting the discrepancy.
    """
    scenario_id: str | None = None
    timestamps: list[float] = []
    current_time_index: int | None = None
    vehicle_tracks: list[tuple[int | None, tuple[MotionState, ...]]] = []
    track_state_lengths: list[int] = []
    total_track_count = 0

    for field_number, wire_type, value in _iter_protobuf_fields(data):
        if field_number == 1 and wire_type == 1:
            timestamps.append(struct.unpack("<d", value)[0])
        elif field_number == 1 and wire_type == 2:
            if len(value) % 8:
                raise ValueError("malformed packed Waymo timestamp field")
            timestamps.extend(
                item[0] for item in struct.iter_unpack("<d", value)
            )
        elif field_number == 2 and wire_type == 2:
            total_track_count += 1
            track_id, object_type, states = _parse_track(value)
            track_state_lengths.append(len(states))
            if object_type == _TYPE_VEHICLE:
                vehicle_tracks.append((track_id, states))
        elif field_number == 5 and wire_type == 2:
            scenario_id = value.decode("utf-8")
        elif field_number == 10 and wire_type == 0:
            current_time_index = int(value)

    if not scenario_id:
        raise ValueError("Waymo Scenario is missing scenario_id")
    if not timestamps or not all(math.isfinite(timestamp) for timestamp in timestamps):
        raise ValueError(f"{scenario_id}: missing or non-finite timestamps")
    if any(
        later <= earlier
        for earlier, later in zip(timestamps, timestamps[1:])
    ):
        raise ValueError(f"{scenario_id}: timestamps are not strictly increasing")
    if not track_state_lengths:
        raise ValueError(f"{scenario_id}: Scenario has no tracks")

    aligned_frame_count = min(len(timestamps), min(track_state_lengths))
    if aligned_frame_count < 2:
        raise ValueError(f"{scenario_id}: fewer than two aligned frames")

    aligned_timestamps = tuple(timestamps[:aligned_frame_count])
    parsed_vehicles: list[VehicleTrack] = []
    seen_track_ids: set[int] = set()
    for track_id, states in vehicle_tracks:
        if track_id is None:
            raise ValueError(f"{scenario_id}: vehicle track is missing its ID")
        if track_id in seen_track_ids:
            raise ValueError(f"{scenario_id}: duplicate vehicle track ID {track_id}")
        seen_track_ids.add(track_id)

        aligned_states = states[:aligned_frame_count]
        for frame_index, state in enumerate(aligned_states):
            if state.valid and not state.has_position:
                raise ValueError(
                    f"{scenario_id}: valid vehicle track {track_id} "
                    f"has no finite position at frame {frame_index}"
                )
        parsed_vehicles.append(
            VehicleTrack(track_id=track_id, states=aligned_states)
        )

    return Scenario(
        scenario_id=scenario_id,
        timestamps=aligned_timestamps,
        vehicles=tuple(parsed_vehicles),
        total_track_count=total_track_count,
        raw_timestamp_count=len(timestamps),
        track_state_count_min=min(track_state_lengths),
        track_state_count_max=max(track_state_lengths),
        current_time_index=current_time_index,
    )


def read_scenarios(paths: Iterable[str | Path]) -> Iterator[tuple[Path, Scenario]]:
    """Read scenarios from one or more uncompressed Waymo TFRecord shards."""
    seen_scenario_ids: set[str] = set()
    for path in paths:
        shard_path = Path(path)
        for payload in read_tfrecord_records(shard_path):
            scenario = parse_scenario(payload)
            if scenario.scenario_id in seen_scenario_ids:
                raise ValueError(
                    f"duplicate scenario_id across shards: {scenario.scenario_id}"
                )
            seen_scenario_ids.add(scenario.scenario_id)
            yield shard_path, scenario
