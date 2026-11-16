import struct

import pytest

from nexgen.waymo import parse_scenario, read_scenarios


def _varint(value: int) -> bytes:
    output = bytearray()
    while value > 0x7F:
        output.append((value & 0x7F) | 0x80)
        value >>= 7
    output.append(value)
    return bytes(output)


def _varint_field(field_number: int, value: int) -> bytes:
    return _varint(field_number << 3) + _varint(value)


def _bytes_field(field_number: int, value: bytes) -> bytes:
    return (
        _varint((field_number << 3) | 2)
        + _varint(len(value))
        + value
    )


def _fixed64_field(field_number: int, value: float) -> bytes:
    return _varint((field_number << 3) | 1) + struct.pack("<d", value)


def _motion_state(x: float, y: float, valid: bool = True) -> bytes:
    return b"".join(
        (
            _fixed64_field(2, x),
            _fixed64_field(3, y),
            _varint_field(11, int(valid)),
        )
    )


def _track(
    track_id: int, object_type: int, states: list[bytes]
) -> bytes:
    output = bytearray(
        _varint_field(1, track_id) + _varint_field(2, object_type)
    )
    for state in states:
        output.extend(_bytes_field(3, state))
    return bytes(output)


def _scenario(
    scenario_id: str,
    timestamps: list[float],
    tracks: list[bytes],
) -> bytes:
    output = bytearray(_bytes_field(5, scenario_id.encode("utf-8")))
    for timestamp in timestamps:
        output.extend(_fixed64_field(1, timestamp))
    for track in tracks:
        output.extend(_bytes_field(2, track))
    return bytes(output)


def _crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
    return crc ^ 0xFFFFFFFF


def _masked_crc32c(data: bytes) -> int:
    crc = _crc32c(data)
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def _write_tfrecord(path, payload: bytes) -> None:
    length = struct.pack("<Q", len(payload))
    path.write_bytes(
        length
        + struct.pack("<I", _masked_crc32c(length))
        + payload
        + struct.pack("<I", _masked_crc32c(payload))
    )


def test_parse_scenario_aligns_timestamps_and_all_track_states() -> None:
    payload = _scenario(
        "scene-a",
        [0.0, 0.1, 0.2],
        [
            _track(
                7,
                1,
                [
                    _motion_state(1.0, 2.0),
                    _motion_state(2.0, 2.0),
                    _motion_state(3.0, 2.0),
                    _motion_state(4.0, 2.0),
                ],
            ),
            _track(
                8,
                2,
                [_motion_state(0.0, 0.0), _motion_state(0.0, 0.0)],
            ),
        ],
    )

    scenario = parse_scenario(payload)

    assert scenario.scenario_id == "scene-a"
    assert scenario.aligned_frame_count == 2
    assert scenario.raw_timestamp_count == 3
    assert scenario.track_state_count_min == 2
    assert scenario.track_state_count_max == 4
    assert scenario.frame_count_mismatch
    assert scenario.vehicles[0].track_id == 7
    assert len(scenario.vehicles[0].states) == 2
    assert scenario.vehicles[0].states[1].x == 2.0


def test_parse_scenario_accepts_packed_timestamps() -> None:
    packed_values = struct.pack("<2d", 3.0, 3.1)
    payload = (
        _bytes_field(5, b"packed-scene")
        + _bytes_field(1, packed_values)
        + _bytes_field(
            2,
            _track(
                4,
                1,
                [_motion_state(0.0, 0.0), _motion_state(1.0, 0.0)],
            ),
        )
    )

    scenario = parse_scenario(payload)

    assert scenario.timestamps == (3.0, 3.1)
    assert scenario.aligned_frame_count == 2


def test_valid_vehicle_state_requires_a_finite_position() -> None:
    bad_state = _varint_field(11, 1)
    payload = _scenario(
        "bad-scene",
        [0.0, 0.1],
        [_track(3, 1, [bad_state, _motion_state(1.0, 1.0)])],
    )

    with pytest.raises(ValueError, match="bad-scene.*no finite position"):
        parse_scenario(payload)


def test_read_scenarios_checks_tfrecord_crc_and_preserves_source_path(tmp_path) -> None:
    shard = tmp_path / "training_20s.tfrecord-00000-of-01000"
    payload = _scenario(
        "scene-a",
        [0.0, 0.1],
        [_track(1, 1, [_motion_state(0.0, 0.0), _motion_state(1.0, 0.0)])],
    )
    _write_tfrecord(shard, payload)

    result = list(read_scenarios([shard]))

    assert len(result) == 1
    assert result[0][0] == shard
    assert result[0][1].scenario_id == "scene-a"

    corrupted = bytearray(shard.read_bytes())
    corrupted[-1] ^= 0x01
    shard.write_bytes(corrupted)
    with pytest.raises(ValueError, match="payload CRC32C mismatch"):
        list(read_scenarios([shard]))


def test_read_scenarios_rejects_duplicate_ids_across_shards(tmp_path) -> None:
    payload = _scenario(
        "same-id",
        [0.0, 0.1],
        [_track(1, 1, [_motion_state(0.0, 0.0), _motion_state(1.0, 0.0)])],
    )
    first = tmp_path / "first.tfrecord"
    second = tmp_path / "second.tfrecord"
    _write_tfrecord(first, payload)
    _write_tfrecord(second, payload)

    with pytest.raises(ValueError, match="duplicate scenario_id"):
        list(read_scenarios([first, second]))
