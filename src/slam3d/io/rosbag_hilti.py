"""Read ROS 2 bags (sqlite3 .db3) without a ROS installation, via the `rosbags` package.

Designed for the Hilti-Trimble 2026 release (/cam0,/cam1 CompressedImage, /imu/data_raw Imu) but
topic names are configurable. Timestamps are taken from message headers (sensor time), not
bag receive time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np


def _typestore():
    from rosbags.typesys import Stores, get_typestore

    return get_typestore(Stores.ROS2_HUMBLE)


def stamp_to_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def bag_topics(bag_dir: str | Path) -> dict:
    from rosbags.rosbag2 import Reader

    with Reader(Path(bag_dir)) as reader:
        return {c.topic: (c.msgtype, c.msgcount) for c in reader.connections}


def iter_compressed_images(bag_dir: str | Path, topics: list[str]) -> Iterator[tuple[str, int, bytes]]:
    """Yield (topic, header_stamp_ns, jpeg_bytes) in bag order."""
    from rosbags.rosbag2 import Reader

    ts = _typestore()
    with Reader(Path(bag_dir)) as reader:
        conns = [c for c in reader.connections if c.topic in topics]
        if not conns:
            raise ValueError(f"none of {topics} in bag {bag_dir}")
        for conn, _t, raw in reader.messages(connections=conns):
            msg = ts.deserialize_cdr(raw, conn.msgtype)
            yield conn.topic, stamp_to_ns(msg.header.stamp), bytes(msg.data)


def iter_synced_pairs(bag_dir, topic0: str, topic1: str, tol_ns: int = 5_000_000):
    """Pair two image topics by header stamp (within tol). Yields (stamp0_ns, stamp1_ns, jpg0, jpg1)."""
    buf0: list = []
    buf1: list = []
    for topic, stamp, data in iter_compressed_images(bag_dir, [topic0, topic1]):
        (buf0 if topic == topic0 else buf1).append((stamp, data))
        while buf0 and buf1:
            s0, d0 = buf0[0]
            s1, d1 = buf1[0]
            if abs(s0 - s1) <= tol_ns:
                buf0.pop(0)
                buf1.pop(0)
                yield s0, s1, d0, d1
            elif s0 < s1:
                buf0.pop(0)
            else:
                buf1.pop(0)


def read_imu(bag_dir, topic: str = "/imu/data_raw") -> np.ndarray:
    """Return (N, 7) array: t_sec, gx, gy, gz [rad/s], ax, ay, az [m/s^2] in the IMU frame."""
    from rosbags.rosbag2 import Reader

    ts = _typestore()
    rows = []
    with Reader(Path(bag_dir)) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            return np.zeros((0, 7))
        for conn, _t, raw in reader.messages(connections=conns):
            m = ts.deserialize_cdr(raw, conn.msgtype)
            rows.append((
                stamp_to_ns(m.header.stamp) * 1e-9,
                m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z,
                m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z,
            ))
    return np.asarray(rows, dtype=np.float64)
