"""货源内存索引：启动时载入 JSONL，查询时用向量化 Haversine + argpartition 取最近 K 条。"""

from __future__ import annotations

import heapq
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


class CargoRepository:
    """货源仓库：维护 pending(未上架) 与 online(已上架) 两级池。"""

    __slots__ = (
        "_path",
        "_pending",
        "_pending_cursor",
        "_online",
        "_online_expire_heap",
        "_online_ids",
        "_online_lat",
        "_online_lng",
        "_online_dirty",
        "_earth_radius_km",
        "_simulation_start_dt",
        "_current_time_minutes",
    )

    def __init__(self, dataset_path: Path, earth_radius_km: float = 6371.0) -> None:
        self._path = dataset_path
        self._pending: list[tuple[int, int, str, dict[str, Any]]] = []
        self._pending_cursor = 0
        self._online: dict[str, tuple[int, dict[str, Any]]] = {}
        self._online_expire_heap: list[tuple[int, str]] = []
        self._online_ids: list[str] = []
        self._online_lat = np.empty(0, dtype=np.float64)
        self._online_lng = np.empty(0, dtype=np.float64)
        self._online_dirty = True
        self._earth_radius_km = earth_radius_km
        self._simulation_start_dt = datetime(2026, 3, 1, 0, 0, 0)
        self._current_time_minutes = 0

    @property
    def size(self) -> int:
        return (len(self._pending) - self._pending_cursor) + len(self._online)

    def load(self) -> None:
        if not self._path.is_file():
            raise FileNotFoundError(f"货源文件不存在: {self._path}")
        pending: list[tuple[int, int, str, dict[str, Any]]] = []
        seen_ids: set[str] = set()
        with self._path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                try:
                    cargo_id = str(obj.get("cargo_id", "")).strip()
                    if not cargo_id:
                        raise ValueError(f"货源文件第 {line_no} 行缺少有效 cargo_id")
                    if cargo_id in seen_ids:
                        raise ValueError(f"货源文件出现重复 cargo_id: {cargo_id}")
                    seen_ids.add(cargo_id)
                    create_minutes = self._to_simulation_minutes(str(obj["create_time"]))
                    remove_minutes = self._to_simulation_minutes(str(obj["remove_time"]))
                    if remove_minutes < create_minutes:
                        remove_minutes = create_minutes
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"货源文件第 {line_no} 行字段无效") from exc
                pending.append((create_minutes, remove_minutes, cargo_id, obj))

        pending.sort(key=lambda item: item[0])
        self._pending = pending
        self._pending_cursor = 0
        self._online = {}
        self._online_expire_heap = []
        self._current_time_minutes = 0
        self._online_dirty = True
        self._rebuild_online_cache_if_needed()
        self.sync_time_minutes(0)

    def get_by_id(self, cargo_id: str) -> dict[str, Any] | None:
        item = self._online.get(cargo_id)
        if item is None:
            return None
        return item[1]

    def remove_by_id(self, cargo_id: str) -> dict[str, Any] | None:
        item = self._online.pop(cargo_id, None)
        if item is None:
            return None
        self._online_dirty = True
        cargo = item[1]
        return cargo

    def nearest_pickup_km(
        self,
        latitude: float,
        longitude: float,
        current_time_minutes: int,
        k: int = 100,
    ) -> list[tuple[float, dict[str, Any]]]:
        """返回当前时刻 online 池中距离最近的至多 k 条。"""
        self.sync_time_minutes(current_time_minutes)
        self._rebuild_online_cache_if_needed()
        n = len(self._online_ids)
        if n == 0:
            return []
        take = min(k, n)
        dists = self._haversine_km(latitude, longitude, self._online_lat, self._online_lng)
        kth = take - 1
        idx = np.argpartition(dists, kth)[:take]
        idx = idx[np.argsort(dists[idx])]
        out: list[tuple[float, dict[str, Any]]] = []
        for i in idx:
            cargo_id = self._online_ids[int(i)]
            record = self._online[cargo_id][1]
            out.append((float(dists[int(i)]), record))
        return out

    def sync_time_minutes(self, current_time_minutes: int) -> None:
        if current_time_minutes < 0:
            current_time_minutes = 0
        if current_time_minutes < self._current_time_minutes:
            return
        self._current_time_minutes = current_time_minutes

        while self._pending_cursor < len(self._pending):
            create_minutes, remove_minutes, cargo_id, record = self._pending[self._pending_cursor]
            if create_minutes > current_time_minutes:
                break
            self._pending_cursor += 1
            if remove_minutes < current_time_minutes:
                continue
            self._online[cargo_id] = (remove_minutes, record)
            heapq.heappush(self._online_expire_heap, (remove_minutes, cargo_id))
            self._online_dirty = True

        while self._online_expire_heap and self._online_expire_heap[0][0] < current_time_minutes:
            remove_minutes, cargo_id = heapq.heappop(self._online_expire_heap)
            online_item = self._online.get(cargo_id)
            if online_item is None:
                continue
            if online_item[0] != remove_minutes:
                continue
            del self._online[cargo_id]
            self._online_dirty = True

    def _rebuild_online_cache_if_needed(self) -> None:
        if not self._online_dirty:
            return
        ids = list(self._online.keys())
        lat: list[float] = []
        lng: list[float] = []
        for cargo_id in ids:
            start = self._online[cargo_id][1]["start"]
            lat.append(float(start["lat"]))
            lng.append(float(start["lng"]))
        self._online_ids = ids
        self._online_lat = np.asarray(lat, dtype=np.float64)
        self._online_lng = np.asarray(lng, dtype=np.float64)
        self._online_dirty = False

    def wall_time_to_simulation_minutes(self, text: str) -> int:
        """将 ``%Y-%m-%d %H:%M:%S`` 墙钟时间转为相对仿真纪元（2026-03-01 00:00）的分钟数。"""
        return self._to_simulation_minutes(text)

    def _to_simulation_minutes(self, text: str) -> int:
        dt = datetime.strptime(text.strip(), "%Y-%m-%d %H:%M:%S")
        delta = dt - self._simulation_start_dt
        return int(delta.total_seconds() // 60)

    def _haversine_km(self, lat: float, lng: float, lat_arr: np.ndarray, lng_arr: np.ndarray) -> np.ndarray:
        """球面大圆距离（公里），向量化。"""
        r = self._earth_radius_km
        p1 = np.radians(lat)
        l1 = np.radians(lng)
        p2 = np.radians(lat_arr)
        l2 = np.radians(lng_arr)
        dp = p2 - p1
        dl = l2 - l1
        sin_dp = np.sin(dp * 0.5)
        sin_dl = np.sin(dl * 0.5)
        h = sin_dp * sin_dp + np.cos(p1) * np.cos(p2) * (sin_dl * sin_dl)
        h = np.minimum(1.0, np.maximum(0.0, h))
        return 2.0 * r * np.arcsin(np.sqrt(h))
