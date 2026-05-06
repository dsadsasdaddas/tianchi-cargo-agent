"""司机状态管理：维护司机基础信息和仿真运行状态。"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

_SIMULATION_EPOCH = datetime(2026, 3, 1, 0, 0, 0)


class DriverStateManager:
    """内存状态管理器。仿真时间单位：分钟（minutes）。"""

    def __init__(self, drivers_path: Path) -> None:
        self._drivers_path = drivers_path
        self._drivers: dict[str, dict[str, Any]] = {}
        self._current_driver_id: str | None = None
        self._simulation_started = False
        self._simulation_progress_minutes = 0
        self._current_order_by_driver: dict[str, str | None] = {}
        self._taken_cargo_ids: set[str] = set()
        self._completed_orders_by_driver: dict[str, int] = {}

    def load(self) -> None:
        if not self._drivers_path.is_file():
            raise FileNotFoundError(f"司机文件不存在: {self._drivers_path}")
        raw = json.loads(self._drivers_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list) or not raw:
            raise ValueError("drivers.json 必须是非空数组")

        drivers: dict[str, dict[str, Any]] = {}
        for idx, item in enumerate(raw, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"drivers.json 第 {idx} 项必须是对象")
            driver_id = item.get("driver_id")
            if not isinstance(driver_id, str) or not driver_id:
                raise ValueError(f"drivers.json 第 {idx} 项缺少有效 driver_id")
            if driver_id in drivers:
                raise ValueError(f"drivers.json 出现重复 driver_id: {driver_id}")
            drivers[driver_id] = item

        self._drivers = drivers
        self._current_driver_id = next(iter(drivers))
        self._simulation_started = False
        self._simulation_progress_minutes = 0
        self._current_order_by_driver = {driver_id: None for driver_id in drivers}
        self._taken_cargo_ids = set()
        self._completed_orders_by_driver = {driver_id: 0 for driver_id in drivers}

    def start_simulation(self, driver_id: str | None, progress_minutes: int = 0) -> dict[str, Any]:
        if progress_minutes < 0:
            raise ValueError("progress_minutes 不能小于 0")
        chosen_id = driver_id or self._current_driver_id
        if not chosen_id or chosen_id not in self._drivers:
            raise ValueError("driver_id 不存在")
        self._current_driver_id = chosen_id
        self._simulation_started = True
        self._simulation_progress_minutes = int(progress_minutes)
        return self.get_system_state()

    def start_simulation_minutes(self, driver_id: str | None, progress_minutes: int = 0) -> dict[str, Any]:
        return self.start_simulation(driver_id=driver_id, progress_minutes=progress_minutes)

    def get_system_state(self) -> dict[str, Any]:
        return {
            "simulation_started": self._simulation_started,
            "simulation_progress_minutes": self._simulation_progress_minutes,
            "simulation_wall_time": self.get_simulation_wall_time(),
            "current_driver_id": self._current_driver_id,
            "drivers_total": len(self._drivers),
        }

    def list_driver_ids(self) -> list[str]:
        return list(self._drivers.keys())

    def get_driver_status(self, driver_id: str) -> dict[str, Any]:
        profile = self._drivers.get(driver_id)
        if profile is None:
            raise KeyError(driver_id)

        simulation_progress_minutes = self._simulation_progress_minutes
        simulation_wall_time = (
            _SIMULATION_EPOCH + timedelta(minutes=int(simulation_progress_minutes))
        ).strftime("%Y-%m-%d %H:%M:%S")

        return {
            "driver_id": driver_id,
            "name": profile.get("name", ""),
            "vehicle_no": profile.get("vehicle_no", ""),
            "truck_length": profile.get("truck_length", ""),
            "preferences": list(profile.get("preferences", [])),
            "current_lat": float(profile.get("current_lat", 0.0)),
            "current_lng": float(profile.get("current_lng", 0.0)),
            "is_current": driver_id == self._current_driver_id,
            "simulation_started": self._simulation_started,
            "simulation_progress_minutes": simulation_progress_minutes,
            "simulation_wall_time": simulation_wall_time,
            "current_order_cargo_id": self._current_order_by_driver.get(driver_id),
            "completed_order_count": self._completed_orders_by_driver.get(driver_id, 0),
        }

    def ensure_active_driver(self, driver_id: str) -> None:
        if driver_id not in self._drivers:
            raise ValueError(f"driver_id 不存在: {driver_id}")
        if not self._simulation_started:
            raise ValueError("模拟尚未开始，请先调用 start_simulation")
        if driver_id != self._current_driver_id:
            raise ValueError(f"当前正在模拟司机为 {self._current_driver_id}，不允许操作 {driver_id}")

    def update_driver_position(self, driver_id: str, latitude: float, longitude: float) -> None:
        self.ensure_active_driver(driver_id)
        profile = self._drivers[driver_id]
        profile["current_lat"] = float(latitude)
        profile["current_lng"] = float(longitude)

    def advance_progress(self, driver_id: str, duration_minutes: int) -> int:
        self.ensure_active_driver(driver_id)
        if duration_minutes < 0:
            raise ValueError("duration_minutes 不能小于 0")
        self._simulation_progress_minutes += int(duration_minutes)
        return self._simulation_progress_minutes

    def get_simulation_progress_minutes(self) -> int:
        return self._simulation_progress_minutes

    def get_simulation_wall_time(self) -> str:
        return (_SIMULATION_EPOCH + timedelta(minutes=int(self._simulation_progress_minutes))).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    def take_order(
        self,
        driver_id: str,
        cargo_id: str,
        duration_minutes: int,
        end_latitude: float,
        end_longitude: float,
    ) -> dict[str, Any]:
        self.ensure_active_driver(driver_id)
        if not cargo_id:
            raise ValueError("cargo_id 不能为空")
        if duration_minutes < 0:
            raise ValueError("duration_minutes 不能小于 0")
        if cargo_id in self._taken_cargo_ids:
            raise ValueError(f"cargo_id 已被占用: {cargo_id}")

        self._taken_cargo_ids.add(cargo_id)
        self._completed_orders_by_driver[driver_id] = self._completed_orders_by_driver.get(driver_id, 0) + 1
        self.advance_progress(driver_id, duration_minutes)
        self._current_order_by_driver[driver_id] = None

        profile = self._drivers[driver_id]
        profile["current_lat"] = float(end_latitude)
        profile["current_lng"] = float(end_longitude)

        return {
            "accepted": True,
            "detail": "接单后已完成、推进时间并更新到卸货地",
            "driver_id": driver_id,
            "cargo_id": cargo_id,
            "simulation_progress_minutes": self._simulation_progress_minutes,
            "simulation_wall_time": self.get_simulation_wall_time(),
        }
