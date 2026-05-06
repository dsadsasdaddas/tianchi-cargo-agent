"""仿真动作纯函数：编排器、决策环境与统计脚本共用同一套规则。"""

from __future__ import annotations

import math
from typing import Any

from simkit.cargo_repository import CargoRepository
from simkit.driver_state_manager import DriverStateManager


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius_km = 6371.0
    p1 = math.radians(lat1)
    l1 = math.radians(lng1)
    p2 = math.radians(lat2)
    l2 = math.radians(lng2)
    dp = p2 - p1
    dl = l2 - l1
    h = math.sin(dp * 0.5) ** 2 + math.cos(p1) * math.cos(p2) * (math.sin(dl * 0.5) ** 2)
    h = min(1.0, max(0.0, h))
    return 2.0 * radius_km * math.asin(math.sqrt(h))


def distance_to_minutes(distance_km: float, speed_km_per_hour: float) -> int:
    if distance_km <= 0:
        return 1
    return max(1, math.ceil((distance_km / speed_km_per_hour) * 60))


def parse_cost_time_to_minutes(cargo: dict[str, Any]) -> int:
    raw_value = cargo.get("cost_time_minutes")
    try:
        minutes = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"货源 cost_time_minutes 数值无效: {raw_value}") from exc
    if minutes < 0:
        raise ValueError(f"货源 cost_time_minutes 不能为负数: {raw_value}")
    return minutes


def _parse_load_window_minutes(cargo: dict[str, Any], repo: CargoRepository) -> tuple[int, int] | None:
    """解析 ``load_time`` 为仿真分钟区间 ``[start, end]``；无字段则不做装货窗约束。"""
    raw = cargo.get("load_time")
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError(f"货源 load_time 须为数组，当前为 {type(raw).__name__}")
    if len(raw) != 2:
        raise ValueError(f"货源 load_time 须为长度 2 的数组: {raw!r}")
    a = str(raw[0]).strip()
    b = str(raw[1]).strip()
    if not a or not b:
        raise ValueError(f"货源 load_time 元素不能为空字符串: {raw!r}")
    start_m = repo.wall_time_to_simulation_minutes(a)
    end_m = repo.wall_time_to_simulation_minutes(b)
    if end_m < start_m:
        raise ValueError(f"货源 load_time 结束早于开始: {raw!r}")
    return (start_m, end_m)


def _estimate_successful_take_order_end_minute(
    t0_minutes: int,
    distance_pickup_km: float,
    reposition_speed_km_per_hour: float,
    cargo: dict[str, Any],
    repo: CargoRepository,
) -> int | None:
    """在能成功装货并完成干线的前提下，预估完单后的 ``simulation_progress_minutes``；若空驶到达时已错过装货窗则返回 None。"""
    dead_minutes = (
        distance_to_minutes(distance_pickup_km, reposition_speed_km_per_hour)
        if distance_pickup_km > 1e-6
        else 0
    )
    arrival_minutes = t0_minutes + dead_minutes
    window = _parse_load_window_minutes(cargo, repo)
    if window is not None:
        load_start_m, load_end_m = window
        if arrival_minutes > load_end_m:
            return None
        ready_minutes = load_start_m if arrival_minutes < load_start_m else arrival_minutes
    else:
        ready_minutes = arrival_minutes
    duration_minutes = parse_cost_time_to_minutes(cargo)
    return ready_minutes + duration_minutes


def normalize_cargo_price_to_yuan(cargo: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(cargo)
    if "price" in normalized:
        normalized["price"] = round(float(normalized["price"]) / 100.0, 2)
    return normalized


def query_cargo(
    repo: CargoRepository,
    manager: DriverStateManager,
    driver_id: str,
    latitude: float,
    longitude: float,
    k: int,
) -> dict[str, Any]:
    manager.ensure_active_driver(driver_id)
    current_time_minutes = manager.get_simulation_progress_minutes()
    pairs = repo.nearest_pickup_km(latitude, longitude, current_time_minutes=current_time_minutes, k=k)
    items = [{"distance_km": float(d), "cargo": normalize_cargo_price_to_yuan(c)} for d, c in pairs]
    return {"driver_id": driver_id, "items": items}


def apply_cargo_query_scan_cost(
    repo: CargoRepository,
    manager: DriverStateManager,
    driver_id: str,
    items_count: int,
    *,
    cargo_view_batch_size: int,
) -> int:
    query_cost_minutes = math.ceil(items_count / cargo_view_batch_size) if items_count else 0
    if query_cost_minutes > 0:
        manager.advance_progress(driver_id, query_cost_minutes)
        repo.sync_time_minutes(manager.get_simulation_progress_minutes())
    return manager.get_simulation_progress_minutes()


def take_order(
    repo: CargoRepository,
    manager: DriverStateManager,
    driver_id: str,
    cargo_id: str,
    reposition_speed_km_per_hour: float,
    simulation_horizon_minutes: int | None = None,
) -> dict[str, Any]:
    """接单：先空驶至装货点；若早于装货时间窗则在装货地等待至窗口开始；晚于窗口结束则本票失败并下掉货源。
    成功时按 cost_time_minutes 完成装运卸并到卸货点。
    若传入 ``simulation_horizon_minutes``，则在动身前预估完单时刻；若超过上界，仍允许接单执行，
    但结果中会标记 ``income_eligible=False``（用于收益脚本不计该单收入）。"""
    cargo = repo.get_by_id(cargo_id)
    if cargo is None:
        raise ValueError(f"cargo_id 不存在: {cargo_id}")
    start = cargo.get("start")
    start_latitude = float(start["lat"])
    start_longitude = float(start["lng"])
    current = manager.get_driver_status(driver_id)
    distance_pickup_km = haversine_km(
        float(current["current_lat"]),
        float(current["current_lng"]),
        start_latitude,
        start_longitude,
    )
    income_eligible = True
    if simulation_horizon_minutes is not None:
        t0_minutes = manager.get_simulation_progress_minutes()
        finish_minutes = _estimate_successful_take_order_end_minute(
            t0_minutes,
            distance_pickup_km,
            reposition_speed_km_per_hour,
            cargo,
            repo,
        )
        if finish_minutes is not None and finish_minutes > simulation_horizon_minutes:
            income_eligible = False
    if distance_pickup_km > 1e-6:
        pickup_minutes = distance_to_minutes(distance_pickup_km, reposition_speed_km_per_hour)
        manager.update_driver_position(driver_id, start_latitude, start_longitude)
        manager.advance_progress(driver_id, pickup_minutes)
        repo.sync_time_minutes(manager.get_simulation_progress_minutes())
    else:
        manager.update_driver_position(driver_id, start_latitude, start_longitude)

    arrival_minutes = manager.get_simulation_progress_minutes()
    window = _parse_load_window_minutes(cargo, repo)
    if window is not None:
        load_start_m, load_end_m = window
        if arrival_minutes > load_end_m:
            removed = repo.remove_by_id(cargo_id)
            if removed is None:
                raise ValueError(f"cargo_id 已从系统移除: {cargo_id}")
            repo.sync_time_minutes(manager.get_simulation_progress_minutes())
            return {
                "accepted": False,
                "detail": "load_time_window_expired",
                "driver_id": driver_id,
                "cargo_id": cargo_id,
                "simulation_progress_minutes": manager.get_simulation_progress_minutes(),
                "simulation_wall_time": manager.get_simulation_wall_time(),
                "pickup_deadhead_km": round(float(distance_pickup_km), 2),
                "haul_distance_km": 0,
            }
        if arrival_minutes < load_start_m:
            wait_minutes = load_start_m - arrival_minutes
            manager.advance_progress(driver_id, wait_minutes)
            repo.sync_time_minutes(manager.get_simulation_progress_minutes())

    duration_minutes = parse_cost_time_to_minutes(cargo)
    end = cargo.get("end")
    end_latitude = float(end["lat"])
    end_longitude = float(end["lng"])
    result = manager.take_order(
        driver_id,
        cargo_id,
        duration_minutes=duration_minutes,
        end_latitude=end_latitude,
        end_longitude=end_longitude,
    )
    removed = repo.remove_by_id(cargo_id)
    if removed is None:
        raise ValueError(f"cargo_id 已从系统移除: {cargo_id}")
    repo.sync_time_minutes(int(result["simulation_progress_minutes"]))
    haul_km = haversine_km(start_latitude, start_longitude, end_latitude, end_longitude)
    result["pickup_deadhead_km"] = round(float(distance_pickup_km), 2)
    result["haul_distance_km"] = round(float(haul_km), 2)
    result["simulation_wall_time"] = manager.get_simulation_wall_time()
    result["income_eligible"] = income_eligible
    return result


def wait(
    repo: CargoRepository,
    manager: DriverStateManager,
    driver_id: str,
    duration_minutes: int,
) -> dict[str, Any]:
    manager.advance_progress(driver_id, duration_minutes)
    repo.sync_time_minutes(manager.get_simulation_progress_minutes())
    return {
        "simulation_progress_minutes": manager.get_simulation_progress_minutes(),
        "simulation_wall_time": manager.get_simulation_wall_time(),
    }


def reposition(
    repo: CargoRepository,
    manager: DriverStateManager,
    driver_id: str,
    latitude: float,
    longitude: float,
    speed_km_per_hour: float,
) -> dict[str, Any]:
    current = manager.get_driver_status(driver_id)
    distance_km = haversine_km(current["current_lat"], current["current_lng"], latitude, longitude)
    duration_minutes = distance_to_minutes(distance_km, speed_km_per_hour)
    manager.update_driver_position(driver_id, latitude, longitude)
    manager.advance_progress(driver_id, duration_minutes)
    repo.sync_time_minutes(manager.get_simulation_progress_minutes())
    return {
        "current_lat": latitude,
        "current_lng": longitude,
        "simulation_progress_minutes": manager.get_simulation_progress_minutes(),
        "simulation_wall_time": manager.get_simulation_wall_time(),
        "distance_km": round(float(distance_km), 2),
    }
