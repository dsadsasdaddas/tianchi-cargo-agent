"""计算 2026 年 3 月每个司机累计收益（与仿真结果 JSONL 对齐）。

输出 `monthly_income_202603.json`：`drivers` 为数组，每项含该司机 `driver_id`、`income`、`token_usage`；
全量汇总在 `summary`。
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_DEMO_ROOT = Path(__file__).resolve().parent
if str(_DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEMO_ROOT))

from simkit.simulation_actions import haversine_km


PROJECT_ROOT = Path(__file__).resolve().parent
SERVER_ROOT = PROJECT_ROOT / "server"
RESULTS_DIR = PROJECT_ROOT / "results"
CARGO_DATASET = SERVER_ROOT / "data" / "cargo_dataset.jsonl"
DRIVERS_DATASET = SERVER_ROOT / "data" / "drivers.json"
CONFIG_PATH = SERVER_ROOT / "config" / "config.json"
OUTPUT_FILE = RESULTS_DIR / "monthly_income_202603.json"
RUN_SUMMARY_FILE = RESULTS_DIR / "run_summary_202603.json"
_SIMULATION_EPOCH = datetime(2026, 3, 1, 0, 0, 0)


def load_cargo_map(path: Path) -> dict[str, dict[str, Any]]:
    cargo_map: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            cargo_id = str(item.get("cargo_id", "")).strip()
            if not cargo_id:
                continue
            try:
                start = item.get("start", {})
                end = item.get("end", {})
                distance_km = haversine_km(
                    float(start["lat"]),
                    float(start["lng"]),
                    float(end["lat"]),
                    float(end["lng"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"货源文件第 {line_no} 行 start/end 坐标无效") from exc
            create_time = str(item.get("create_time", "")).strip()
            remove_time = str(item.get("remove_time", "")).strip()
            if not create_time or not remove_time:
                raise ValueError(f"货源文件第 {line_no} 行缺少 create_time/remove_time")
            create_minutes = int((_SIMULATION_EPOCH.fromisoformat(create_time.replace(" ", "T")) - _SIMULATION_EPOCH).total_seconds() // 60)
            remove_minutes = int((_SIMULATION_EPOCH.fromisoformat(remove_time.replace(" ", "T")) - _SIMULATION_EPOCH).total_seconds() // 60)
            cost_time_minutes = int(item.get("cost_time_minutes", 0) or 0)
            load_window = item.get("load_time")
            load_start_minutes: int | None = None
            load_end_minutes: int | None = None
            if load_window is not None:
                if not isinstance(load_window, list) or len(load_window) != 2:
                    raise ValueError(f"货源文件第 {line_no} 行 load_time 格式无效")
                left = str(load_window[0]).strip()
                right = str(load_window[1]).strip()
                if not left or not right:
                    raise ValueError(f"货源文件第 {line_no} 行 load_time 为空")
                load_start_minutes = int((_SIMULATION_EPOCH.fromisoformat(left.replace(" ", "T")) - _SIMULATION_EPOCH).total_seconds() // 60)
                load_end_minutes = int((_SIMULATION_EPOCH.fromisoformat(right.replace(" ", "T")) - _SIMULATION_EPOCH).total_seconds() // 60)
                if load_end_minutes < load_start_minutes:
                    # 数据集中存在少量异常时间窗（结束早于开始），这里降级为“无装货窗约束”避免统计脚本整体中断。
                    load_start_minutes = None
                    load_end_minutes = None
            cargo_map[cargo_id] = {
                "price": float(item.get("price", 0.0)) / 100.0,
                "distance_km": distance_km,
                "create_minutes": create_minutes,
                "remove_minutes": remove_minutes,
                "start_lat": float(start["lat"]),
                "start_lng": float(start["lng"]),
                "end_lat": float(end["lat"]),
                "end_lng": float(end["lng"]),
                "cost_time_minutes": cost_time_minutes,
                "load_start_minutes": load_start_minutes,
                "load_end_minutes": load_end_minutes,
            }
    return cargo_map


def load_driver_cost_map(path: Path) -> dict[str, float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("drivers.json 必须为数组")
    cost_map: dict[str, float] = {}
    for item in raw:
        driver_id = str(item.get("driver_id", "")).strip()
        if not driver_id:
            continue
        cost_per_km = float(item.get("cost_per_km", 0.0))
        if cost_per_km < 0:
            raise ValueError(f"driver {driver_id} 的 cost_per_km 不能为负数")
        cost_map[driver_id] = cost_per_km
    return cost_map


def load_driver_preferences_map(path: Path) -> dict[str, list[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("drivers.json 必须为数组")
    out: dict[str, list[str]] = {}
    for item in raw:
        driver_id = str(item.get("driver_id", "")).strip()
        if not driver_id:
            continue
        prefs = item.get("preferences") or []
        if not isinstance(prefs, list):
            prefs = []
        out[driver_id] = [str(x) for x in prefs if isinstance(x, str)]
    return out


def iter_result_files(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("actions_202603_*.jsonl"))


def load_simulate_time_seconds(path: Path) -> float | None:
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    value = raw.get("simulate_time_seconds")
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    return None


def load_reposition_speed_km_per_hour(path: Path) -> float:
    raw = json.loads(path.read_text(encoding="utf-8"))
    value = raw.get("reposition_speed_km_per_hour")
    if not isinstance(value, (int, float)) or float(value) <= 0:
        raise ValueError("config.json 缺少有效字段 reposition_speed_km_per_hour")
    return float(value)


def load_simulation_duration_days(path: Path) -> int:
    raw = json.loads(path.read_text(encoding="utf-8"))
    value = raw.get("simulation_duration_days")
    if not isinstance(value, int) or value <= 0:
        raise ValueError("run_summary_202603.json 缺少有效字段 simulation_duration_days")
    # 结算口径封顶 30 天：信任 run_summary 提供的天数，但不会超过赛题月度上限。
    return min(int(value), 30)


def _nearly_equal(a: float, b: float, eps: float = 1e-4) -> bool:
    return abs(float(a) - float(b)) <= eps


def _distance_minutes(distance_km: float, speed_km_per_hour: float) -> int:
    if distance_km <= 0:
        return 1
    import math
    return max(1, int(math.ceil((distance_km / speed_km_per_hour) * 60.0)))


def _iter_day_segments(start_min: int, end_min: int) -> list[tuple[int, int]]:
    """按天拆分 [start_min, end_min) 区间，返回 [(day_idx, segment_minutes), ...]。"""
    out: list[tuple[int, int]] = []
    if end_min <= start_min:
        return out
    cur = start_min
    while cur < end_min:
        day_idx = cur // 1440
        day_end = (day_idx + 1) * 1440
        seg_end = min(day_end, end_min)
        out.append((day_idx, seg_end - cur))
        cur = seg_end
    return out


def _interval_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def _build_step_contexts(file_path: Path) -> list[dict[str, Any]]:
    ctxs: list[dict[str, Any]] = []
    prev_end_minutes = 0
    with file_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            row = line.strip()
            if not row:
                continue
            record: dict[str, Any] = json.loads(row)
            step_elapsed = int(record.get("step_elapsed_minutes", -1))
            query_scan_cost = int(record.get("query_scan_cost_minutes", -1))
            action_exec_cost = int(record.get("action_exec_cost_minutes", -1))
            result = record.get("result", {})
            end_minutes = int(result.get("simulation_progress_minutes", -1))
            if min(step_elapsed, query_scan_cost, action_exec_cost, end_minutes) < 0:
                raise ValueError(f"{file_path.name} 第 {line_no} 行缺少步骤耗时字段")
            step_start = prev_end_minutes
            action_start = step_start + query_scan_cost
            action_end = action_start + action_exec_cost
            pos_before = record.get("position_before", {}) or {}
            pos_after = record.get("position_after", {}) or {}
            ctxs.append(
                {
                    "line_no": line_no,
                    "action_name": str((record.get("action") or {}).get("action", "")).strip().lower(),
                    "params": (record.get("action") or {}).get("params", {}) or {},
                    "result": result if isinstance(result, dict) else {},
                    "step_start": step_start,
                    "action_start": action_start,
                    "action_end": action_end,
                    "step_end": end_minutes,
                    "action_exec_cost": action_exec_cost,
                    "before_lat": float(pos_before.get("lat", 0.0)),
                    "before_lng": float(pos_before.get("lng", 0.0)),
                    "after_lat": float(pos_after.get("lat", 0.0)),
                    "after_lng": float(pos_after.get("lng", 0.0)),
                }
            )
            prev_end_minutes = end_minutes
    return ctxs


def _apply_daily_penalty(violation_count: int, amount_per_violation: float, cap_amount: float) -> float:
    if violation_count <= 0:
        return 0.0
    return float(min(violation_count * amount_per_violation, cap_amount))


def _evaluate_preferences(
    driver_id: str,
    file_path: Path,
    preferences: list[str],
    simulation_duration_days: int,
) -> tuple[float, dict[str, Any]]:
    """按司机偏好规则计算罚分。返回 (penalty, details)。"""
    if driver_id not in {"D006", "D007", "D008", "D009", "D010"}:
        return 0.0, {"rules": []}
    ctxs = _build_step_contexts(file_path)
    if not ctxs:
        return 0.0, {"rules": []}

    rules: list[dict[str, Any]] = []
    total_penalty = 0.0
    days = list(range(simulation_duration_days))

    if driver_id == "D006":
        # 每天至少 5 小时连续休息（用 wait 的 action_exec 近似休息时段）
        violation_days = 0
        for day in days:
            intervals: list[tuple[int, int]] = []
            for c in ctxs:
                if c["action_name"] != "wait" or c["action_exec_cost"] <= 0:
                    continue
                # 休息按整步区间计算：查看货源耗时不打断连续休息
                start = c["step_start"]
                end = c["step_end"]
                d0 = day * 1440
                d1 = d0 + 1440
                s = max(start, d0)
                e = min(end, d1)
                if e > s:
                    intervals.append((s, e))
            intervals.sort()
            merged: list[tuple[int, int]] = []
            for s, e in intervals:
                if not merged or s > merged[-1][1]:
                    merged.append((s, e))
                else:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            longest = 0
            for s, e in merged:
                longest = max(longest, e - s)
            if longest < 300:
                violation_days += 1
        penalty = _apply_daily_penalty(violation_days, amount_per_violation=400.0, cap_amount=4000.0)
        total_penalty += penalty
        rules.append(
            {
                "rule": "每天至少连续休息5小时",
                "preference_text": preferences[0] if preferences else "",
                "violations": violation_days,
                "penalty": penalty,
            }
        )

    if driver_id == "D007":
        # 每天 23:00-04:00 休息：不接单也不空驶（按动作执行区间与窗口相交判断，包含跨窗）
        violation_days_set: set[int] = set()
        for c in ctxs:
            if c["action_name"] not in {"take_order", "reposition"}:
                continue
            a_start = c["action_start"]
            a_end = c["action_end"]
            for day in days:
                w1s = day * 1440 + 23 * 60
                w1e = (day + 1) * 1440
                w2s = (day + 1) * 1440
                w2e = (day + 1) * 1440 + 4 * 60
                if _interval_overlap(a_start, a_end, w1s, w1e) or _interval_overlap(a_start, a_end, w2s, w2e):
                    violation_days_set.add(day)
        violations = len(violation_days_set)
        penalty = _apply_daily_penalty(violations, amount_per_violation=500.0, cap_amount=5000.0)
        total_penalty += penalty
        rules.append(
            {
                "rule": "23:00-04:00休息（不接单不空驶）",
                "preference_text": preferences[0] if preferences else "",
                "violations": violations,
                "penalty": penalty,
            }
        )

    if driver_id == "D008":
        # 月度至少 2 天完全不出车（仅 take_order/reposition 视为出车，wait 不算）
        active_minutes_by_day: dict[int, int] = {d: 0 for d in days}
        for c in ctxs:
            if c["action_name"] not in {"take_order", "reposition"}:
                continue
            for d, seg in _iter_day_segments(c["action_start"], c["action_end"]):
                active_minutes_by_day[d] = active_minutes_by_day.get(d, 0) + seg
        off_days = sum(1 for d in days if active_minutes_by_day.get(d, 0) == 0)
        ok = off_days >= 2
        penalty = 0.0 if ok else 3000.0
        total_penalty += penalty
        rules.append(
            {
                "rule": "每月至少2天完全不出车",
                "preference_text": preferences[0] if preferences else "",
                "off_days": off_days,
                "penalty": penalty,
            }
        )

    if driver_id == "D009":
        # 每天 23:00 前回到家附近 1km；到家后至次日 08:00 不再接单或空驶
        home_lat, home_lng = 23.12, 113.28
        violation_days = 0
        for day in days:
            t23 = day * 1440 + 23 * 60
            t8_next = (day + 1) * 1440 + 8 * 60
            last_pos: tuple[float, float] | None = None
            for c in ctxs:
                if c["step_end"] <= t23:
                    last_pos = (c["after_lat"], c["after_lng"])
            if last_pos is None:
                # 若当天 23:00 前无记录，使用首条 before 位置近似
                first = ctxs[0]
                last_pos = (first["before_lat"], first["before_lng"])
            home_ok = haversine_km(last_pos[0], last_pos[1], home_lat, home_lng) <= 1.0
            quiet_ok = True
            for c in ctxs:
                if c["action_name"] in {"take_order", "reposition"} and _interval_overlap(
                    c["action_start"], c["action_end"], t23, t8_next
                ):
                    quiet_ok = False
                    break
            if not (home_ok and quiet_ok):
                violation_days += 1
        penalty = _apply_daily_penalty(violation_days, amount_per_violation=600.0, cap_amount=6000.0)
        total_penalty += penalty
        rules.append(
            {
                "rule": "每天23:00前回家且到次日08:00不再接单或空驶",
                "preference_text": preferences[0] if preferences else "",
                "violations": violation_days,
                "penalty": penalty,
            }
        )

    if driver_id == "D010":
        # 规则 1：每月到固定点 >= 5 天（同一天多次算 1 次）
        target_lat, target_lng = 23.13, 113.26
        visit_days: set[int] = set()
        for c in ctxs:
            day = c["step_end"] // 1440
            if haversine_km(c["after_lat"], c["after_lng"], target_lat, target_lng) <= 1.0:
                visit_days.add(day)
        visit_ok = len(visit_days) >= 5
        penalty_visit = 0.0 if visit_ok else 3000.0
        total_penalty += penalty_visit
        rules.append(
            {
                "rule": "每月至少5天到达目标点",
                "preference_text": preferences[0] if len(preferences) > 0 else "",
                "visit_days": len(visit_days),
                "penalty": penalty_visit,
            }
        )

        # 规则 2：禁止进入区域（圆心 23.30,113.52 半径 2km）
        center_lat, center_lng, radius_km = 23.30, 113.52, 2.0
        entered = False
        for c in ctxs:
            if haversine_km(c["before_lat"], c["before_lng"], center_lat, center_lng) <= radius_km:
                entered = True
                break
            if haversine_km(c["after_lat"], c["after_lng"], center_lat, center_lng) <= radius_km:
                entered = True
                break
        penalty_zone = 3000.0 if entered else 0.0
        total_penalty += penalty_zone
        rules.append(
            {
                "rule": "禁止进入指定禁入区域",
                "preference_text": preferences[1] if len(preferences) > 1 else "",
                "entered": entered,
                "penalty": penalty_zone,
            }
        )

    return total_penalty, {"rules": rules}


def _validate_and_compute_income_by_driver(
    file_path: Path,
    cargo_map: dict[str, dict[str, Any]],
    cost_per_km: float,
    reposition_speed_km_per_hour: float,
    simulation_horizon_minutes: int | None = None,
) -> tuple[dict[str, float], dict[str, int]]:
    income = {"gross_income": 0.0, "distance_km": 0.0, "cost": 0.0, "net_income": 0.0}
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0}
    prev_end_minutes = 0
    with file_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            row = line.strip()
            if not row:
                continue
            record: dict[str, Any] = json.loads(row)
            action = record.get("action", {})
            params = action.get("params", {})
            result = record.get("result", {})
            action_name = str(action.get("action", "")).strip().lower()
            if action_name not in {"wait", "reposition", "take_order"}:
                raise ValueError(f"{file_path.name} 第 {line_no} 行 action 非法: {action_name}")

            raw_usage = record.get("token_usage", {})
            if not isinstance(raw_usage, dict):
                raise ValueError(f"{file_path.name} 第 {line_no} 行 token_usage 非法")
            token_usage["prompt_tokens"] += int(raw_usage.get("prompt_tokens", 0))
            token_usage["completion_tokens"] += int(raw_usage.get("completion_tokens", 0))
            token_usage["reasoning_tokens"] += int(raw_usage.get("reasoning_tokens", 0))
            token_usage["total_tokens"] += int(raw_usage.get("total_tokens", 0))

            end_minutes = int(result.get("simulation_progress_minutes", -1))
            if end_minutes < 0:
                raise ValueError(f"{file_path.name} 第 {line_no} 行缺少 simulation_progress_minutes")
            step_elapsed = int(record.get("step_elapsed_minutes", -1))
            if step_elapsed < 0:
                raise ValueError(f"{file_path.name} 第 {line_no} 行缺少 step_elapsed_minutes")
            query_scan_cost = int(record.get("query_scan_cost_minutes", -1))
            action_exec_cost = int(record.get("action_exec_cost_minutes", -1))
            if query_scan_cost < 0 or action_exec_cost < 0:
                raise ValueError(f"{file_path.name} 第 {line_no} 行缺少 query/action cost 字段")
            if step_elapsed != query_scan_cost + action_exec_cost:
                raise ValueError(f"{file_path.name} 第 {line_no} 行耗时不一致")
            if end_minutes - prev_end_minutes != step_elapsed:
                raise ValueError(f"{file_path.name} 第 {line_no} 行时间推进不一致")
            step_start_minutes = prev_end_minutes
            action_start_minutes = step_start_minutes + query_scan_cost

            pos_before = record.get("position_before", {})
            pos_after = record.get("position_after", {})
            if not isinstance(pos_before, dict) or not isinstance(pos_after, dict):
                raise ValueError(f"{file_path.name} 第 {line_no} 行缺少位置字段")
            before_lat = float(pos_before.get("lat"))
            before_lng = float(pos_before.get("lng"))
            after_lat = float(pos_after.get("lat"))
            after_lng = float(pos_after.get("lng"))

            if action_name == "wait":
                wait_minutes = int((params or {}).get("duration_minutes", 1))
                if action_exec_cost != wait_minutes:
                    raise ValueError(f"{file_path.name} 第 {line_no} 行 wait 时间不一致")
                if (not _nearly_equal(before_lat, after_lat)) or (not _nearly_equal(before_lng, after_lng)):
                    raise ValueError(f"{file_path.name} 第 {line_no} 行 wait 不应改变位置")

            elif action_name == "reposition":
                target_lat = float((params or {}).get("latitude"))
                target_lng = float((params or {}).get("longitude"))
                if (not _nearly_equal(after_lat, target_lat)) or (not _nearly_equal(after_lng, target_lng)):
                    raise ValueError(f"{file_path.name} 第 {line_no} 行 reposition 终点位置错误")
                expected_km = haversine_km(before_lat, before_lng, target_lat, target_lng)
                expected_minutes = _distance_minutes(expected_km, reposition_speed_km_per_hour)
                if action_exec_cost != expected_minutes:
                    raise ValueError(f"{file_path.name} 第 {line_no} 行 reposition 时间不一致")
                income["distance_km"] += float(result.get("distance_km", 0.0))

            elif action_name == "take_order":
                cargo_id = str((params or {}).get("cargo_id", "")).strip()
                if not cargo_id:
                    raise ValueError(f"{file_path.name} 第 {line_no} 行 take_order 缺少 cargo_id")
                cargo = cargo_map.get(cargo_id)
                if cargo is None:
                    raise ValueError(f"{file_path.name} 第 {line_no} 行 cargo_id 不存在: {cargo_id}")
                if not (int(cargo["create_minutes"]) <= action_start_minutes <= int(cargo["remove_minutes"])):
                    if bool(result.get("accepted", False)):
                        raise ValueError(f"{file_path.name} 第 {line_no} 行接单时点不在货源有效期")

                accepted = bool(result.get("accepted", False))
                if accepted:
                    if (not _nearly_equal(after_lat, float(cargo["end_lat"]))) or (
                        not _nearly_equal(after_lng, float(cargo["end_lng"]))
                    ):
                        raise ValueError(f"{file_path.name} 第 {line_no} 行接单后位置错误")
                    pickup_km = haversine_km(before_lat, before_lng, float(cargo["start_lat"]), float(cargo["start_lng"]))
                    pickup_minutes = _distance_minutes(pickup_km, reposition_speed_km_per_hour) if pickup_km > 1e-6 else 0
                    arrival_minutes = action_start_minutes + pickup_minutes
                    wait_minutes = 0
                    load_start_minutes = cargo.get("load_start_minutes")
                    load_end_minutes = cargo.get("load_end_minutes")
                    if isinstance(load_start_minutes, int) and isinstance(load_end_minutes, int):
                        if arrival_minutes > load_end_minutes:
                            raise ValueError(f"{file_path.name} 第 {line_no} 行成功接单但已超装货时间窗")
                        wait_minutes = max(0, load_start_minutes - arrival_minutes)
                    expected_exec = pickup_minutes + wait_minutes + int(cargo["cost_time_minutes"])
                    if action_exec_cost != expected_exec:
                        raise ValueError(f"{file_path.name} 第 {line_no} 行接单耗时不一致")
                    # 不信任日志中的可选标记，收益资格由脚本按仿真上界自行判定。
                    income_eligible = (
                        simulation_horizon_minutes is None or int(end_minutes) <= int(simulation_horizon_minutes)
                    )
                    if income_eligible:
                        income["gross_income"] += float(cargo["price"])
                    income["distance_km"] += float(result.get("pickup_deadhead_km", 0.0) or 0.0)
                    haul_km = float(result.get("haul_distance_km", 0.0) or 0.0)
                    if haul_km <= 0:
                        haul_km = float(cargo["distance_km"])
                    income["distance_km"] += haul_km

            prev_end_minutes = end_minutes

    income["cost"] = income["distance_km"] * cost_per_km
    income["net_income"] = income["gross_income"] - income["cost"]
    for key in ("gross_income", "distance_km", "cost", "net_income"):
        income[key] = round(float(income[key]), 2)
    return income, token_usage


def compute_income(
    files: list[Path],
    cargo_map: dict[str, dict[str, Any]],
    driver_cost_map: dict[str, float],
    driver_preferences_map: dict[str, list[str]],
    reposition_speed_km_per_hour: float,
    simulation_duration_days: int,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, int]], dict[str, int], dict[str, str], dict[str, dict[str, Any]]]:
    stats: dict[str, dict[str, float]] = {}
    token_stats: dict[str, dict[str, int]] = {}
    validation_errors: dict[str, str] = {}
    preference_details_by_driver: dict[str, dict[str, Any]] = {}
    zero_income = {"gross_income": 0.0, "distance_km": 0.0, "cost": 0.0, "net_income": 0.0}
    zero_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0}
    simulation_horizon_minutes = int(simulation_duration_days) * 24 * 60
    for file_path in files:
        driver_id = file_path.name.split("_")[2]
        cost_per_km = float(driver_cost_map.get(driver_id, 0.0))
        try:
            income_item, token_item = _validate_and_compute_income_by_driver(
                file_path,
                cargo_map,
                cost_per_km=cost_per_km,
                reposition_speed_km_per_hour=reposition_speed_km_per_hour,
                simulation_horizon_minutes=simulation_horizon_minutes,
            )
            preference_penalty, preference_details = _evaluate_preferences(
                driver_id,
                file_path,
                driver_preferences_map.get(driver_id, []),
                simulation_duration_days=simulation_duration_days,
            )
            income_item["preference_penalty"] = round(float(preference_penalty), 2)
            income_item["net_income"] = round(float(income_item["net_income"] - preference_penalty), 2)
            preference_details_by_driver[driver_id] = preference_details
            stats[driver_id] = income_item
            token_stats[driver_id] = token_item
        except Exception as exc:
            stats[driver_id] = dict(zero_income)
            token_stats[driver_id] = dict(zero_tokens)
            validation_errors[driver_id] = f"{type(exc).__name__}: {exc}"
            preference_details_by_driver[driver_id] = {"rules": []}
    total_token_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
    }
    for token_item in token_stats.values():
        total_token_usage["prompt_tokens"] += int(token_item["prompt_tokens"])
        total_token_usage["completion_tokens"] += int(token_item["completion_tokens"])
        total_token_usage["reasoning_tokens"] += int(token_item["reasoning_tokens"])
        total_token_usage["total_tokens"] += int(token_item["total_tokens"])
    return stats, token_stats, total_token_usage, validation_errors, preference_details_by_driver


def build_drivers_payload(
    income: dict[str, dict[str, float]],
    token_by_driver: dict[str, dict[str, int]],
    validation_errors: dict[str, str],
    preference_details_by_driver: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """按司机合并收入与 Token，输出稳定排序的列表。"""
    default_income = {"gross_income": 0.0, "distance_km": 0.0, "cost": 0.0, "preference_penalty": 0.0, "net_income": 0.0}
    default_tokens = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
    }
    driver_ids = sorted(set(income) | set(token_by_driver))
    rows: list[dict[str, Any]] = []
    for driver_id in driver_ids:
        inc = {**default_income, **income.get(driver_id, {})}
        tok = {**default_tokens, **token_by_driver.get(driver_id, {})}
        rows.append(
            {
                "driver_id": driver_id,
                "income": inc,
                "token_usage": tok,
                "calculation_aborted": driver_id in validation_errors,
                "validation_error": validation_errors.get(driver_id),
                "preference_check": preference_details_by_driver.get(driver_id, {"rules": []}),
            }
        )
    return rows


def main() -> None:
    if not CARGO_DATASET.is_file():
        raise FileNotFoundError(f"缺少货源数据: {CARGO_DATASET}")
    if not DRIVERS_DATASET.is_file():
        raise FileNotFoundError(f"缺少司机数据: {DRIVERS_DATASET}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cargo_map = load_cargo_map(CARGO_DATASET)
    driver_cost_map = load_driver_cost_map(DRIVERS_DATASET)
    driver_preferences_map = load_driver_preferences_map(DRIVERS_DATASET)
    reposition_speed_km_per_hour = load_reposition_speed_km_per_hour(CONFIG_PATH)
    simulation_duration_days = load_simulation_duration_days(RUN_SUMMARY_FILE)
    result_files = iter_result_files(RESULTS_DIR)
    income, token_by_driver, total_token_usage, validation_errors, preference_details_by_driver = compute_income(
        result_files,
        cargo_map,
        driver_cost_map,
        driver_preferences_map,
        reposition_speed_km_per_hour=reposition_speed_km_per_hour,
        simulation_duration_days=simulation_duration_days,
    )
    drivers = build_drivers_payload(income, token_by_driver, validation_errors, preference_details_by_driver)
    total_net_income = round(sum(float(d["income"]["net_income"]) for d in drivers), 2)
    total_preference_penalty = round(sum(float(d["income"].get("preference_penalty", 0.0)) for d in drivers), 2)
    simulate_time_seconds = load_simulate_time_seconds(RUN_SUMMARY_FILE)
    payload = {
        "month": "2026-03",
        "simulate_time_seconds": simulate_time_seconds,
        "result_files_count": len(result_files),
        "drivers": drivers,
        "summary": {
            "total_net_income_all_drivers": total_net_income,
            "total_preference_penalty": total_preference_penalty,
            "total_token_usage": total_token_usage,
            "failed_driver_count": len(validation_errors),
            "failed_drivers": validation_errors,
        },
        "cost_meaning": "cost = distance_km * cost_per_km (driver cost per km)",
        "cost_metric": "net_income = gross_income - (distance_km * cost_per_km)",
    }
    OUTPUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
