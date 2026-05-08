"""模型决策服务：依赖 `simkit.ports.SimulationApiPort`，由评测进程注入具体环境。"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime
from typing import Any

from simkit.ports import SimulationApiPort

SIMULATION_EPOCH = datetime(2026, 3, 1, 0, 0, 0)
DEFAULT_REPOSITION_SPEED_KM_PER_HOUR = 60.0
DEFAULT_COST_PER_KM = 1.5
TIME_VALUE_YUAN_PER_MINUTE = 0.2
PROMPT_CANDIDATE_LIMIT = 10


class ModelDecisionService:
    """基于大模型的单步决策：拉取状态与候选货源，请求补全并解析为结构化动作。"""

    def __init__(self, api: SimulationApiPort) -> None:
        self._api = api
        self._logger = logging.getLogger("agent.decision_service")
        self._driver_memory: dict[str, dict[str, Any]] = {}

    def decide(self, driver_id: str) -> dict[str, Any]:
        status = self._api.get_driver_status(driver_id)
        lat = float(status["current_lat"])
        lng = float(status["current_lng"])
        normalized_preferences = _normalize_preferences(status.get("preferences", []))
        preferences = [p["text"] for p in normalized_preferences]
        preference_constraints = _extract_preference_constraints(preferences)

        rest_action = self._daily_rest_guard(
            driver_id=driver_id,
            current_minutes=int(status["simulation_progress_minutes"]),
            preference_constraints=preference_constraints,
        )
        if rest_action is not None:
            return rest_action

        cargo_resp = self._api.query_cargo(driver_id=driver_id, latitude=lat, longitude=lng)
        items = cargo_resp.get("items", [])
        status_after_query = self._api.get_driver_status(driver_id)
        real_time = int(status_after_query["simulation_progress_minutes"])
        self._logger.info(
            "decision input driver_id=%s time_min=%s real_time=%s loc=(%.5f,%.5f) cargo_items=%s",
            driver_id,
            status.get("simulation_progress_minutes"),
            real_time,
            lat,
            lng,
            len(items),
        )

        valid_items: list[dict[str, Any]] = []
        for item in items:
            cargo = item.get("cargo", {})
            remove_time = cargo.get("remove_time")
            if remove_time and _wall_time_to_minutes(str(remove_time)) < real_time:
                continue
            metrics = _estimate_cargo_metrics(item, real_time, lat, lng)
            if not metrics.get("feasible", False):
                continue
            enriched_item = dict(item)
            enriched_item["_metrics"] = metrics
            valid_items.append(enriched_item)
        self._logger.info("cargo filter: %d -> %d (removed %d expired/load_time)", len(items), len(valid_items), len(items) - len(valid_items))
        if not valid_items:
            self._logger.info("no valid cargo, waiting 15 minutes")
            return {"action": "wait", "params": {"duration_minutes": 15}}

        valid_items.sort(key=lambda x: float((x.get("_metrics") or {}).get("estimated_score", 0.0)), reverse=True)
        prompt_items = valid_items[:PROMPT_CANDIDATE_LIMIT]
        allowed_cargo_ids = {str((item.get("cargo") or {}).get("cargo_id", "")).strip() for item in prompt_items}
        allowed_cargo_ids.discard("")
        best_cargo_id = str((prompt_items[0].get("cargo") or {}).get("cargo_id", "")).strip()
        prompt = self._build_prompt(
            driver_id=driver_id,
            status=status,
            items=prompt_items,
            real_time=real_time,
            preferences=preferences,
            preference_constraints=preference_constraints,
        )
        self._logger.info("prompt_content driver_id=%s prompt=%s", driver_id, prompt[:500])
        model_resp = self._api.model_chat_completion(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "你是货运调度决策器。只输出JSON："
                            '{"action":"take_order|reposition|wait","params":{...}}。'
                            "take_order.params 必须包含候选里的 cargo_id；wait.params 必须包含正整数 duration_minutes；"
                            "reposition.params 必须包含 latitude 和 longitude。"
                            "cargo_candidates 已经过代码过滤，并按 score 从高到低排序。"
                            "默认选择 candidate[0] 执行 take_order。"
                            "只有所有候选 score <= 0，或 pref 明确显示风险时，才选择 wait。"
                            "不要重新解析偏好文本；只使用 score、net、pref 和候选排序。"
                            "禁止编造 cargo_id，禁止输出 markdown、解释或额外文本。"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "response_format": {"type": "json_object"},
                "enable_thinking": True,
            }
        )
        action = self._parse_action(model_resp)
        if action.get("action") == "take_order":
            cargo_id = str((action.get("params") or {}).get("cargo_id", "")).strip()
            if cargo_id not in allowed_cargo_ids:
                self._logger.warning("model returned cargo_id outside prompt candidates: %s; fallback take best", cargo_id)
                return {"action": "take_order", "params": {"cargo_id": best_cargo_id}}
        self._logger.info("decision output driver_id=%s action=%s params=%s", driver_id, action.get("action"), action.get("params"))
        return action

    def _daily_rest_guard(self, *, driver_id: str, current_minutes: int, preference_constraints: dict[str, Any]) -> dict[str, Any] | None:
        daily_rest = preference_constraints.get("daily_rest")
        if not isinstance(daily_rest, dict):
            return None
        try:
            required_minutes = int(daily_rest.get("min_continuous_minutes", 0) or 0)
        except (TypeError, ValueError):
            return None
        if required_minutes <= 0:
            return None
        memory = self._driver_memory.setdefault(driver_id, {})
        rested_days: set[int] = memory.setdefault("daily_rest_days", set())
        day_idx = int(current_minutes) // 1440
        if day_idx in rested_days:
            return None
        minute_of_day = int(current_minutes) % 1440
        remaining_today = 1440 - minute_of_day
        if remaining_today >= required_minutes:
            duration_minutes = required_minutes
            rested_days.add(day_idx)
            satisfied_day = day_idx
        else:
            duration_minutes = remaining_today + required_minutes
            rested_days.add(day_idx + 1)
            satisfied_day = day_idx + 1
        self._logger.info("daily_rest_guard driver_id=%s day=%s duration=%s required=%s current_min=%s", driver_id, satisfied_day, duration_minutes, required_minutes, current_minutes)
        return {"action": "wait", "params": {"duration_minutes": int(duration_minutes)}}

    def _build_prompt(self, driver_id: str, status: dict[str, Any], items: list[dict[str, Any]], real_time: int, preferences: list[str] | None = None, preference_constraints: dict[str, Any] | None = None) -> str:
        cargo_candidates: list[dict[str, Any]] = []
        if preferences is None:
            normalized_preferences = _normalize_preferences(status.get("preferences", []))
            preferences = [p["text"] for p in normalized_preferences]
        preference_constraints = preference_constraints or _extract_preference_constraints(preferences)
        for item in items:
            cargo = item.get("cargo", {})
            metrics = item.get("_metrics") if isinstance(item.get("_metrics"), dict) else _estimate_cargo_metrics(item, real_time, float(status.get("current_lat", 0) or 0), float(status.get("current_lng", 0) or 0))
            cargo_candidates.append({
                "cargo_id": cargo.get("cargo_id"),
                "score": metrics.get("estimated_score"),
                "net": metrics.get("estimated_net_after_distance_cost"),
                "price": metrics.get("price_yuan"),
                "pickup_km": metrics.get("pickup_km"),
                "haul_km": metrics.get("haul_km"),
                "finish_min": metrics.get("finish_min"),
                "pref": metrics.get("preference_tag", "ok"),
            })
        decision_context: dict[str, Any] = {
            "driver_id": driver_id,
            "simulation_progress_minutes": real_time,
            "driver_status": {"current_lat": status.get("current_lat"), "current_lng": status.get("current_lng"), "truck_length": status.get("truck_length"), "completed_order_count": status.get("completed_order_count")},
            "rules": [
                "cargo_candidates are sorted by score descending; candidate[0] is the default choice.",
                "score includes distance and time costs; future code may also include preference penalties.",
                "prefer take_order for candidate[0] when score > 0 and pref is ok.",
                "wait only when no positive-score candidate is available or pref indicates risk.",
            ],
            "cargo_candidates": cargo_candidates,
        }
        decision_context["preference_constraints"] = _constraints_for_prompt(preference_constraints)
        decision_context["rules"].extend([
            "preference_constraints are parsed by code from driver preferences; do not re-parse raw preference text.",
            "If a candidate does not explicitly conflict with parsed constraints, treat it as selectable.",
        ])
        return json.dumps(decision_context, ensure_ascii=False)

    def _parse_action(self, model_resp: dict[str, Any]) -> dict[str, Any]:
        choices = model_resp.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("模型返回缺少 choices")
        content = choices[0].get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("模型返回 content 为空")
        action = json.loads(content)
        if not isinstance(action, dict):
            raise ValueError("模型返回动作不是JSON对象")
        action_name = str(action.get("action", "")).strip().lower()
        params = action.get("params")
        if action_name not in {"take_order", "reposition", "wait"}:
            raise ValueError(f"模型返回未知action: {action_name}")
        if not isinstance(params, dict):
            raise ValueError("模型返回 params 必须是对象")
        if action_name == "take_order":
            cargo_id = str(params.get("cargo_id", "")).strip()
            if not cargo_id:
                raise ValueError("take_order 缺少有效 cargo_id")
            return {"action": "take_order", "params": {"cargo_id": cargo_id}}
        if action_name == "reposition":
            return {"action": "reposition", "params": {"latitude": float(params["latitude"]), "longitude": float(params["longitude"])}}
        duration_minutes = int(params["duration_minutes"])
        if duration_minutes <= 0:
            raise ValueError("wait.duration_minutes 必须为正整数")
        return {"action": "wait", "params": {"duration_minutes": duration_minutes}}


_TIME_RE = re.compile(r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)(?!\d)")
_COORD_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*[,，]\s*(-?\d+(?:\.\d+)?)")
_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")
_CN_NUMBERS = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _safe_float(value: Any) -> float | None:
    """宽松转换数值字段；缺失或非法时返回 None，避免偏好标准化中断决策。"""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_preferences(raw_preferences: Any) -> list[dict[str, Any]]:
    """把运行时返回的 preferences 统一成结构化列表。

    兼容 list[str]、list[dict]、单个 str/dict、None。dict 正文优先从
    content/text/description/rule/preference 读取，并保留罚金与生效窗口字段。
    """
    if raw_preferences is None:
        return []
    raw_items = raw_preferences if isinstance(raw_preferences, list) else [raw_preferences]

    normalized: list[dict[str, Any]] = []
    for item in raw_items:
        if item is None:
            continue
        if isinstance(item, dict):
            text = (
                item.get("content")
                or item.get("text")
                or item.get("description")
                or item.get("rule")
                or item.get("preference")
                or ""
            )
            text = str(text).strip()
            if not text:
                continue
            normalized.append(
                {
                    "text": text,
                    "penalty_amount": _safe_float(item.get("penalty_amount")),
                    "penalty_cap": _safe_float(item.get("penalty_cap")),
                    "start_time": str(item.get("start_time")).strip() if item.get("start_time") else None,
                    "end_time": str(item.get("end_time")).strip() if item.get("end_time") else None,
                    "raw": item,
                }
            )
            continue

        text = str(item).strip()
        if not text:
            continue
        normalized.append(
            {
                "text": text,
                "penalty_amount": None,
                "penalty_cap": None,
                "start_time": None,
                "end_time": None,
                "raw": item,
            }
        )
    return normalized


def _wall_time_to_minutes(wall_time_str: str) -> int:
    return int((datetime.strptime(wall_time_str.strip(), "%Y-%m-%d %H:%M:%S") - SIMULATION_EPOCH).total_seconds() // 60)


def _pickup_minutes(distance_km: float, speed_km_per_hour: float = 60.0) -> int:
    return 0 if distance_km <= 1e-6 else max(1, math.ceil((distance_km / speed_km_per_hour) * 60))


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius_km = 6371.0
    p1, l1, p2, l2 = map(math.radians, [lat1, lng1, lat2, lng2])
    h = math.sin((p2 - p1) * 0.5) ** 2 + math.cos(p1) * math.cos(p2) * math.sin((l2 - l1) * 0.5) ** 2
    return 2.0 * radius_km * math.asin(math.sqrt(min(1.0, max(0.0, h))))


def _parse_load_window(load_time: Any) -> tuple[int, int] | None:
    if load_time is None:
        return None
    return _wall_time_to_minutes(str(load_time[0])), _wall_time_to_minutes(str(load_time[1]))


def _estimate_cargo_metrics(item: dict[str, Any], real_time: int, driver_lat: float = 0.0, driver_lng: float = 0.0) -> dict[str, Any]:
    cargo = item.get("cargo") or {}
    start = cargo.get("start") or {}
    pickup_km = _haversine_km(driver_lat, driver_lng, float(start["lat"]), float(start["lng"]))
    pickup_minutes = _pickup_minutes(pickup_km, DEFAULT_REPOSITION_SPEED_KM_PER_HOUR)
    arrival_min = int(real_time) + pickup_minutes
    feasible, risk_reasons = True, []
    load_wait_minutes, ready_min = 0, arrival_min
    load_window = _parse_load_window(cargo.get("load_time"))
    if load_window is not None:
        load_start_min, load_end_min = load_window
        if arrival_min > load_end_min:
            feasible = False; risk_reasons.append("miss_load_window")
        else:
            load_wait_minutes = max(0, load_start_min - arrival_min); ready_min = arrival_min + load_wait_minutes
    remove_time = cargo.get("remove_time")
    if remove_time and _wall_time_to_minutes(str(remove_time)) < ready_min:
        feasible = False; risk_reasons.append("expires_before_ready")
    haul_minutes = int(cargo.get("cost_time_minutes", 0) or 0)
    end = cargo.get("end") or {}
    haul_km = _haversine_km(float(start["lat"]), float(start["lng"]), float(end["lat"]), float(end["lng"]))
    finish_min = ready_min + haul_minutes
    total_exec_minutes = max(0, finish_min - int(real_time))
    price_yuan = float(cargo.get("price", 0.0) or 0.0)
    distance_cost = (pickup_km + haul_km) * DEFAULT_COST_PER_KM
    estimated_net_after_distance_cost = price_yuan - distance_cost
    estimated_score = estimated_net_after_distance_cost - total_exec_minutes * TIME_VALUE_YUAN_PER_MINUTE
    return {"feasible": feasible, "risk_reasons": risk_reasons, "price_yuan": round(price_yuan, 2), "pickup_km": round(pickup_km, 2), "haul_km": round(haul_km, 2), "pickup_minutes": int(pickup_minutes), "load_wait_minutes": int(load_wait_minutes), "haul_minutes": int(haul_minutes), "arrival_min": int(arrival_min), "ready_min": int(ready_min), "finish_min": int(finish_min), "total_exec_minutes": int(total_exec_minutes), "distance_cost": round(distance_cost, 2), "estimated_net_after_distance_cost": round(estimated_net_after_distance_cost, 2), "estimated_score": round(estimated_score, 2)}


def _all_time_texts(text: str) -> list[str]:
    return [f"{int(h):02d}:{m}" for h, m in _TIME_RE.findall(text)]


def _first_number(text: str) -> float | None:
    m = _NUM_RE.search(text)
    if m: return float(m.group(1))
    for k, v in _CN_NUMBERS.items():
        if k in text: return float(v)
    return None


def _number_before_unit(text: str, unit_pattern: str) -> float | None:
    m = re.search(rf"(\d+(?:\.\d+)?)\s*{unit_pattern}", text)
    if m: return float(m.group(1))
    m = re.search(rf"([一二两三四五六七八九十])\s*{unit_pattern}", text)
    return float(_CN_NUMBERS.get(m.group(1), 0)) if m else None


def _first_coord(text: str) -> dict[str, float] | None:
    m = _COORD_RE.search(text)
    return {"lat": float(m.group(1)), "lng": float(m.group(2))} if m else None


def _radius_km(text: str, default: float | None = None) -> float | None:
    for pat in [r"半径\s*(\d+(?:\.\d+)?)\s*(?:km|公里)", r"附近\s*(\d+(?:\.\d+)?)\s*(?:km|公里)", r"(\d+(?:\.\d+)?)\s*(?:km|公里)\s*内"]:
        m = re.search(pat, text, re.IGNORECASE)
        if m: return float(m.group(1))
    return default


def _constraints_for_prompt(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"raw_preferences", "source", "notes"}: continue
            compact = _constraints_for_prompt(item)
            if compact not in (None, {}, []): out[key] = compact
        return out
    if isinstance(value, list):
        return [item for item in (_constraints_for_prompt(x) for x in value) if item not in (None, {}, [])]
    return value


def _extract_preference_constraints(preferences: list[str]) -> dict[str, Any]:
    constraints: dict[str, Any] = {"raw_preferences": preferences, "daily_rest": None, "quiet_windows": [], "monthly_off_days": None, "home_return": None, "target_visits": None, "forbidden_zones": [], "notes": []}
    for pref in preferences:
        text = str(pref).strip()
        if not text: continue
        times = _all_time_texts(text); coord = _first_coord(text)
        if "连续" in text and "休息" in text:
            hours = _number_before_unit(text, "小时")
            if hours: constraints["daily_rest"] = {"min_continuous_minutes": int(hours * 60), "source": text}
        if "完全不出车" in text or ("不出车" in text and ("天" in text or "月" in text)):
            days = _number_before_unit(text, "天") or _first_number(text)
            if days: constraints["monthly_off_days"] = {"min_days": int(days), "source": text}
        if ("回家" in text or "回到家" in text or "家坐标" in text) and coord:
            home: dict[str, Any] = {"lat": coord["lat"], "lng": coord["lng"], "radius_km": _radius_km(text, default=1.0), "source": text}
            if times: home["return_before"] = times[0]
            if len(times) >= 2:
                home["quiet_until"] = times[1]; home["forbid_actions_after_return"] = ["take_order", "reposition"]
            constraints["home_return"] = home
        if ("目标点" in text or "固定地点" in text or "到一个固定" in text) and coord:
            visits = _number_before_unit(text, "次") or _first_number(text)
            constraints["target_visits"] = {"lat": coord["lat"], "lng": coord["lng"], "min_visit_days": int(visits) if visits else None, "same_day_count_once": "同一天" in text, "radius_km": _radius_km(text, default=1.0), "source": text}
        if ("禁入" in text or "不想进入" in text or "禁止进入" in text) and coord:
            constraints["forbidden_zones"].append({"lat": coord["lat"], "lng": coord["lng"], "radius_km": _radius_km(text, default=2.0), "source": text})
        if len(times) >= 2 and any(key in text for key in ("不接单", "不空驶", "休整", "不再接单", "不再空驶")):
            forbid_actions: list[str] = []
            if "不接单" in text or "不再接单" in text: forbid_actions.append("take_order")
            if "不空驶" in text or "不再空驶" in text or ("不" in text and "空驶" in text): forbid_actions.append("reposition")
            constraints["quiet_windows"].append({"start": times[0], "end": times[1], "forbid_actions": sorted(set(forbid_actions or ["take_order", "reposition"])), "source": text})
    constraints["notes"].append("Treat parsed constraints as hard constraints when selecting an action.")
    return constraints
