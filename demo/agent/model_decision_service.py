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
DIRECT_TAKE_SCORE_THRESHOLD = 80.0
DIRECT_TAKE_NET_THRESHOLD = 150.0


class ModelDecisionService:
    """基于大模型的单步决策：拉取状态与候选货源，请求补全并解析为结构化动作。"""

    def __init__(self, api: SimulationApiPort) -> None:
        self._api = api
        self._logger = logging.getLogger("agent.decision_service")

    def decide(self, driver_id: str) -> dict[str, Any]:
        status = self._api.get_driver_status(driver_id)
        lat = float(status["current_lat"])
        lng = float(status["current_lng"])
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
        # 过滤掉赶不上装货窗的货源、扫描期间已下架的货源，并补充执行时间/收益估算。
        valid_items = []
        for item in items:
            cargo = item.get("cargo", {})
            remove_time = cargo.get("remove_time")
            if remove_time:
                remove_min = _wall_time_to_minutes(remove_time)
                if remove_min < real_time:
                    continue
            metrics = _estimate_cargo_metrics(item, real_time)
            if not metrics.get("feasible", False):
                continue
            enriched_item = dict(item)
            enriched_item["_metrics"] = metrics
            valid_items.append(enriched_item)
        self._logger.info(
            "cargo filter: %d -> %d (removed %d expired load_time)",
            len(items),
            len(valid_items),
            len(items) - len(valid_items),
        )
        if not valid_items:
            self._logger.info("no valid cargo, waiting 15 minutes")
            return {"action": "wait", "params": {"duration_minutes": 15}}

        valid_items.sort(key=lambda x: float((x.get("_metrics") or {}).get("estimated_score", 0.0)), reverse=True)
        prompt_items = valid_items[:PROMPT_CANDIDATE_LIMIT]
        allowed_cargo_ids = {str((item.get("cargo") or {}).get("cargo_id", "")).strip() for item in prompt_items}
        allowed_cargo_ids.discard("")
        preferences = [str(x) for x in (status.get("preferences", []) or [])]
        best_item = prompt_items[0]
        best_cargo_id = str((best_item.get("cargo") or {}).get("cargo_id", "")).strip()
        best_metrics = best_item.get("_metrics") or {}
        best_score = float(best_metrics.get("estimated_score", 0.0) or 0.0)
        best_net = float(best_metrics.get("estimated_net_after_distance_cost", 0.0) or 0.0)
        if not preferences and best_cargo_id and best_score >= DIRECT_TAKE_SCORE_THRESHOLD and best_net >= DIRECT_TAKE_NET_THRESHOLD:
            self._logger.info(
                "direct take_order driver_id=%s cargo_id=%s score=%.2f net=%.2f",
                driver_id,
                best_cargo_id,
                best_score,
                best_net,
            )
            return {"action": "take_order", "params": {"cargo_id": best_cargo_id}}
        prompt = self._build_prompt(driver_id=driver_id, status=status, items=prompt_items, real_time=real_time)
        self._logger.info("prompt_content driver_id=%s prompt=%s", driver_id, prompt[:500])
        model_resp = self._api.model_chat_completion(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "你是货运调度决策器。"
                            "只允许输出一个JSON对象，格式必须是"
                            '{"action":"take_order|reposition|wait","params":{...}}。'
                            "禁止输出markdown、解释或额外文本。"
                            "当action是take_order时，params必须包含cargo_id字符串；"
                            "当action是reposition时，params必须包含latitude和longitude数值；"
                            "当action是wait时，params必须包含duration_minutes正整数。"
                            "simulation_progress_minutes 为自 2026-03-01 00:00:00 起的仿真经过分钟数。"
                            "候选货源含 load_time 为装货时间窗 [开始,结束]（墙钟）；"
                            "若当前仿真时刻晚于窗结束则 take_order 会失败。"
                            "若订单无法在仿真上界前完成，动作仍可能执行，但收益不会被计入，应避免。"
                            "cargo_candidates 已通过基础可行性过滤，并按 estimated_score 从高到低排序。"
                            "优先选择 estimated_score 最高且不明确违反 preference_constraints 的 take_order。"
                            "不要臆造未给出的隐藏偏好或额外风险。"
                            "只有当所有候选收益为非正，或明确违反已解析的硬约束时，才选择 wait。"
                            "take_order 只能选择 cargo_candidates 中出现的 cargo_id，禁止编造 cargo_id。"
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
                self._logger.warning("model returned cargo_id outside prompt candidates: %s; fallback wait", cargo_id)
                return {"action": "wait", "params": {"duration_minutes": 15}}
        self._logger.info(
            "decision output driver_id=%s action=%s params=%s",
            driver_id,
            action.get("action"),
            action.get("params"),
        )
        return action

    def _build_prompt(self, driver_id: str, status: dict[str, Any], items: list[dict[str, Any]], real_time: int) -> str:
        cargo_candidates: list[dict[str, Any]] = []
        preferences = [str(x) for x in (status.get("preferences", []) or [])]
        preference_constraints = _extract_preference_constraints(preferences)
        for item in items:
            cargo = item.get("cargo", {})
            metrics = item.get("_metrics")
            if not isinstance(metrics, dict):
                metrics = _estimate_cargo_metrics(item, real_time)
            cargo_candidates.append(
                {
                    "cargo_id": cargo.get("cargo_id"),
                    "price_yuan": metrics.get("price_yuan"),
                    "estimated_score": metrics.get("estimated_score"),
                    "estimated_net_after_distance_cost": metrics.get("estimated_net_after_distance_cost"),
                    "pickup_km": metrics.get("pickup_km"),
                    "haul_km": metrics.get("haul_km"),
                    "pickup_minutes": metrics.get("pickup_minutes"),
                    "load_wait_minutes": metrics.get("load_wait_minutes"),
                    "haul_minutes": metrics.get("haul_minutes"),
                    "arrival_min": metrics.get("arrival_min"),
                    "ready_min": metrics.get("ready_min"),
                    "finish_min": metrics.get("finish_min"),
                    "total_exec_minutes": metrics.get("total_exec_minutes"),
                    "cost_time_minutes": cargo.get("cost_time_minutes"),
                    "load_time": cargo.get("load_time"),
                    "start": cargo.get("start"),
                    "end": cargo.get("end"),
                    "distance_km": item.get("distance_km"),
                }
            )
        decision_context = {
            "driver_id": driver_id,
            "simulation_progress_minutes": real_time,
            "driver_status": {
                "current_lat": status.get("current_lat"),
                "current_lng": status.get("current_lng"),
                "truck_length": status.get("truck_length"),
                "completed_order_count": status.get("completed_order_count"),
            },
            "decision_rules": [
                "price_yuan is already in yuan; do NOT divide it by 100.",
                "estimated_net_after_distance_cost subtracts pickup+haul distance cost. estimated_score also subtracts time opportunity cost.",
                "cargo_candidates are sorted by estimated_score descending; candidate[0] is the default best cargo.",
                "Prefer take_order for the highest estimated_score candidate unless there is an explicit hard-constraint violation.",
                "Only take orders with positive estimated_net_after_distance_cost unless preference or positioning benefits are clear.",
                "Taking orders is strongly preferred over waiting. Only wait if no candidate has positive expected net income after penalties.",
                "When waiting, use short durations (10-15 minutes) to check for new cargo sooner.",
            ],
            "cargo_candidates": cargo_candidates,
        }
        if preferences:
            decision_context["preferences"] = preferences
            decision_context["preference_constraints"] = preference_constraints
            decision_context["decision_rules"].extend(
                [
                    "preference_constraints are parsed hints from preferences; obey explicit constraints but do not infer extra hidden rules.",
                    "If a candidate does not explicitly conflict with parsed constraints, treat it as selectable.",
                ]
            )
        return json.dumps(decision_context, ensure_ascii=False)

    def _parse_action(self, model_resp: dict[str, Any]) -> dict[str, Any]:
        choices = model_resp.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("模型返回缺少 choices")
        message = choices[0].get("message", {})
        content = message.get("content")
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
            latitude = float(params["latitude"])
            longitude = float(params["longitude"])
            return {"action": "reposition", "params": {"latitude": latitude, "longitude": longitude}}
        duration_minutes = int(params["duration_minutes"])
        if duration_minutes <= 0:
            raise ValueError("wait.duration_minutes 必须为正整数")
        return {"action": "wait", "params": {"duration_minutes": duration_minutes}}


# ---------------------------------------------------------------------------
# 工具函数：时间/坐标/偏好解析
# ---------------------------------------------------------------------------

_TIME_RE = re.compile(r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)(?!\d)")
_COORD_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*[,，]\s*(-?\d+(?:\.\d+)?)")
_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")

_CN_NUMBERS = {
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}


def _wall_time_to_minutes(wall_time_str: str) -> int:
    dt = datetime.strptime(wall_time_str.strip(), "%Y-%m-%d %H:%M:%S")
    delta = dt - SIMULATION_EPOCH
    return int(delta.total_seconds() // 60)


def _pickup_minutes(distance_km: float, speed_km_per_hour: float = 60.0) -> int:
    if distance_km <= 1e-6:
        return 0
    return max(1, math.ceil((distance_km / speed_km_per_hour) * 60))


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
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


def _parse_load_window(load_time: Any) -> tuple[int, int] | None:
    if load_time is None:
        return None
    if not isinstance(load_time, list) or len(load_time) != 2:
        return None
    start_text = str(load_time[0]).strip()
    end_text = str(load_time[1]).strip()
    if not start_text or not end_text:
        return None
    start_min = _wall_time_to_minutes(start_text)
    end_min = _wall_time_to_minutes(end_text)
    if end_min < start_min:
        return None
    return start_min, end_min


def _estimate_cargo_metrics(item: dict[str, Any], real_time: int) -> dict[str, Any]:
    """估算 take_order 的执行时间与简化收益。

    与 simkit.take_order 保持一致：
    - 接货空驶距离 <= 1e-6 km 时 pickup_minutes = 0；
    - 否则按 ceil(distance / speed * 60)，最少 1 分钟；
    - 早到装货窗则等待到 load_start；晚于 load_end 则不可行；
    - 干线耗时使用 cargo.cost_time_minutes。

    注意 query_cargo 已把 cargo.price 从“分”转成“元”，这里不能再 /100。
    """
    cargo = item.get("cargo") or {}
    pickup_km = float(item.get("distance_km", 0.0) or 0.0)
    pickup_minutes = _pickup_minutes(pickup_km, DEFAULT_REPOSITION_SPEED_KM_PER_HOUR)
    arrival_min = int(real_time) + pickup_minutes

    feasible = True
    risk_reasons: list[str] = []
    load_wait_minutes = 0
    ready_min = arrival_min
    load_window = _parse_load_window(cargo.get("load_time"))
    if cargo.get("load_time") is not None and load_window is None:
        feasible = False
        risk_reasons.append("invalid_load_time")
    elif load_window is not None:
        load_start_min, load_end_min = load_window
        if arrival_min > load_end_min:
            feasible = False
            risk_reasons.append("miss_load_window")
        else:
            load_wait_minutes = max(0, load_start_min - arrival_min)
            ready_min = arrival_min + load_wait_minutes

    # Demo 引擎会在接货空驶/等待装货期间同步货源上下线；如果货源在真正开始干线前
    # 已过 remove_time，repo.remove_by_id 会失败并导致本步无收益但时间已被推进。
    # 因此除了“决策时在线”，还要确保货源至少存活到 ready_min。
    remove_time = cargo.get("remove_time")
    if remove_time:
        try:
            remove_min = _wall_time_to_minutes(str(remove_time))
            if remove_min < ready_min:
                feasible = False
                risk_reasons.append("expires_before_ready")
        except ValueError:
            feasible = False
            risk_reasons.append("invalid_remove_time")

    try:
        haul_minutes = int(cargo.get("cost_time_minutes", 0) or 0)
    except (TypeError, ValueError):
        haul_minutes = 0
        feasible = False
        risk_reasons.append("invalid_cost_time")
    if haul_minutes < 0:
        feasible = False
        risk_reasons.append("negative_cost_time")

    start = cargo.get("start") or {}
    end = cargo.get("end") or {}
    try:
        start_lat = float(start["lat"])
        start_lng = float(start["lng"])
        end_lat = float(end["lat"])
        end_lng = float(end["lng"])
        haul_km = _haversine_km(start_lat, start_lng, end_lat, end_lng)
    except (KeyError, TypeError, ValueError):
        haul_km = 0.0
        feasible = False
        risk_reasons.append("invalid_start_end")

    finish_min = ready_min + max(0, haul_minutes)
    total_exec_minutes = max(0, finish_min - int(real_time))
    price_yuan = float(cargo.get("price", 0.0) or 0.0)
    distance_cost = (pickup_km + haul_km) * DEFAULT_COST_PER_KM
    estimated_net_after_distance_cost = price_yuan - distance_cost
    estimated_score = estimated_net_after_distance_cost - total_exec_minutes * TIME_VALUE_YUAN_PER_MINUTE

    return {
        "feasible": feasible,
        "risk_reasons": risk_reasons,
        "price_yuan": round(price_yuan, 2),
        "pickup_km": round(pickup_km, 2),
        "haul_km": round(haul_km, 2),
        "pickup_minutes": int(pickup_minutes),
        "load_wait_minutes": int(load_wait_minutes),
        "haul_minutes": int(max(0, haul_minutes)),
        "arrival_min": int(arrival_min),
        "ready_min": int(ready_min),
        "finish_min": int(finish_min),
        "total_exec_minutes": int(total_exec_minutes),
        "distance_cost": round(distance_cost, 2),
        "estimated_net_after_distance_cost": round(estimated_net_after_distance_cost, 2),
        "estimated_score": round(estimated_score, 2),
    }


def _time_text_to_min(text: str) -> int | None:
    match = _TIME_RE.search(text)
    if not match:
        return None
    return int(match.group(1)) * 60 + int(match.group(2))


def _all_time_texts(text: str) -> list[str]:
    return [f"{int(h):02d}:{m}" for h, m in _TIME_RE.findall(text)]


def _first_number(text: str) -> float | None:
    match = _NUM_RE.search(text)
    if match:
        return float(match.group(1))
    for key, value in _CN_NUMBERS.items():
        if key in text:
            return float(value)
    return None


def _number_before_unit(text: str, unit_pattern: str) -> float | None:
    match = re.search(rf"(\d+(?:\.\d+)?)\s*{unit_pattern}", text)
    if match:
        return float(match.group(1))
    match = re.search(rf"([一二两三四五六七八九十])\s*{unit_pattern}", text)
    if match:
        return float(_CN_NUMBERS.get(match.group(1), 0))
    return None


def _first_coord(text: str) -> dict[str, float] | None:
    match = _COORD_RE.search(text)
    if not match:
        return None
    return {"lat": float(match.group(1)), "lng": float(match.group(2))}


def _radius_km(text: str, default: float | None = None) -> float | None:
    match = re.search(r"半径\s*(\d+(?:\.\d+)?)\s*(?:km|公里)", text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r"附近\s*(\d+(?:\.\d+)?)\s*(?:km|公里)", text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:km|公里)\s*内", text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return default


def _extract_preference_constraints(preferences: list[str]) -> dict[str, Any]:
    """把官方注入的自然语言偏好抽取成轻量结构化提示。

    不按 driver_id 写死，只识别常见的时间窗、连续休息、休息天、回家、目标点、禁入区。
    未识别文本仍保留在 raw_preferences 里交给模型理解。
    """
    constraints: dict[str, Any] = {
        "raw_preferences": preferences,
        "daily_rest": None,
        "quiet_windows": [],
        "monthly_off_days": None,
        "home_return": None,
        "target_visits": None,
        "forbidden_zones": [],
        "notes": [],
    }
    for pref in preferences:
        text = str(pref).strip()
        if not text:
            continue
        times = _all_time_texts(text)
        coord = _first_coord(text)

        if "连续" in text and "休息" in text:
            hours = _number_before_unit(text, "小时")
            if hours:
                constraints["daily_rest"] = {
                    "min_continuous_minutes": int(hours * 60),
                    "source": text,
                }

        if "完全不出车" in text or ("不出车" in text and ("天" in text or "月" in text)):
            days = _number_before_unit(text, "天") or _first_number(text)
            if days:
                constraints["monthly_off_days"] = {
                    "min_days": int(days),
                    "source": text,
                }

        if ("回家" in text or "回到家" in text or "家坐标" in text) and coord:
            home: dict[str, Any] = {
                "lat": coord["lat"],
                "lng": coord["lng"],
                "radius_km": _radius_km(text, default=1.0),
                "source": text,
            }
            if times:
                home["return_before"] = times[0]
            if len(times) >= 2:
                home["quiet_until"] = times[1]
                home["forbid_actions_after_return"] = ["take_order", "reposition"]
            constraints["home_return"] = home

        if ("目标点" in text or "固定地点" in text or "到一个固定" in text) and coord:
            visits = _number_before_unit(text, "次") or _first_number(text)
            constraints["target_visits"] = {
                "lat": coord["lat"],
                "lng": coord["lng"],
                "min_visit_days": int(visits) if visits else None,
                "same_day_count_once": "同一天" in text,
                "radius_km": _radius_km(text, default=1.0),
                "source": text,
            }

        if ("禁入" in text or "不想进入" in text or "禁止进入" in text) and coord:
            constraints["forbidden_zones"].append(
                {
                    "lat": coord["lat"],
                    "lng": coord["lng"],
                    "radius_km": _radius_km(text, default=2.0),
                    "source": text,
                }
            )

        if len(times) >= 2 and any(key in text for key in ("不接单", "不空驶", "休整", "不再接单", "不再空驶")):
            forbid_actions: list[str] = []
            if "不接单" in text or "不再接单" in text:
                forbid_actions.append("take_order")
            if "不空驶" in text or "不再空驶" in text or ("不" in text and "空驶" in text):
                forbid_actions.append("reposition")
            if not forbid_actions:
                forbid_actions = ["take_order", "reposition"]
            constraints["quiet_windows"].append(
                {
                    "start": times[0],
                    "end": times[1],
                    "forbid_actions": sorted(set(forbid_actions)),
                    "source": text,
                }
            )

    constraints["notes"].append("Treat parsed constraints as hard constraints when selecting an action.")
    return constraints
