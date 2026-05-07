"""模型决策服务：依赖 `simkit.ports.SimulationApiPort`，由评测进程注入具体环境。"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from typing import Any

from simkit.ports import SimulationApiPort

SIMULATION_EPOCH = datetime(2026, 3, 1, 0, 0, 0)


def _wall_time_to_minutes(wall_time_str: str) -> int:
    dt = datetime.strptime(wall_time_str.strip(), "%Y-%m-%d %H:%M:%S")
    delta = dt - SIMULATION_EPOCH
    return int(delta.total_seconds() // 60)


def _pickup_minutes(distance_km: float, speed_km_per_hour: float = 60.0) -> int:
    if distance_km <= 1e-6:
        return 0
    return max(1, math.ceil((distance_km / speed_km_per_hour) * 60))


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
        # 过滤掉赶不上装货窗的货源，以及扫描期间已下架的货源
        valid_items = []
        for item in items:
            cargo = item.get("cargo", {})
            # 检查 remove_time：扫描期间可能已下架
            remove_time = cargo.get("remove_time")
            if remove_time:
                remove_min = _wall_time_to_minutes(remove_time)
                if remove_min < real_time:
                    continue  # 已下架，跳过
            # 检查 load_time：赶不上装货窗
            load_time = cargo.get("load_time")
            if load_time is not None and isinstance(load_time, list) and len(load_time) == 2:
                load_end = _wall_time_to_minutes(str(load_time[1]))
                distance_km = float(item.get("distance_km", 0))
                pickup_min = _pickup_minutes(distance_km)
                arrival = real_time + pickup_min
                if arrival > load_end:
                    continue  # 赶不上装货窗，跳过
            valid_items.append(item)
        self._logger.info(
            "cargo filter: %d -> %d (removed %d expired load_time)",
            len(items),
            len(valid_items),
            len(items) - len(valid_items),
        )
        # 没有可用货源，直接休息（不调模型）
        if not valid_items:
            self._logger.info("no valid cargo, waiting 30 minutes")
            return {"action": "wait", "params": {"duration_minutes": 30}}
        # 有可用货源，调模型决策（只传过滤后的货源）
        prompt = self._build_prompt(driver_id=driver_id, status=status, items=valid_items, real_time=real_time)
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
                            "若接单后无法在仿真总时长内完成装货与干线，take_order 会失败（detail 含 simulation_horizon_exceeded），且不推进时间与位置。"
                            "preferences 是司机的个性化偏好，违反会扣钱。决策时必须优先遵守偏好约束，避免产生扣罚。"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "response_format": {"type": "json_object"},
                "enable_thinking": False,
            }
        )
        action = self._parse_action(model_resp)
        self._logger.info(
            "decision output driver_id=%s action=%s params=%s",
            driver_id,
            action.get("action"),
            action.get("params"),
        )
        return action

    def _build_prompt(self, driver_id: str, status: dict[str, Any], items: list[dict[str, Any]], real_time: int) -> str:
        cargo_candidates: list[dict[str, Any]] = []
        for item in items[:20]:
            cargo = item.get("cargo", {})
            cargo_candidates.append(
                {
                    "cargo_id": cargo.get("cargo_id"),
                    "price": cargo.get("price"),
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
            "preferences": list(status.get("preferences", [])),
            "cargo_candidates": cargo_candidates,
        }
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
