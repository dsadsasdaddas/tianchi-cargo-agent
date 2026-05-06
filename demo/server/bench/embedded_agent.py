"""进程内决策：连接 simkit 与模型网关，构造可注入主循环的决策引擎。"""

from __future__ import annotations

import json
import logging
from typing import Any

from agent.model_decision_service import ModelDecisionService
from simkit import simulation_actions
from simkit.cargo_repository import CargoRepository
from simkit.driver_state_manager import DriverStateManager
from simkit.ports import SimulationApiPort

from .model_gateway_client import ModelGatewayClient


class EmbeddedDecisionEnvironment:
    """决策一步内可调用的环境：司机/货源查询 + 模型补全。"""

    def __init__(
        self,
        repo: CargoRepository,
        manager: DriverStateManager,
        model_gateway: ModelGatewayClient,
        *,
        nearest_cargo_limit: int = 100,
        cargo_view_batch_size: int = 10,
    ) -> None:
        self._repo = repo
        self._manager = manager
        self._model_gateway = model_gateway
        self._nearest_cargo_limit = nearest_cargo_limit
        self._cargo_view_batch_size = cargo_view_batch_size
        self._logger = logging.getLogger("bench.embedded_agent.environment")
        self._last_model_usage = self._empty_usage()

    def get_driver_status(self, driver_id: str) -> dict[str, Any]:
        return self._manager.get_driver_status(driver_id)

    def query_cargo(self, driver_id: str, latitude: float, longitude: float) -> dict[str, Any]:
        sim_min_before = self._manager.get_simulation_progress_minutes()
        raw = simulation_actions.query_cargo(
            self._repo,
            self._manager,
            driver_id,
            latitude,
            longitude,
            k=self._nearest_cargo_limit,
        )
        items = raw.get("items", [])
        if not isinstance(items, list):
            raise TypeError("query_cargo 返回的 items 必须为列表")
        simulation_actions.apply_cargo_query_scan_cost(
            self._repo,
            self._manager,
            driver_id,
            len(items),
            cargo_view_batch_size=self._cargo_view_batch_size,
        )
        sim_min_after = self._manager.get_simulation_progress_minutes()
        scan_cost_minutes = sim_min_after - sim_min_before
        self._logger.info(
            "query_cargo ok driver_id=%s items=%s scan_cost_min=%s",
            driver_id,
            len(items),
            scan_cost_minutes,
        )
        return raw

    def model_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        resp = self._model_gateway.chat_completion(payload)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            raise ValueError("模型网关返回不是 JSON 对象")
        usage = self._extract_model_usage(data)
        self._last_model_usage["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
        self._last_model_usage["completion_tokens"] += int(usage.get("completion_tokens", 0))
        self._last_model_usage["reasoning_tokens"] += int(usage.get("reasoning_tokens", 0))
        self._last_model_usage["total_tokens"] += int(usage.get("total_tokens", 0))
        self._logger.info("model_chat_completion ok")
        return data

    def reset_last_model_usage(self) -> None:
        self._last_model_usage = self._empty_usage()

    def get_last_model_usage(self) -> dict[str, int]:
        return dict(self._last_model_usage)

    @staticmethod
    def _empty_usage() -> dict[str, int]:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
        }

    @staticmethod
    def _extract_model_usage(model_resp: dict[str, Any]) -> dict[str, int]:
        usage = model_resp.get("usage", {})
        if not isinstance(usage, dict):
            return EmbeddedDecisionEnvironment._empty_usage()
        completion_details = usage.get("completion_tokens_details") or {}
        reasoning_tokens_raw = 0
        if isinstance(completion_details, dict):
            reasoning_tokens_raw = completion_details.get("reasoning_tokens") or 0
        prompt_tokens = max(0, int(usage.get("prompt_tokens", 0)))
        completion_tokens = max(0, int(usage.get("completion_tokens", 0)))
        total_tokens = max(0, int(usage.get("total_tokens", 0)))
        reasoning_tokens = max(0, int(reasoning_tokens_raw))
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": total_tokens,
        }


class EmbeddedAgentDecisionEngine:
    """主循环调用的固定接口。"""

    def __init__(self, decision_service: ModelDecisionService, environment: EmbeddedDecisionEnvironment) -> None:
        self._decision_service = decision_service
        self._environment = environment
        self._logger = logging.getLogger("bench.embedded_agent.engine")

    def decide(self, driver_id: str) -> dict[str, Any]:
        self._environment.reset_last_model_usage()
        action = self._decision_service.decide(driver_id)
        # 服务端强制覆盖 token 统计，避免使用选手侧上报值。
        action["model_usage"] = self._environment.get_last_model_usage()
        usage = action["model_usage"]
        params = action.get("params", {})
        params_compact = json.dumps(params, ensure_ascii=False, separators=(",", ":")) if isinstance(params, dict) else str(params)
        self._logger.info(
            "decision driver_id=%s action=%s params=%s tokens prompt=%s completion=%s reasoning=%s total=%s",
            driver_id,
            action.get("action"),
            params_compact,
            int(usage.get("prompt_tokens", 0)),
            int(usage.get("completion_tokens", 0)),
            int(usage.get("reasoning_tokens", 0)),
            int(usage.get("total_tokens", 0)),
        )
        return action


def build_embedded_agent_decision_engine(
    repo: CargoRepository,
    manager: DriverStateManager,
    model_gateway: ModelGatewayClient,
) -> EmbeddedAgentDecisionEngine:
    environment = EmbeddedDecisionEnvironment(
        repo=repo,
        manager=manager,
        model_gateway=model_gateway,
    )
    environment_port: SimulationApiPort = environment
    decision_service = ModelDecisionService(environment_port)
    return EmbeddedAgentDecisionEngine(decision_service, environment)
