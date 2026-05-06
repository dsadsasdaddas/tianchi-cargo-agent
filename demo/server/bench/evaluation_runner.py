"""一次性评测编排：加载配置 → 仿真主循环；结果写入 `results_dir` 即结束。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from simkit.cargo_repository import CargoRepository
from simkit.driver_state_manager import DriverStateManager

from .embedded_agent import build_embedded_agent_decision_engine
from .model_gateway_client import ModelGatewayClient
from .settings import AppSettings, load_settings
from .simulation_orchestrator import SimulationOrchestrator


class EvaluationRunner:
    """配置驱动：跑完仿真并落盘动作日志与 run_summary；不含收益统计。"""

    def __init__(self, config_path: Path | None = None, max_steps: int | None = None) -> None:
        self._config_path = config_path
        self._max_steps = max_steps
        self._logger = logging.getLogger("bench.evaluation_runner")

    def run(self) -> dict[str, Any]:
        settings = load_settings(self._config_path)
        self._configure_logging(settings)
        self._logger.info(
            "evaluation begin config=%s max_steps=%s",
            self._config_path or "default",
            self._max_steps if self._max_steps is not None else "(config)",
        )

        repo = CargoRepository(settings.cargo_dataset_path)
        repo.load()
        manager = DriverStateManager(settings.drivers_path)
        manager.load()

        model_gateway = ModelGatewayClient(
            api_url=settings.model_api_url,
            api_key=settings.model_api_key,
            default_model_name=settings.model_name,
            timeout_seconds=settings.model_timeout_seconds,
        )
        try:
            embedded_engine = build_embedded_agent_decision_engine(
                repo=repo,
                manager=manager,
                model_gateway=model_gateway,
            )
            orchestrator = SimulationOrchestrator(
                cargo_repository=repo,
                driver_state_manager=manager,
                agent_decision=embedded_engine,
                results_dir=settings.results_dir,
                reposition_speed_km_per_hour=settings.reposition_speed_km_per_hour,
                simulation_max_steps=settings.simulation_max_steps,
                simulation_duration_days=settings.simulation_duration_days,
            )

            manager.start_simulation_minutes(driver_id=None, progress_minutes=0)
            summary = orchestrator.run(max_steps=self._max_steps)
            self._logger.info("simulation finished results_dir=%s summary=%s", settings.results_dir, summary)
            return summary
        finally:
            model_gateway.close()

    def _configure_logging(self, settings: AppSettings) -> None:
        settings.log_dir.mkdir(parents=True, exist_ok=True)
        log_file = settings.log_dir / "server_runtime.log"
        fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        handlers = [
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ]
        for handler in handlers:
            handler.setFormatter(fmt)
        logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)
