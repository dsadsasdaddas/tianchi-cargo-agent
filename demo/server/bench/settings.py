"""加载服务端评测配置。

- 数据路径（货源、司机）相对 `demo/server`。
- 结果与日志路径（`results_dir` / `log_dir`）相对 `demo` 根目录，与 `server` 平级。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

# `demo/server` 根目录（本包位于 `demo/server/bench/`）
SERVER_ROOT = Path(__file__).resolve().parent.parent
# `demo` 根目录（与 `server` 平级，用于 `results/` 等输出）
DEMO_ROOT = SERVER_ROOT.parent
DEFAULT_CONFIG_PATH = SERVER_ROOT / "config" / "config.json"


def _resolve_model_api_key(raw_value: object) -> str:
    """优先使用环境变量，避免密钥写入仓库或共享屏幕时泄露。"""
    for name in ("DASHSCOPE_API_KEY", "TIANCHI_MODEL_API_KEY"):
        v = os.environ.get(name, "").strip()
        if v:
            return v
    if isinstance(raw_value, str) and raw_value.strip():
        return raw_value.strip()
    raise ValueError(
        "未配置模型 API Key：请设置环境变量 DASHSCOPE_API_KEY（或 TIANCHI_MODEL_API_KEY），"
        "或在 config.json 中填写 model_api_key；切勿将含真实密钥的 config 提交到公开仓库。"
    )


@dataclass(frozen=True)
class AppSettings:
    cargo_dataset_path: Path
    drivers_path: Path
    reposition_speed_km_per_hour: float
    results_dir: Path
    log_dir: Path
    simulation_max_steps: int
    simulation_duration_days: int
    model_api_url: str
    model_api_key: str
    model_name: str
    model_timeout_seconds: float


def load_settings(config_path: Path | None = None) -> AppSettings:
    path = config_path or DEFAULT_CONFIG_PATH
    raw = json.loads(path.read_text(encoding="utf-8"))
    cargo_rel = raw.get("cargo_dataset_path")
    if not cargo_rel or not isinstance(cargo_rel, str):
        raise ValueError("config.json 缺少有效字段 cargo_dataset_path（字符串）")
    drivers_rel = raw.get("drivers_path")
    if not drivers_rel or not isinstance(drivers_rel, str):
        raise ValueError("config.json 缺少有效字段 drivers_path（字符串）")
    reposition_speed = raw.get("reposition_speed_km_per_hour")
    if not isinstance(reposition_speed, (float, int)) or float(reposition_speed) <= 0:
        raise ValueError("config.json 缺少有效字段 reposition_speed_km_per_hour（正数）")
    results_rel = raw.get("results_dir")
    if not isinstance(results_rel, str) or not results_rel.strip():
        raise ValueError("config.json 缺少有效字段 results_dir（字符串）")
    log_rel = raw.get("log_dir", "results/logs")
    if not isinstance(log_rel, str) or not log_rel.strip():
        raise ValueError("config.json 缺少有效字段 log_dir（字符串）")
    simulation_max_steps = raw.get("simulation_max_steps")
    if not isinstance(simulation_max_steps, int) or simulation_max_steps <= 0:
        raise ValueError("config.json 缺少有效字段 simulation_max_steps（正整数）")
    simulation_duration_days = raw.get("simulation_duration_days", 31)
    if not isinstance(simulation_duration_days, int) or simulation_duration_days <= 0:
        raise ValueError("config.json 缺少有效字段 simulation_duration_days（正整数）")
    model_api_url = raw.get("model_api_url")
    if not isinstance(model_api_url, str) or not model_api_url.strip():
        raise ValueError("config.json 缺少有效字段 model_api_url（字符串）")
    model_api_key = _resolve_model_api_key(raw.get("model_api_key", ""))
    model_name = raw.get("model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("config.json 缺少有效字段 model_name（字符串）")
    model_timeout_seconds = raw.get("model_timeout_seconds", 60)
    if not isinstance(model_timeout_seconds, (int, float)) or float(model_timeout_seconds) <= 0:
        raise ValueError("config.json 缺少有效字段 model_timeout_seconds（正数）")
    cargo_path = Path(cargo_rel)
    drivers_path = Path(drivers_rel)
    results_path = Path(results_rel)
    log_path = Path(log_rel)
    if not cargo_path.is_absolute():
        cargo_path = SERVER_ROOT / cargo_path
    if not drivers_path.is_absolute():
        drivers_path = SERVER_ROOT / drivers_path
    if not results_path.is_absolute():
        results_path = DEMO_ROOT / results_path
    if not log_path.is_absolute():
        log_path = DEMO_ROOT / log_path
    return AppSettings(
        cargo_dataset_path=cargo_path.resolve(),
        drivers_path=drivers_path.resolve(),
        reposition_speed_km_per_hour=float(reposition_speed),
        results_dir=results_path.resolve(),
        log_dir=log_path.resolve(),
        simulation_max_steps=simulation_max_steps,
        simulation_duration_days=simulation_duration_days,
        model_api_url=model_api_url.strip(),
        model_api_key=model_api_key.strip(),
        model_name=model_name.strip(),
        model_timeout_seconds=float(model_timeout_seconds),
    )
