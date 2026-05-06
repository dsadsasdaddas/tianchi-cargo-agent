"""评测入口：配置 `config/config.json` 后执行本脚本，结果写入与 `server` 平级的 `demo/results/`（无 HTTP 服务）。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_SERVER_ROOT = Path(__file__).resolve().parent
_DEMO_ROOT = _SERVER_ROOT.parent
if str(_DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEMO_ROOT))
if str(_SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVER_ROOT))

from bench.evaluation_runner import EvaluationRunner


def main() -> int:
    parser = argparse.ArgumentParser(description="离线仿真评测（结果写入 demo/results/）")
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="配置文件路径，默认使用 server/config/config.json",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        metavar="N",
        help="全局最大步数上限（调试用；省略则使用 config 中 simulation_max_steps）",
    )
    args = parser.parse_args()
    config_path = Path(args.config) if args.config else None
    try:
        runner = EvaluationRunner(config_path=config_path, max_steps=args.max_steps)
        runner.run()
        return 0
    except Exception:
        logging.exception("evaluation failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
