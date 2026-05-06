# 评测 Demo（统一说明）

本目录用于离线仿真与收益计算：

- `server/`：评测编排入口与运行配置
- `agent/`：选手决策逻辑
- `simkit/`：仿真状态、动作规则与跨模块协议
- `calc_monthly_income.py`：依据 `demo/results/` 中的仿真输出做收益与特定偏好规则校验（偏好变化时，计算脚本中的规则也需同步调整）

依赖方向：`simkit` ← `agent` ← `server`。

## 目录结构

```text
demo/
├─ agent/                        # 决策实现（如 model_decision_service.py）
├─ simkit/                       # 仿真规则、状态管理、ports 协议
├─ results/                      # 动作日志、history、run_summary、收益输出（与 server 平级）
├─ server/
│  ├─ main.py                    # 仿真入口（无 HTTP 服务）
│  ├─ bench/                     # 评测编排实现
│  ├─ config/
│  │  ├─ config.example.json
│  │  └─ config.json
│  └─ data/                      # drivers / cargo 数据
└─ calc_monthly_income.py        # 收益计算与动作合法性校验
```

## 快速开始

1) 安装依赖

```bash
cd demo/server
pip install -r requirements.txt
```

2) 配置

- 复制 `server/config/config.example.json` 为 `server/config/config.json`
- 推荐使用环境变量 `DASHSCOPE_API_KEY`

3) 运行仿真

```bash
cd demo/server
python main.py
```

结果会写入 `demo/results/`（包括 `actions_*.jsonl`、`history/`、`run_summary_202603.json` 等）。

4) 计算收益

```bash
cd demo
python calc_monthly_income.py
```

输出文件：`demo/results/monthly_income_202603.json`。

## 结果文件说明

- `actions_*.jsonl`：逐步动作记录（含时间推进、位置前后、query/action耗时、结果）
- `history/`：每次仿真开始前归档的旧结果（按时间戳分子目录）
- `run_summary_202603.json`：本次仿真汇总（含仿真天数）
- `monthly_income_202603.json`：司机收益、token、偏好罚分、校验信息

## 安全注意

- 不要把真实密钥写入仓库（`config.json` 建议本地维护）
- 运行产物建议不入库（`demo/results/` 已在 `.gitignore` 中处理）
