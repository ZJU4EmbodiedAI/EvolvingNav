#!/usr/bin/env python3
"""Train and evaluate P4D-Belief on the frozen 30-day HSSD dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from readyagent.p4d_belief.data import audit_training_contract, load_catalog
from readyagent.p4d_belief.models import ModelConfig
from readyagent.p4d_belief.training import (
    TrainConfig,
    evaluate_classical_baselines,
    evaluate_model,
    summarize_neural,
    train_model,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=("p4d", "direct", "gru"), default=["p4d"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--use-instance-identity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the observable target instance UUID as a stable identity embedding.",
    )
    parser.add_argument("--use-compatibility", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--routine-train-fraction",
        type=float,
        default=0.5,
        help="Expected Routine fraction in the Routine/Static weighted sampler.",
    )
    parser.add_argument(
        "--validation-subset",
        choices=("main", "routine_all", "routine_exact", "static"),
        default="main",
        help="Subset whose NLL selects the early-stopping checkpoint.",
    )
    parser.add_argument(
        "--unknown-train-boost",
        type=float,
        default=1.0,
        help="Within-Routine sampling multiplier for true Unknown targets.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-split", choices=("val", "test"), default="val")
    parser.add_argument("--skip-classical", action="store_true")
    return parser.parse_args()


def write_report(
    output: Path,
    *,
    audit: dict,
    classical: dict,
    aggregate: dict,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> None:
    lines = [
        "# P4D-Belief 30天训练与测试报告",
        "",
        "## 实验设置",
        "",
        f"- 模型：Query-Conditioned Continuous-Time Transformer，{model_config.layers}层，hidden={model_config.hidden_dim}，heads={model_config.heads}",
        f"- 目标身份特征：{'启用稳定instance embedding' if model_config.use_instance_identity else '仅使用object category'}。",
        f"- 训练：Routine Exact/No-transition + Static；Routine采样占比={train_config.routine_train_fraction:.0%}；AdamW，lr={train_config.learning_rate}，batch={train_config.batch_size}",
        f"- 早停依据：验证集 `{train_config.validation_subset}` 的状态NLL。",
        f"- Unknown正样本训练采样倍率：{train_config.unknown_train_boost:g}×。",
        "- 模型输入不含 world type、grounding quality、GT activity、隐藏变化或监督字段；可选instance embedding只编码查询中已知的目标身份。",
        "- P4D-Belief仅使用最终状态NLL；Bayesian Update不参与训练。",
        "",
        "## 训练前审计",
        "",
        f"- 审计通过：{audit['passed']}",
        f"- Routine发生隐藏事件后回到last state的记录：{audit['routine_return_to_last_records']}",
        "- `y_moved`表示查询状态与last state是否不同，不表示期间是否曾发生移动。",
        "",
        "## Seed 聚合指标",
        "",
        "| 方法 | 子集 | Top-1 | Top-3 | NLL | ECE | 平均检查位置 | Persistence AUROC |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, subsets in classical.items():
        for subset in ("routine_exact", "routine_all", "random", "static"):
            row = subsets[subset]
            lines.append(
                f"| {method} | {subset} | {row['top1']:.3f} | {row['top3']:.3f} | {row['nll']:.3f} | {row['ece']:.3f} | {row['mean_receptacles_checked']:.2f} | {row['persistence_auroc']:.3f} |"
            )
    for method, subsets in aggregate.items():
        for subset in ("routine_exact", "routine_all", "random", "static"):
            row = subsets[subset]
            def cell(key: str) -> str:
                return f"{row[key]['mean']:.3f}±{row[key]['std']:.3f}"
            lines.append(
                f"| {method} | {subset} | {cell('top1')} | {cell('top3')} | {cell('nll')} | {cell('ece')} | {cell('mean_receptacles_checked')} | {cell('persistence_auroc')} |"
            )
    lines.extend(
        [
            "",
            "## 专项训练指标",
            "",
            "| 方法 | 子集 | 样本 | Top-1 | Top-3 | NLL | 平均检查位置 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method, subsets in classical.items():
        for subset in ("routine_unknown", "routine_returned"):
            if subset not in subsets:
                continue
            row = subsets[subset]
            lines.append(
                f"| {method} | {subset} | {row['records']} | {row['top1']:.3f} | {row['top3']:.3f} | {row['nll']:.3f} | {row['mean_receptacles_checked']:.2f} |"
            )
    for method, subsets in aggregate.items():
        for subset in ("routine_unknown", "routine_returned"):
            if subset not in subsets:
                continue
            row = subsets[subset]
            def diagnostic_cell(key: str) -> str:
                return f"{row[key]['mean']:.3f}±{row[key]['std']:.3f}"
            lines.append(
                f"| {method} | {subset} | {row['records']} | {diagnostic_cell('top1')} | {diagnostic_cell('top3')} | {diagnostic_cell('nll')} | {diagnostic_cell('mean_receptacles_checked')} |"
            )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 当前负证据的confidence/coverage来自受控模拟，不代表真实视觉检测器。",
            "- 数据仅有一个HSSD场景；共享语义位置编码避免了state-ID embedding，但尚不能证明跨场景泛化。",
            "- Unknown正标签表示对象物理离开建模场景，不表示遮挡或检测失败。",
            "- `returned_to_last`用于区分期间发生移动与查询时最终Persistence；它不直接作为状态预测模型输入。",
            "- 生成器GT Activity不进入模型；上下文只来自机器人可观察的巡检聚合。",
        ]
    )
    (output / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = arguments()
    dataset_root = args.dataset_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    catalog = load_catalog(dataset_root)
    audit = audit_training_contract(dataset_root)
    if not audit["passed"]:
        raise RuntimeError("training contract audit failed")
    (output / "training_contract_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    model_config = ModelConfig(
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        use_instance_identity=args.use_instance_identity,
        use_compatibility=args.use_compatibility,
    )
    train_config = TrainConfig(
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        patience=args.patience,
        gradient_clip=args.gradient_clip,
        routine_train_fraction=args.routine_train_fraction,
        validation_subset=args.validation_subset,
        unknown_train_boost=args.unknown_train_boost,
    )
    run_config = {
        "dataset_root": str(dataset_root),
        "output": str(output),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "models": args.models,
        "seeds": args.seeds,
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
    }
    (output / "config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    classical = {} if args.skip_classical else evaluate_classical_baselines(dataset_root, catalog)
    (output / "classical_baselines.json").write_text(
        json.dumps(classical, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    seed_results: dict[str, list[dict]] = {name: [] for name in args.models}
    training_results: list[dict] = []
    for name in args.models:
        for seed in args.seeds:
            run_dir = output / "checkpoints" / name / f"seed_{seed}"
            print(f"TRAIN model={name} seed={seed} device={device}", flush=True)
            model, training = train_model(
                name=name,
                seed=seed,
                dataset_root=dataset_root,
                catalog=catalog,
                model_config=model_config,
                train_config=train_config,
                device=device,
                output_dir=run_dir,
            )
            metrics = evaluate_model(
                model,
                dataset_root=dataset_root,
                catalog=catalog,
                split=args.eval_split,
                device=device,
                batch_size=args.eval_batch_size,
                include_update=name == "p4d",
            )
            (run_dir / f"{args.eval_split}_metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            seed_results[name].append(metrics)
            training_results.append(training)
            print(
                f"DONE model={name} seed={seed} epoch={training['best_epoch']} "
                f"routine_exact_top1={metrics['routine_exact']['top1']:.4f}",
                flush=True,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    aggregate = summarize_neural(seed_results)
    (output / "neural_aggregate.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "training_summary.json").write_text(
        json.dumps(training_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(
        output,
        audit=audit,
        classical=classical,
        aggregate=aggregate,
        model_config=model_config,
        train_config=train_config,
    )
    print(json.dumps({"status": "OK", "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
