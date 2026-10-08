#!/usr/bin/env python3
"""Jointly fine-tune P4D state belief and chronological transition head."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from evolvingnav.transition_model import (
    TransitionHead, event_horizon_pairs, transition_nll,
)
from readyagent.p4d_belief.data import PackedQueries, load_catalog
from readyagent.p4d_belief.models import ModelConfig, state_nll
from readyagent.p4d_belief.training import build_model


class ChronologicalDataset(Dataset):
    def __init__(self, root: Path, split: str, horizons_s: tuple[float, ...],
                 max_pairs: int | None = None) -> None:
        self.records = PackedQueries(root / f"records/packed/{split}.npz")
        arrays = self.records.arrays
        events = []
        for world_id, name in ((0, "routine"), (1, "random")):
            events_path = root / f"worlds/{name}/private_gt/hidden_transitions.jsonl"
            with events_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        event = json.loads(line)
                        event["world_id"] = world_id
                        events.append(event)
        self.pairs = event_horizon_pairs(
            instance_ids=arrays["instance_uuid"],
            world_ids=arrays["meta_world_variant_id"],
            query_times_s=arrays["query_time_days"].astype(float) * 86400.0,
            query_states=arrays["y_current_state"], events=events,
            horizons_s=horizons_s,
        )
        if max_pairs is not None:
            self.pairs = self.pairs[:max_pairs]
        if not self.pairs:
            raise ValueError(f"no chronological pairs in {split}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        source, anchor_time, horizon, source_state, destination_state = self.pairs[item]
        batch = self.records[source]
        candidates = self.records.arrays["candidate_state_ids"][source].tolist()
        source_time = float(self.records.arrays["query_time_days"][source]) * 86400.0
        anchor_days = anchor_time / 86400.0
        phase = 2 * math.pi * anchor_days
        batch["query_time_days"] = torch.tensor(anchor_days, dtype=torch.float32)
        batch["query_time_of_day_sin_cos"] = torch.tensor(
            [math.sin(phase), math.cos(phase)], dtype=torch.float32
        )
        batch["query_weekday_id"] = torch.tensor(
            (int(batch["query_weekday_id"]) + math.floor(anchor_days)
             - math.floor(source_time / 86400.0)) % 7, dtype=torch.long
        )
        batch["elapsed_since_last_positive_days"] = (
            batch["elapsed_since_last_positive_days"]
            + torch.tensor((anchor_time - source_time) / 86400.0, dtype=torch.float32)
        )
        batch["target_candidate_index"] = torch.tensor(candidates.index(source_state))
        batch["y_current_state"] = torch.tensor(source_state)
        batch["transition_source_index"] = torch.tensor(candidates.index(source_state))
        batch["transition_destination_index"] = torch.tensor(candidates.index(destination_state))
        batch["transition_horizon_s"] = torch.tensor(horizon, dtype=torch.float32)
        return batch


def epoch(model, head, loader, optimizer, device, gradient_clip: float) -> float:
    training = optimizer is not None
    model.train(training)
    head.train(training)
    total, count = 0.0, 0
    for raw in loader:
        batch = {key: value.to(device) for key, value in raw.items()}
        with torch.set_grad_enabled(training):
            output = model(batch)
            kernel = head(output["context"], output["candidates"],
                          batch["transition_horizon_s"], batch["candidate_mask"])
            loss = state_nll(output["probabilities"], batch["target_candidate_index"])
            loss = loss + transition_nll(
                kernel, batch["transition_source_index"],
                batch["transition_destination_index"],
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(head.parameters()), gradient_clip
                )
                optimizer.step()
        total += float(loss.detach()) * len(batch["transition_horizon_s"])
        count += len(batch["transition_horizon_s"])
    return total / count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--belief-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizons-s", nargs="+", type=float,
                        default=[2.0, 10.0, 30.0, 60.0, 120.0, 300.0])
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    catalog = load_catalog(args.dataset)
    original = torch.load(args.belief_checkpoint, map_location="cpu", weights_only=True)
    model = build_model("p4d", catalog, ModelConfig(**original["model_config"]))
    model.load_state_dict(original["model"])
    model.to(device)
    head = TransitionHead(int(original["model_config"]["hidden_dim"])).to(device)
    train = ChronologicalDataset(args.dataset, "train", tuple(args.horizons_s), args.max_pairs)
    val = ChronologicalDataset(args.dataset, "val", tuple(args.horizons_s), args.max_pairs)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=args.batch_size)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()), lr=3e-4, weight_decay=1e-2
    )
    best, stale, state = float("inf"), 0, None
    for number in range(1, args.epochs + 1):
        train_loss = epoch(model, head, train_loader, optimizer, device, 1.0)
        val_loss = epoch(model, head, val_loader, None, device, 1.0)
        print(f"epoch={number} train={train_loss:.5f} val={val_loss:.5f}", flush=True)
        if val_loss < best - 1e-5:
            best, stale = val_loss, 0
            state = (copy.deepcopy(model.state_dict()), copy.deepcopy(head.state_dict()))
        else:
            stale += 1
        if stale >= args.patience:
            break
    if state is None:
        raise RuntimeError("transition training produced no checkpoint")
    args.output.mkdir(parents=True)
    torch.save({
        "model": state[0], "transition_head": state[1],
        "model_config": original["model_config"], "best_validation_loss": best,
        "horizons_s": args.horizons_s,
    }, args.output / "best.pt")
    (args.output / "summary.json").write_text(json.dumps({
        "train_pairs": len(train), "val_pairs": len(val),
        "best_validation_loss": best,
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
