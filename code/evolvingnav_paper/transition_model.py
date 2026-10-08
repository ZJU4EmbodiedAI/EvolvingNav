"""Learned row-conditional transition head for N4 (Equations 10 and 19)."""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import defaultdict

import numpy as np
import torch
from torch import nn


class TransitionHead(nn.Module):
    """Shares the belief backbone's query/candidate embeddings; has separate parameters."""

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim + 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, context: torch.Tensor, candidates: torch.Tensor,
                horizon_s: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
        batch, states, hidden = candidates.shape
        if context.shape != (batch, hidden) or horizon_s.shape != (batch,):
            raise ValueError("transition context or horizon has incorrect shape")
        if torch.any(horizon_s < 0):
            raise ValueError("transition horizon must be nonnegative")
        phase = horizon_s.float() * (2 * math.pi / 86400.0)
        time = torch.stack((torch.log1p(horizon_s.float()),
                            torch.sin(phase), torch.cos(phase),
                            horizon_s.float() / 86400.0), dim=-1)
        features = torch.cat((
            context[:, None, None, :].expand(-1, states, states, -1),
            candidates[:, :, None, :].expand(-1, -1, states, -1),
            candidates[:, None, :, :].expand(-1, states, -1, -1),
            time[:, None, None, :].expand(-1, states, states, -1),
        ), dim=-1)
        logits = self.mlp(features).squeeze(-1)
        logits = logits.masked_fill(~candidate_mask[:, None, :].bool(), -1e4)
        probabilities = torch.softmax(logits, dim=-1)
        return probabilities * candidate_mask[:, :, None].float()


def transition_nll(kernel: torch.Tensor, source_index: torch.Tensor,
                   destination_index: torch.Tensor) -> torch.Tensor:
    rows = kernel[torch.arange(len(kernel), device=kernel.device), source_index.long()]
    selected = rows.gather(1, destination_index.long()[:, None]).squeeze(1)
    return -selected.clamp_min(1e-9).log().mean()


def chronological_pairs(*, instance_ids: np.ndarray, world_ids: np.ndarray,
                        times_s: np.ndarray, states: np.ndarray,
                        max_horizon_s: float) -> list[tuple[int, int, float, int, int]]:
    """Create train-only H<=ta, s(ta), delta, s(ta+delta) labels."""
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, (instance, world) in enumerate(zip(instance_ids, world_ids, strict=True)):
        groups[(str(instance), int(world))].append(index)
    pairs = []
    for indices in groups.values():
        ordered = sorted(indices, key=lambda i: float(times_s[i]))
        for source, destination in zip(ordered, ordered[1:], strict=False):
            horizon = float(times_s[destination] - times_s[source])
            if 0 < horizon <= max_horizon_s:
                pairs.append((source, destination, horizon,
                              int(states[source]), int(states[destination])))
    return pairs


def event_horizon_pairs(*, instance_ids: np.ndarray, world_ids: np.ndarray,
                        query_times_s: np.ndarray, query_states: np.ndarray,
                        events: list[dict], horizons_s: tuple[float, ...]
                        ) -> list[tuple[int, float, float, int, int]]:
    """Sample short stationary and event-crossing tuples from one split's causal snapshots."""
    if not horizons_s or any(h <= 0 for h in horizons_s):
        raise ValueError("horizons must be positive")
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    event_groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for index, (instance, world) in enumerate(zip(instance_ids, world_ids, strict=True)):
        groups[(str(instance), int(world))].append(index)
    for event in events:
        event_groups[(str(event["instance_uuid"]), int(event.get("world_id", 0)))].append(event)
    pairs: list[tuple[int, float, float, int, int]] = []
    for (instance, world), indices in groups.items():
        ordered = sorted(indices, key=lambda i: float(query_times_s[i]))
        times = [float(query_times_s[i]) for i in ordered]
        relevant = sorted(event_groups.get((instance, world), []),
                          key=lambda event: float(event["event_time_s"]))
        event_times = [float(event["event_time_s"]) for event in relevant]
        for position, source in enumerate(ordered):
            horizon = float(horizons_s[position % len(horizons_s)])
            anchor = times[position]
            if anchor + horizon > times[-1]:
                continue
            if bisect_right(event_times, anchor + horizon) == bisect_right(event_times, anchor):
                state = int(query_states[source])
                pairs.append((source, anchor, horizon, state, state))
        for event_index, event in enumerate(relevant):
            event_time = float(event["event_time_s"])
            for horizon in horizons_s:
                anchor = event_time - horizon / 2.0
                future = anchor + horizon
                if anchor < times[0] or future > times[-1]:
                    continue
                if (event_index > 0 and event_times[event_index - 1] >= anchor) or (
                    event_index + 1 < len(event_times) and event_times[event_index + 1] <= future
                ):
                    continue
                source_position = bisect_right(times, anchor) - 1
                if source_position < 0:
                    continue
                pairs.append((
                    ordered[source_position], anchor, float(horizon),
                    int(event["source_state_id"]), int(event["destination_state_id"]),
                ))
    return pairs


class NeuralTransition:
    """Inference adapter that never receives transition labels or mobility tags."""

    dynamic = True

    def __init__(self, belief_model: nn.Module, head: TransitionHead,
                 packed_batch: dict[str, torch.Tensor]) -> None:
        self.model = belief_model.eval()
        self.head = head.eval()
        self.batch = {key: value.clone() for key, value in packed_batch.items()}
        if self.batch["candidate_state_ids"].shape[0] != 1:
            raise ValueError("N4 transition adapter expects one active episode")
        self.elapsed_s = 0.0

    def add_candidate(self, state_id: int) -> None:
        present = self.batch["candidate_state_ids"][0].tolist()
        if state_id in present:
            return
        self.batch["candidate_state_ids"] = torch.cat((
            self.batch["candidate_state_ids"],
            torch.tensor([[state_id]], dtype=self.batch["candidate_state_ids"].dtype),
        ), dim=1)
        self.batch["candidate_mask"] = torch.cat((
            self.batch["candidate_mask"],
            torch.ones((1, 1), dtype=self.batch["candidate_mask"].dtype),
        ), dim=1)

    def matrix(self, states: list[int], elapsed_s: float) -> np.ndarray:
        if elapsed_s < 0:
            raise ValueError("time cannot go backwards")
        if elapsed_s == 0:
            return np.eye(len(states), dtype=float)
        device = next(self.model.parameters()).device
        batch = {key: value.to(device) for key, value in self.batch.items()}
        batch["query_time_days"] = batch["query_time_days"] + self.elapsed_s / 86400.0
        batch["query_weekday_id"] = batch["query_time_days"].floor().long() % 7
        batch["elapsed_since_last_positive_days"] = (
            batch["elapsed_since_last_positive_days"] + self.elapsed_s / 86400.0
        )
        day_phase = 2 * math.pi * batch["query_time_days"]
        batch["query_time_of_day_sin_cos"] = torch.stack(
            (day_phase.sin(), day_phase.cos()), dim=-1
        )
        present = batch["candidate_state_ids"][0].tolist()
        index = [present.index(state) for state in states]
        allowed = torch.zeros_like(batch["candidate_mask"])
        allowed[:, index] = True
        batch["candidate_mask"] = batch["candidate_mask"].bool() & allowed.bool()
        with torch.inference_mode():
            context, candidates = self.model.backbone(batch)
            kernel = self.head(
                context, candidates, torch.tensor([elapsed_s], device=device),
                batch["candidate_mask"],
            )[0]
        return kernel[index][:, index].cpu().numpy()

    def advance_clock(self, elapsed_s: float) -> None:
        if elapsed_s < 0:
            raise ValueError("time cannot go backwards")
        self.elapsed_s += elapsed_s
