"""Query-conditioned continuous-time pointer models for P4D-Belief."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from readyagent.p4d_belief.data import Catalog


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 128
    layers: int = 3
    heads: int = 4
    dropout: float = 0.15
    use_instance_identity: bool = False
    use_compatibility: bool = False


class SemanticLocationEncoder(nn.Module):
    """Shared location encoder with no scene-specific state-ID embedding."""

    def __init__(self, catalog: Catalog, hidden_dim: int) -> None:
        super().__init__()
        region_count = len(catalog.schema["region_category_to_id"])
        receptacle_count = len(catalog.schema["receptacle_category_to_id"])
        semantic_dim = hidden_dim // 4
        self.region_embedding = nn.Embedding(region_count, semantic_dim)
        self.receptacle_embedding = nn.Embedding(receptacle_count, semantic_dim)
        self.project = nn.Sequential(
            nn.Linear(semantic_dim * 2 + 4, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        center = catalog.center_xyz.float()
        known = ~catalog.is_unknown.bool()
        mean = center[known].mean(dim=0)
        std = center[known].std(dim=0, unbiased=int(known.sum()) > 1).clamp_min(1e-3)
        self.register_buffer("region_category", catalog.region_category.long())
        self.register_buffer("receptacle_category", catalog.receptacle_category.long())
        self.register_buffer("center", center)
        self.register_buffer("center_mean", mean)
        self.register_buffer("center_std", std)
        self.register_buffer("is_unknown", catalog.is_unknown.float())

    def base_nodes(self, batch=None) -> torch.Tensor:
        runtime = batch is not None and "location_center_xyz" in batch
        center = batch["location_center_xyz"] if runtime else self.center
        region = batch["location_region_category"] if runtime else self.region_category
        receptacle = batch["location_receptacle_category"] if runtime else self.receptacle_category
        unknown = batch["location_is_unknown"] if runtime else self.is_unknown
        normalized_center = (center - self.center_mean) / self.center_std
        return self.project(
            torch.cat(
                (
                    self.region_embedding(region.long()),
                    self.receptacle_embedding(receptacle.long()),
                    normalized_center,
                    unknown.float().unsqueeze(-1),
                ),
                dim=-1,
            )
        )


class RelativeTimeEncoderLayer(nn.Module):
    """Transformer layer with a learned, per-head continuous-time bias."""

    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.time_bias = nn.Linear(2, heads, bias=False)
        self.attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, tokens: torch.Tensor, times_days: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        delta_hours = (times_days[:, :, None] - times_days[:, None, :]) * 24.0
        time_features = torch.stack(
            (torch.sign(delta_hours), torch.log1p(delta_hours.abs())), dim=-1
        )
        bias = self.time_bias(time_features).permute(0, 3, 1, 2)
        key_mask = (~valid)[:, None, None, :]
        bias = bias.masked_fill(key_mask, -1e4)
        batch, length = valid.shape
        attention_mask = bias.reshape(batch * self.heads, length, length)
        attended, _ = self.attention(
            tokens, tokens, tokens, attn_mask=attention_mask, need_weights=False
        )
        tokens = self.norm1(tokens + self.dropout(attended))
        return self.norm2(tokens + self.feed_forward(tokens))


class HistoryInputs(nn.Module):
    @staticmethod
    def gather(nodes, indices):
        if nodes.ndim == 2:
            return nodes[indices]
        batch = torch.arange(nodes.shape[0], device=nodes.device)
        return nodes[batch[:, None], indices] if indices.ndim == 2 else nodes[batch, indices]

    def __init__(self, catalog: Catalog, config: ModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_dim
        category_count = len(catalog.schema["category_to_id"])
        instance_count = len(catalog.schema.get("instance_uuid_to_id", {}))
        self.location = SemanticLocationEncoder(catalog, hidden)
        self.object_embedding = nn.Embedding(category_count, hidden)
        self.instance_embedding = (
            nn.Embedding(instance_count, hidden)
            if config.use_instance_identity and instance_count
            else None
        )
        self.event_embedding = nn.Embedding(3, hidden)
        self.weekday_embedding = nn.Embedding(7, hidden // 4)
        self.quality_projection = nn.Sequential(
            nn.Linear(6, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, hidden)
        )
        self.time_projection = nn.Linear(5, hidden)
        self.query_projection = nn.Sequential(
            nn.Linear(hidden + hidden // 4 + 4, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.history_summary_projection = nn.Sequential(
            nn.Linear(7, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.observable_context_projection = nn.Sequential(
            nn.Linear(category_count + 5, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.query_context_norm = nn.LayerNorm(hidden)
        self.candidate_geometry = nn.Sequential(
            nn.Linear(5, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.token_norm = nn.LayerNorm(hidden)

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        base_nodes = self.location.base_nodes(batch)
        event_type = batch["event_type"].long()
        history_mask = batch["history_mask"].bool()
        observed = batch["observed_state_id"].long()
        inspected = batch["candidate_state_id"].long()
        location_id = torch.where(event_type == 1, observed, inspected).clamp_min(0)
        location_token = self.gather(base_nodes, location_id)
        query_days = batch["query_time_days"].float()
        event_days = batch["event_time_days"].float()
        age_hours = ((query_days[:, None] - event_days) * 24.0).clamp_min(0.0)
        time_features = torch.stack(
            (
                torch.log1p(age_hours),
                torch.sin(2 * math.pi * event_days),
                torch.cos(2 * math.pi * event_days),
                torch.sin(2 * math.pi * event_days / 7.0),
                torch.cos(2 * math.pi * event_days / 7.0),
            ),
            dim=-1,
        )
        object_token = self.object_embedding(batch["target_category_id"].long())
        if self.instance_embedding is not None:
            identity = self.instance_embedding(batch["target_instance_id"].long())
            if "target_instance_known" in batch:
                identity = identity * batch["target_instance_known"][:, None]
            object_token = object_token + identity
        tokens = self.token_norm(
            location_token
            + self.event_embedding(event_type)
            + self.quality_projection(batch["evidence_features"].float())
            + self.time_projection(time_features)
            + object_token[:, None, :]
        )
        query_features = torch.cat(
            (
                object_token,
                self.weekday_embedding(batch["query_weekday_id"].long()),
                batch["query_time_of_day_sin_cos"].float(),
                torch.log1p(batch["elapsed_since_last_positive_days"].float())[:, None],
                torch.ones_like(query_days)[:, None],
            ),
            dim=-1,
        )
        positive = (event_type == 1) & history_mask
        inspection = (event_type == 2) & history_mask
        positive_count = positive.sum(dim=1).float()
        inspection_count = inspection.sum(dim=1).float()
        state_one_hot = torch.nn.functional.one_hot(
            observed.clamp_min(0), num_classes=base_nodes.shape[-2]
        ).bool()
        distinct_states = (state_one_hot & positive[:, :, None]).any(dim=1).sum(dim=1).float()
        previous = torch.full(
            (event_type.shape[0],), -1, dtype=torch.long, device=event_type.device
        )
        change_count = torch.zeros_like(positive_count)
        for position in range(event_type.shape[1]):
            current = observed[:, position]
            is_positive = positive[:, position]
            change_count += (
                is_positive & (previous >= 0) & (current != previous)
            ).float()
            previous = torch.where(is_positive, current, previous)
        inspected_time = torch.where(
            inspection, event_days, torch.full_like(event_days, -1e9)
        ).max(dim=1).values
        time_since_inspection = torch.where(
            inspection.any(dim=1),
            (query_days - inspected_time).clamp_min(0.0),
            batch["elapsed_since_last_positive_days"].float(),
        )
        negative_strength = batch["evidence_features"].float()[:, :, 5]
        negative_sum = (negative_strength * inspection.float()).sum(dim=1)
        negative_mean = negative_sum / inspection_count.clamp_min(1.0)
        negative_max = negative_strength.masked_fill(~inspection, 0.0).max(dim=1).values
        summary = torch.stack(
            (
                torch.log1p(positive_count),
                torch.log1p(inspection_count),
                torch.log1p(distinct_states),
                torch.log1p(change_count),
                torch.log1p(time_since_inspection * 24.0),
                negative_mean,
                negative_max,
            ),
            dim=-1,
        )
        last_node = self.gather(base_nodes, batch["last_state"].long())
        observable_context = torch.zeros_like(last_node)
        if "context_mask" in batch:
            context_mask = batch["context_mask"].bool()
            context_age_hours = (
                (query_days[:, None] - batch["context_time_days"].float()) * 24.0
            ).clamp_min(0.0)
            context_features = torch.cat(
                (
                    batch["context_category_counts"].float(),
                    batch["context_observation_features"].float(),
                    torch.log1p(context_age_hours)[:, :, None],
                ),
                dim=-1,
            )
            projected_context = self.observable_context_projection(context_features)
            observable_context = (
                projected_context * context_mask[:, :, None].float()
            ).sum(dim=1) / context_mask.sum(dim=1).clamp_min(1)[:, None]
        query_token = self.query_context_norm(
            self.query_projection(query_features)
            + last_node
            + self.history_summary_projection(summary)
            + observable_context
        )
        return tokens, query_token, base_nodes, history_mask

    def candidate_nodes(
        self, base_nodes: torch.Tensor, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        candidate_ids = batch["candidate_state_ids"].long().clamp_min(0)
        nodes = self.gather(base_nodes, candidate_ids)
        geometry_nodes = batch.get("location_center_xyz", self.location.center)
        centers = self.gather(geometry_nodes, candidate_ids)
        last_center = self.gather(geometry_nodes, batch["last_state"].long())
        relative = (centers - last_center[:, None, :]) / self.location.center_std
        distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
        is_last = (
            candidate_ids == batch["last_state"].long()[:, None]
        ).float()[:, :, None]
        # Geometry is added without introducing a scene-specific state embedding.
        geometry = torch.cat((relative, distance, is_last), dim=-1)
        return nodes + self.candidate_geometry(geometry)


class CTTransformerBackbone(nn.Module):
    def __init__(self, catalog: Catalog, config: ModelConfig) -> None:
        super().__init__()
        self.inputs = HistoryInputs(catalog, config)
        self.layers = nn.ModuleList(
            RelativeTimeEncoderLayer(config.hidden_dim, config.heads, config.dropout)
            for _ in range(config.layers)
        )

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history, query, base_nodes, history_mask = self.inputs(batch)
        tokens = torch.cat((history, query[:, None, :]), dim=1)
        query_valid = torch.ones(
            (history_mask.shape[0], 1), dtype=torch.bool, device=history_mask.device
        )
        valid = torch.cat((history_mask, query_valid), dim=1)
        times = torch.cat(
            (batch["event_time_days"].float(), batch["query_time_days"].float()[:, None]),
            dim=1,
        )
        for layer in self.layers:
            tokens = layer(tokens, times, valid)
        candidates = self.inputs.candidate_nodes(base_nodes, batch)
        return tokens[:, -1], candidates


class GRUBackbone(nn.Module):
    def __init__(self, catalog: Catalog, config: ModelConfig) -> None:
        super().__init__()
        self.inputs = HistoryInputs(catalog, config)
        self.gru = nn.GRU(
            config.hidden_dim,
            config.hidden_dim,
            num_layers=2,
            dropout=config.dropout,
            batch_first=True,
        )

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history, query, base_nodes, history_mask = self.inputs(batch)
        lengths = history_mask.sum(dim=1).long()
        sequence = torch.zeros(
            history.shape[0], history.shape[1] + 1, history.shape[2],
            device=history.device, dtype=history.dtype,
        )
        sequence[:, : history.shape[1]] = history
        sequence[torch.arange(len(lengths), device=history.device), lengths] = query
        packed = nn.utils.rnn.pack_padded_sequence(
            sequence, (lengths + 1).cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        candidates = self.inputs.candidate_nodes(base_nodes, batch)
        return hidden[-1], candidates


class PointerBase(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.query_pointer = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.candidate_pointer = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = hidden_dim**-0.5

    def pointer_logits(
        self, query: torch.Tensor, candidates: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        logits = torch.einsum(
            "bd,bmd->bm", self.query_pointer(query), self.candidate_pointer(candidates)
        ) * self.scale
        return logits.masked_fill(~mask.bool(), -1e4)


class DirectPointer(PointerBase):
    """Direct softmax comparator using the same CT Transformer and pointer."""

    def __init__(self, catalog: Catalog, config: ModelConfig) -> None:
        super().__init__(CTTransformerBackbone(catalog, config), config.hidden_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        query, candidates = self.backbone(batch)
        logits = self.pointer_logits(query, candidates, batch["candidate_mask"])
        probabilities = torch.softmax(logits, dim=-1)
        last_probability = probabilities.gather(
            1, batch["last_candidate_index"].long()[:, None]
        ).squeeze(1)
        return {"probabilities": probabilities, "rho": last_probability, "logits": logits}


class GRUDirectPointer(PointerBase):
    def __init__(self, catalog: Catalog, config: ModelConfig) -> None:
        super().__init__(GRUBackbone(catalog, config), config.hidden_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        query, candidates = self.backbone(batch)
        logits = self.pointer_logits(query, candidates, batch["candidate_mask"])
        probabilities = torch.softmax(logits, dim=-1)
        rho = probabilities.gather(
            1, batch["last_candidate_index"].long()[:, None]
        ).squeeze(1)
        return {"probabilities": probabilities, "rho": rho, "logits": logits}


class P4DBelief(PointerBase):
    """Persistence-relocation factorized pointer with a single final-state NLL."""

    def __init__(self, catalog: Catalog, config: ModelConfig) -> None:
        super().__init__(CTTransformerBackbone(catalog, config), config.hidden_dim)
        self.persistence = nn.Linear(config.hidden_dim, 1)
        self.compatibility = (
            nn.Sequential(
                nn.Linear(config.hidden_dim * 2, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, 1),
            )
            if config.use_compatibility else None
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        query, candidates = self.backbone(batch)
        rho = torch.sigmoid(self.persistence(query)).squeeze(-1)
        relocation_mask = batch["candidate_mask"].bool().clone()
        relocation_mask.scatter_(
            1, batch["last_candidate_index"].long()[:, None], False
        )
        relocation_logits = self.pointer_logits(query, candidates, relocation_mask)
        if self.compatibility is not None:
            paired = torch.cat(
                (query[:, None, :].expand_as(candidates), candidates), dim=-1
            )
            relocation_logits = relocation_logits + self.compatibility(paired).squeeze(-1)
            relocation_logits = relocation_logits.masked_fill(~relocation_mask, -1e4)
        relocation = torch.softmax(relocation_logits, dim=-1)
        probabilities = relocation * (1.0 - rho[:, None])
        probabilities.scatter_(
            1, batch["last_candidate_index"].long()[:, None], rho[:, None]
        )
        probabilities = probabilities * batch["candidate_mask"].float()
        return {
            "probabilities": probabilities,
            "rho": rho,
            "relocation_probabilities": relocation,
            "relocation_mask": relocation_mask,
            "context": query,
            "candidates": candidates,
        }


def state_nll(probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    selected = probabilities.gather(1, targets.long()[:, None]).squeeze(1)
    return -torch.log(selected.clamp_min(1e-9)).mean()


def bayesian_update(
    prior: torch.Tensor,
    checked_candidate: torch.Tensor,
    *,
    observed: torch.Tensor,
    confidence: torch.Tensor,
    coverage: torch.Tensor,
) -> torch.Tensor:
    """Apply one parameter-free visibility-aware observation update."""

    likelihood = torch.ones_like(prior)
    strength = (confidence * coverage).clamp(0.0, 1.0)
    rows = torch.arange(prior.shape[0], device=prior.device)
    not_seen_likelihood = (1.0 - strength).clamp_min(1e-6)
    likelihood[rows, checked_candidate.long()] = torch.where(
        observed.bool(),
        torch.ones_like(strength),
        not_seen_likelihood,
    )
    if observed.any():
        detected_rows = rows[observed.bool()]
        likelihood[detected_rows] = 1e-6
        likelihood[detected_rows, checked_candidate[observed.bool()].long()] = strength[
            observed.bool()
        ].clamp_min(1e-6)
    posterior = prior * likelihood
    return posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(1e-12)
