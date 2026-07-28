"""PathMIL model components used by training, evaluation and visualization."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors


class AttentionNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.layers(features), features


class GatedAttentionNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        attention_layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        ]
        gate_layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
        ]
        if dropout > 0:
            attention_layers.append(nn.Dropout(dropout))
            gate_layers.append(nn.Dropout(dropout))

        self.attention = nn.Sequential(*attention_layers)
        self.gate = nn.Sequential(*gate_layers)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.output(self.attention(features) * self.gate(features))
        return scores, features


class DynamicPathEncoder(nn.Module):
    """Build differentiable local paths from patch features and coordinates."""

    def __init__(
        self,
        embed_dim: int,
        attention_dim: int = 256,
        spatial_dim: int = 32,
        depth: int = 2,
        neighbor_count: int = 8,
        temperature: float = 1.0,
        local_weight: float = 1.0,
        spatial_weight: float = 0.25,
        global_weight: float = 0.5,
    ) -> None:
        super().__init__()
        if depth < 0:
            raise ValueError("depth must be non-negative.")
        if neighbor_count < 1:
            raise ValueError("neighbor_count must be at least 1.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        self.depth = depth
        self.neighbor_count = neighbor_count
        self.temperature = temperature
        self.local_weight = local_weight
        self.spatial_weight = spatial_weight
        self.global_weight = global_weight

        self.query_projection = nn.Linear(embed_dim, attention_dim)
        self.key_projection = nn.Linear(embed_dim, attention_dim)
        self.value_projection = nn.Linear(embed_dim, embed_dim)
        self.global_projection = nn.Linear(embed_dim, 1)
        self.spatial_projection = nn.Sequential(
            nn.Linear(1, spatial_dim),
            nn.ReLU(inplace=True),
            nn.Linear(spatial_dim, 1),
        )

    @staticmethod
    def _soft_orthogonality_loss(weight: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(weight, p=2, dim=1, eps=1e-6)
        gram = normalized @ normalized.transpose(0, 1)
        identity = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
        return (gram - identity).pow(2).sum() / gram.size(0)

    def projection_decorrelation_loss(self) -> torch.Tensor:
        return self._soft_orthogonality_loss(
            self.query_projection.weight
        ) + self._soft_orthogonality_loss(self.key_projection.weight)

    @staticmethod
    def _projection_stats(weight: torch.Tensor, prefix: str) -> dict[str, float]:
        normalized = F.normalize(weight, p=2, dim=1, eps=1e-6)
        gram = normalized @ normalized.transpose(0, 1)
        mask = ~torch.eye(gram.size(0), device=gram.device, dtype=torch.bool)
        off_diagonal = gram[mask]
        if off_diagonal.numel() == 0:
            return {
                f"{prefix}_offdiag_abs_mean": 0.0,
                f"{prefix}_offdiag_square_mean": 0.0,
                f"{prefix}_offdiag_abs_max": 0.0,
            }
        return {
            f"{prefix}_offdiag_abs_mean": off_diagonal.abs().mean().item(),
            f"{prefix}_offdiag_square_mean": off_diagonal.pow(2).mean().item(),
            f"{prefix}_offdiag_abs_max": off_diagonal.abs().max().item(),
        }

    @torch.no_grad()
    def projection_decorrelation_stats(self) -> dict[str, float]:
        stats = self._projection_stats(self.query_projection.weight, "query")
        stats.update(self._projection_stats(self.key_projection.weight, "key"))
        return stats

    def _nearest_neighbors(
        self,
        coordinates: torch.Tensor,
        minimum_count: int = 1,
    ) -> torch.Tensor:
        patch_count = coordinates.size(0)
        if patch_count == 1:
            return torch.zeros((1, 1), dtype=torch.long, device=coordinates.device)

        neighbor_count = min(
            max(self.neighbor_count, minimum_count),
            patch_count - 1,
        )
        coordinates_array = coordinates.detach().cpu().numpy()
        search = NearestNeighbors(n_neighbors=neighbor_count + 1)
        raw_indices = search.fit(coordinates_array).kneighbors(
            coordinates_array,
            return_distance=False,
        )
        indices = torch.as_tensor(raw_indices, dtype=torch.long)
        row_indices = torch.arange(patch_count).unsqueeze(1)
        indices = indices[indices != row_indices].reshape(
            patch_count,
            neighbor_count,
        )
        return indices.to(device=coordinates.device)

    @staticmethod
    def _expand_blocked_neighborhoods(
        expanded_candidates: torch.Tensor,
        local_neighbor_count: int,
        visited: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visited_mask = (
            expanded_candidates.unsqueeze(-1) == visited.unsqueeze(1)
        ).any(dim=-1)
        if local_neighbor_count == expanded_candidates.size(1):
            return expanded_candidates, visited_mask

        blocked = visited_mask[:, :local_neighbor_count].all(dim=1)
        outside_local_neighborhood = torch.arange(
            expanded_candidates.size(1),
            device=expanded_candidates.device,
        ).unsqueeze(0) >= local_neighbor_count
        restrict_to_local = (
            ~blocked.unsqueeze(1) & outside_local_neighborhood
        )
        return expanded_candidates, visited_mask | restrict_to_local

    def forward(
        self,
        features: torch.Tensor,
        coordinates: torch.Tensor,
        return_indices: bool = False,
    ):
        if features.ndim != 2:
            raise ValueError("features must have shape [patch_count, embed_dim].")
        if coordinates.ndim != 2 or coordinates.size(1) != 2:
            raise ValueError("coordinates must have shape [patch_count, 2].")
        if features.size(0) != coordinates.size(0):
            raise ValueError("features and coordinates must contain the same patches.")
        if features.size(0) == 0:
            raise ValueError("PathMIL cannot process an empty patch bag.")

        query = self.query_projection(features)
        key = self.key_projection(features)
        value = self.value_projection(features)
        global_score = self.global_projection(features).squeeze(-1)
        coordinates = coordinates.to(device=features.device, dtype=features.dtype)

        patch_count = features.size(0)
        effective_depth = min(self.depth, patch_count - 1)
        neighbors = self._nearest_neighbors(
            coordinates,
            minimum_count=effective_depth,
        )
        local_neighbor_count = min(
            self.neighbor_count,
            neighbors.size(1),
        )
        current = torch.arange(patch_count, device=features.device)
        batch_index = torch.arange(patch_count, device=features.device)
        path_indices = [current.unsqueeze(1)]
        path_features = [value[current].unsqueeze(1)]

        for _ in range(effective_depth):
            expanded_candidates = neighbors[current]
            visited = torch.cat(path_indices, dim=1)
            candidates, visited_mask = self._expand_blocked_neighborhoods(
                expanded_candidates,
                local_neighbor_count,
                visited,
            )
            candidate_keys = key[candidates]
            local_score = (
                query[current].unsqueeze(1) * candidate_keys
            ).sum(-1) / math.sqrt(candidate_keys.size(-1))

            displacement = (
                coordinates[candidates] - coordinates[current].unsqueeze(1)
            )
            distance = displacement.norm(dim=-1, keepdim=True)
            spatial_score = self.spatial_projection(distance).squeeze(-1)
            spatial_score = spatial_score - spatial_score.mean(dim=1, keepdim=True)
            candidate_global_score = global_score[candidates]

            score = (
                self.local_weight * local_score
                + self.spatial_weight * spatial_score
                + self.global_weight * candidate_global_score
            )
            masked_score = score.masked_fill(visited_mask, -torch.inf)
            soft_selection = F.softmax(
                masked_score / self.temperature,
                dim=1,
            )
            selected_offset = masked_score.argmax(dim=1)
            hard_selection = F.one_hot(
                selected_offset,
                num_classes=candidates.size(1),
            ).to(dtype=soft_selection.dtype)
            straight_through_selection = (
                hard_selection - soft_selection.detach() + soft_selection
            )

            next_feature = torch.sum(
                straight_through_selection.unsqueeze(-1) * value[candidates],
                dim=1,
            )
            current = candidates[batch_index, selected_offset]
            path_indices.append(current.unsqueeze(1))
            path_features.append(next_feature.unsqueeze(1))

        encoded_paths = torch.cat(path_features, dim=1)
        if return_indices:
            return encoded_paths, torch.cat(path_indices, dim=1)
        return encoded_paths


class GatedPositionalEmbedding(nn.Module):
    """Apply gated multi-scale depthwise convolutions on a pseudo-grid."""

    def __init__(self, embed_dim: int, gate_hidden_ratio: float = 0.5) -> None:
        super().__init__()
        self.convolution_7 = nn.Conv2d(
            embed_dim, embed_dim, 7, padding=3, groups=embed_dim
        )
        self.convolution_5 = nn.Conv2d(
            embed_dim, embed_dim, 5, padding=2, groups=embed_dim
        )
        self.convolution_3 = nn.Conv2d(
            embed_dim, embed_dim, 3, padding=1, groups=embed_dim
        )
        gate_dim = max(1, int(embed_dim * gate_hidden_ratio))
        self.token_gate = nn.Sequential(
            nn.Linear(embed_dim, gate_dim),
            nn.ReLU(inplace=True),
            nn.Linear(gate_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        batch_size, token_count, embed_dim = tokens.shape
        if token_count != height * width:
            raise ValueError("token_count must equal height * width.")

        gate = self.token_gate(tokens).transpose(1, 2).reshape(
            batch_size, 1, height, width
        )
        grid = tokens.transpose(1, 2).reshape(
            batch_size, embed_dim, height, width
        )
        multi_scale = (
            self.convolution_7(grid)
            + self.convolution_5(grid)
            + self.convolution_3(grid)
        )
        output = grid + gate * multi_scale
        return output.flatten(2).transpose(1, 2)


class PathMIL(nn.Module):
    """Dual-stream multiple-instance model with dynamic path aggregation."""

    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 512,
        attention_dim: int = 256,
        dropout: float = 0.25,
        n_classes: int = 2,
        instance_sample_count: int = 8,
        instance_loss_fn: nn.Module | None = None,
        include_negative_instances: bool = False,
        path_depth: int = 2,
        path_attention_dim: int = 256,
        path_spatial_dim: int = 32,
        path_neighbor_count: int = 8,
        path_temperature: float = 1.0,
        path_local_weight: float = 1.0,
        path_spatial_weight: float = 0.25,
        path_global_weight: float = 0.5,
        gated_attention: bool = True,
    ) -> None:
        super().__init__()
        if n_classes < 2:
            raise ValueError("n_classes must be at least 2.")
        if instance_sample_count < 1:
            raise ValueError("instance_sample_count must be at least 1.")

        self.n_classes = n_classes
        self.instance_sample_count = instance_sample_count
        self.include_negative_instances = include_negative_instances
        self.instance_loss_fn = instance_loss_fn or nn.CrossEntropyLoss()

        self.path_encoder = DynamicPathEncoder(
            embed_dim=embed_dim,
            attention_dim=path_attention_dim,
            spatial_dim=path_spatial_dim,
            depth=path_depth,
            neighbor_count=path_neighbor_count,
            temperature=path_temperature,
            local_weight=path_local_weight,
            spatial_weight=path_spatial_weight,
            global_weight=path_global_weight,
        )
        self.path_attention = GatedAttentionNetwork(
            input_dim=embed_dim,
            hidden_dim=attention_dim,
            dropout=dropout,
        )
        self.patch_projection = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.path_projection = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.feature_fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.positional_embedding = GatedPositionalEmbedding(hidden_dim)

        attention_class = GatedAttentionNetwork if gated_attention else AttentionNetwork
        self.attention_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.slide_attention = attention_class(
            input_dim=hidden_dim,
            hidden_dim=attention_dim,
            dropout=dropout,
        )
        self.slide_classifier = nn.Linear(hidden_dim, n_classes)
        self.instance_classifiers = nn.ModuleList(
            nn.Linear(hidden_dim, 2) for _ in range(n_classes)
        )

    @staticmethod
    def _targets(length: int, value: int, device: torch.device) -> torch.Tensor:
        return torch.full((length,), value, device=device, dtype=torch.long)

    def _evaluate_in_class(
        self,
        attention: torch.Tensor,
        features: torch.Tensor,
        classifier: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_count = min(self.instance_sample_count, features.size(0))
        positive_indices = torch.topk(attention, sample_count, dim=1).indices[0]
        negative_indices = torch.topk(-attention, sample_count, dim=1).indices[0]
        instances = torch.cat(
            [features[positive_indices], features[negative_indices]],
            dim=0,
        )
        targets = torch.cat(
            [
                self._targets(sample_count, 1, features.device),
                self._targets(sample_count, 0, features.device),
            ]
        )
        logits = classifier(instances)
        predictions = logits.argmax(dim=1)
        return self.instance_loss_fn(logits, targets), predictions, targets

    def _evaluate_out_of_class(
        self,
        attention: torch.Tensor,
        features: torch.Tensor,
        classifier: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_count = min(self.instance_sample_count, features.size(0))
        indices = torch.topk(attention, sample_count, dim=1).indices[0]
        targets = self._targets(sample_count, 0, features.device)
        logits = classifier(features[indices])
        predictions = logits.argmax(dim=1)
        return self.instance_loss_fn(logits, targets), predictions, targets

    def _instance_supervision(
        self,
        attention: torch.Tensor,
        features: torch.Tensor,
        label: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        one_hot_label = F.one_hot(
            label.view(-1)[0],
            num_classes=self.n_classes,
        )
        losses = []
        predictions = []
        targets = []

        for class_index, classifier in enumerate(self.instance_classifiers):
            if one_hot_label[class_index].item() == 1:
                result = self._evaluate_in_class(attention, features, classifier)
            elif self.include_negative_instances:
                result = self._evaluate_out_of_class(attention, features, classifier)
            else:
                continue
            loss, class_predictions, class_targets = result
            losses.append(loss)
            predictions.append(class_predictions)
            targets.append(class_targets)

        if not losses:
            raise RuntimeError("No instance-supervision loss was produced.")
        instance_loss = torch.stack(losses).mean()
        return {
            "instance_loss": instance_loss,
            "instance_predictions": torch.cat(predictions),
            "instance_targets": torch.cat(targets),
        }

    @staticmethod
    def _pad_to_square(tokens: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        patch_count = tokens.size(1)
        side_length = math.ceil(math.sqrt(patch_count))
        padding = side_length * side_length - patch_count
        if padding > 0:
            repeat_count = math.ceil(padding / patch_count)
            repeated = tokens.repeat(1, repeat_count, 1)[:, :padding, :]
            tokens = torch.cat([tokens, repeated], dim=1)
        return tokens, side_length, side_length

    def forward(
        self,
        features: torch.Tensor,
        coordinates: torch.Tensor,
        label: torch.Tensor | None = None,
        evaluate_instances: bool = False,
        return_features: bool = False,
        attention_only: bool = False,
    ):
        encoded_paths = self.path_encoder(features, coordinates)
        patch_count, path_length, embed_dim = encoded_paths.shape

        path_steps = encoded_paths.reshape(patch_count * path_length, embed_dim)
        path_scores, path_steps = self.path_attention(path_steps)
        path_scores = path_scores.reshape(patch_count, path_length, 1)
        path_steps = path_steps.reshape(patch_count, path_length, embed_dim)
        path_features = (F.softmax(path_scores, dim=1) * path_steps).sum(dim=1)

        patch_features = self.patch_projection(features)
        path_features = self.path_projection(path_features)
        fused_features = self.feature_fusion(
            torch.cat([patch_features, path_features], dim=1)
        )

        padded, height, width = self._pad_to_square(fused_features.unsqueeze(0))
        fused_features = self.positional_embedding(padded, height, width)[
            :, :patch_count, :
        ].squeeze(0)

        attention_features = self.attention_projection(fused_features)
        attention, embedded_features = self.slide_attention(attention_features)
        attention = attention.transpose(1, 0)
        if attention_only:
            return attention

        normalized_attention = F.softmax(attention, dim=1)
        slide_features = normalized_attention @ embedded_features
        logits = self.slide_classifier(slide_features)
        probabilities = F.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1, keepdim=True)

        output = {
            "logits": logits,
            "probabilities": probabilities,
            "predictions": predictions,
            "attention": attention,
        }
        if evaluate_instances:
            if label is None:
                raise ValueError("label is required when evaluate_instances=True.")
            output.update(
                self._instance_supervision(
                    normalized_attention,
                    embedded_features,
                    label,
                )
            )
        if return_features:
            output["features"] = slide_features
        return output
