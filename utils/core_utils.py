"""Training, validation and evaluation orchestration for PathMIL."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from dataset_modules.dataset_generic import save_splits
from models import PathMIL
from utils.utils import (
    build_split_loader,
    classification_error,
    create_optimizer,
    print_model_summary,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ClassAccuracy:
    """Track correct predictions and counts independently for each class."""

    def __init__(self, class_count: int) -> None:
        self.class_count = class_count
        self.correct = np.zeros(class_count, dtype=np.int64)
        self.count = np.zeros(class_count, dtype=np.int64)

    def update(self, predictions, targets) -> None:
        predictions = np.asarray(
            torch.as_tensor(predictions).detach().cpu()
        ).astype(int).reshape(-1)
        targets = np.asarray(torch.as_tensor(targets).detach().cpu()).astype(int).reshape(
            -1
        )
        if predictions.size != targets.size:
            raise ValueError("predictions and targets must have the same length.")

        for class_index in np.unique(targets):
            if not 0 <= class_index < self.class_count:
                raise ValueError(f"Target class {class_index} is out of range.")
            mask = targets == class_index
            self.count[class_index] += int(mask.sum())
            self.correct[class_index] += int((predictions[mask] == targets[mask]).sum())

    def summary(self, class_index: int) -> tuple[float | None, int, int]:
        count = int(self.count[class_index])
        correct = int(self.correct[class_index])
        accuracy = None if count == 0 else correct / count
        return accuracy, correct, count


class EarlyStopping:
    """Save the best validation-loss checkpoint and stop after no improvement."""

    def __init__(
        self,
        patience: int,
        minimum_epoch: int,
        checkpoint_path: Path,
        verbose: bool = True,
    ) -> None:
        if patience < 1:
            raise ValueError("patience must be at least 1.")
        if minimum_epoch < 0:
            raise ValueError("minimum_epoch must be non-negative.")
        self.patience = patience
        self.minimum_epoch = minimum_epoch
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose
        self.best_loss = float("inf")
        self.epochs_without_improvement = 0

    def update(self, epoch: int, validation_loss: float, model: nn.Module) -> bool:
        if validation_loss < self.best_loss:
            if self.verbose:
                print(
                    f"Validation loss improved: {self.best_loss:.6f} -> "
                    f"{validation_loss:.6f}. Saving checkpoint."
                )
            self.best_loss = validation_loss
            self.epochs_without_improvement = 0
            torch.save(model.state_dict(), self.checkpoint_path)
            return False

        self.epochs_without_improvement += 1
        if self.verbose:
            print(
                "No validation improvement: "
                f"{self.epochs_without_improvement}/{self.patience}"
            )
        return (
            epoch >= self.minimum_epoch
            and self.epochs_without_improvement >= self.patience
        )


@dataclass
class EvaluationResult:
    loss: float
    error: float
    auc: float
    class_accuracy: ClassAccuracy
    instance_loss: float | None = None
    instance_accuracy: ClassAccuracy | None = None
    records: dict[str, dict[str, Any]] | None = None


class SmoothTop1SVMLoss(nn.Module):
    """Device-independent smooth multiclass hinge loss."""

    def __init__(
        self,
        class_count: int,
        margin: float = 1.0,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if class_count < 2:
            raise ValueError("class_count must be at least 2.")
        if margin < 0:
            raise ValueError("margin must be non-negative.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.class_count = class_count
        self.margin = margin
        self.temperature = temperature

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if logits.ndim != 2 or logits.size(1) != self.class_count:
            raise ValueError(
                f"logits must have shape [batch_size, {self.class_count}]."
            )
        targets = targets.to(device=logits.device, dtype=torch.long)
        true_scores = logits.gather(1, targets.unsqueeze(1))
        margins = logits - true_scores + self.margin
        margins = margins.scatter(1, targets.unsqueeze(1), 0.0)
        return (
            self.temperature
            * torch.logsumexp(margins / self.temperature, dim=1)
        ).mean()


def create_loss(name: str, class_count: int) -> nn.Module:
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "svm":
        return SmoothTop1SVMLoss(class_count)
    raise ValueError(f"Unsupported loss: {name}")


def build_model(
    args,
    instance_loss: nn.Module | None = None,
    device: torch.device = DEVICE,
) -> nn.Module:
    if instance_loss is None:
        instance_loss = nn.CrossEntropyLoss()
    model = PathMIL(
        embed_dim=args.embed_dim,
        hidden_dim=getattr(args, "hidden_dim", 512),
        attention_dim=getattr(args, "attention_dim", 256),
        dropout=args.drop_out,
        n_classes=args.n_classes,
        instance_sample_count=getattr(args, "instance_sample_count", 8),
        instance_loss_fn=instance_loss,
        include_negative_instances=getattr(
            args,
            "include_negative_instances",
            False,
        ),
        path_depth=getattr(args, "path_depth", 2),
        path_attention_dim=getattr(args, "path_attention_dim", 256),
        path_spatial_dim=getattr(args, "path_spatial_dim", 32),
        path_neighbor_count=getattr(args, "path_neighbor_count", 8),
        path_temperature=getattr(args, "path_temperature", 1.0),
        path_local_weight=getattr(args, "path_local_weight", 1.0),
        path_spatial_weight=getattr(args, "path_spatial_weight", 0.25),
        path_global_weight=getattr(args, "path_global_weight", 0.5),
    )
    return model.to(device)


def _forward_batch(
    model: nn.Module,
    features: torch.Tensor,
    coordinates: torch.Tensor,
    labels: torch.Tensor | None = None,
    evaluate_instances: bool = False,
) -> dict[str, torch.Tensor]:
    return model(
        features,
        coordinates,
        label=labels,
        evaluate_instances=evaluate_instances,
    )


def _projection_loss(model: nn.Module) -> torch.Tensor:
    base_model = model.module if hasattr(model, "module") else model
    path_encoder = getattr(base_model, "path_encoder", None)
    if path_encoder is None:
        return next(model.parameters()).new_zeros(())
    return path_encoder.projection_decorrelation_loss()


def _projection_stats(model: nn.Module) -> dict[str, float]:
    base_model = model.module if hasattr(model, "module") else model
    path_encoder = getattr(base_model, "path_encoder", None)
    if path_encoder is None:
        return {}
    return path_encoder.projection_decorrelation_stats()


def _write_class_accuracy(writer, prefix: str, tracker: ClassAccuracy, step: int) -> None:
    for class_index in range(tracker.class_count):
        accuracy, _, _ = tracker.summary(class_index)
        if accuracy is not None:
            writer.add_scalar(
                f"{prefix}/class_{class_index}_accuracy",
                accuracy,
                step,
            )


def _create_summary_writer(log_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        from tensorboardX import SummaryWriter
    return SummaryWriter(str(log_dir), flush_secs=15)


def _print_class_accuracy(title: str, tracker: ClassAccuracy) -> None:
    for class_index in range(tracker.class_count):
        accuracy, correct, count = tracker.summary(class_index)
        print(
            f"{title} class {class_index}: accuracy={accuracy}, "
            f"correct={correct}/{count}"
        )


def train_one_epoch(
    epoch: int,
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    classification_loss: nn.Module,
    args,
    writer=None,
) -> dict[str, float]:
    """Run one optimization epoch and return averaged scalar metrics."""
    start_time = time.perf_counter()
    model.train()
    slide_accuracy = ClassAccuracy(args.n_classes)
    instance_accuracy = ClassAccuracy(2)
    totals = {
        "slide_loss": 0.0,
        "instance_loss": 0.0,
        "supervised_loss": 0.0,
        "projection_loss": 0.0,
        "total_loss": 0.0,
        "error": 0.0,
        "gradient_norm": 0.0,
    }
    batch_count = 0
    instance_batch_count = 0

    use_instance_supervision = args.instance_supervision

    for batch_index, (features, labels, coordinates) in enumerate(loader):
        features = features.to(DEVICE)
        labels = labels.to(DEVICE)
        coordinates = coordinates.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)

        output = _forward_batch(
            model,
            features,
            coordinates,
            labels,
            evaluate_instances=use_instance_supervision,
        )
        slide_loss = classification_loss(output["logits"], labels)

        if use_instance_supervision:
            instance_loss = output["instance_loss"]
            supervised_loss = (
                args.slide_loss_weight * slide_loss
                + (1.0 - args.slide_loss_weight) * instance_loss
            )
            instance_accuracy.update(
                output["instance_predictions"],
                output["instance_targets"],
            )
            totals["instance_loss"] += instance_loss.detach().item()
            instance_batch_count += 1
        else:
            instance_loss = slide_loss.new_zeros(())
            supervised_loss = slide_loss

        projection_loss = _projection_loss(model)
        total_loss = (
            supervised_loss + args.projection_reg_weight * projection_loss
        )
        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                "Non-finite loss detected at "
                f"epoch={epoch}, batch={batch_index}."
            )

        total_loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float("inf"),
        )
        optimizer.step()

        slide_accuracy.update(output["predictions"], labels)
        totals["slide_loss"] += slide_loss.detach().item()
        totals["supervised_loss"] += supervised_loss.detach().item()
        totals["projection_loss"] += projection_loss.detach().item()
        totals["total_loss"] += total_loss.detach().item()
        totals["error"] += classification_error(output["predictions"], labels)
        totals["gradient_norm"] += float(gradient_norm)
        batch_count += 1

        if args.print_every > 0 and (batch_index + 1) % args.print_every == 0:
            print(
                f"epoch={epoch} batch={batch_index + 1} "
                f"slide_loss={slide_loss.detach().item():.4f} "
                f"instance_loss={instance_loss.detach().item():.4f} "
                f"projection_loss={projection_loss.detach().item():.6f} "
                f"total_loss={total_loss.detach().item():.4f}"
            )

    if batch_count == 0:
        raise RuntimeError("The training loader returned no batches.")

    averaged = {
        name: value / batch_count
        for name, value in totals.items()
        if name != "instance_loss"
    }
    averaged["instance_loss"] = (
        totals["instance_loss"] / instance_batch_count
        if instance_batch_count
        else 0.0
    )
    averaged["projection_weighted"] = (
        args.projection_reg_weight * averaged["projection_loss"]
    )
    averaged["accuracy"] = 1.0 - averaged["error"]
    averaged["learning_rate"] = float(optimizer.param_groups[0]["lr"])
    averaged["epoch_seconds"] = time.perf_counter() - start_time

    print(
        f"Train epoch {epoch}: total_loss={averaged['total_loss']:.4f}, "
        f"slide_loss={averaged['slide_loss']:.4f}, "
        f"instance_loss={averaged['instance_loss']:.4f}, "
        f"projection_loss={averaged['projection_loss']:.6f}, "
        f"error={averaged['error']:.4f}"
    )
    _print_class_accuracy("Train", slide_accuracy)
    if instance_batch_count:
        _print_class_accuracy("Train instance", instance_accuracy)

    projection_stats = _projection_stats(model)
    if projection_stats:
        print(
            "Projection statistics: "
            + ", ".join(f"{key}={value:.6f}" for key, value in projection_stats.items())
        )

    if writer is not None:
        for name, value in averaged.items():
            writer.add_scalar(f"train/{name}", value, epoch)
        for name, value in projection_stats.items():
            writer.add_scalar(f"train/projection_{name}", value, epoch)
        _write_class_accuracy(writer, "train", slide_accuracy, epoch)
        if instance_batch_count:
            _write_class_accuracy(
                writer,
                "train_instance",
                instance_accuracy,
                epoch,
            )
    return averaged


def _compute_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    class_count = probabilities.shape[1]
    if class_count == 2:
        if np.unique(labels).size < 2:
            return float("nan")
        return float(roc_auc_score(labels, probabilities[:, 1]))

    class_auc = []
    for class_index in range(class_count):
        binary_labels = (labels == class_index).astype(int)
        if np.unique(binary_labels).size < 2:
            class_auc.append(float("nan"))
        else:
            class_auc.append(
                roc_auc_score(binary_labels, probabilities[:, class_index])
            )
    return float(np.nanmean(class_auc))


def evaluate_model(
    model: nn.Module,
    loader,
    classification_loss: nn.Module,
    args,
    split_name: str,
    epoch: int | None = None,
    writer=None,
    collect_records: bool = False,
) -> EvaluationResult:
    """Evaluate one split with shared validation/test metric logic."""
    model.eval()
    slide_accuracy = ClassAccuracy(args.n_classes)
    instance_accuracy = ClassAccuracy(2)
    probabilities = []
    labels = []
    total_loss = 0.0
    total_error = 0.0
    total_instance_loss = 0.0
    instance_batch_count = 0
    records: dict[str, dict[str, Any]] = {}
    slide_ids = (
        loader.dataset.slide_data["slide_id"] if collect_records else None
    )
    evaluate_instances = (
        split_name == "validation" and args.instance_supervision
    )

    with torch.inference_mode():
        for batch_index, (features, batch_labels, coordinates) in enumerate(loader):
            features = features.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            coordinates = coordinates.to(DEVICE)
            output = _forward_batch(
                model,
                features,
                coordinates,
                batch_labels,
                evaluate_instances=evaluate_instances,
            )
            loss = classification_loss(output["logits"], batch_labels)
            total_loss += loss.item()
            total_error += classification_error(
                output["predictions"],
                batch_labels,
            )
            slide_accuracy.update(output["predictions"], batch_labels)

            batch_probabilities = (
                output["probabilities"].detach().cpu().numpy().reshape(-1)
            )
            batch_label = int(batch_labels.view(-1)[0].item())
            probabilities.append(batch_probabilities)
            labels.append(batch_label)

            if evaluate_instances:
                total_instance_loss += output["instance_loss"].item()
                instance_batch_count += 1
                instance_accuracy.update(
                    output["instance_predictions"],
                    output["instance_targets"],
                )

            if collect_records and slide_ids is not None:
                slide_id = slide_ids.iloc[batch_index]
                records[str(slide_id)] = {
                    "slide_id": np.asarray(slide_id),
                    "prob": batch_probabilities,
                    "label": batch_label,
                }

    batch_count = len(labels)
    if batch_count == 0:
        raise RuntimeError(f"The {split_name} loader returned no batches.")

    probabilities_array = np.stack(probabilities)
    labels_array = np.asarray(labels)
    result = EvaluationResult(
        loss=total_loss / batch_count,
        error=total_error / batch_count,
        auc=_compute_auc(labels_array, probabilities_array),
        class_accuracy=slide_accuracy,
        instance_loss=(
            total_instance_loss / instance_batch_count
            if instance_batch_count
            else None
        ),
        instance_accuracy=instance_accuracy if instance_batch_count else None,
        records=records if collect_records else None,
    )

    print(
        f"{split_name.capitalize()}: loss={result.loss:.4f}, "
        f"error={result.error:.4f}, auc={result.auc:.4f}"
    )
    _print_class_accuracy(split_name.capitalize(), slide_accuracy)
    if result.instance_accuracy is not None:
        _print_class_accuracy(
            f"{split_name.capitalize()} instance",
            result.instance_accuracy,
        )

    if writer is not None and epoch is not None:
        writer.add_scalar(f"{split_name}/loss", result.loss, epoch)
        writer.add_scalar(f"{split_name}/error", result.error, epoch)
        writer.add_scalar(f"{split_name}/accuracy", 1.0 - result.error, epoch)
        writer.add_scalar(f"{split_name}/auc", result.auc, epoch)
        if result.instance_loss is not None:
            writer.add_scalar(
                f"{split_name}/instance_loss",
                result.instance_loss,
                epoch,
            )
        _write_class_accuracy(writer, split_name, slide_accuracy, epoch)
    return result


def train_fold(datasets, fold: int, args):
    """Train, validate and test one cross-validation fold."""
    print(f"\nTraining fold {fold}")
    result_dir = Path(args.results_dir)
    log_dir = result_dir / "tensorboard" / f"fold_{fold}"
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = result_dir / f"s_{fold}_checkpoint.pt"

    writer = None
    if args.log_data:
        writer = _create_summary_writer(log_dir)
        writer.add_text("run/device", str(DEVICE), 0)
        effective_config = getattr(args, "effective_config_yaml", "")
        if effective_config:
            writer.add_text(
                "run/config",
                f"```yaml\n{effective_config}\n```",
                0,
            )

    train_split, validation_split, test_split = datasets
    save_splits(
        datasets,
        ["train", "validation", "test"],
        str(result_dir / f"splits_{fold}.csv"),
    )
    print(
        f"Samples: train={len(train_split)}, "
        f"validation={len(validation_split)}, test={len(test_split)}"
    )

    classification_loss = create_loss(
        args.classification_loss,
        args.n_classes,
    )
    instance_loss = create_loss(args.instance_loss, 2)
    model = build_model(args, instance_loss)
    print_model_summary(model)
    if writer is not None:
        total_parameters = sum(
            parameter.numel() for parameter in model.parameters()
        )
        trainable_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        writer.add_scalar("model/total_parameters", total_parameters, 0)
        writer.add_scalar(
            "model/trainable_parameters",
            trainable_parameters,
            0,
        )
    optimizer = create_optimizer(model, args)

    train_loader = build_split_loader(
        train_split,
        training=True,
        testing=args.testing,
        weighted=args.weighted_sample,
    )
    validation_loader = build_split_loader(
        validation_split,
        testing=args.testing,
    )
    test_loader = build_split_loader(test_split, testing=args.testing)

    stopper = None
    if args.early_stopping:
        stopper = EarlyStopping(
            patience=args.early_stopping_patience,
            minimum_epoch=args.early_stopping_min_epoch,
            checkpoint_path=checkpoint_path,
        )

    try:
        for epoch in range(args.max_epochs):
            train_one_epoch(
                epoch,
                model,
                train_loader,
                optimizer,
                classification_loss,
                args,
                writer,
            )
            validation = evaluate_model(
                model,
                validation_loader,
                classification_loss,
                args,
                split_name="validation",
                epoch=epoch,
                writer=writer,
            )
            if writer is not None:
                writer.flush()
            if stopper is not None and stopper.update(
                epoch,
                validation.loss,
                model,
            ):
                print(f"Early stopping at epoch {epoch}.")
                break

        if stopper is None:
            torch.save(model.state_dict(), checkpoint_path)
        else:
            model.load_state_dict(
                torch.load(checkpoint_path, map_location=DEVICE)
            )

        validation = evaluate_model(
            model,
            validation_loader,
            classification_loss,
            args,
            split_name="validation",
        )
        test = evaluate_model(
            model,
            test_loader,
            classification_loss,
            args,
            split_name="test",
            writer=writer,
            collect_records=True,
        )

        if writer is not None:
            writer.add_scalar("final/validation_loss", validation.loss, 0)
            writer.add_scalar("final/validation_error", validation.error, 0)
            writer.add_scalar(
                "final/validation_accuracy",
                1.0 - validation.error,
                0,
            )
            writer.add_scalar("final/validation_auc", validation.auc, 0)
            writer.add_scalar("final/test_loss", test.loss, 0)
            writer.add_scalar("final/test_error", test.error, 0)
            writer.add_scalar("final/test_accuracy", 1.0 - test.error, 0)
            writer.add_scalar("final/test_auc", test.auc, 0)

        return (
            test.records,
            test.auc,
            validation.auc,
            1.0 - test.error,
            1.0 - validation.error,
        )
    finally:
        if writer is not None:
            writer.close()
