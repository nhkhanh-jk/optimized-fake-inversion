"""
FakeInversion - Evaluation
============================
Comprehensive evaluation: per-model metrics, generalization testing,
confusion matrices, and result tables.
"""

import os
import sys
import json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
from torch.cuda.amp import autocast

import config
from utils import setup_logger, load_checkpoint
from classifier.model import FakeInversionClassifier
from classifier.dataset import FakeInversionDataset, EvalTransform


logger = setup_logger("evaluate")


class Evaluator:
    """Comprehensive evaluator for FakeInversion detector."""

    def __init__(
        self,
        model_path: str = None,
        device: torch.device = None,
    ):
        """Initialize the evaluator.

        Args:
            model_path: Path to trained model checkpoint.
            device: Compute device.
        """
        self.device = device or config.get_device()
        self.model_path = model_path or os.path.join(
            config.CHECKPOINTS_DIR, "best_model.pt"
        )
        self.model = None

    def load_model(self):
        """Load the trained classifier."""
        logger.info(f"Loading model from: {self.model_path}")

        self.model = FakeInversionClassifier().to(self.device)

        ckpt = load_checkpoint(self.model_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        best_auc = ckpt.get("best_auc", "N/A")
        epoch = ckpt.get("epoch", "N/A")
        logger.info(f"  Loaded: epoch={epoch}, best_auc={best_auc}")

    @torch.no_grad()
    def predict_dataset(self, dataset: FakeInversionDataset) -> dict:
        """Run predictions on an entire dataset.

        Returns:
            Dict with labels, probs, preds, model_names.
        """
        from torch.utils.data import DataLoader

        loader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        all_labels = []
        all_probs = []
        all_preds = []
        all_model_names = []

        for features, labels, model_names in loader:
            features = features.to(self.device)

            with autocast(enabled=config.USE_AMP):
                logits = self.model(features)

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs > 0.5).long()

            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_model_names.extend(model_names)

        return {
            "labels": np.array(all_labels),
            "probs": np.array(all_probs),
            "preds": np.array(all_preds),
            "model_names": all_model_names,
        }

    def compute_metrics(self, labels, probs, preds) -> dict:
        """Compute classification metrics.

        Returns:
            Dict with accuracy, auc, ap, confusion_matrix, etc.
        """
        metrics = {}

        metrics["accuracy"] = accuracy_score(labels, preds)
        metrics["num_samples"] = len(labels)
        metrics["num_real"] = int((labels == 0).sum())
        metrics["num_fake"] = int((labels == 1).sum())

        try:
            metrics["auc"] = roc_auc_score(labels, probs)
        except ValueError:
            metrics["auc"] = float("nan")

        try:
            metrics["ap"] = average_precision_score(labels, probs)
        except ValueError:
            metrics["ap"] = float("nan")

        cm = confusion_matrix(labels, preds, labels=[0, 1])
        metrics["confusion_matrix"] = cm.tolist()

        # True positive rate, false positive rate
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["tpr"] = tp / max(tp + fn, 1)  # Sensitivity / Recall
            metrics["fpr"] = fp / max(fp + tn, 1)  # False positive rate
            metrics["tnr"] = tn / max(tn + fp, 1)  # Specificity
            metrics["precision"] = tp / max(tp + fp, 1)

        return metrics

    def evaluate_overall(self, features_dir: str) -> dict:
        """Evaluate overall performance.

        Args:
            features_dir: Directory with .pt feature files.

        Returns:
            Dict with overall metrics.
        """
        if self.model is None:
            self.load_model()

        dataset = FakeInversionDataset(features_dir, transform=EvalTransform())
        logger.info(f"Evaluating {len(dataset)} samples from {features_dir}")

        results = self.predict_dataset(dataset)
        metrics = self.compute_metrics(
            results["labels"], results["probs"], results["preds"]
        )

        logger.info(f"  Overall Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Overall AUC:      {metrics['auc']:.4f}")
        logger.info(f"  Overall AP:       {metrics['ap']:.4f}")

        return metrics

    def evaluate_per_model(self, features_dir: str) -> dict:
        """Evaluate per-model performance (generalization test).

        Args:
            features_dir: Directory with .pt feature files.

        Returns:
            Dict mapping model_name → metrics.
        """
        if self.model is None:
            self.load_model()

        dataset = FakeInversionDataset(features_dir, transform=EvalTransform())
        results = self.predict_dataset(dataset)

        # Group by model
        model_groups = defaultdict(lambda: {"labels": [], "probs": [], "preds": []})

        for i, model_name in enumerate(results["model_names"]):
            model_groups[model_name]["labels"].append(results["labels"][i])
            model_groups[model_name]["probs"].append(results["probs"][i])
            model_groups[model_name]["preds"].append(results["preds"][i])

        # Compute per-model metrics
        per_model_metrics = {}

        logger.info("\n" + "=" * 80)
        logger.info(f"{'Model':>20s} | {'Samples':>8s} | {'Acc':>8s} | {'AUC':>8s} | {'AP':>8s} | {'TPR':>8s}")
        logger.info("-" * 80)

        for model_name in sorted(model_groups.keys()):
            group = model_groups[model_name]
            labels = np.array(group["labels"])
            probs = np.array(group["probs"])
            preds = np.array(group["preds"])

            metrics = self.compute_metrics(labels, probs, preds)
            per_model_metrics[model_name] = metrics

            # Highlight unseen models (not used in training)
            marker = "  " if model_name == config.TRAIN_SOURCE_MODEL else "★ "

            logger.info(
                f"{marker}{model_name:>18s} | {metrics['num_samples']:>8d} | "
                f"{metrics['accuracy']:>8.4f} | {metrics['auc']:>8.4f} | "
                f"{metrics['ap']:>8.4f} | {metrics.get('tpr', 0):>8.4f}"
            )

        logger.info("=" * 80)
        logger.info("  ★ = unseen model (not used in training)")

        # Compute averages
        avg_acc = np.mean([m["accuracy"] for m in per_model_metrics.values()])
        avg_auc = np.nanmean([m["auc"] for m in per_model_metrics.values()])
        avg_ap = np.nanmean([m["ap"] for m in per_model_metrics.values()])

        logger.info(f"\n  Average Accuracy: {avg_acc:.4f}")
        logger.info(f"  Average AUC:      {avg_auc:.4f}")
        logger.info(f"  Average AP:       {avg_ap:.4f}")

        per_model_metrics["_average"] = {
            "accuracy": avg_acc,
            "auc": avg_auc,
            "ap": avg_ap,
        }

        return per_model_metrics

    def save_results(self, results: dict, filename: str = "evaluation_results.json"):
        """Save evaluation results to JSON."""
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        filepath = os.path.join(config.RESULTS_DIR, filename)

        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        serializable = json.loads(json.dumps(results, default=convert))

        with open(filepath, "w") as f:
            json.dump(serializable, f, indent=2)

        logger.info(f"📊 Results saved to: {filepath}")

    def get_roc_data(self, features_dir: str) -> dict:
        """Get ROC curve data for plotting.

        Returns:
            Dict mapping model_name → {fpr, tpr, auc}.
        """
        if self.model is None:
            self.load_model()

        dataset = FakeInversionDataset(features_dir, transform=EvalTransform())
        results = self.predict_dataset(dataset)

        model_groups = defaultdict(lambda: {"labels": [], "probs": []})
        for i, model_name in enumerate(results["model_names"]):
            model_groups[model_name]["labels"].append(results["labels"][i])
            model_groups[model_name]["probs"].append(results["probs"][i])

        roc_data = {}
        for model_name, group in model_groups.items():
            labels = np.array(group["labels"])
            probs = np.array(group["probs"])

            if len(np.unique(labels)) < 2:
                continue

            fpr, tpr, _ = roc_curve(labels, probs)
            auc = roc_auc_score(labels, probs)

            roc_data[model_name] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": auc,
            }

        return roc_data


def run_full_evaluation(
    model_path: str = None,
    test_dir: str = None,
    eval_dir: str = None,
):
    """Run the complete evaluation pipeline.

    Args:
        model_path: Path to trained model checkpoint.
        test_dir: Directory with test features.
        eval_dir: Directory with evaluation features (all models).
    """
    evaluator = Evaluator(model_path=model_path)

    # Test set evaluation (same distribution as training)
    if test_dir is None:
        test_dir = os.path.join(config.FEATURES_DIR, "test")

    if os.path.exists(test_dir):
        logger.info("\n📋 TEST SET EVALUATION (same distribution)")
        test_metrics = evaluator.evaluate_overall(test_dir)
        evaluator.save_results(
            {"test_overall": test_metrics}, "test_results.json"
        )

    # Generalization evaluation (all models)
    if eval_dir is None:
        eval_dir = os.path.join(config.FEATURES_DIR, "eval")

    if os.path.exists(eval_dir):
        logger.info("\n📋 GENERALIZATION EVALUATION (all models)")
        per_model = evaluator.evaluate_per_model(eval_dir)
        evaluator.save_results(per_model, "per_model_results.json")

        # Save ROC data
        roc_data = evaluator.get_roc_data(eval_dir)
        evaluator.save_results(roc_data, "roc_data.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate FakeInversion detector")
    parser.add_argument("--model", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--eval_dir", type=str, default=None)
    args = parser.parse_args()

    run_full_evaluation(
        model_path=args.model,
        test_dir=args.test_dir,
        eval_dir=args.eval_dir,
    )
