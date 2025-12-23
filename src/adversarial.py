"""
WarrantProof Adversarial Module - PGD Attack Generator for Robust Training

SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY

This module implements Projected Gradient Descent (PGD) attack generation
for adversarial training of the KAN (Kolmogorov-Arnold Network) fraud detector.

Key Insight:
Fraudsters can compute gradients. If they know the model, they can perturb
features to cross the decision boundary (evasion attack).

Solution: Generate adversarial examples during training. Force model to be
robust to worst-case perturbations within epsilon ball.

OMEGA Citation:
"Create a 'Red Team' generator that produces 'Adversarial Examples' (PGD attack)"
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
import random
import math

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from .core import (
    TENANT_ID,
    DISCLAIMER,
    ADVERSARIAL_EPSILON,
    ADVERSARIAL_PGD_STEPS,
    KAN_ROBUST_ACCURACY_TARGET,
    dual_hash,
    emit_receipt,
    StopRuleException,
)


@dataclass
class AdversarialExample:
    """An adversarial perturbation of original input."""
    original: List[float]
    perturbed: List[float]
    perturbation: List[float]
    epsilon: float
    attack_type: str
    original_prediction: float = 0.0
    adversarial_prediction: float = 0.0
    evasion_success: bool = False


def pgd_attack(
    model_fn: Callable[[List[float]], float],
    x: List[float],
    y_true: int,
    epsilon: float = ADVERSARIAL_EPSILON,
    steps: int = ADVERSARIAL_PGD_STEPS,
    step_size: Optional[float] = None
) -> AdversarialExample:
    """
    Projected Gradient Descent attack.
    Perturb x by up to epsilon in direction of gradient to cause misclassification.

    Args:
        model_fn: Function that takes input and returns P(fraud)
        x: Input features
        y_true: True label (0=legitimate, 1=fraud)
        epsilon: Maximum perturbation (L-infinity norm)
        steps: Number of PGD iterations
        step_size: Step size per iteration (default: epsilon/steps)

    Returns:
        AdversarialExample with perturbed input
    """
    if step_size is None:
        step_size = epsilon / steps

    x_adv = list(x)
    original_pred = model_fn(x)

    for _ in range(steps):
        # Estimate gradient numerically
        gradient = _estimate_gradient(model_fn, x_adv)

        # Update adversarial example
        if y_true == 1:
            # If true fraud, try to make it look legitimate (reduce P(fraud))
            for i in range(len(x_adv)):
                x_adv[i] -= step_size * _sign(gradient[i])
        else:
            # If true legitimate, try to make it look like fraud (increase P(fraud))
            for i in range(len(x_adv)):
                x_adv[i] += step_size * _sign(gradient[i])

        # Project back onto epsilon ball (L-infinity constraint)
        for i in range(len(x_adv)):
            x_adv[i] = max(x[i] - epsilon, min(x[i] + epsilon, x_adv[i]))
            # Also clamp to valid feature ranges [0, 1] for normalized features
            x_adv[i] = max(0.0, min(1.0, x_adv[i]))

    adversarial_pred = model_fn(x_adv)

    # Check if attack succeeded
    if y_true == 1:
        evasion_success = adversarial_pred < 0.5  # Fraud misclassified as legitimate
    else:
        evasion_success = adversarial_pred >= 0.5  # Legitimate misclassified as fraud

    perturbation = [x_adv[i] - x[i] for i in range(len(x))]

    return AdversarialExample(
        original=x,
        perturbed=x_adv,
        perturbation=perturbation,
        epsilon=epsilon,
        attack_type="PGD",
        original_prediction=original_pred,
        adversarial_prediction=adversarial_pred,
        evasion_success=evasion_success,
    )


def fgsm_attack(
    model_fn: Callable[[List[float]], float],
    x: List[float],
    y_true: int,
    epsilon: float = ADVERSARIAL_EPSILON
) -> AdversarialExample:
    """
    Fast Gradient Sign Method (single-step PGD).
    Faster but weaker than full PGD.

    Args:
        model_fn: Function that takes input and returns P(fraud)
        x: Input features
        y_true: True label
        epsilon: Maximum perturbation

    Returns:
        AdversarialExample with perturbed input
    """
    gradient = _estimate_gradient(model_fn, x)
    original_pred = model_fn(x)

    x_adv = list(x)
    for i in range(len(x_adv)):
        if y_true == 1:
            x_adv[i] -= epsilon * _sign(gradient[i])
        else:
            x_adv[i] += epsilon * _sign(gradient[i])
        # Clamp to valid range
        x_adv[i] = max(0.0, min(1.0, x_adv[i]))

    adversarial_pred = model_fn(x_adv)

    if y_true == 1:
        evasion_success = adversarial_pred < 0.5
    else:
        evasion_success = adversarial_pred >= 0.5

    perturbation = [x_adv[i] - x[i] for i in range(len(x))]

    return AdversarialExample(
        original=x,
        perturbed=x_adv,
        perturbation=perturbation,
        epsilon=epsilon,
        attack_type="FGSM",
        original_prediction=original_pred,
        adversarial_prediction=adversarial_pred,
        evasion_success=evasion_success,
    )


def generate_adversarial_dataset(
    transactions: List[dict],
    model_fn: Callable[[List[float]], float],
    feature_extractor: Callable[[dict], List[float]],
    epsilon: float = ADVERSARIAL_EPSILON
) -> List[Tuple[List[float], int]]:
    """
    For each transaction, generate adversarial perturbation.
    Return augmented dataset for adversarial training.

    Args:
        transactions: Original transactions
        model_fn: Model prediction function
        feature_extractor: Function to extract features from transaction
        epsilon: Perturbation bound

    Returns:
        List of (perturbed_features, true_label) tuples
    """
    augmented = []

    for tx in transactions:
        # Extract features
        x = feature_extractor(tx)
        y_true = 1 if tx.get("_is_fraud") or tx.get("classification") == "fraudulent" else 0

        # Include original
        augmented.append((x, y_true))

        # Generate adversarial example
        adv = pgd_attack(model_fn, x, y_true, epsilon)
        augmented.append((adv.perturbed, y_true))

    return augmented


def evaluate_robustness(
    model_fn: Callable[[List[float]], float],
    test_data: List[Tuple[List[float], int]],
    epsilon: float = ADVERSARIAL_EPSILON
) -> dict:
    """
    Measure accuracy under PGD attack.

    Args:
        model_fn: Model prediction function
        test_data: List of (features, label) tuples
        epsilon: Attack epsilon

    Returns:
        Robustness evaluation dict
    """
    clean_correct = 0
    adversarial_correct = 0
    attack_successes = 0

    for x, y_true in test_data:
        # Clean accuracy
        pred = model_fn(x)
        predicted_label = 1 if pred >= 0.5 else 0
        if predicted_label == y_true:
            clean_correct += 1

        # Adversarial accuracy
        adv = pgd_attack(model_fn, x, y_true, epsilon)
        adv_pred_label = 1 if adv.adversarial_prediction >= 0.5 else 0
        if adv_pred_label == y_true:
            adversarial_correct += 1
        else:
            attack_successes += 1

    total = len(test_data)
    clean_accuracy = clean_correct / total if total > 0 else 0
    robust_accuracy = adversarial_correct / total if total > 0 else 0
    attack_success_rate = attack_successes / total if total > 0 else 0

    return {
        "clean_accuracy": round(clean_accuracy, 4),
        "robust_accuracy": round(robust_accuracy, 4),
        "attack_success_rate": round(attack_success_rate, 4),
        "epsilon": epsilon,
        "samples_evaluated": total,
        "meets_target": robust_accuracy >= KAN_ROBUST_ACCURACY_TARGET,
    }


def _estimate_gradient(
    model_fn: Callable[[List[float]], float],
    x: List[float],
    delta: float = 1e-4
) -> List[float]:
    """
    Estimate gradient numerically using finite differences.

    Args:
        model_fn: Model prediction function
        x: Input point
        delta: Step size for finite differences

    Returns:
        Estimated gradient
    """
    gradient = []
    base_output = model_fn(x)

    for i in range(len(x)):
        x_plus = list(x)
        x_plus[i] += delta

        output_plus = model_fn(x_plus)
        grad_i = (output_plus - base_output) / delta
        gradient.append(grad_i)

    return gradient


def _sign(x: float) -> float:
    """Sign function: returns -1, 0, or 1."""
    if x > 0:
        return 1.0
    elif x < 0:
        return -1.0
    return 0.0


def emit_adversarial_receipt(adv_example: AdversarialExample) -> dict:
    """
    Emit adversarial_receipt documenting attack result.

    Args:
        adv_example: AdversarialExample to document

    Returns:
        adversarial_receipt dict
    """
    # Calculate gradient norm (L2)
    perturbation_norm = math.sqrt(sum(p**2 for p in adv_example.perturbation))

    return emit_receipt("adversarial", {
        "tenant_id": TENANT_ID,
        "attack_type": adv_example.attack_type,
        "perturbation_epsilon": adv_example.epsilon,
        "evasion_success": adv_example.evasion_success,
        "gradient_norm": round(perturbation_norm, 6),
        "original_prediction": round(adv_example.original_prediction, 4),
        "adversarial_prediction": round(adv_example.adversarial_prediction, 4),
        "prediction_shift": round(abs(adv_example.adversarial_prediction - adv_example.original_prediction), 4),
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


# === STOPRULES ===

def stoprule_robust_accuracy_low(accuracy: float) -> None:
    """If accuracy < 85% under attack, retrain."""
    if accuracy < KAN_ROBUST_ACCURACY_TARGET:
        emit_receipt("anomaly", {
            "metric": "robust_accuracy_low",
            "accuracy": accuracy,
            "target": KAN_ROBUST_ACCURACY_TARGET,
            "delta": accuracy - KAN_ROBUST_ACCURACY_TARGET,
            "action": "retrain_with_adversarial",
            "classification": "deviation",
            "simulation_flag": DISCLAIMER,
        })
        raise StopRuleException(
            f"Robust accuracy {accuracy:.2%} < target {KAN_ROBUST_ACCURACY_TARGET:.2%}"
        )


def stoprule_epsilon_violated(perturbation: List[float], epsilon: float) -> None:
    """If perturbations exceed epsilon, invalid attack."""
    max_perturbation = max(abs(p) for p in perturbation) if perturbation else 0
    if max_perturbation > epsilon * 1.01:  # Allow 1% tolerance
        emit_receipt("anomaly", {
            "metric": "epsilon_violated",
            "max_perturbation": max_perturbation,
            "epsilon": epsilon,
            "action": "fix_attack",
            "classification": "violation",
            "simulation_flag": DISCLAIMER,
        })
        raise StopRuleException(
            f"Perturbation {max_perturbation} exceeds epsilon {epsilon}"
        )


# === MODULE SELF-TEST ===

if __name__ == "__main__":
    import sys

    print(f"# WarrantProof Adversarial Module Self-Test", file=sys.stderr)
    print(f"# {DISCLAIMER}", file=sys.stderr)

    # Simple test model: linear classifier
    def simple_model(x: List[float]) -> float:
        """P(fraud) = sigmoid(sum of features - 0.5)"""
        z = sum(x) - len(x) * 0.5
        return 1 / (1 + math.exp(-z))

    # Test 1: PGD attack on legitimate sample
    x_legit = [0.3, 0.3, 0.3, 0.3, 0.3]  # Low values = legitimate
    y_legit = 0

    adv_legit = pgd_attack(simple_model, x_legit, y_legit, epsilon=0.1)
    print(f"# PGD on legitimate: orig={adv_legit.original_prediction:.3f}, adv={adv_legit.adversarial_prediction:.3f}", file=sys.stderr)
    print(f"# Evasion success: {adv_legit.evasion_success}", file=sys.stderr)

    # Test 2: PGD attack on fraud sample
    x_fraud = [0.7, 0.7, 0.7, 0.7, 0.7]  # High values = fraud
    y_fraud = 1

    adv_fraud = pgd_attack(simple_model, x_fraud, y_fraud, epsilon=0.1)
    print(f"# PGD on fraud: orig={adv_fraud.original_prediction:.3f}, adv={adv_fraud.adversarial_prediction:.3f}", file=sys.stderr)

    # Test 3: FGSM attack (faster, single-step)
    adv_fgsm = fgsm_attack(simple_model, x_legit, y_legit, epsilon=0.1)
    print(f"# FGSM: orig={adv_fgsm.original_prediction:.3f}, adv={adv_fgsm.adversarial_prediction:.3f}", file=sys.stderr)

    # Test 4: Evaluate robustness
    test_data = [
        ([0.3, 0.3, 0.3, 0.3, 0.3], 0),
        ([0.4, 0.4, 0.4, 0.4, 0.4], 0),
        ([0.7, 0.7, 0.7, 0.7, 0.7], 1),
        ([0.8, 0.8, 0.8, 0.8, 0.8], 1),
    ]
    robustness = evaluate_robustness(simple_model, test_data, epsilon=0.1)
    print(f"# Clean accuracy: {robustness['clean_accuracy']:.2%}", file=sys.stderr)
    print(f"# Robust accuracy: {robustness['robust_accuracy']:.2%}", file=sys.stderr)

    # Test 5: Receipt emission
    receipt = emit_adversarial_receipt(adv_legit)
    assert receipt["receipt_type"] == "adversarial"
    assert "evasion_success" in receipt

    # Test 6: Verify epsilon constraint
    for p in adv_legit.perturbation:
        assert abs(p) <= adv_legit.epsilon * 1.01, "Perturbation exceeded epsilon"

    print(f"# PASS: adversarial module self-test", file=sys.stderr)
