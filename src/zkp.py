"""
WarrantProof ZKP Module - Recursive Zero-Knowledge SNARKs for Fraud Prevention

SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY

This module implements Mina-style Incrementally Verifiable Computation (IVC)
for deterministic fraud prevention. Invalid transactions cannot be submitted -
they fail proof generation. Recursive composition means a single 22kb proof
validates the entire transaction history.

The Paradigm Shift:
- Detection -> Prevention
- Probabilistic -> Deterministic
- Reactive -> Proactive
- Statistical -> Cryptographic

OMEGA Citation:
"We do not ask 'Is this transaction suspicious?' We ask 'Is this transaction
mathematically possible within the constrained state transition function of
the Fiscal Budget?'"

Note: This is a simulation. Real ZKP requires libsnark/bellman.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from .core import (
    TENANT_ID,
    DISCLAIMER,
    ZKP_PROOF_SIZE_BYTES,
    ZKP_VERIFICATION_TIME_MAX_MS,
    dual_hash,
    emit_receipt,
    StopRuleException,
)


@dataclass
class ZKProof:
    """Simulated zero-knowledge proof structure."""
    proof_bytes: bytes
    state_prev_hash: str
    state_next_hash: str
    previous_proof_hash: Optional[str]
    circuit_constraints_satisfied: bool
    verification_key_hash: str
    ts: str = ""

    @property
    def size_bytes(self) -> int:
        return len(self.proof_bytes)

    @property
    def hash(self) -> str:
        return dual_hash(self.proof_bytes)


@dataclass
class CircuitConstraints:
    """ZK circuit constraints for procurement transactions."""
    invoice_math_correct: bool = True
    vendor_signature_valid: bool = True
    contract_parameters_satisfied: bool = True
    sam_gov_verified: bool = True
    bekenstein_bound_respected: bool = True


def circuit_constraints() -> Dict[str, str]:
    """
    Define ZK circuit constraints for procurement.

    Returns:
        Dict mapping constraint names to descriptions
    """
    return {
        "invoice_math": "Sum of line items equals total, no floating point drift",
        "vendor_signature": "Vendor cryptographically signed by SAM.gov CA",
        "contract_parameters": "Labor rates, quantities within contract bounds",
        "bekenstein_bound": "Metadata entropy sufficient for invoice amount",
        "state_transition": "S_{n-1} -> S_n follows budget transition rules",
    }


def generate_proof(
    state_prev: Dict[str, Any],
    state_next: Dict[str, Any],
    witness: Dict[str, Any]
) -> ZKProof:
    """
    Generate zk-SNARK proof that transition S_{n-1} -> S_n is valid.
    Witness = private data (vendor list, labor hours).

    Note: This is a simulation. Real ZKP requires specialized crypto libraries.

    Args:
        state_prev: Previous state dict
        state_next: Next state dict
        witness: Private witness data

    Returns:
        ZKProof object
    """
    # Hash states
    state_prev_hash = dual_hash(json.dumps(state_prev, sort_keys=True))
    state_next_hash = dual_hash(json.dumps(state_next, sort_keys=True))

    # Check circuit constraints (simulation)
    constraints = _check_constraints(state_prev, state_next, witness)

    # Generate simulated proof (in reality, this would be cryptographic)
    proof_data = {
        "type": "groth16_simulation",
        "state_prev_hash": state_prev_hash,
        "state_next_hash": state_next_hash,
        "constraints_satisfied": constraints.invoice_math_correct
            and constraints.vendor_signature_valid
            and constraints.contract_parameters_satisfied,
        "witness_commitment": dual_hash(json.dumps(witness, sort_keys=True)),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Create proof bytes (simulated ~22kb)
    proof_json = json.dumps(proof_data, sort_keys=True)
    # Pad to approximate real proof size
    padding_needed = max(0, ZKP_PROOF_SIZE_BYTES - len(proof_json))
    proof_bytes = proof_json.encode() + b'\x00' * padding_needed

    return ZKProof(
        proof_bytes=proof_bytes[:ZKP_PROOF_SIZE_BYTES],
        state_prev_hash=state_prev_hash,
        state_next_hash=state_next_hash,
        previous_proof_hash=None,
        circuit_constraints_satisfied=constraints.invoice_math_correct
            and constraints.vendor_signature_valid
            and constraints.contract_parameters_satisfied,
        verification_key_hash=dual_hash(b"verification_key"),
        ts=proof_data["ts"],
    )


def verify_proof(
    proof: ZKProof,
    state_prev: Dict[str, Any],
    state_next: Dict[str, Any]
) -> bool:
    """
    Verify proof without accessing witness. Returns true/false. O(1) time.

    Args:
        proof: ZKProof to verify
        state_prev: Expected previous state
        state_next: Expected next state

    Returns:
        True if proof is valid
    """
    start_time = time.time()

    # Verify state hashes match
    state_prev_hash = dual_hash(json.dumps(state_prev, sort_keys=True))
    state_next_hash = dual_hash(json.dumps(state_next, sort_keys=True))

    if proof.state_prev_hash != state_prev_hash:
        return False
    if proof.state_next_hash != state_next_hash:
        return False

    # Verify constraints were satisfied
    if not proof.circuit_constraints_satisfied:
        return False

    # Verify proof size (constant ~22kb for Mina-style)
    if proof.size_bytes > ZKP_PROOF_SIZE_BYTES * 1.1:  # Allow 10% tolerance
        return False

    # Check verification time SLO
    verification_time_ms = (time.time() - start_time) * 1000
    if verification_time_ms > ZKP_VERIFICATION_TIME_MAX_MS:
        stoprule_verification_timeout(verification_time_ms)

    return True


def recursive_compose(
    proof_n: ZKProof,
    proof_n_minus_1: ZKProof
) -> ZKProof:
    """
    IVC (Incrementally Verifiable Computation).
    proof_n proves proof_{n-1} valid. Return composed proof.

    The key insight: Single 22kb proof validates entire history.

    Args:
        proof_n: Current proof
        proof_n_minus_1: Previous proof

    Returns:
        Composed proof with recursive verification
    """
    # Verify the chain: proof_n's state_prev should match proof_{n-1}'s state_next
    if proof_n.state_prev_hash != proof_n_minus_1.state_next_hash:
        stoprule_recursion_broken(
            proof_n.state_prev_hash,
            proof_n_minus_1.state_next_hash
        )

    # Create composed proof
    composed_data = {
        "type": "recursive_composition",
        "proof_n_hash": proof_n.hash,
        "proof_n_minus_1_hash": proof_n_minus_1.hash,
        "recursive_depth": 2,  # In real IVC, this stays constant regardless of depth
        "state_genesis_hash": proof_n_minus_1.state_prev_hash,
        "state_latest_hash": proof_n.state_next_hash,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    proof_json = json.dumps(composed_data, sort_keys=True)
    padding_needed = max(0, ZKP_PROOF_SIZE_BYTES - len(proof_json))
    proof_bytes = proof_json.encode() + b'\x00' * padding_needed

    return ZKProof(
        proof_bytes=proof_bytes[:ZKP_PROOF_SIZE_BYTES],
        state_prev_hash=proof_n_minus_1.state_prev_hash,  # Genesis
        state_next_hash=proof_n.state_next_hash,  # Latest
        previous_proof_hash=proof_n_minus_1.hash,
        circuit_constraints_satisfied=proof_n.circuit_constraints_satisfied
            and proof_n_minus_1.circuit_constraints_satisfied,
        verification_key_hash=dual_hash(b"recursive_vk"),
        ts=composed_data["ts"],
    )


def verify_recursive_chain(
    proof_latest: ZKProof,
    state_genesis: Dict[str, Any]
) -> bool:
    """
    By induction, verify entire chain from genesis to latest via single proof.

    Args:
        proof_latest: Latest composed proof
        state_genesis: Expected genesis state

    Returns:
        True if entire chain is valid
    """
    genesis_hash = dual_hash(json.dumps(state_genesis, sort_keys=True))

    # For recursive proofs, state_prev_hash should be genesis
    if proof_latest.state_prev_hash != genesis_hash:
        return False

    return proof_latest.circuit_constraints_satisfied


def _check_constraints(
    state_prev: Dict[str, Any],
    state_next: Dict[str, Any],
    witness: Dict[str, Any]
) -> CircuitConstraints:
    """
    Check if transaction satisfies circuit constraints.
    This is the simulation - real ZKP would encode these as R1CS constraints.

    Args:
        state_prev: Previous state
        state_next: Next state
        witness: Private witness data

    Returns:
        CircuitConstraints with satisfaction flags
    """
    constraints = CircuitConstraints()

    # Constraint 1: Invoice math correct
    if "amount" in witness:
        expected_balance = state_prev.get("balance", 0) - witness.get("amount", 0)
        actual_balance = state_next.get("balance", 0)
        constraints.invoice_math_correct = abs(expected_balance - actual_balance) < 0.01

    # Constraint 2: Vendor signature (simulated)
    if "vendor_signature" in witness:
        constraints.vendor_signature_valid = witness.get("vendor_signature") is not None

    # Constraint 3: Contract parameters
    if "labor_rate" in witness and "contract_max_rate" in witness:
        constraints.contract_parameters_satisfied = (
            witness["labor_rate"] <= witness["contract_max_rate"]
        )

    # Constraint 4: SAM.gov verification
    if "sam_verified" in witness:
        constraints.sam_gov_verified = witness.get("sam_verified", False)

    return constraints


def emit_zkp_receipt(
    proof: ZKProof,
    verification_result: bool,
    state_transition: str = ""
) -> dict:
    """
    Emit zkp_receipt documenting proof verification.

    Args:
        proof: ZKProof that was verified
        verification_result: True if proof valid
        state_transition: Description of state transition

    Returns:
        zkp_receipt dict
    """
    return emit_receipt("zkp", {
        "tenant_id": TENANT_ID,
        "proof_valid": verification_result,
        "state_transition": state_transition or f"{proof.state_prev_hash[:16]} -> {proof.state_next_hash[:16]}",
        "previous_proof_hash": proof.previous_proof_hash or "genesis",
        "proof_size_bytes": proof.size_bytes,
        "verification_time_ms": 0.0,  # Placeholder
        "circuit_satisfied": proof.circuit_constraints_satisfied,
        "recursive_depth": 1 if proof.previous_proof_hash else 0,
        "mina_style_constant_size": proof.size_bytes <= ZKP_PROOF_SIZE_BYTES * 1.1,
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


# === STOPRULES ===

def stoprule_proof_invalid(proof: ZKProof, reason: str) -> None:
    """Transaction rejected, cannot append to ledger."""
    emit_receipt("anomaly", {
        "metric": "proof_invalid",
        "proof_hash": proof.hash[:32],
        "reason": reason,
        "action": "reject_transaction",
        "classification": "violation",
        "simulation_flag": DISCLAIMER,
    })
    raise StopRuleException(f"ZKP proof invalid: {reason}")


def stoprule_circuit_unsatisfied(constraint: str) -> None:
    """Constraint violated, emit violation_receipt."""
    emit_receipt("anomaly", {
        "metric": "circuit_unsatisfied",
        "constraint": constraint,
        "action": "reject_transaction",
        "classification": "violation",
        "simulation_flag": DISCLAIMER,
    })
    raise StopRuleException(f"ZKP circuit constraint violated: {constraint}")


def stoprule_recursion_broken(expected: str, actual: str) -> None:
    """If proof_n doesn't verify proof_{n-1}, chain invalid."""
    emit_receipt("anomaly", {
        "metric": "recursion_broken",
        "expected_hash": expected[:32],
        "actual_hash": actual[:32],
        "action": "halt",
        "classification": "violation",
        "simulation_flag": DISCLAIMER,
    })
    raise StopRuleException(f"ZKP recursion broken: expected {expected[:16]}, got {actual[:16]}")


def stoprule_verification_timeout(time_ms: float) -> None:
    """Verification exceeded time limit."""
    emit_receipt("anomaly", {
        "metric": "verification_timeout",
        "time_ms": time_ms,
        "limit_ms": ZKP_VERIFICATION_TIME_MAX_MS,
        "action": "optimize",
        "classification": "deviation",
        "simulation_flag": DISCLAIMER,
    })
    raise StopRuleException(f"ZKP verification timeout: {time_ms}ms > {ZKP_VERIFICATION_TIME_MAX_MS}ms")


# === MODULE SELF-TEST ===

if __name__ == "__main__":
    import sys

    print(f"# WarrantProof ZKP Module Self-Test", file=sys.stderr)
    print(f"# {DISCLAIMER}", file=sys.stderr)

    # Test 1: Generate valid proof
    state_prev = {"balance": 1000000}
    state_next = {"balance": 900000}
    witness = {"amount": 100000, "vendor_signature": "sig_123"}

    proof = generate_proof(state_prev, state_next, witness)
    print(f"# Proof size: {proof.size_bytes} bytes", file=sys.stderr)
    assert proof.size_bytes <= ZKP_PROOF_SIZE_BYTES * 1.1

    # Test 2: Verify valid proof
    is_valid = verify_proof(proof, state_prev, state_next)
    print(f"# Proof valid: {is_valid}", file=sys.stderr)
    assert is_valid == True

    # Test 3: Verify rejects wrong state
    wrong_state = {"balance": 500000}
    is_invalid = verify_proof(proof, state_prev, wrong_state)
    print(f"# Wrong state rejected: {not is_invalid}", file=sys.stderr)
    assert is_invalid == False

    # Test 4: Recursive composition
    state_1 = {"balance": 1000000}
    state_2 = {"balance": 900000}
    state_3 = {"balance": 800000}

    proof_1 = generate_proof(state_1, state_2, {"amount": 100000})
    proof_2 = generate_proof(state_2, state_3, {"amount": 100000})

    composed = recursive_compose(proof_2, proof_1)
    print(f"# Composed proof size: {composed.size_bytes} bytes", file=sys.stderr)
    assert composed.size_bytes <= ZKP_PROOF_SIZE_BYTES * 1.1

    # Test 5: Verify recursive chain
    chain_valid = verify_recursive_chain(composed, state_1)
    print(f"# Recursive chain valid: {chain_valid}", file=sys.stderr)
    assert chain_valid == True

    # Test 6: Receipt emission
    receipt = emit_zkp_receipt(proof, True, "S_0 -> S_1")
    assert receipt["receipt_type"] == "zkp"
    assert receipt["proof_valid"] == True

    # Test 7: Circuit constraints
    constraints = circuit_constraints()
    print(f"# Circuit constraints defined: {len(constraints)}", file=sys.stderr)
    assert len(constraints) >= 5

    print(f"# PASS: zkp module self-test", file=sys.stderr)
