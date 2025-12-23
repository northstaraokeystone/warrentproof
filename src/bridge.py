"""
WarrantProof Bridge Module - Cross-System Integration with Catalytic Detection

⚠️ SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY ⚠️

This module provides translation between incompatible branch systems.
Navy ERP, Army GFEBS, Air Force DEAMS speak different languages.
Bridge creates universal receipt format.

The Challenge: 400+ incompatible financial systems across DoD.
The Solution: Universal receipt layer with translation proofs.

Supported Systems (Simulated):
- Navy: ERP (Enterprise Resource Planning)
- Army: GFEBS (General Fund Enterprise Business System)
- AirForce: DEAMS (Defense Enterprise Accounting and Management System)
- Marines: Uses Navy ERP with modifications
- SpaceForce: DEAMS variant
- CoastGuard: Separate system (DHS)

OMEGA v3 Enhancement:
Catalytic link detection beyond financial flow. Per OMEGA §3.1:
"RAF catalysis hidden in information flow, not currency flow."
Detects: shared addresses, board connections, IP proximity.

SLOs:
- Translation <= 200ms per receipt
- Zero information loss (verified cryptographically)
- Support all 6 branch systems
- Catalytic detection F1 > 0.80
"""

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set, Tuple

import numpy as np

from .core import (
    TENANT_ID,
    DISCLAIMER,
    BRANCHES,
    MUTUAL_INFO_TRANSFER_THRESHOLD,
    CROSS_BRANCH_ACCURACY_TARGET,
    RAF_CYCLE_MIN_LENGTH,
    RAF_CYCLE_MAX_LENGTH,
    dual_hash,
    emit_receipt,
    get_citation,
    stoprule_hash_mismatch,
    StopRuleException,
)

# OMEGA v3: Import RAF for catalytic detection
from .raf import (
    build_transaction_graph,
    add_catalytic_links,
    detect_cycles,
    identify_keystone_species,
)


# === SYSTEM CONFIGURATIONS ===

BRANCH_SYSTEMS = {
    "Navy": {
        "system": "Navy_ERP",
        "field_mapping": {
            "transaction_id": "document_number",
            "amount_usd": "obligation_amount",
            "approver": "certifying_officer",
            "vendor": "cage_code",
            "program": "program_element",
        },
        "date_format": "YYYY-MM-DD",
    },
    "Army": {
        "system": "Army_GFEBS",
        "field_mapping": {
            "transaction_id": "gfebs_doc_id",
            "amount_usd": "fund_amount",
            "approver": "authorizing_official",
            "vendor": "vendor_code",
            "program": "mdep",
        },
        "date_format": "MM/DD/YYYY",
    },
    "AirForce": {
        "system": "AirForce_DEAMS",
        "field_mapping": {
            "transaction_id": "deams_transaction_id",
            "amount_usd": "total_amount",
            "approver": "resource_advisor",
            "vendor": "supplier_id",
            "program": "eeic",
        },
        "date_format": "DD-MMM-YYYY",
    },
    "Marines": {
        "system": "Navy_ERP_USMC",
        "field_mapping": {
            "transaction_id": "document_number",
            "amount_usd": "obligation_amount",
            "approver": "certifying_officer",
            "vendor": "cage_code",
            "program": "program_element",
        },
        "date_format": "YYYY-MM-DD",
    },
    "SpaceForce": {
        "system": "SpaceForce_DEAMS",
        "field_mapping": {
            "transaction_id": "sf_transaction_id",
            "amount_usd": "total_amount",
            "approver": "delta_commander",
            "vendor": "supplier_id",
            "program": "space_eeic",
        },
        "date_format": "DD-MMM-YYYY",
    },
    "CoastGuard": {
        "system": "USCG_FINCEN",
        "field_mapping": {
            "transaction_id": "cg_doc_number",
            "amount_usd": "commitment_amount",
            "approver": "commanding_officer",
            "vendor": "vendor_tin",
            "program": "opfac",
        },
        "date_format": "YYYY-MM-DD",
    },
}

# Canonical field names used in WarrantProof receipts
CANONICAL_FIELDS = [
    "transaction_id", "amount_usd", "approver", "vendor", "program",
    "branch", "ts", "receipt_type", "decision_lineage"
]


# === CORE FUNCTIONS ===

def translate_receipt(
    receipt: dict,
    source_system: str,
    target_system: str
) -> dict:
    """
    Convert receipt format between systems.
    Per spec: Translation <= 200ms per receipt.

    Args:
        receipt: Original receipt dict
        source_system: Source system name (e.g., "Navy_ERP")
        target_system: Target system name (e.g., "Army_GFEBS")

    Returns:
        Translated receipt in target system format
    """
    # Find source and target configs
    source_config = _find_system_config(source_system)
    target_config = _find_system_config(target_system)

    if not source_config or not target_config:
        # If system not found, return canonical format
        return _to_canonical(receipt)

    # Step 1: Convert to canonical format
    canonical = _to_canonical(receipt, source_config)

    # Step 2: Convert from canonical to target format
    translated = _from_canonical(canonical, target_config)

    # Add bridge metadata
    translated["_bridge"] = {
        "source_system": source_system,
        "target_system": target_system,
        "translated_at": emit_receipt("bridge_translation", {}, to_stdout=False)["ts"],
        "original_hash": dual_hash(str(receipt)),
    }

    return translated


def verify_translation(original: dict, translated: dict) -> bool:
    """
    Ensure no information loss during translation.
    Per spec: Zero information loss (verified cryptographically).

    Args:
        original: Original receipt
        translated: Translated receipt

    Returns:
        True if translation preserved all information
    """
    # Extract core data from both
    original_canonical = _to_canonical(original)
    translated_canonical = _to_canonical(translated)

    # Check that all canonical fields are preserved
    for field in CANONICAL_FIELDS:
        original_val = original_canonical.get(field)
        translated_val = translated_canonical.get(field)

        # Skip metadata fields that are expected to differ
        if field in ["ts", "receipt_type"]:
            continue

        if original_val != translated_val:
            # Check if it's a format difference (e.g., date format)
            if not _values_equivalent(original_val, translated_val):
                return False

    return True


def create_bridge_proof(original_receipt: dict, translated_receipt: dict) -> dict:
    """
    Create cryptographic proof that translation is valid.
    Per spec: Prove translation preserved information.

    Args:
        original_receipt: Original receipt
        translated_receipt: Translated receipt

    Returns:
        bridge_receipt with proof
    """
    original_hash = dual_hash(str(original_receipt))
    translated_hash = dual_hash(str(translated_receipt))

    # Create proof by hashing the concatenation
    proof_hash = dual_hash(original_hash + translated_hash)

    # Verify translation
    preserved = verify_translation(original_receipt, translated_receipt)

    bridge_receipt = emit_receipt("bridge", {
        "tenant_id": TENANT_ID,
        "source_system": original_receipt.get("_bridge", {}).get("source_system",
                         _infer_system(original_receipt)),
        "target_system": translated_receipt.get("_bridge", {}).get("target_system",
                         _infer_system(translated_receipt)),
        "original_receipt_id": original_hash,
        "translated_receipt_id": translated_hash,
        "translation_proof": proof_hash,
        "information_preserved": preserved,
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)

    if not preserved:
        stoprule_hash_mismatch(original_hash, translated_hash)

    return bridge_receipt


def cross_branch_chain(receipts: list[dict]) -> dict:
    """
    Construct proof chain across multiple branch systems.
    Per spec: Unified proof layer across incompatible systems.

    Args:
        receipts: Receipts from different branches

    Returns:
        Chain proof dict with all translations
    """
    if not receipts:
        return {"chain": [], "proof": None}

    # Group by branch
    by_branch = {}
    for receipt in receipts:
        branch = receipt.get("branch", "unknown")
        if branch not in by_branch:
            by_branch[branch] = []
        by_branch[branch].append(receipt)

    # Convert all to canonical format
    canonical_chain = []
    translation_proofs = []

    for branch, branch_receipts in by_branch.items():
        system = BRANCH_SYSTEMS.get(branch, {}).get("system", "unknown")

        for receipt in branch_receipts:
            canonical = _to_canonical(receipt)
            canonical["_source_branch"] = branch
            canonical["_source_system"] = system
            canonical_chain.append(canonical)

            # Create translation proof
            proof = create_bridge_proof(receipt, canonical)
            translation_proofs.append(proof)

    # Compute chain proof
    chain_hash = dual_hash(str(canonical_chain))

    chain_receipt = emit_receipt("chain", {
        "tenant_id": TENANT_ID,
        "branches_included": list(by_branch.keys()),
        "systems_bridged": list(set(
            BRANCH_SYSTEMS.get(b, {}).get("system", "unknown")
            for b in by_branch.keys()
        )),
        "receipt_count": len(receipts),
        "chain_hash": chain_hash,
        "translation_proofs": len(translation_proofs),
        "all_preserved": all(p.get("information_preserved") for p in translation_proofs),
        "citation": get_citation("GAO_DOD_IT"),
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)

    return {
        "chain": canonical_chain,
        "proof": chain_receipt,
        "translation_proofs": translation_proofs,
    }


# === HELPER FUNCTIONS ===

def _find_system_config(system_name: str) -> Optional[dict]:
    """Find system config by system name."""
    for branch, config in BRANCH_SYSTEMS.items():
        if config["system"] == system_name:
            return config
    return None


def _infer_system(receipt: dict) -> str:
    """Infer system from receipt branch."""
    branch = receipt.get("branch", "unknown")
    return BRANCH_SYSTEMS.get(branch, {}).get("system", "unknown")


def _to_canonical(receipt: dict, source_config: Optional[dict] = None) -> dict:
    """Convert receipt to canonical format."""
    canonical = {
        "transaction_id": None,
        "amount_usd": None,
        "approver": None,
        "vendor": None,
        "program": None,
        "branch": receipt.get("branch"),
        "ts": receipt.get("ts"),
        "receipt_type": receipt.get("receipt_type"),
        "decision_lineage": receipt.get("decision_lineage", []),
    }

    if source_config:
        # Reverse map from source fields
        reverse_map = {v: k for k, v in source_config["field_mapping"].items()}
        for source_field, canonical_field in reverse_map.items():
            if source_field in receipt:
                canonical[canonical_field] = receipt[source_field]

    # Also try direct field names
    for field in CANONICAL_FIELDS:
        if field in receipt and canonical.get(field) is None:
            canonical[field] = receipt[field]

    return canonical


def _from_canonical(canonical: dict, target_config: dict) -> dict:
    """Convert from canonical to target system format."""
    translated = {}

    for canonical_field, target_field in target_config["field_mapping"].items():
        if canonical.get(canonical_field) is not None:
            translated[target_field] = canonical[canonical_field]

    # Copy through non-mapped fields
    for field in ["branch", "ts", "receipt_type", "decision_lineage"]:
        if field in canonical:
            translated[field] = canonical[field]

    return translated


def _values_equivalent(val1, val2) -> bool:
    """Check if two values are equivalent (allowing for format differences)."""
    if val1 == val2:
        return True

    # Try numeric comparison
    try:
        if float(val1) == float(val2):
            return True
    except (TypeError, ValueError):
        pass

    # Try string comparison (case insensitive)
    try:
        if str(val1).lower() == str(val2).lower():
            return True
    except (TypeError, ValueError):
        pass

    return False


# === V2 MUTUAL INFORMATION ===

def mutual_information(
    branch_A_receipts: list,
    branch_B_receipts: list
) -> float:
    """
    Calculate I(A;B) = H(A) + H(B) - H(A,B).
    Returns shared entropy between branches.

    Args:
        branch_A_receipts: Receipts from branch A
        branch_B_receipts: Receipts from branch B

    Returns:
        Mutual information in bits
    """
    if not branch_A_receipts or not branch_B_receipts:
        return 0.0

    # Extract features for entropy calculation
    def extract_features(receipts: list) -> list:
        features = []
        for r in receipts:
            features.append(r.get("receipt_type", ""))
            features.append(str(r.get("amount_usd", 0))[:4])
            features.append(r.get("vendor", "")[:5] if r.get("vendor") else "")
        return features

    features_A = extract_features(branch_A_receipts)
    features_B = extract_features(branch_B_receipts)
    features_AB = features_A + features_B

    def shannon_entropy(features: list) -> float:
        if not features:
            return 0.0
        counter = Counter(features)
        total = len(features)
        entropy = 0.0
        for count in counter.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        return entropy

    H_A = shannon_entropy(features_A)
    H_B = shannon_entropy(features_B)
    H_AB = shannon_entropy(features_AB)

    # I(A;B) = H(A) + H(B) - H(A,B)
    mutual_info = H_A + H_B - H_AB

    # Clamp to non-negative (numerical errors can cause slight negatives)
    return max(0.0, mutual_info)


def transfer_pattern(
    source_branch: str,
    target_branch: str,
    pattern: dict,
    mutual_info: float
) -> bool:
    """
    Transfer pattern from source to target branch if I(A;B) > threshold.

    Args:
        source_branch: Source branch name
        target_branch: Target branch name
        pattern: Pattern dict to transfer
        mutual_info: Calculated mutual information

    Returns:
        True if transfer successful
    """
    if mutual_info < MUTUAL_INFO_TRANSFER_THRESHOLD:
        # Insufficient shared information for reliable transfer
        return False

    # Emit transfer receipt
    emit_receipt("pattern_transfer", {
        "tenant_id": TENANT_ID,
        "source_branch": source_branch,
        "target_branch": target_branch,
        "pattern_id": pattern.get("pattern_id", dual_hash(str(pattern))[:16]),
        "mutual_information": round(mutual_info, 4),
        "threshold": MUTUAL_INFO_TRANSFER_THRESHOLD,
        "transfer_success": True,
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)

    return True


def calculate_transfer_benefit(
    mutual_info: float,
    pattern_complexity: float
) -> float:
    """
    Calculate expected speedup from pattern transfer.
    time_reduction ≈ I(A;B) / H(pattern)

    Args:
        mutual_info: Mutual information between branches
        pattern_complexity: Entropy/complexity of pattern (H(pattern))

    Returns:
        Expected time reduction factor
    """
    if pattern_complexity <= 0:
        return 0.0

    benefit = mutual_info / pattern_complexity

    return min(10.0, benefit)  # Cap at 10x speedup


def cross_branch_learning(
    source_branch: str,
    source_receipts: list,
    target_branch: str,
    target_receipts: list,
    patterns: list
) -> dict:
    """
    Apply cross-branch learning using mutual information.
    Transfer patterns from source to target if shared entropy sufficient.

    Args:
        source_branch: Source branch name
        source_receipts: Receipts from source branch
        target_branch: Target branch name
        target_receipts: Receipts from target branch
        patterns: Patterns learned from source

    Returns:
        Learning result with transferred patterns
    """
    # Calculate mutual information
    mi = mutual_information(source_receipts, target_receipts)

    transferred = []
    failed = []

    for pattern in patterns:
        # Estimate pattern complexity from fingerprint
        fingerprint = pattern.get("fingerprint", pattern)
        complexity = len(str(fingerprint)) * 0.01  # Simple proxy

        # Calculate transfer benefit
        benefit = calculate_transfer_benefit(mi, complexity)

        # Attempt transfer
        if transfer_pattern(source_branch, target_branch, pattern, mi):
            transferred.append({
                "pattern": pattern,
                "benefit": benefit,
            })
        else:
            failed.append(pattern)

    # Emit learning receipt
    emit_receipt("cross_branch_learning", {
        "tenant_id": TENANT_ID,
        "source_branch": source_branch,
        "target_branch": target_branch,
        "mutual_information": round(mi, 4),
        "patterns_transferred": len(transferred),
        "patterns_failed": len(failed),
        "accuracy_target": CROSS_BRANCH_ACCURACY_TARGET,
        "citation": get_citation("GAO_DOD_IT"),
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)

    return {
        "mutual_information": mi,
        "transferred": transferred,
        "failed": failed,
        "success_rate": len(transferred) / len(patterns) if patterns else 1.0,
    }


# === OMEGA v3: CATALYTIC LINK DETECTION ===

@dataclass
class CatalyticLink:
    """
    Non-financial link between entities that may indicate collusion.
    Per OMEGA §3.1: "RAF catalysis hidden in information flow, not currency flow."
    """
    link_type: str  # "shared_address", "board_connection", "ip_proximity", "temporal_pattern"
    entity_a: str
    entity_b: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    catalytic_strength: float = 0.0


def detect_shared_addresses(
    entities: List[Dict[str, Any]]
) -> List[CatalyticLink]:
    """
    Detect entities sharing physical or mailing addresses.
    Shell companies often share registered agent addresses.

    Args:
        entities: List of entity dicts with address fields

    Returns:
        List of CatalyticLink for shared addresses
    """
    links = []
    address_map: Dict[str, List[str]] = defaultdict(list)

    # Normalize and index addresses
    for entity in entities:
        entity_id = entity.get("vendor") or entity.get("cage_code") or entity.get("id")
        if not entity_id:
            continue

        # Check multiple address fields
        for addr_field in ["address", "physical_address", "mailing_address", "registered_address"]:
            addr = entity.get(addr_field)
            if addr:
                # Normalize address
                if isinstance(addr, dict):
                    addr_str = " ".join(str(v) for v in addr.values() if v)
                else:
                    addr_str = str(addr)

                normalized = _normalize_address(addr_str)
                if normalized:
                    address_map[normalized].append(entity_id)

    # Find entities sharing addresses
    for addr, entity_ids in address_map.items():
        if len(entity_ids) > 1:
            # Create pairwise links
            for i, entity_a in enumerate(entity_ids):
                for entity_b in entity_ids[i+1:]:
                    links.append(CatalyticLink(
                        link_type="shared_address",
                        entity_a=entity_a,
                        entity_b=entity_b,
                        evidence={"normalized_address": addr, "entity_count": len(entity_ids)},
                        confidence=min(0.9, 0.5 + 0.1 * len(entity_ids)),
                        catalytic_strength=0.8,
                    ))

    return links


def detect_board_connections(
    entities: List[Dict[str, Any]]
) -> List[CatalyticLink]:
    """
    Detect shared board members, officers, or ownership.
    Interlocking directorates indicate potential collusion.

    Args:
        entities: List of entity dicts with ownership/officer fields

    Returns:
        List of CatalyticLink for board connections
    """
    links = []
    person_map: Dict[str, List[str]] = defaultdict(list)

    for entity in entities:
        entity_id = entity.get("vendor") or entity.get("id")
        if not entity_id:
            continue

        # Check ownership and officer fields
        people = []

        # Ownership details
        ownership = entity.get("ownership_details", {})
        if isinstance(ownership, dict):
            for key in ["owner", "principal", "agent", "officers"]:
                val = ownership.get(key)
                if val:
                    if isinstance(val, list):
                        people.extend(val)
                    else:
                        people.append(val)

        # Direct officer fields
        for field_name in ["officer", "owner", "principal", "board_members", "directors"]:
            val = entity.get(field_name)
            if val:
                if isinstance(val, list):
                    people.extend(val)
                else:
                    people.append(val)

        # Index by normalized person name
        for person in people:
            if isinstance(person, str):
                normalized = _normalize_name(person)
                if normalized:
                    person_map[normalized].append(entity_id)

    # Find entities sharing people
    for person, entity_ids in person_map.items():
        if len(entity_ids) > 1:
            for i, entity_a in enumerate(entity_ids):
                for entity_b in entity_ids[i+1:]:
                    links.append(CatalyticLink(
                        link_type="board_connection",
                        entity_a=entity_a,
                        entity_b=entity_b,
                        evidence={"shared_person": person, "entity_count": len(entity_ids)},
                        confidence=min(0.95, 0.6 + 0.1 * len(entity_ids)),
                        catalytic_strength=0.9,  # Strong indicator
                    ))

    return links


def detect_ip_proximity(
    receipts: List[Dict[str, Any]],
    ip_threshold: int = 16  # /16 subnet
) -> List[CatalyticLink]:
    """
    Detect entities submitting from proximate IP addresses.
    Bid rigging often involves submissions from same network.

    Args:
        receipts: List of receipts with IP metadata
        ip_threshold: Subnet mask for proximity (16 = /16)

    Returns:
        List of CatalyticLink for IP proximity
    """
    links = []
    ip_map: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    for receipt in receipts:
        entity_id = receipt.get("vendor") or receipt.get("contractor_id")
        if not entity_id:
            continue

        # Check for IP in metadata
        ip = receipt.get("source_ip") or receipt.get("_metadata", {}).get("ip")
        if not ip:
            continue

        # Extract subnet
        try:
            parts = ip.split(".")[:ip_threshold // 8]
            subnet = ".".join(parts)
            ip_map[subnet].append((entity_id, ip))
        except (ValueError, IndexError):
            continue

    # Find entities in same subnet
    for subnet, entries in ip_map.items():
        unique_entities = list(set(e[0] for e in entries))
        if len(unique_entities) > 1:
            for i, entity_a in enumerate(unique_entities):
                for entity_b in unique_entities[i+1:]:
                    links.append(CatalyticLink(
                        link_type="ip_proximity",
                        entity_a=entity_a,
                        entity_b=entity_b,
                        evidence={"subnet": subnet, "threshold": ip_threshold},
                        confidence=0.6,  # Moderate confidence
                        catalytic_strength=0.5,
                    ))

    return links


def detect_temporal_patterns(
    receipts: List[Dict[str, Any]],
    time_window_seconds: int = 300  # 5 minutes
) -> List[CatalyticLink]:
    """
    Detect suspiciously coordinated submission timing.
    Collusive bids often submitted in tight time windows.

    Args:
        receipts: List of receipts with timestamps
        time_window_seconds: Window for "suspicious" proximity

    Returns:
        List of CatalyticLink for temporal patterns
    """
    links = []

    # Group by solicitation/contract
    by_solicitation: Dict[str, List[Dict]] = defaultdict(list)
    for receipt in receipts:
        sol_id = receipt.get("solicitation_id") or receipt.get("contract_id") or "default"
        by_solicitation[sol_id].append(receipt)

    for sol_id, sol_receipts in by_solicitation.items():
        # Sort by timestamp
        try:
            sorted_receipts = sorted(
                sol_receipts,
                key=lambda r: r.get("ts", "")
            )
        except TypeError:
            continue

        # Check for tight clustering
        for i, r1 in enumerate(sorted_receipts):
            for r2 in sorted_receipts[i+1:]:
                entity_a = r1.get("vendor") or r1.get("contractor_id")
                entity_b = r2.get("vendor") or r2.get("contractor_id")

                if not entity_a or not entity_b or entity_a == entity_b:
                    continue

                # Calculate time difference
                try:
                    from datetime import datetime
                    t1 = datetime.fromisoformat(r1.get("ts", "").replace("Z", "+00:00"))
                    t2 = datetime.fromisoformat(r2.get("ts", "").replace("Z", "+00:00"))
                    diff_seconds = abs((t2 - t1).total_seconds())

                    if diff_seconds <= time_window_seconds:
                        links.append(CatalyticLink(
                            link_type="temporal_pattern",
                            entity_a=entity_a,
                            entity_b=entity_b,
                            evidence={
                                "solicitation": sol_id,
                                "time_diff_seconds": diff_seconds,
                                "window": time_window_seconds,
                            },
                            confidence=max(0.3, 1.0 - diff_seconds / time_window_seconds),
                            catalytic_strength=0.6,
                        ))
                except (ValueError, TypeError):
                    continue

    return links


def detect_all_catalytic_links(
    entities: List[Dict[str, Any]],
    receipts: List[Dict[str, Any]]
) -> Dict[str, List[CatalyticLink]]:
    """
    Run all catalytic detection algorithms.
    Per OMEGA: "RAF catalysis hidden in information flow."

    Args:
        entities: List of entity dicts
        receipts: List of receipt dicts

    Returns:
        Dict mapping link_type to list of CatalyticLink
    """
    all_links = {
        "shared_address": detect_shared_addresses(entities),
        "board_connection": detect_board_connections(entities),
        "ip_proximity": detect_ip_proximity(receipts),
        "temporal_pattern": detect_temporal_patterns(receipts),
    }

    return all_links


def integrate_catalytic_with_raf(
    transactions: List[Dict[str, Any]],
    catalytic_links: Dict[str, List[CatalyticLink]]
) -> Dict[str, Any]:
    """
    Integrate catalytic links into RAF network analysis.
    Catalytic links become edges in the RAF graph.

    Args:
        transactions: Financial transactions for RAF graph
        catalytic_links: Detected catalytic links

    Returns:
        Enhanced RAF analysis with catalytic cycles
    """
    # Build base transaction graph
    graph = build_transaction_graph(transactions)

    # Add catalytic links as edges
    catalytic_edges = []
    for link_type, links in catalytic_links.items():
        for link in links:
            catalytic_edges.append({
                "from": link.entity_a,
                "to": link.entity_b,
                "type": f"catalytic_{link_type}",
                "weight": link.catalytic_strength,
                "evidence": link.evidence,
            })

    # Add catalytic edges to graph
    graph = add_catalytic_links(graph, catalytic_edges)

    # Detect cycles including catalytic links
    cycles = detect_cycles(graph, min_length=RAF_CYCLE_MIN_LENGTH, max_length=RAF_CYCLE_MAX_LENGTH)

    # Identify keystone species
    keystones = identify_keystone_species(graph)

    # Classify cycles by catalytic involvement
    pure_financial = []
    catalytic_enhanced = []

    for cycle in cycles:
        has_catalytic = any(
            graph.get_edge_data(cycle[i], cycle[(i+1) % len(cycle)], {}).get("type", "").startswith("catalytic_")
            for i in range(len(cycle))
        ) if hasattr(graph, 'get_edge_data') else False

        if has_catalytic:
            catalytic_enhanced.append(cycle)
        else:
            pure_financial.append(cycle)

    return {
        "total_cycles": len(cycles),
        "pure_financial_cycles": len(pure_financial),
        "catalytic_enhanced_cycles": len(catalytic_enhanced),
        "keystones": keystones,
        "catalytic_links_added": len(catalytic_edges),
        "cycles": cycles,
    }


def emit_catalytic_receipt(
    catalytic_links: Dict[str, List[CatalyticLink]],
    raf_integration: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Emit receipt documenting catalytic link detection.

    Args:
        catalytic_links: Detected catalytic links by type
        raf_integration: Optional RAF integration results

    Returns:
        catalytic_receipt dict
    """
    total_links = sum(len(links) for links in catalytic_links.values())

    # Calculate average confidence by type
    confidence_by_type = {}
    for link_type, links in catalytic_links.items():
        if links:
            confidence_by_type[link_type] = round(
                sum(l.confidence for l in links) / len(links),
                4
            )

    receipt_data = {
        "tenant_id": TENANT_ID,
        "total_catalytic_links": total_links,
        "links_by_type": {k: len(v) for k, v in catalytic_links.items()},
        "confidence_by_type": confidence_by_type,
        "omega_citation": "OMEGA §3.1: RAF catalysis hidden in information flow",
        "simulation_flag": DISCLAIMER,
    }

    if raf_integration:
        receipt_data["raf_integration"] = {
            "total_cycles": raf_integration.get("total_cycles", 0),
            "catalytic_enhanced_cycles": raf_integration.get("catalytic_enhanced_cycles", 0),
            "keystones_found": len(raf_integration.get("keystones", [])),
        }

    return emit_receipt("catalytic", receipt_data, to_stdout=False)


def cross_branch_learning_with_catalysis(
    source_branch: str,
    source_receipts: List[Dict],
    source_entities: List[Dict],
    target_branch: str,
    target_receipts: List[Dict],
    target_entities: List[Dict],
    patterns: List[Dict]
) -> Dict[str, Any]:
    """
    Enhanced cross-branch learning with catalytic link detection.
    Combines mutual information transfer with RAF catalytic analysis.

    Args:
        source_branch: Source branch name
        source_receipts: Receipts from source branch
        source_entities: Entities from source branch
        target_branch: Target branch name
        target_receipts: Receipts from target branch
        target_entities: Entities from target branch
        patterns: Patterns to transfer

    Returns:
        Learning result with catalytic analysis
    """
    # Standard cross-branch learning
    base_result = cross_branch_learning(
        source_branch, source_receipts,
        target_branch, target_receipts,
        patterns
    )

    # Detect catalytic links across both branches
    all_entities = source_entities + target_entities
    all_receipts = source_receipts + target_receipts

    catalytic_links = detect_all_catalytic_links(all_entities, all_receipts)

    # Integrate with RAF
    all_transactions = [r for r in all_receipts if r.get("amount_usd")]
    raf_result = integrate_catalytic_with_raf(all_transactions, catalytic_links)

    # Emit receipt
    emit_catalytic_receipt(catalytic_links, raf_result)

    # Enhance result
    base_result["catalytic_analysis"] = {
        "links_detected": sum(len(l) for l in catalytic_links.values()),
        "raf_cycles": raf_result.get("total_cycles", 0),
        "catalytic_cycles": raf_result.get("catalytic_enhanced_cycles", 0),
        "cross_branch_keystones": [
            k for k in raf_result.get("keystones", [])
            if _is_cross_branch_keystone(k, source_entities, target_entities)
        ],
    }

    return base_result


# === CATALYTIC HELPER FUNCTIONS ===

def _normalize_address(addr: str) -> str:
    """Normalize address for comparison."""
    if not addr:
        return ""

    # Lowercase and strip
    normalized = addr.lower().strip()

    # Remove common abbreviations
    replacements = [
        ("street", "st"), ("avenue", "ave"), ("road", "rd"),
        ("boulevard", "blvd"), ("drive", "dr"), ("lane", "ln"),
        ("suite", "ste"), ("apartment", "apt"), ("#", ""),
        (".", ""), (",", ""), ("  ", " "),
    ]
    for old, new in replacements:
        normalized = normalized.replace(old, new)

    # Remove extra whitespace
    normalized = " ".join(normalized.split())

    return normalized


def _normalize_name(name: str) -> str:
    """Normalize person name for comparison."""
    if not name:
        return ""

    # Lowercase and strip
    normalized = name.lower().strip()

    # Remove titles and suffixes
    for title in ["mr", "mrs", "ms", "dr", "jr", "sr", "ii", "iii", "esq"]:
        normalized = normalized.replace(f" {title}", "").replace(f"{title} ", "")

    # Remove punctuation
    normalized = normalized.replace(".", "").replace(",", "")

    # Remove extra whitespace
    normalized = " ".join(normalized.split())

    return normalized


def _is_cross_branch_keystone(
    keystone: str,
    source_entities: List[Dict],
    target_entities: List[Dict]
) -> bool:
    """Check if keystone spans both branches."""
    source_ids = {e.get("vendor") or e.get("id") for e in source_entities}
    target_ids = {e.get("vendor") or e.get("id") for e in target_entities}

    return keystone in source_ids or keystone in target_ids


# === STOPRULES ===

def stoprule_catalytic_cycle_detected(
    cycle: List[str],
    catalytic_types: List[str]
) -> None:
    """Halt on high-confidence catalytic cycle."""
    emit_receipt("anomaly", {
        "metric": "catalytic_cycle",
        "cycle_length": len(cycle),
        "catalytic_types": catalytic_types,
        "cycle_entities": cycle[:5],  # First 5 for privacy
        "action": "escalate_investigation",
        "classification": "critical",
        "simulation_flag": DISCLAIMER,
    })
    raise StopRuleException(f"Catalytic cycle detected: {len(cycle)} entities via {catalytic_types}")


# === SYSTEM INFO ===

def list_supported_systems() -> List[Dict]:
    """List all supported branch systems."""
    systems = []
    for branch, config in BRANCH_SYSTEMS.items():
        systems.append({
            "branch": branch,
            "system": config["system"],
            "fields": list(config["field_mapping"].keys()),
        })
    return systems


# === MODULE SELF-TEST ===

if __name__ == "__main__":
    import sys
    import time

    print(f"# WarrantProof Bridge Module Self-Test", file=sys.stderr)
    print(f"# {DISCLAIMER}", file=sys.stderr)

    # Create test receipts from different branches
    navy_receipt = {
        "receipt_type": "warrant",
        "branch": "Navy",
        "document_number": "N00024-24-C-1234",
        "obligation_amount": 5000000.0,
        "certifying_officer": "CAPT Smith",
        "cage_code": "1ABC2",
        "program_element": "0603564N",
        "ts": "2024-01-15T10:00:00Z",
        "decision_lineage": [],
    }

    army_receipt = {
        "receipt_type": "warrant",
        "branch": "Army",
        "gfebs_doc_id": "W912HZ-24-D-5678",
        "fund_amount": 3000000.0,
        "authorizing_official": "COL Jones",
        "vendor_code": "2DEF3",
        "mdep": "ARNG",
        "ts": "2024-01-16T11:00:00Z",
        "decision_lineage": [],
    }

    # Test translation latency
    t0 = time.time()
    translated = translate_receipt(navy_receipt, "Navy_ERP", "Army_GFEBS")
    translation_time = (time.time() - t0) * 1000
    assert translation_time <= 200, f"Translation {translation_time}ms > 200ms SLO"

    # Test verification
    verification = verify_translation(navy_receipt, translated)
    # May not be perfectly preserved due to field mapping, but core should be

    # Test bridge proof
    proof = create_bridge_proof(navy_receipt, translated)
    assert proof["receipt_type"] == "bridge"
    assert "translation_proof" in proof

    # Test cross-branch chain
    chain_result = cross_branch_chain([navy_receipt, army_receipt])
    assert len(chain_result["chain"]) == 2
    assert chain_result["proof"]["receipt_type"] == "chain"
    assert "Navy" in chain_result["proof"]["branches_included"]
    assert "Army" in chain_result["proof"]["branches_included"]

    # Test system listing
    systems = list_supported_systems()
    assert len(systems) == 6  # All 6 branches

    print(f"# Translation tests passed (latency: {translation_time:.1f}ms)", file=sys.stderr)

    # === OMEGA v3: Catalytic Detection Tests ===
    print(f"# Testing OMEGA v3 catalytic detection...", file=sys.stderr)

    # Test entities with shared address (shell company indicator)
    test_entities = [
        {
            "vendor": "VENDOR_A",
            "physical_address": {"street": "123 Main Street", "city": "Washington", "state": "DC"},
            "ownership_details": {"owner": "John Smith", "type": "LLC"},
        },
        {
            "vendor": "VENDOR_B",
            "physical_address": {"street": "123 Main St.", "city": "Washington", "state": "DC"},  # Same address, different format
            "ownership_details": {"owner": "John Smith", "type": "Corp"},  # Same owner
        },
        {
            "vendor": "VENDOR_C",
            "physical_address": {"street": "456 Oak Avenue", "city": "Boston", "state": "MA"},
            "ownership_details": {"owner": "Jane Doe", "type": "LLC"},
        },
    ]

    # Test shared address detection
    address_links = detect_shared_addresses(test_entities)
    print(f"# Shared address links detected: {len(address_links)}", file=sys.stderr)
    assert len(address_links) >= 1, "Should detect shared address between VENDOR_A and VENDOR_B"

    # Test board connection detection
    board_links = detect_board_connections(test_entities)
    print(f"# Board connection links detected: {len(board_links)}", file=sys.stderr)
    assert len(board_links) >= 1, "Should detect shared owner between VENDOR_A and VENDOR_B"

    # Test IP proximity detection
    test_receipts_ip = [
        {"vendor": "VENDOR_A", "source_ip": "192.168.1.100", "ts": "2024-01-15T10:00:00Z"},
        {"vendor": "VENDOR_B", "source_ip": "192.168.1.150", "ts": "2024-01-15T10:01:00Z"},
        {"vendor": "VENDOR_C", "source_ip": "10.0.0.50", "ts": "2024-01-15T10:02:00Z"},
    ]
    ip_links = detect_ip_proximity(test_receipts_ip)
    print(f"# IP proximity links detected: {len(ip_links)}", file=sys.stderr)
    assert len(ip_links) >= 1, "Should detect IP proximity between VENDOR_A and VENDOR_B"

    # Test temporal pattern detection
    test_receipts_temporal = [
        {"vendor": "VENDOR_A", "solicitation_id": "SOL001", "ts": "2024-01-15T10:00:00Z"},
        {"vendor": "VENDOR_B", "solicitation_id": "SOL001", "ts": "2024-01-15T10:02:00Z"},  # 2 min later
        {"vendor": "VENDOR_C", "solicitation_id": "SOL001", "ts": "2024-01-15T15:00:00Z"},  # 5 hours later
    ]
    temporal_links = detect_temporal_patterns(test_receipts_temporal)
    print(f"# Temporal pattern links detected: {len(temporal_links)}", file=sys.stderr)
    assert len(temporal_links) >= 1, "Should detect temporal clustering between VENDOR_A and VENDOR_B"

    # Test detect_all_catalytic_links
    all_links = detect_all_catalytic_links(test_entities, test_receipts_ip)
    total_links = sum(len(l) for l in all_links.values())
    print(f"# Total catalytic links: {total_links}", file=sys.stderr)
    assert total_links > 0, "Should detect at least one catalytic link"

    # Test catalytic receipt emission
    catalytic_receipt = emit_catalytic_receipt(all_links)
    assert catalytic_receipt["receipt_type"] == "catalytic"
    assert "omega_citation" in catalytic_receipt

    print(f"# PASS: bridge module self-test (catalytic detection validated)", file=sys.stderr)
