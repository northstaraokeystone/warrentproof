"""
Gov-OS Scenarios - Test Scenarios for Fraud Detection

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY

v5.1 Scenarios:
- contagion: Cross-domain super-graph detection test
"""

from .contagion import (
    run_contagion_test,
    build_super_graph,
    inject_shell_entity,
    SHELL_ENTITY_ID,
)

__all__ = [
    "run_contagion_test",
    "build_super_graph",
    "inject_shell_entity",
    "SHELL_ENTITY_ID",
]
