"""
Agent package for Forest Intelligence & Early-Warning AI.

Provides the InvestigationOrchestrator, which coordinates the NLP, ML, CV,
and historical-retrieval modules into a single evidence-collection workflow.

Evidence *fusion* and final assessment generation are implemented separately
(Phase 3) and consume the OrchestrationResult produced here.
"""

from backend.agent.orchestrator import InvestigationOrchestrator, OrchestrationResult
from backend.agent.fusion import EvidenceFusion

__all__ = ["InvestigationOrchestrator", "OrchestrationResult", "EvidenceFusion"]
