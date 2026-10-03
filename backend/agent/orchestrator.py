"""
Investigation Orchestrator — Agent Layer

Coordinates the NLP, ML, CV, and historical-retrieval modules for a single
investigation request.  This module is ONLY an orchestrator:

  - It calls existing module interfaces without duplicating their logic.
  - It decides which tools to invoke based on available inputs.
  - It collects and preserves raw module outputs unchanged.
  - It records missing evidence and tool failures explicitly.
  - It does NOT generate a final assessment (Phase 3 / fusion step).
  - It does NOT fabricate results for unavailable or failed tools.
  - It does NOT trigger real-world actions.

Required fields for ML to run:
    temperature, humidity, rainfall, wind_speed, vegetation_dryness
All five must be present in the environment dict; the tool is skipped otherwise.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("forest_intelligence.agent")

# Exact set of keys the ML model requires.
_ML_REQUIRED_FIELDS = frozenset(
    {"temperature", "humidity", "rainfall", "wind_speed", "vegetation_dryness"}
)


@dataclass
class OrchestrationResult:
    """
    Structured container for evidence collected by the orchestrator.

    Fields
    ------
    incident : dict
        Raw output from the NLP pipeline.  Always present (NLP always runs).
    risk : dict or None
        Raw (score, level) pair as a dict from the ML module, or None when ML
        was not run or failed.
    vision : dict or None
        Raw output from the CV analyzer, or None when CV was not run or failed.
    historical_matches : list of dict
        Reports returned by the retrieval module.  Empty list when none found.
    missing_evidence : list of str
        Human-readable reasons why a tool was not run (e.g. no image provided,
        incomplete environment data).
    tool_errors : dict of str → str
        Maps tool name to error description for tools that failed at runtime.
    """

    incident: Dict[str, Any]
    risk: Optional[Dict[str, Any]] = None
    vision: Optional[Dict[str, Any]] = None
    historical_matches: List[Dict[str, Any]] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)
    tool_errors: Dict[str, str] = field(default_factory=dict)


class InvestigationOrchestrator:
    """
    Orchestrates NLP, ML, CV, and historical-retrieval tools for a single
    investigation request.

    Usage
    -----
    orchestrator = InvestigationOrchestrator()
    result: OrchestrationResult = orchestrator.investigate(
        incident_text="Smoke observed near Zone 17 …",
        image_path="data/sample/sample_test_fire.png",   # or None
        environment={"temperature": 37, …},              # or None
    )

    The returned OrchestrationResult is consumed by the evidence-fusion layer
    (Phase 3) which produces the final InvestigateResponse.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def investigate(
        self,
        incident_text: str,
        image_path: Optional[str],
        environment: Optional[Dict[str, Any]],
    ) -> OrchestrationResult:
        """
        Run all applicable tools for the given inputs and return collected
        evidence.

        Parameters
        ----------
        incident_text : str
            Raw incident report text.  Must be non-empty; NLP always runs.
        image_path : str or None
            File-system path to an image.  CV runs only when provided.
        environment : dict or None
            Environmental data dict.  ML runs only when all five required keys
            are present.

        Returns
        -------
        OrchestrationResult
        """
        logger.info("Investigation started (image=%s, env=%s)",
                    "provided" if image_path else "absent",
                    "provided" if environment else "absent")

        result = OrchestrationResult(incident={})

        # ---- 1. NLP (always) -----------------------------------------
        result.incident = self._run_nlp(incident_text)

        # ---- 2. Historical retrieval (always) ------------------------
        result.historical_matches = self._run_retrieval(incident_text)

        # ---- 3. ML (conditional on complete environment) -------------
        result.risk, ml_missing, ml_error = self._run_ml(environment)
        if ml_missing:
            result.missing_evidence.append(ml_missing)
        if ml_error:
            result.tool_errors["ml"] = ml_error

        # ---- 4. CV (conditional on image path) -----------------------
        result.vision, cv_missing, cv_error = self._run_cv(image_path)
        if cv_missing:
            result.missing_evidence.append(cv_missing)
        if cv_error:
            result.tool_errors["cv"] = cv_error

        logger.info(
            "Investigation complete — risk=%s, vision=%s, "
            "historical=%d match(es), missing=%s, errors=%s",
            "present" if result.risk else "absent",
            "present" if result.vision else "absent",
            len(result.historical_matches),
            result.missing_evidence or "none",
            list(result.tool_errors.keys()) or "none",
        )
        return result

    # ------------------------------------------------------------------
    # Private tool runners
    # ------------------------------------------------------------------

    def _run_nlp(self, incident_text: str) -> Dict[str, Any]:
        """
        Call the NLP pipeline.  NLP is a mandatory tool; failures propagate.
        """
        from backend.nlp.pipeline import analyze_incident_text  # deferred import

        logger.debug("Running NLP pipeline on incident text")
        result = analyze_incident_text(incident_text)
        logger.debug("NLP complete — type=%s, severity=%s",
                     result.get("incident_type"), result.get("severity"))
        return result

    def _run_retrieval(self, incident_text: str) -> List[Dict[str, Any]]:
        """
        Call the historical retrieval module.  Always runs; returns [] on
        empty results or unexpected failure.
        """
        from backend.nlp.retrieval import retriever  # singleton instance

        logger.debug("Running historical retrieval")
        try:
            matches = retriever.retrieve(incident_text)
            logger.debug("Retrieval returned %d match(es)", len(matches))
            return matches if matches else []
        except Exception as exc:  # pragma: no cover — unexpected retrieval error
            logger.error("Historical retrieval failed unexpectedly: %s", exc)
            return []

    def _run_ml(
        self, environment: Optional[Dict[str, Any]]
    ) -> tuple:
        """
        Call the ML risk predictor when all required environmental fields are
        present.

        Returns
        -------
        (risk_dict | None, missing_reason | None, error_description | None)
        """
        # ---- gate: environment must be provided ----
        if not environment:
            logger.info("ML skipped — no environment data provided")
            return None, "No environmental data provided; ML risk prediction skipped.", None

        # ---- gate: all required keys must be present ----
        missing_keys = _ML_REQUIRED_FIELDS - set(environment.keys())
        if missing_keys:
            sorted_missing = sorted(missing_keys)
            reason = (
                f"Incomplete environmental data — missing field(s): "
                f"{', '.join(sorted_missing)}; ML risk prediction skipped."
            )
            logger.info("ML skipped — missing env keys: %s", sorted_missing)
            return None, reason, None

        # ---- run the model ----
        logger.debug("Running ML risk prediction")
        try:
            from backend.ml.predict import predict_risk  # deferred import

            score, level = predict_risk(
                temperature=float(environment["temperature"]),
                humidity=float(environment["humidity"]),
                rainfall=float(environment["rainfall"]),
                wind_speed=float(environment["wind_speed"]),
                vegetation_dryness=float(environment["vegetation_dryness"]),
            )
            logger.debug("ML prediction — score=%.2f, level=%s", score, level)
            return {"risk_score": score, "risk_level": level}, None, None

        except (ValueError, TypeError, KeyError) as exc:
            error_msg = f"ML prediction failed due to invalid input: {exc}"
            logger.warning(error_msg)
            return None, None, error_msg

        except Exception as exc:
            error_msg = f"ML prediction failed unexpectedly: {exc}"
            logger.error(error_msg)
            return None, None, error_msg

    def _run_cv(
        self, image_path: Optional[str]
    ) -> tuple:
        """
        Call the CV analyzer when an image path is provided.

        Returns
        -------
        (vision_dict | None, missing_reason | None, error_description | None)
        """
        if not image_path:
            logger.info("CV skipped — no image path provided")
            return None, "No image provided; visual analysis skipped.", None

        logger.debug("Running CV analysis on: %s", image_path)
        try:
            from backend.cv.analyzer import analyze_image  # deferred import

            vision_result = analyze_image(image_path)
            logger.debug(
                "CV analysis complete — fire=%s, smoke=%s, confidence=%.3f",
                vision_result.get("fire_detected"),
                vision_result.get("smoke_detected"),
                vision_result.get("confidence", 0.0),
            )
            return vision_result, None, None

        except FileNotFoundError as exc:
            error_msg = f"Image file not found: {exc}"
            logger.warning("CV failed — %s", error_msg)
            return None, None, error_msg

        except ValueError as exc:
            error_msg = f"Image validation error: {exc}"
            logger.warning("CV failed — %s", error_msg)
            return None, None, error_msg

        except Exception as exc:
            error_msg = f"CV analysis failed unexpectedly: {exc}"
            logger.error("CV failed — %s", error_msg)
            return None, None, error_msg
