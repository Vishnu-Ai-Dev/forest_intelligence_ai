"""
Evidence Fusion Layer

Consumes the structured evidence collected by the InvestigationOrchestrator
and generates a deterministic, grounded, text-based assessment.
This module does NOT use LLMs, does not fabricate data, and ensures all
claims are strictly bounded by the provided evidence.
"""

from backend.agent.orchestrator import OrchestrationResult

class EvidenceFusion:
    """
    Synthesizes multi-modal evidence into a single human-readable assessment.
    """

    def assess(self, result: OrchestrationResult) -> str:
        """
        Generates the assessment string based ONLY on the populated fields
        in the OrchestrationResult.
        """
        assessment_parts = []

        # --- 1. NLP / Incident Evidence ---
        inc = result.incident
        inc_type = inc.get("incident_type")
        severity = inc.get("severity")
        location = inc.get("location")
        time = inc.get("time")
        conditions = inc.get("conditions", [])

        # Build base incident sentence
        if inc_type and inc_type != "unknown":
            type_str = inc_type.replace("_", " ")
            if severity and severity != "Unknown":
                sentence1 = f"The incident is classified as a {severity.lower()} {type_str}."
            else:
                sentence1 = f"The incident is classified as a {type_str}."
        else:
            if severity and severity != "Unknown":
                sentence1 = f"An incident of {severity.lower()} severity was reported."
            else:
                sentence1 = "An unclassified incident was reported."

        # Add location/time if present
        loc_time_parts = []
        if location:
            loc_time_parts.append(f"at {location}")
        if time:
            loc_time_parts.append(f"around {time}")
        
        if loc_time_parts:
            # Drop the period, add the location/time, and put the period back
            sentence1 = sentence1[:-1] + " " + " ".join(loc_time_parts) + "."
            
        assessment_parts.append(sentence1)

        # Conditions
        if conditions:
            cond_str = ", ".join(str(c).lower() for c in conditions)
            assessment_parts.append(
                f"Reported conditions ({cond_str}) indicate factors relevant to the assessment."
            )

        # --- 2. ML / Risk Evidence ---
        if result.risk:
            risk_level = result.risk.get("risk_level", "UNKNOWN")
            risk_score = result.risk.get("risk_score", 0.0)
            assessment_parts.append(
                f"The predictive model estimates {risk_level} fire risk ({risk_score:.2f})."
            )

        # --- 3. CV / Vision Evidence ---
        if result.vision:
            fire = result.vision.get("fire_detected", False)
            smoke = result.vision.get("smoke_detected", False)
            anomaly = result.vision.get("anomaly_detected", False)
            conf = result.vision.get("confidence", 0.0)

            if fire and smoke:
                assessment_parts.append(
                    f"Computer vision detected fire and smoke with confidence {conf:.2f}, providing supporting visual evidence."
                )
            elif fire:
                assessment_parts.append(
                    f"Computer vision detected fire with confidence {conf:.2f}, providing supporting visual evidence."
                )
            elif smoke:
                assessment_parts.append(
                    f"Computer vision detected smoke with confidence {conf:.2f}, providing supporting visual evidence."
                )
            elif anomaly:
                assessment_parts.append(
                    f"Computer vision detected a visual anomaly with confidence {conf:.2f}."
                )
            else:
                assessment_parts.append(
                    "Computer vision did not detect fire or smoke in the provided image."
                )

        # --- 4. Historical Evidence ---
        if result.historical_matches:
            # We just reference the fact that there's context, perhaps extracting the type or severity of the top match
            top_match = result.historical_matches[0]
            desc = top_match.get("text", "A previous event")
            if len(desc) > 40:
                desc = desc[:37] + "..."
                
            assessment_parts.append(
                f"A historical incident (e.g., '{desc}') was retrieved as contextual evidence; "
                "this does not establish that the historical incident is related to the current event."
            )

        return " ".join(assessment_parts)
