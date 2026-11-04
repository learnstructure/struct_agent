"""
Which state of the model an analysis result describes.

Stored analysis results are not invalidated when the model changes, so a design check could combine the
current section capacities with the demands of an earlier model (resized sections, new loads). Analysis tools
stamp every result they produce with a fingerprint of the model inputs; ``run_design`` refuses a result whose
fingerprint no longer matches the model.
"""

from __future__ import annotations

import functools
import hashlib
from typing import Any, Callable


# Seismic basis fields an analysis result depends on beyond the model: the drift limits and amplification of the
# stored ELF drift checks and the frame system. Spectral values, R and rho act through the loads and combinations
# (part of the model); the procedure and case ids change when an RSA follows an ELF analysis without changing it.
BASIS_FIELDS = {"system", "cd", "importance_factor", "risk_category"}


def model_fingerprint(project: Any, with_combinations: bool = True) -> str:
    """Hash of the inputs an analysis of ``project`` depends on: the structural model, the seismic system
    coefficients of the recorded basis, and the units. A static analysis (``with_combinations``) depends on the
    load combinations but not on the mass source (which an RSA defines); a dynamic analysis the other way round."""
    exclude = {"metadata", "analysis_cases"} | ({"mass_source"} if with_combinations else {"load_combinations"})
    basis = getattr(project, "seismic_basis", None)
    parts = (project.model.model_dump_json(exclude=exclude),
             "" if basis is None else basis.model_dump_json(include=BASIS_FIELDS),
             str(getattr(project.metadata, "units", "")))
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()


def is_current(project: Any, result: Any) -> bool:
    """False when ``result`` was computed from an earlier state of the model (unstamped results pass)."""
    stamp = getattr(result, "model_fingerprint", None)
    if stamp is None:
        return True
    res_type = getattr(result, "result_type", "")
    if res_type in ("modal", "rsa", "time_history") or getattr(result, "modal_result", None) is not None:
        return stamp == model_fingerprint(project, with_combinations=False)
    return stamp in (model_fingerprint(project, True), model_fingerprint(project, False))


def stamps_results(with_combinations: bool) -> Callable:
    """Decorator for analysis tools: stamp each result the tool adds or replaces on ``project.analysis_results``.

    Static analyses depend on the model's load combinations; response-spectrum and time-history analyses
    build their own combinations, so their stamp leaves them out.
    """
    def decorate(tool: Callable) -> Callable:
        @functools.wraps(tool)
        def wrapper(project: Any, *args: Any, **kwargs: Any) -> Any:
            results = getattr(project, "analysis_results", None)
            before = dict(results) if results is not None else {}
            out = tool(project, *args, **kwargs)
            if getattr(out, "ok", False) and getattr(project, "model", None) is not None:
                an_type = str(kwargs.get("analysis_type", "")).lower()
                is_dynamic = an_type in ("modal", "rsa", "time_history") or not with_combinations
                fp_static = model_fingerprint(project, with_combinations=True)
                fp_dynamic = model_fingerprint(project, with_combinations=False)
                for key, res in project.analysis_results.items():
                    if before.get(key) is not res:
                        res_type = getattr(res, "result_type", "")
                        if is_dynamic or res_type in ("modal", "rsa", "time_history") or getattr(res, "modal_result", None) is not None:
                            res.model_fingerprint = fp_dynamic
                        else:
                            res.model_fingerprint = fp_static
            return out
        return wrapper
    return decorate
