"""Compatibility helpers for importing llama_stack APIs across versions."""

from __future__ import annotations

# The llama_stack APIs were moved under a separate `llama_stack_api` package
# upstream. Prefer the new package layout and fall back to the legacy one so
# this provider can run with both.
try:  # Current dedicated llama_stack_api package (preferred)
    from llama_stack_api import (
        Api,
        Benchmark,
        BenchmarkConfig,
        BenchmarksProtocolPrivate,
        Eval,
        EvalCandidate,
        EvaluateResponse,
        EvaluateRowsRequest,
        Job,
        JobCancelRequest,
        JobResultRequest,
        JobStatus,
        JobStatusRequest,
        ListBenchmarksResponse,
        ProviderSpec,
        RemoteProviderSpec,
        RunEvalRequest,
        ScoringResult,
        json_schema_type,
    )
    from llama_stack_api.eval.compat import (
        resolve_evaluate_rows_request,
        resolve_job_cancel_request,
        resolve_job_result_request,
        resolve_job_status_request,
        resolve_run_eval_request,
    )
except ModuleNotFoundError:  # Legacy llama_stack layout (pre-0.5.x)
    # Note: Request objects (RunEvalRequest, etc.) were introduced in 0.5.x
    # Legacy versions only support individual parameter style
    from llama_stack.apis.benchmarks import Benchmark, ListBenchmarksResponse
    from llama_stack.apis.common.job_types import Job, JobStatus
    from llama_stack.apis.datatypes import Api
    from llama_stack.apis.eval import (
        BenchmarkConfig,
        Eval,
        EvalCandidate,
        EvaluateResponse,
    )
    from llama_stack.apis.scoring import ScoringResult
    from llama_stack.providers.datatypes import (
        BenchmarksProtocolPrivate,
        ProviderSpec,
        RemoteProviderSpec,
    )
    from llama_stack.schema_utils import json_schema_type

    # Request objects don't exist in pre-0.5.x, set to None
    RunEvalRequest = None  # type: ignore
    EvaluateRowsRequest = None  # type: ignore
    JobStatusRequest = None  # type: ignore
    JobCancelRequest = None  # type: ignore
    JobResultRequest = None  # type: ignore

    # Compat resolve functions don't exist in pre-0.5.x, set to None
    resolve_run_eval_request = None  # type: ignore
    resolve_evaluate_rows_request = None  # type: ignore
    resolve_job_status_request = None  # type: ignore
    resolve_job_cancel_request = None  # type: ignore
    resolve_job_result_request = None  # type: ignore

__all__ = [
    "Api",
    "Benchmark",
    "BenchmarkConfig",
    "BenchmarksProtocolPrivate",
    "Eval",
    "EvalCandidate",
    "EvaluateResponse",
    "EvaluateRowsRequest",
    "Job",
    "JobCancelRequest",
    "JobResultRequest",
    "JobStatus",
    "JobStatusRequest",
    "ListBenchmarksResponse",
    "ProviderSpec",
    "RemoteProviderSpec",
    "RunEvalRequest",
    "ScoringResult",
    "json_schema_type",
    "resolve_evaluate_rows_request",
    "resolve_job_cancel_request",
    "resolve_job_result_request",
    "resolve_job_status_request",
    "resolve_run_eval_request",
]
