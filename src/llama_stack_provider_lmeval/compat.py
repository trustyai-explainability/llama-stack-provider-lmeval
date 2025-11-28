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
        Job,
        JobStatus,
        ListBenchmarksResponse,
        ProviderSpec,
        RemoteProviderSpec,
        ScoringResult,
        json_schema_type,
    )
except ModuleNotFoundError:  # Legacy llama_stack layout
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

__all__ = [
    "Api",
    "Benchmark",
    "BenchmarkConfig",
    "BenchmarksProtocolPrivate",
    "Eval",
    "EvalCandidate",
    "EvaluateResponse",
    "Job",
    "JobStatus",
    "ListBenchmarksResponse",
    "ProviderSpec",
    "RemoteProviderSpec",
    "ScoringResult",
    "json_schema_type",
]
