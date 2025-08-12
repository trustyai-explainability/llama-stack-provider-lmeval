"""Configuration classes for the LMEval provider."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from llama_stack.apis.eval import BenchmarkConfig, EvalCandidate
from llama_stack.schema_utils import json_schema_type
from pydantic import Field

from .errors import LMEvalConfigError


@json_schema_type
@dataclass
class TLSConfig:
    """TLS configuration for the LMEval provider."""

    enable: bool
    cert_file: Optional[str] = None
    cert_secret: Optional[str] = None

    def __post_init__(self):
        """Validate the configuration"""
        if self.enable and (self.cert_file is not None or self.cert_secret is not None):
            # If both cert_file and cert_secret are provided, both must be set
            if not (self.cert_file is not None and self.cert_secret is not None):
                raise LMEvalConfigError(
                    "Both cert_file and cert_secret must be set when TLS is enabled and certificates are specified"
                )


@json_schema_type
@dataclass
class LMEvalBenchmarkConfig(BenchmarkConfig):
    """Configuration for LMEval benchmarkd

    The metadata field can contain custom configurations such as:

    - custom_task: For specifying custom task sources (e.g., from git)
    - env: Dictionary of environment variables to pass to the evaluation pod
           (e.g., {'DK_BENCH_DATASET_PATH': '', 'JUDGE_MODEL_URL': ''})
    - tokenizer: Custom tokenizer to use for the model
    """

    # K8s specific configuration
    model: str = Field(description="Name of the model")
    eval_candidate: EvalCandidate
    # FIXME: mode is only present temporarily and for debug purposes, it will be removed
    # mode: str = Field(description="Mode of the benchmark", default="production")
    env_vars: Optional[List[Dict[str, str]]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate the configuration"""
        if not self.model:
            raise ValueError("model must be provided")


@json_schema_type
@dataclass
class K8sLMEvalConfig:
    """Configuration for Kubernetes LMEvalJob CR"""

    model: str
    model_args: Optional[List[Dict[str, str]]] = field(default_factory=list)
    task_list: Optional[Dict[str, List[str]]] = None
    log_samples: bool = True
    namespace: str = "default"
    env_vars: Optional[List[Dict[str, str]]] = None

    def __post_init__(self):
        """Validate the configuration"""
        if not self.task_list or not self.task_list.get("taskNames"):
            raise ValueError("taskList.taskNames must be provided")

        if not self.model:
            raise ValueError("model must be provided")


@json_schema_type
@dataclass
class LMEvalEvalProviderConfig:
    """LMEval Provider configuration"""

    use_k8s: bool = True
    # FIXME: Hardcoded just for debug purposes
    base_url: str = "http://llamastack-service:8321"
    namespace: str | None = None
    kubeconfig_path: Optional[str] = None
    # Service account to use for Kubernetes deployment
    service_account: Optional[str] = None
    # Default tokenizer to use when none is specified in the ModelCandidate
    default_tokenizer: str = "google/flan-t5-base"
    metadata: Optional[Dict[str, Any]] = None
    # TLS configuration - structured approach
    tls: Optional[TLSConfig] = None

    def __post_init__(self):
        """Validate the configuration"""
        if not isinstance(self.use_k8s, bool):
            raise LMEvalConfigError("use_k8s must be a boolean")
        if self.use_k8s is False:
            raise LMEvalConfigError(
                "Only Kubernetes LMEval backend is supported at the moment"
            )


__all__ = [
    "TLSConfig",
    "LMEvalBenchmarkConfig",
    "K8sLMEvalConfig",
    "LMEvalEvalProviderConfig",
]
