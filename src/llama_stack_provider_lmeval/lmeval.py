"""LMEval provider implementation for Llama Stack."""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from kubernetes import client as k8s_client
from kubernetes import config as k8s_config
from kubernetes.client.rest import ApiException
from llama_stack.apis.benchmarks import Benchmark, ListBenchmarksResponse
from llama_stack.apis.common.job_types import Job, JobStatus
from llama_stack.apis.eval import BenchmarkConfig, Eval, EvaluateResponse
from llama_stack.apis.scoring import ScoringResult
from llama_stack.providers.datatypes import BenchmarksProtocolPrivate
from pydantic import BaseModel

from .config import LMEvalEvalProviderConfig
from .errors import LMEvalConfigError, LMEvalTaskNameError

logger = logging.getLogger(__name__)


def _get_tls_config_from_env(provider_config=None) -> str | bool | None:
    """Get TLS configuration from environment variables with provider config fallback.

    Args:
        provider_config: Optional provider configuration object for fallback

    Returns:
        TLS configuration: True for verify=True, string path for certificate file, or None if not configured
    """
    tls_enabled = os.environ.get("TRUSTYAI_LMEVAL_TLS", "").lower() == "true"
    if not tls_enabled:
        # Fallback to provider config if environment variables not set
        if (
            provider_config
            and hasattr(provider_config, "tls")
            and provider_config.tls is not None
        ) and provider_config.tls.enable:
            if (
                provider_config.tls.cert_file is not None
                and provider_config.tls.cert_secret is not None
            ):
                # Both are set, return the full path where the certificate will be mounted
                mount_path = "/etc/ssl/certs"
                full_cert_path = f"{mount_path}/{provider_config.tls.cert_file}"
                logger.debug(
                    "Using TLS configuration from provider config: %s", full_cert_path
                )
                return full_cert_path
            else:
                # TLS enabled but no certificates specified
                logger.debug("Using TLS configuration from provider config: True")
                return True

        return None

    cert_file = os.environ.get("TRUSTYAI_LMEVAL_CERT_FILE")
    cert_secret = os.environ.get("TRUSTYAI_LMEVAL_CERT_SECRET")

    if cert_file and cert_secret:
        # Both are set, return the full path where the certificate will be mounted
        mount_path = "/etc/ssl/certs"
        full_cert_path = f"{mount_path}/{cert_file}"
        logger.debug(
            "Using TLS configuration from environment variables: %s", full_cert_path
        )
        return full_cert_path
    elif cert_file or cert_secret:
        # Only one is set, this is invalid configuration
        missing_var = (
            "TRUSTYAI_LMEVAL_CERT_SECRET" if cert_file else "TRUSTYAI_LMEVAL_CERT_FILE"
        )
        logger.error(
            "Invalid TLS configuration: %s is set but %s is missing. "
            "Both environment variables must be set when TRUSTYAI_LMEVAL_TLS is True.",
            "TRUSTYAI_LMEVAL_CERT_FILE" if cert_file else "TRUSTYAI_LMEVAL_CERT_SECRET",
            missing_var,
        )

        # Check if we can fall back to provider config
        if (
            provider_config
            and hasattr(provider_config, "tls")
            and provider_config.tls is not None
            and provider_config.tls.enable
        ):
            if (
                provider_config.tls.cert_file is not None
                and provider_config.tls.cert_secret is not None
            ):
                mount_path = "/etc/ssl/certs"
                full_cert_path = f"{mount_path}/{provider_config.tls.cert_file}"
                logger.warning(
                    "Falling back to provider config TLS due to incomplete environment variables: %s",
                    full_cert_path,
                )
                return full_cert_path
            else:
                logger.warning(
                    "Falling back to provider config TLS (verify=True) due to incomplete environment variables"
                )
                return True
        else:
            logger.error(
                "Cannot fall back to provider config TLS. TLS verification will be disabled."
            )
            return None
    else:
        # Neither is set, return True for verify=True
        logger.debug("No TLS certificate files specified, using verify=True")
        return True


def _create_tls_volume_config(
    provider_config=None,
) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]] | None]:
    """Create volume mount and volume configuration for TLS certificates.

    Args:
        provider_config: Optional provider configuration object for fallback

    Returns:
        Tuple of (volume_mounts, volumes) or (None, None) if TLS is not configured
    """
    tls_enabled = os.environ.get("TRUSTYAI_LMEVAL_TLS", "").lower() == "true"
    cert_file = os.environ.get("TRUSTYAI_LMEVAL_CERT_FILE")
    cert_secret = os.environ.get("TRUSTYAI_LMEVAL_CERT_SECRET")

    # If environment variables not set, check provider config
    if (
        not tls_enabled
        and provider_config
        and hasattr(provider_config, "tls")
        and provider_config.tls is not None
    ) and provider_config.tls.enable:
        tls_enabled = True
        cert_file = provider_config.tls.cert_file
        cert_secret = provider_config.tls.cert_secret

    if not tls_enabled:
        return None, None

    # Create TLSConfig object from environment variables for validation
    try:
        from .config import TLSConfig

        tls_config = TLSConfig(
            enable=True, cert_file=cert_file, cert_secret=cert_secret
        )
    except Exception as e:
        logger.warning(
            "TLS configuration validation failed: %s. No volumes will be created.",
            str(e),
        )
        return None, None

    # If validation passed but no certificates specified, no volumes needed
    if tls_config.cert_file is None or tls_config.cert_secret is None:
        logger.debug("TLS enabled but no certificates specified, no volumes created")
        return None, None

    # Mount path is predefined as /etc/ssl/certs/
    mount_path = "/etc/ssl/certs"

    # Create volume mount
    volume_mounts = [{"name": "tls-cert", "mountPath": mount_path, "readOnly": True}]

    # Create volume
    volumes = [
        {
            "name": "tls-cert",
            "secret": {
                "secretName": tls_config.cert_secret,
                "items": [{"key": tls_config.cert_file, "path": tls_config.cert_file}],
            },
        }
    ]

    logger.info(
        "Created TLS volume config: mount=%s, secret=%s, cert_file=%s",
        mount_path,
        tls_config.cert_secret,
        tls_config.cert_file,
    )
    return volume_mounts, volumes


class ModelArg(BaseModel):
    """Model argument for the LMEval CR."""

    name: str
    value: str


class ContainerConfig(BaseModel):
    """Container configuration for the LMEval CR."""

    env: list[dict[str, Any]] | None = None
    volumeMounts: list[dict[str, Any]] | None = None


class PodConfig(BaseModel):
    """Pod configuration for the LMEval CR."""

    container: ContainerConfig
    serviceAccountName: str | None = None
    volumes: list[dict[str, Any]] | None = None


class GitSource(BaseModel):
    """Git source for custom tasks."""

    url: str
    branch: str | None = None
    commit: str | None = None
    path: str | None = None

    def model_dump(self, **kwargs):
        result = super().model_dump(**kwargs)
        return {k: v for k, v in result.items() if v is not None}


class CustomTaskSource(BaseModel):
    """Source of custom tasks."""

    git: GitSource

    def model_dump(self, **kwargs):
        return {"git": self.git.model_dump(**kwargs)}


class CustomTasks(BaseModel):
    """Custom tasks configuration."""

    source: CustomTaskSource

    def model_dump(self, **kwargs):
        return {"source": self.source.model_dump(**kwargs)}


class TaskList(BaseModel):
    """Task list configuration for the LMEval Custom Resource."""

    taskNames: list[str]
    customTasks: CustomTasks | None = None

    def model_dump(self, **kwargs):
        result = super().model_dump(**kwargs)
        if result.get("customTasks") is None:
            del result["customTasks"]
        return result


class LMEvalSpec(BaseModel):
    """Specification for the LMEval Custom Resource."""

    allowOnline: bool = True
    allowCodeExecution: bool = True
    model: str = "local-completions"
    taskList: TaskList
    logSamples: bool = True
    batchSize: str = "1"
    limit: str | None = None
    modelArgs: list[ModelArg]
    pod: PodConfig | None = None
    offline: dict[str, Any] | None = None

    def model_dump(self, **kwargs):
        result = super().model_dump(**kwargs)

        if "taskList" in result:
            task_list = result["taskList"]
            if "customTasks" in task_list and task_list["customTasks"] is None:
                del task_list["customTasks"]

        if "offline" in result and result["offline"] is None:
            del result["offline"]

        return result


class LMEvalMetadata(BaseModel):
    """Metadata for the LMEval Custom Resource."""

    name: str
    namespace: str


class LMEvalCR(BaseModel):
    """LMEval Custom Resource model."""

    apiVersion: str = "trustyai.opendatahub.io/v1alpha1"
    kind: str = "LMEvalJob"
    metadata: LMEvalMetadata
    spec: LMEvalSpec

    def model_dump(self, **kwargs):
        result = super().model_dump(**kwargs)

        task_list = result.get("spec", {}).get("taskList", {})

        if task_list.get("customTasks") is None:
            del task_list["customTasks"]

        return result


class LMEvalCRBuilder:
    """An utility class which creates LMEval Custom Resources from BenchmarkConfigs."""

    def __init__(self, namespace: str = "default", service_account: str | None = None):
        """Initialize the LMEvalCRBuilder.

        Args:
            namespace: The Kubernetes namespace to use
            service_account: Optional service account to use for the LMEval Custom Resource
        """
        self._namespace = namespace
        self._service_account = service_account
        self._config = None

    @staticmethod
    def _build_openai_url(base_url: str) -> str:
        """Build OpenAI-compatible URL from base URL.

        Args:
            base_url: The base URL for the model service

        Returns:
            OpenAI-compatible URL with proper path structure
        """
        # Strip trailing slashes
        cleaned_url = base_url.rstrip("/")

        # Check if URL already ends with v1
        if cleaned_url.endswith("/v1"):
            return f"{cleaned_url}/completions"
        else:
            return f"{cleaned_url}/v1/completions"

    def _create_model_args(
        self, base_url: str, benchmark_config: BenchmarkConfig
    ) -> list[ModelArg]:
        """Create model arguments for the LMEvalJob CR."""
        model_args = [
            ModelArg(
                name="base_url",
                value=self._build_openai_url(base_url) if base_url is not None else "",
            ),
        ]

        # Add model name if specified in benchmark config
        if hasattr(benchmark_config, "model") and benchmark_config.model:
            model_args.append(ModelArg(name="model", value=benchmark_config.model))

        # Add custom model args from benchmark config, avoiding duplicate keys
        if hasattr(benchmark_config, "model_args") and benchmark_config.model_args:
            existing_arg_names = {arg.name for arg in model_args}
            for arg in benchmark_config.model_args:
                if arg.name not in existing_arg_names:
                    model_args.append(arg)
                else:
                    # Optionally, update the value for existing keys instead of skipping
                    for i, existing_arg in enumerate(model_args):
                        if existing_arg.name == arg.name:
                            model_args[i] = arg
                            break

        # Add TLS configuration
        env_tls_config = _get_tls_config_from_env(self._config)
        if env_tls_config is not None:
            model_args.append(
                ModelArg(name="verify_certificate", value=str(env_tls_config))
            )

        return model_args

    def _collect_env_vars(
        self, task_config: BenchmarkConfig, stored_benchmark: Benchmark | None
    ) -> list[dict[str, Any]]:
        """Collect environment variables.

        Args:
            task_config: Task configuration
            stored_benchmark: Optional stored benchmark

        Returns:
            List of environment variables
        """
        env_vars = []
        if hasattr(task_config, "env_vars") and task_config.env_vars:
            env_vars = task_config.env_vars

        # Get environment variables from metadata
        if hasattr(task_config, "metadata") and task_config.metadata:
            metadata_env = task_config.metadata.get("env")
            if metadata_env and isinstance(metadata_env, dict):
                logger.debug(
                    "Found environment variables in metadata: %s", metadata_env
                )
                for key, value in metadata_env.items():
                    if isinstance(value, dict) and "secret" in value:
                        # Handle Kubernetes secret reference structure
                        env_vars.append({"name": key, "secret": value["secret"]})
                        logger.debug(
                            "Added secret environment variable from metadata: %s", key
                        )
                    else:
                        # Handle simple string value
                        env_vars.append({"name": key, "value": str(value)})
                        logger.debug(
                            "Added environment variable from metadata: %s", key
                        )

        # Get environment variables from stored benchmark metadata
        if (
            not env_vars
            and stored_benchmark
            and hasattr(stored_benchmark, "metadata")
            and stored_benchmark.metadata
        ):
            metadata_env = stored_benchmark.metadata.get("env")
            if metadata_env and isinstance(metadata_env, dict):
                logger.debug(
                    "Found environment variables in stored benchmark metadata: %s",
                    metadata_env,
                )
                for key, value in metadata_env.items():
                    if isinstance(value, dict) and "secret" in value:
                        # Handle Kubernetes secret reference structure
                        env_vars.append({"name": key, "secret": value["secret"]})
                        logger.debug(
                            "Added secret environment variable from stored benchmark metadata: %s",
                            key,
                        )
                    else:
                        # Handle simple string value
                        env_vars.append({"name": key, "value": str(value)})
                        logger.debug(
                            "Added environment variable from stored benchmark metadata: %s",
                            key,
                        )

        return env_vars

    def _extract_task_name(self, benchmark_id: str) -> str:
        """Extract task name from benchmark ID.

        Args:
            benchmark_id: The benchmark id

        Returns:
            Task name

        Raises:
            LMEvalTaskNameError: If task name is empty or invalid
        """
        task_name_parts = benchmark_id.split("::")
        task_name = task_name_parts[-1].strip() if task_name_parts else ""
        if not task_name:
            raise LMEvalTaskNameError(
                f"Invalid benchmark_id '{benchmark_id}': task name is empty or invalid"
            )

        return task_name

    def _create_pod_config(self, env_vars: list[dict[str, Any]]) -> PodConfig | None:
        """Create pod configuration with environment variables.

        Args:
            env_vars: List of environment variables

        Returns:
            PodConfig object or None if no config needed
        """
        if not env_vars and not self._service_account:
            # Check if we need TLS volumes even without env vars
            volume_mounts, volumes = _create_tls_volume_config(self._config)
            if volume_mounts and volumes:
                container_config = ContainerConfig(volumeMounts=volume_mounts)
                pod_config = PodConfig(
                    container=container_config,
                    serviceAccountName=self._service_account,
                    volumes=volumes,
                )
                logger.info("Created pod config with TLS volumes")
                return pod_config
            return None

        # Process environment variables to handle both simple values and secret references
        processed_env_vars = []
        for env_var in env_vars:
            env_entry = {"name": env_var["name"]}

            # Check if this env var has a secret reference
            if "secret" in env_var and env_var["secret"]:
                # Custom secret structure: name/value/secret
                secret_ref = env_var["secret"]
                if (
                    isinstance(secret_ref, dict)
                    and "name" in secret_ref
                    and "key" in secret_ref
                ):
                    env_entry["valueFrom"] = {
                        "secretKeyRef": {
                            "name": secret_ref["name"],
                            "key": secret_ref["key"],
                        }
                    }
                else:
                    # Invalid secret structure, fall back to simple value
                    logger.warning(
                        "Invalid secret structure for env var '%s'. "
                        "Expected a dict with 'name' and 'key'. Falling back to simple value.",
                        env_var.get("name", "<unknown>"),
                    )
                    env_entry["value"] = str(env_var.get("value", ""))
            else:
                # Handle value field (simple or complex structures)
                value = env_var.get("value")
                if (
                    isinstance(value, str)
                    and value.startswith("{")
                    and value.endswith("}")
                ):
                    # A stringified dict, parse it
                    try:
                        import ast

                        parsed_value = ast.literal_eval(value)
                        if (
                            isinstance(parsed_value, dict)
                            and "valueFrom" in parsed_value
                        ):
                            # Use the parsed valueFrom structure
                            env_entry |= parsed_value
                        else:
                            # Use as simple string value
                            env_entry["value"] = value
                    except (ValueError, SyntaxError):
                        # If parsing fails, use as simple string value
                        env_entry["value"] = value
                elif isinstance(value, dict):
                    # Direct dict structure (e.g., valueFrom)
                    env_entry.update(value)
                else:
                    # Simple string value
                    env_entry["value"] = str(value) if value is not None else ""

            processed_env_vars.append(env_entry)

        # Get TLS volume configuration
        volume_mounts, volumes = _create_tls_volume_config(self._config)

        # Add environment variables to the container config
        container_config = ContainerConfig(
            env=processed_env_vars or None, volumeMounts=volume_mounts
        )

        # Add service account to the pod config
        pod_config = PodConfig(
            container=container_config,
            serviceAccountName=self._service_account,
            volumes=volumes,
        )

        if env_vars:
            env_var_names = [env_var["name"] for env_var in processed_env_vars]
            logger.debug(
                "Setting pod environment variables: %s", ", ".join(env_var_names)
            )

        if volume_mounts and volumes:
            logger.info("Added TLS volume configuration to pod")

        return pod_config

    def _extract_git_source(
        self, task_config: BenchmarkConfig, stored_benchmark: Benchmark | None
    ) -> dict[str, Any] | None:
        """Extract git source data from task config or stored benchmark.

        Args:
            task_config: Task configuration
            stored_benchmark: Optional stored benchmark

        Returns:
            Git source data or None if not available
        """
        # Check task_config metadata first
        if (
            hasattr(task_config, "metadata")
            and task_config.metadata
            and "custom_task" in task_config.metadata
        ):
            custom_task_data = task_config.metadata.get("custom_task", {})
            git_data = custom_task_data.get("git", {})

            if git_data and "url" in git_data:
                logger.debug(
                    "Found git URL in task_config metadata: %s", git_data.get("url")
                )
                return {
                    "url": git_data.get("url"),
                    "branch": git_data.get("branch"),
                    "commit": git_data.get("commit"),
                    "path": git_data.get("path"),
                }

        # Check in stored_benchmark
        if (
            stored_benchmark
            and hasattr(stored_benchmark, "metadata")
            and stored_benchmark.metadata
        ):
            custom_task_data = stored_benchmark.metadata.get("custom_task", {})
            git_data = custom_task_data.get("git", {})
            if git_data and "url" in git_data:
                logger.debug(
                    "Found git URL in stored_benchmark metadata: %s",
                    git_data.get("url"),
                )
                return {
                    "url": git_data.get("url"),
                    "branch": git_data.get("branch"),
                    "commit": git_data.get("commit"),
                    "path": git_data.get("path"),
                }

        return None

    def _extract_pvc_name(
        self, task_config: BenchmarkConfig, stored_benchmark: Benchmark | None
    ) -> str | None:
        """Get PVC name from metadata with structure input.storage.pvc.

        Args:
            task_config: Task configuration
            stored_benchmark: Optional stored benchmark

        Returns:
            PVC name or None if not available
        """
        if hasattr(task_config, "metadata") and task_config.metadata:
            input_data = task_config.metadata.get("input", {})
            if isinstance(input_data, dict):
                storage_data = input_data.get("storage", {})
                if isinstance(storage_data, dict):
                    pvc_name = storage_data.get("pvc")
                    if pvc_name and isinstance(pvc_name, str):
                        logger.debug(
                            "Found PVC name in task_config metadata: %s", pvc_name
                        )
                        return pvc_name

        if (
            stored_benchmark
            and hasattr(stored_benchmark, "metadata")
            and stored_benchmark.metadata
        ):
            input_data = stored_benchmark.metadata.get("input", {})
            if isinstance(input_data, dict):
                storage_data = input_data.get("storage", {})
                if isinstance(storage_data, dict):
                    pvc_name = storage_data.get("pvc")
                    if pvc_name and isinstance(pvc_name, str):
                        logger.debug(
                            "Found PVC name in stored_benchmark metadata: %s", pvc_name
                        )
                        return pvc_name

        return None

    def create_cr(
        self,
        benchmark_id: str,
        task_config: BenchmarkConfig,
        base_url: str | None = None,
        limit: str | None = None,
        stored_benchmark: Benchmark | None = None,
    ) -> dict:
        """Create LMEval Custom Resource from a Llama Stack BenchmarkConfig.

        Args:
            benchmark_id: The benchmark identifier
            task_config: Configuration for the evaluation task
            base_url: Optional base URL for the model service
            limit: Optional limit for number of examples to evaluate
            stored_benchmark: Optional stored benchmark to retrieve metadata

        Returns:
            dict: LMEval CR specification

        Raises:
            LMEvalConfigError: If the configuration is invalid
        """
        logger.info("CREATE_CR: Starting CR creation for benchmark %s", benchmark_id)
        logger.info("CREATE_CR: Task config type: %s", type(task_config))
        logger.info(
            "CREATE_CR: Task config has metadata: %s", hasattr(task_config, "metadata")
        )
        if hasattr(task_config, "metadata"):
            logger.info(
                "CREATE_CR: Task config metadata content: %s", task_config.metadata
            )
        # Validate model candidate
        eval_candidate = task_config.eval_candidate
        if eval_candidate.type != "model":
            raise LMEvalConfigError("LMEval only supports model candidates for now")

        # Extract TLS configuration - prioritise environment variables over benchmark config
        benchmark_tls = None

        # Check for TLS configuration in benchmark config
        if (
            hasattr(task_config, "tls")
            and task_config.tls is not None
            and isinstance(task_config.tls, str | bool)
        ):
            benchmark_tls = task_config.tls
            logger.debug(
                "Found TLS configuration in benchmark config: %s", benchmark_tls
            )

        logger.info("Final benchmark_tls value for model args: %s", benchmark_tls)

        # Create model args
        model_args = self._create_model_args(base_url, task_config)

        if (
            hasattr(stored_benchmark, "metadata")
            and stored_benchmark.metadata
            and "tokenizer" in stored_benchmark.metadata
        ):
            tokenizer_value = stored_benchmark.metadata.get("tokenizer")
            if isinstance(tokenizer_value, str) and tokenizer_value:
                logger.debug(
                    "Using custom tokenizer from metadata: %s", tokenizer_value
                )
                model_args.append(ModelArg(name="tokenizer", value=tokenizer_value))

        # Add tokenized_requests parameter if present in metadata
        if (
            hasattr(stored_benchmark, "metadata")
            and stored_benchmark.metadata
            and "tokenized_requests" in stored_benchmark.metadata
        ):
            tokenized_requests_value = stored_benchmark.metadata.get(
                "tokenized_requests"
            )
            if (
                isinstance(tokenized_requests_value, bool | str)
                and tokenized_requests_value is not None
            ):
                value_str = str(tokenized_requests_value)
                logger.debug("Using tokenized_requests from metadata: %s", value_str)
                model_args.append(ModelArg(name="tokenized_requests", value=value_str))

        # Collect environment variables
        env_vars = self._collect_env_vars(task_config, stored_benchmark)

        # Extract task name
        task_name = self._extract_task_name(benchmark_id)

        # Create pod config
        pod_config = self._create_pod_config(env_vars)

        # Generate UUID-based id

        job_id = str(uuid.uuid4())

        # Create task list
        task_list_params = {"taskNames": [task_name]}

        # Extract git source data
        git_source_data = self._extract_git_source(task_config, stored_benchmark)

        # Extract PVC name
        pvc_name = self._extract_pvc_name(task_config, stored_benchmark)

        # Create CR spec parameters
        spec_params = {
            "taskList": TaskList(**task_list_params),
            "modelArgs": model_args,
            "pod": pod_config,
        }

        if limit is not None:
            spec_params["limit"] = limit

        # Create CR
        cr = LMEvalCR(
            metadata=LMEvalMetadata(
                name=f"lmeval-llama-stack-job-{job_id[:8]}", namespace=self._namespace
            ),
            spec=LMEvalSpec(**spec_params),
        )

        cr_dict = cr.model_dump()

        if pvc_name:
            logger.info("Setting up offline storage with PVC: %s", pvc_name)
            if "offline" in cr_dict["spec"] and cr_dict["spec"]["offline"] is None:
                logger.warning("Removing null offline field from CR spec")
                del cr_dict["spec"]["offline"]

            cr_dict["spec"]["offline"] = {"storage": {"pvcName": pvc_name}}
            logger.debug("Added offline storage to CR with PVC: %s", pvc_name)

        # Add custom tasks to CR if git source data is available
        if git_source_data:
            logger.info("Adding customTasks to CR with git data: %s", git_source_data)

            custom_tasks_section = {"source": {"git": {}}}

            for key, value in git_source_data.items():
                if value is not None:
                    custom_tasks_section["source"]["git"][key] = value

            cr_dict["spec"]["taskList"]["customTasks"] = custom_tasks_section

            logger.debug(
                "Added customTasks to CR: %s",
                json.dumps(custom_tasks_section, indent=2),
            )
        else:
            logger.warning("No git source data found for customTasks")

        logger.debug("Final LMEval Custom Resource: %s", json.dumps(cr_dict, indent=2))

        return cr_dict


def _resolve_namespace(config: LMEvalEvalProviderConfig) -> str:
    """Resolve the namespace.
    1. Namespace check in the provider config
    2. If missing, read from environment variable TRUSTYAI_LM_EVAL_NAMESPACE
    3. If all above missing, use current namespace from service account or pod environment.

    Args:
        config: The LMEval provider configuration

    Returns:
        The resolved namespace string

    """  # noqa: D205, E501
    # Check if namespace is explicitly set in the provider config
    if config.namespace:
        logger.debug("Using namespace from provider config: %s", config.namespace)
        return config.namespace.strip()

    # Check from environment variable
    env_namespace = os.getenv("TRUSTYAI_LM_EVAL_NAMESPACE")  # noqa: Q000
    if env_namespace:
        logger.debug("Using namespace from environment variable: %s", env_namespace)
        return env_namespace

    # Check from service account namespace file
    service_account_namespace_path = (
        "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
    )
    if Path(service_account_namespace_path).exists():
        try:
            with open(service_account_namespace_path) as f:
                namespace = f.read().strip()
                if namespace:
                    logger.debug("Using namespace from service account: %s", namespace)
                    return namespace
        except OSError as e:
            logger.warning("Failed to read namespace from service account: %s", e)

    # Check for POD_NAMESPACE environment variable
    pod_namespace = os.getenv("POD_NAMESPACE")
    if pod_namespace:
        logger.debug(
            "Using namespace from POD_NAMESPACE environment variable: %s", pod_namespace
        )
        return pod_namespace

    # Check for NAMESPACE environment variable
    alt_namespace = os.getenv("NAMESPACE")  # noqa: Q000
    if alt_namespace:
        logger.debug(
            "Using namespace from NAMESPACE environment variable: %s", alt_namespace
        )
        return alt_namespace

    # No namespace found - fail explicitly
    raise LMEvalConfigError(  # noqa: TRY003
        "Unable to determine namespace. Please specify one of the following:\n"  # noqa: EM101
        "1. Set 'namespace' in your run.yaml provider config\n"
        "2. Set TRUSTYAI_LM_EVAL_NAMESPACE environment variable\n"
        "3. Ensure pod has access to service account namespace file\n"
        "4. Set POD_NAMESPACE or NAMESPACE environment variables",
    )


class LMEval(Eval, BenchmarksProtocolPrivate):
    """LMEval provider implementation for Kubernetes-based evaluations."""

    def __init__(self, config: LMEvalEvalProviderConfig):
        self._config = config

        self._namespace = _resolve_namespace(self._config)

        logger.debug("LMEval provider initialized with namespace: %s", self._namespace)
        logger.debug("LMEval provider config values: %s", vars(self._config))
        self.benchmarks = {}
        self._jobs: list[Job] = []
        self._job_metadata = {}

        self._k8s_client = None
        self._k8s_custom_api = None
        if self.use_k8s:
            self._init_k8s_client()
            logger.debug(
                "Initialized Kubernetes client with namespace: %s", self._namespace
            )
            self._cr_builder = LMEvalCRBuilder(
                namespace=self._namespace,
                service_account=getattr(self._config, "service_account", None),
            )
            self._cr_builder._config = self._config

    def _init_k8s_client(self):
        """Initialize the Kubernetes client."""
        # FIXME: Support in-cluster kubeconfig only?
        try:
            k8s_config.load_incluster_config()
            logger.debug("Loaded Kubernetes config from within the cluster")
        except k8s_config.ConfigException:
            try:
                k8s_config.load_kube_config()
                logger.debug("Loaded Kubernetes config from kubeconfig file")
            except k8s_config.ConfigException as e:
                logger.error("Failed to load Kubernetes config: %s", e)
                raise LMEvalConfigError(
                    f"Failed to initialize Kubernetes client: {e}"
                ) from e

        self._k8s_client = k8s_client.ApiClient()
        self._k8s_custom_api = k8s_client.CustomObjectsApi(self._k8s_client)

    @property
    def use_k8s(self) -> bool:
        """Check if K8s mode is enabled."""
        return getattr(self._config, "use_k8s", True)

    async def initialize(self):
        logger.debug("Initializing Base LMEval")

    async def list_benchmarks(self) -> ListBenchmarksResponse:
        """List all registered benchmarks.

        Returns:
            ListBenchmarksResponse: Response containing all registered benchmarks
        """
        return ListBenchmarksResponse(data=list(self.benchmarks.values()))

    async def get_benchmark(self, benchmark_id: str) -> Benchmark | None:
        """Get a specific benchmark by ID.

        Args:
            benchmark_id: The benchmark identifier

        Returns:
            Optional[Benchmark]: The benchmark if found, None otherwise
        """
        benchmark = self.benchmarks.get(benchmark_id)
        if benchmark:
            logger.debug(
                "Retrieved benchmark %s with metadata: %s",
                benchmark_id,
                benchmark.metadata,
            )
            if "custom_task" in benchmark.metadata:
                logger.debug(
                    "Benchmark %s has custom_task: %s",
                    benchmark_id,
                    benchmark.metadata.get("custom_task"),
                )
        return benchmark

    async def register_benchmark(self, benchmark: Benchmark) -> None:
        """Register a benchmark for evaluation.

        Args:
            benchmark: The benchmark to register
        """
        logger.debug("Registering benchmark: %s", benchmark.identifier)
        # FIXME: Remove debug logging
        if hasattr(benchmark, "metadata") and benchmark.metadata:
            logger.debug("Registering benchmark with metadata: %s", benchmark.metadata)
            if "custom_task" in benchmark.metadata:
                logger.debug(
                    "Benchmark has custom_task: %s",
                    benchmark.metadata.get("custom_task"),
                )
        self.benchmarks[benchmark.identifier] = benchmark

    def _get_job_id(self) -> str:
        """Generate a unique job ID.

        Returns:
            Unique job ID string
        """

        return f"lmeval-job-{str(uuid.uuid4())}"

    def _deploy_lmeval_cr(self, cr: dict, job_id: str) -> None:
        """Deploy LMEval custom resource to Kubernetes.

        Args:
            cr: Custom resource definition
            job_id: Job ID to track the deployment

        Raises:
            LMEvalConfigError: If deployment fails
        """
        group = "trustyai.opendatahub.io"
        version = "v1alpha1"
        plural = "lmevaljobs"

        if "spec" in cr:
            pvc_name = None

            if hasattr(self._cr_builder, "_config") and hasattr(
                self._cr_builder._config, "metadata"
            ):
                config_metadata = self._cr_builder._config.metadata
                if (
                    config_metadata
                    and "input" in config_metadata
                    and "storage" in config_metadata.get("input", {})
                ):
                    storage_data = config_metadata["input"]["storage"]
                    if isinstance(storage_data, dict) and "pvc" in storage_data:
                        pvc_name = storage_data["pvc"]

            if pvc_name:
                if "offline" in cr["spec"] and cr["spec"]["offline"] is None:
                    logger.warning(
                        "Removing null offline field from CR spec before deployment"
                    )
                    del cr["spec"]["offline"]

                cr["spec"]["offline"] = {"storage": {"pvcName": pvc_name}}
                logger.debug(
                    "Ensured offline storage in CR before deployment: %s", pvc_name
                )

        cr_yaml = yaml.dump(cr, default_flow_style=False)
        try:
            cr_dict = yaml.safe_load(cr_yaml)
            if (
                "spec" in cr_dict
                and "offline" in cr_dict["spec"]
                and cr_dict["spec"]["offline"] is None
            ):
                logger.warning("Found null offline field in YAML, fixing it")
                del cr_dict["spec"]["offline"]

            if pvc_name:
                cr_dict["spec"]["offline"] = {"storage": {"pvcName": pvc_name}}
                logger.debug("Re-added offline storage in YAML: %s", pvc_name)

            cr_yaml = yaml.dump(cr_dict, default_flow_style=False)
            cr = cr_dict
        except Exception as e:
            logger.error("Error fixing YAML: %s", e)

        logger.info(
            "Deploying LMEvalJob Custom Resource to Kubernetes namespace: %s",
            self._namespace,
        )
        logger.info("Full Custom Resource being submitted: \n%s", cr_yaml)

        try:
            response = self._k8s_custom_api.create_namespaced_custom_object(
                group=group,
                version=version,
                namespace=self._namespace,
                plural=plural,
                body=cr,
            )

            if (
                not response
                or not isinstance(response, dict)
                or "metadata" not in response
            ):
                logger.error("Invalid response from Kubernetes API")
                raise LMEvalConfigError("Invalid response from Kubernetes API")

            logger.debug(
                "Successfully deployed LMEval CR to Kubernetes: %s",
                response["metadata"]["name"],
            )

            self._job_metadata[job_id]["k8s_name"] = cr["metadata"]["name"]

        except ApiException as e:
            logger.error("Failed to deploy LMEval CR to Kubernetes: %s", e)
            raise LMEvalConfigError(f"Failed to deploy LMEval CR: {e}") from e

    def _process_benchmark_config(
        self, benchmark_config: BenchmarkConfig
    ) -> str | None:
        """Process benchmark configuration for limit.

        Args:
            benchmark_config: Benchmark configuration

        Returns:
            Example limit as string or None
        """
        if (
            not hasattr(benchmark_config, "num_examples")
            or benchmark_config.num_examples is None
        ):
            return None

        config_limit = str(benchmark_config.num_examples)
        logger.debug("Using example limit from config: %s", config_limit)
        return config_limit

    def _process_environment_vars(self, benchmark_config: BenchmarkConfig) -> None:
        """Process environment variables from metadata.

        Args:
            benchmark_config: Benchmark configuration
        """
        if not (
            hasattr(benchmark_config, "metadata")
            and benchmark_config.metadata
            and "env" in benchmark_config.metadata
        ):
            return

        env_data = benchmark_config.metadata.get("env", {})
        if not isinstance(env_data, dict):
            return

        if (
            not hasattr(benchmark_config, "env_vars")
            or benchmark_config.env_vars is None
        ):
            benchmark_config.env_vars = []

        for key, value in env_data.items():
            benchmark_config.env_vars.append({"name": key, "value": str(value)})

        logger.debug(
            "Added environment variables from metadata.env to benchmark_config"
        )

    async def run_eval(
        self,
        benchmark_id: str,
        benchmark_config: BenchmarkConfig,
        limit: str = "2",
    ) -> dict[str, str]:
        """Run an evaluation for a specific benchmark and configuration.

        Args:
            benchmark_id: The benchmark id
            benchmark_config: Configuration for the evaluation task
            limit: The maximum number of examples to evaluate (default: 2)

        Returns:
            Dict containing job_id for evaluation tracking
        """
        if not self.use_k8s:
            raise NotImplementedError("Non-K8s evaluation not implemented yet")

        if not isinstance(benchmark_config, BenchmarkConfig):
            raise LMEvalConfigError("K8s mode requires BenchmarkConfig")

        stored_benchmark = await self.get_benchmark(benchmark_id)
        logger.info("Running evaluation for benchmark %s", benchmark_id)

        config_limit = self._process_benchmark_config(benchmark_config)
        self._process_environment_vars(benchmark_config)

        if hasattr(benchmark_config, "metadata") and benchmark_config.metadata:
            logger.debug(
                "Benchmark config metadata: %s",
                json.dumps(benchmark_config.metadata, indent=2),
            )
            if (
                "input" in benchmark_config.metadata
                and "storage" in benchmark_config.metadata.get("input", {})
            ):
                logger.debug(
                    "Found storage config in metadata: %s",
                    benchmark_config.metadata["input"]["storage"],
                )

        cr = self._cr_builder.create_cr(
            benchmark_id=benchmark_id,
            task_config=benchmark_config,
            base_url=getattr(self._config, "base_url", None),
            limit=config_limit or limit,
            stored_benchmark=stored_benchmark,
        )

        logger.debug(
            "Generated LMEvalJob Custom Resource for benchmark %s", benchmark_id
        )

        if (
            "spec" in cr
            and hasattr(benchmark_config, "metadata")
            and benchmark_config.metadata
        ):
            input_data = benchmark_config.metadata.get("input", {})
            if (
                isinstance(input_data, dict)
                and "storage" in input_data
                and "pvc" in input_data.get("storage", {})
            ):
                pvc_name = input_data["storage"]["pvc"]

                if "offline" in cr["spec"] and cr["spec"]["offline"] is None:
                    logger.warning(
                        "Removing null offline field from CR spec in run_eval"
                    )
                    del cr["spec"]["offline"]

                cr["spec"]["offline"] = {"storage": {"pvcName": pvc_name}}
                logger.debug("Ensured offline storage in CR with PVC: %s", pvc_name)

        job_id = self._get_job_id()

        job = Job(
            job_id=job_id,
            status=JobStatus.scheduled,
            metadata={"created_at": datetime.now().isoformat()},
        )
        self._jobs.append(job)
        self._job_metadata[job_id] = {}

        try:
            self._deploy_lmeval_cr(cr, job_id)
        except Exception as e:
            job.status = JobStatus.failed
            self._job_metadata[job_id]["error"] = str(e)
            raise

        return {"job_id": job_id}

    async def evaluate_rows(
        self,
        benchmark_id: str,
        input_rows: list[dict[str, Any]],
        scoring_functions: list[str],
        benchmark_config: BenchmarkConfig,
    ) -> EvaluateResponse:
        """Evaluate a list of rows on a benchmark.

        Args:
            benchmark_id: The ID of the benchmark to run the evaluation on.
            input_rows: The rows to evaluate.
            scoring_functions: The scoring functions to use for the evaluation.
            benchmark_config: The configuration for the benchmark.

        Returns:
            EvaluateResponse: Object containing generations and scores
        """
        if not self.use_k8s:
            raise NotImplementedError("Non-K8s evaluation not implemented yet")

        # FIXME: Placeholder
        generations = []
        for row in input_rows:
            generation = {**row, "generated_answer": "Placeholder answer from LMEval"}
            generations.append(generation)

        scores = {}
        for scoring_fn in scoring_functions:
            score_rows = [{"score": 0.5} for _ in input_rows]
            scores[scoring_fn] = ScoringResult(
                aggregated_results={"accuracy": 0.5}, score_rows=score_rows
            )

        return EvaluateResponse(generations=generations, scores=scores)

    async def job_status(self, benchmark_id: str, job_id: str) -> dict[str, str] | None:
        """Get the status of a running evaluation job.

        Args:
            benchmark_id: The benchmark id
            job_id: The job id

        Returns:
            Dict with current status of the job
        """
        if not self.use_k8s:
            raise NotImplementedError("Non-K8s evaluation not implemented yet")

        job = next((j for j in self._jobs if j.job_id == job_id), None)
        if not job:
            logger.warning("Job %s not found", job_id)
            return None

        try:
            job_metadata = self._job_metadata.get(job_id, {})
            k8s_name = job_metadata.get("k8s_name")
            if not k8s_name:
                logger.warning("No K8s resource name found for job %s", job_id)
                return {"job_id": job_id, "status": JobStatus.scheduled}

            group = "trustyai.opendatahub.io"
            version = "v1alpha1"
            plural = "lmevaljobs"

            cr = self._k8s_custom_api.get_namespaced_custom_object(
                group=group,
                version=version,
                namespace=self._namespace,
                plural=plural,
                name=k8s_name,
            )

            status = cr.get("status", {})
            state = status.get("state", "")
            reason = status.get("reason", "")
            message = status.get("message", "")

            logger.debug(
                "Job %s status: state=%s, reason=%s, message=%s",
                job_id,
                state,
                reason,
                message,
            )

            job_status = JobStatus.scheduled
            if state == "Complete":
                job_status = (
                    JobStatus.failed if reason == "Failed" else JobStatus.completed
                )
            elif state in ("Running", "Pending", "Scheduled"):
                job_status = (
                    JobStatus.in_progress if state == "Running" else JobStatus.scheduled
                )
            elif state == "Cancelled":
                job_status = JobStatus.cancelled

            # Update the job status
            job.status = job_status

            return {"job_id": job_id, "status": job_status}

        except ApiException as e:
            logger.error("Failed to get job status from Kubernetes: %s", e)
            return {"job_id": job_id, "status": JobStatus.scheduled}

    async def job_cancel(self, benchmark_id: str, job_id: str) -> None:
        """Cancel a running evaluation job.

        Args:
            benchmark_id: The benchmark identifier
            job_id: The job identifier
        """
        if not self.use_k8s:
            raise NotImplementedError("Non-K8s evaluation not implemented yet")

        job = next((j for j in self._jobs if j.job_id == job_id), None)
        if not job:
            logger.warning("Job %s not found", job_id)
            return

        try:
            job_metadata = self._job_metadata.get(job_id, {})
            k8s_name = job_metadata.get("k8s_name")
            if not k8s_name:
                logger.warning("No K8s resource name found for job %s", job_id)
                return

            # Delete the LMEvalJob
            group = "trustyai.opendatahub.io"
            version = "v1alpha1"
            plural = "lmevaljobs"

            self._k8s_custom_api.delete_namespaced_custom_object(
                group=group,
                version=version,
                namespace=self._namespace,
                plural=plural,
                name=k8s_name,
            )

            # Update job status
            job.status = JobStatus.cancelled
            logger.info(
                "Successfully cancelled job %s (Kubernetes resource: %s)",
                job_id,
                k8s_name,
            )

        except ApiException as e:
            logger.error("Failed to cancel job in Kubernetes: %s", e)
            raise LMEvalConfigError(f"Failed to cancel job: {e}") from e

    def _get_job_and_k8s_name(self, job_id: str) -> tuple[Job | None, str | None]:
        """Get job and Kubernetes resource name.

        Args:
            job_id: The job ID

        Returns:
            (job, k8s_name)
        """
        job = next((j for j in self._jobs if j.job_id == job_id), None)
        if not job:
            logger.warning("Job %s not found", job_id)
            return None, None

        job_metadata = self._job_metadata.get(job_id, {})
        k8s_name = job_metadata.get("k8s_name")
        if not k8s_name:
            logger.warning("No K8s resource name found for job %s", job_id)
            return job, None

        return job, k8s_name

    def _get_k8s_cr(self, k8s_name: str) -> dict[str, Any] | None:
        """Get Kubernetes custom resource.

        Args:
            k8s_name: Kubernetes resource name

        Returns:
            Custom resource as dictionary or None if not found
        """
        try:
            group = "trustyai.opendatahub.io"
            version = "v1alpha1"
            plural = "lmevaljobs"

            return self._k8s_custom_api.get_namespaced_custom_object(
                group=group,
                version=version,
                namespace=self._namespace,
                plural=plural,
                name=k8s_name,
            )
        except ApiException as e:
            logger.error("Failed to get custom resource: %s", e)
            return None

    def _parse_evaluation_results(
        self, results_str: str
    ) -> tuple[dict[str, ScoringResult], list[dict[str, Any]], dict[str, Any]]:
        """Parse evaluation results from JSON string.

        Args:
            results_str: JSON string containing results

        Returns:
            (scores, generations, metadata)
        """
        try:
            results = json.loads(results_str)
            logger.debug("Successfully parsed results JSON")
        except json.JSONDecodeError as e:
            logger.error("Failed to parse results JSON: %s", e)
            raise ValueError(f"Invalid JSON in results field: {e}") from e

        # Extract scores
        scores = {}
        if "results" in results and isinstance(results["results"], dict):
            for task_name, metrics in results["results"].items():
                if not isinstance(metrics, dict):
                    continue

                for metric_key, metric_value in metrics.items():
                    if (
                        not isinstance(metric_value, int | float)
                        or metric_key == "alias"
                    ):
                        continue

                    metric_name = (
                        metric_key.split(",")[0] if "," in metric_key else metric_key
                    )
                    score_rows = [{"score": metric_value}]
                    scores[f"{task_name}:{metric_name}"] = ScoringResult(
                        aggregated_results={metric_name: metric_value},
                        score_rows=score_rows,
                    )

        # Extract generations
        generations = []
        if "samples" in results and isinstance(results["samples"], list):
            for sample in results["samples"]:
                generation = {
                    "generated_answer": sample.get("output", ""),
                    "input": sample["input"] if "input" in sample else None,
                }
                generations.append(generation)

        # Extract metadata
        metadata = {
            "task_info": {
                "task_name": list(results.get("results", {}).keys()),
                "n_samples": results.get("n-samples", {}),
                "higher_is_better": results.get("higher_is_better", {}),
            },
            "model_info": {
                "model_name": results.get("model_name", ""),
                "model_source": results.get("model_source", ""),
                "evaluation_time": results.get("total_evaluation_time_seconds", ""),
            },
            "config": results.get("config", {}),
        }

        return scores, generations, metadata

    async def job_result(self, benchmark_id: str, job_id: str) -> EvaluateResponse:
        """Get the results of a completed evaluation job.

        Args:
            benchmark_id: The benchmark id
            job_id: The job id

        Returns:
            EvaluateResponse: Results of the evaluation
        """
        if not self.use_k8s:
            return EvaluateResponse(
                generations=[],
                scores={},
                metadata={"error": "Non-K8s implementation not available"},
            )

        # Get job and k8s name
        job, k8s_name = self._get_job_and_k8s_name(job_id)
        if not job:
            return EvaluateResponse(generations=[], scores={})
        if not k8s_name:
            return EvaluateResponse(generations=[], scores={})

        # Get custom resource
        cr = self._get_k8s_cr(k8s_name)
        if not cr:
            return EvaluateResponse(
                generations=[],
                scores={},
                metadata={"error": "Failed to get custom resource"},
            )

        # Extract status information
        status = cr.get("status", {})
        state = status.get("state", "")
        reason = status.get("reason", "")
        message = status.get("message", "")

        # Check if job is complete
        if state != "Complete":
            logger.warning("Job %s is not complete yet (state: %s)", job_id, state)
            return EvaluateResponse(
                generations=[],
                scores={},
                metadata={"state": state, "message": message or "Job not complete"},
            )

        # Get results JSON
        results_str = status.get("results", "")
        if not results_str:
            logger.warning("No results found in job %s", job_id)
            return EvaluateResponse(
                generations=[],
                scores={},
                metadata={
                    "state": state,
                    "reason": reason,
                    "message": "No results found",
                },
            )

        try:
            # Parse results
            scores, generations, result_metadata = self._parse_evaluation_results(
                results_str
            )

            # Add status information to metadata
            metadata = {
                "state": state,
                "reason": reason,
                "message": message,
                **result_metadata,
            }

            # Update job status
            job.status = JobStatus.completed

            return EvaluateResponse(
                generations=generations, scores=scores, metadata=metadata
            )

        except Exception as e:
            logger.error("Error retrieving job results: %s", e)
            return EvaluateResponse(
                generations=[], scores={}, metadata={"error": str(e)}
            )

    async def shutdown(self) -> None:
        """Clean up resources when shutting down."""
        logger.info("Shutting down LMEval provider")
        if self._k8s_client:
            self._k8s_client.close()
            logger.debug("Closed Kubernetes client connection")
