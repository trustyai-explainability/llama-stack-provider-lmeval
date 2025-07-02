from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


class ModelArg(BaseModel):
    """Model argument for the LMEval CR."""

    name: str
    value: str


class ContainerConfig(BaseModel):
    """Container configuration for the LMEval CR."""

    env: Optional[List[Dict[str, Any]]] = None


class PodConfig(BaseModel):
    """Pod configuration for the LMEval CR."""

    container: ContainerConfig
    serviceAccountName: Optional[str] = None


class GitSource(BaseModel):
    """Git source for custom tasks."""

    url: str
    branch: Optional[str] = None
    commit: Optional[str] = None
    path: Optional[str] = None

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

    taskNames: List[str]
    customTasks: Optional[CustomTasks] = None

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
    limit: Optional[str] = None
    modelArgs: List[ModelArg]
    pod: Optional[PodConfig] = None
    offline: Optional[Dict[str, Any]] = None

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

    def __init__(
        self, namespace: str = "default", service_account: Optional[str] = None
    ):
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
            return f"{cleaned_url}/openai/v1/completions"
        else:
            return f"{cleaned_url}/v1/openai/v1/completions"

    def _create_model_args(
        self, model_name: str, base_url: Optional[str] = None
    ) -> List[ModelArg]:
        """Create model arguments for the CR.

        Args:
            model_name: The model name to use
            base_url: Optional base URL for the model service

        Returns:
            List of ModelArg objects
        """
        model_args = [ModelArg(name="model", value=model_name)]

        if base_url:
            openai_base_url = self._build_openai_url(base_url)
            model_args.append(ModelArg(name="base_url", value=openai_base_url))

        # Add TLS configuration (if present in config)
        if hasattr(self._config, "tls") and self._config.tls is not None:
            tls_value = str(self._config.tls)
            logger.debug(
                f"Adding TLS configuration to CR: verify_certificate={tls_value}"
            )
            model_args.append(ModelArg(name="verify_certificate", value=tls_value))

        model_args.append(ModelArg(name="num_concurrent", value="1"))

        return model_args

    def _collect_env_vars(
        self, task_config: BenchmarkConfig, stored_benchmark: Optional[Benchmark]
    ) -> List[Dict[str, Any]]:
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
                logger.debug(f"Found environment variables in metadata: {metadata_env}")
                for key, value in metadata_env.items():
                    if isinstance(value, dict) and "secret" in value:
                        # Handle Kubernetes secret reference structure
                        env_vars.append({"name": key, "secret": value["secret"]})
                        logger.debug(
                            f"Added secret environment variable from metadata: {key}"
                        )
                    else:
                        # Handle simple string value
                        env_vars.append({"name": key, "value": str(value)})
                        logger.debug(
                            f"Added environment variable from metadata: {key}"
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
                    f"Found environment variables in stored benchmark metadata: {metadata_env}"
                )
                for key, value in metadata_env.items():
                    if isinstance(value, dict) and "secret" in value:
                        # Handle Kubernetes secret reference structure
                        env_vars.append({"name": key, "secret": value["secret"]})
                        logger.debug(
                            f"Added secret environment variable from stored benchmark metadata: {key}"
                        )
                    else:
                        # Handle simple string value
                        env_vars.append({"name": key, "value": str(value)})
                        logger.debug(
                            f"Added environment variable from stored benchmark metadata: {key}"
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

    def _create_pod_config(self, env_vars: List[Dict[str, Any]]) -> Optional[PodConfig]:
        """Create pod configuration with environment variables.

        Args:
            env_vars: List of environment variables

        Returns:
            PodConfig object or None if no config needed
        """
        if not env_vars and not self._service_account:
            return None

        # Process environment variables to handle both simple values and secret references
        processed_env_vars = []
        for env_var in env_vars:
            env_entry = {"name": env_var["name"]}
            
            # Check if this env var has a secret reference
            if "secret" in env_var and env_var["secret"]:
                # Custom secret structure: name/value/secret
                secret_ref = env_var["secret"]
                if isinstance(secret_ref, dict) and "name" in secret_ref and "key" in secret_ref:
                    env_entry["valueFrom"] = {
                        "secretKeyRef": {
                            "name": secret_ref["name"],
                            "key": secret_ref["key"]
                        }
                    }
                else:
                    # Invalid secret structure, fall back to simple value
                    logger.warning(
                        f"Invalid secret structure for env var '{env_var.get('name', '<unknown>')}'. "
                        "Expected a dict with 'name' and 'key'. Falling back to simple value."
                    )
                    env_entry["value"] = str(env_var.get("value", ""))
            else:
                # Handle value field (simple or complex structures)
                value = env_var.get("value")
                if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
                    # A stringified dict, parse it
                    try:
                        import ast
                        parsed_value = ast.literal_eval(value)
                        if isinstance(parsed_value, dict) and "valueFrom" in parsed_value:
                            # Use the parsed valueFrom structure
                            env_entry.update(parsed_value)
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

        # Add environment variables to the container config
        container_config = ContainerConfig(env=processed_env_vars or None)

        # Add service account to the pod config
        pod_config = PodConfig(
            container=container_config, serviceAccountName=self._service_account
        )

        if env_vars:
            env_var_names = [env_var["name"] for env_var in processed_env_vars]
            logger.debug(
                f"Setting pod environment variables: {', '.join(env_var_names)}"
            )

        return pod_config

    def _extract_git_source(
        self, task_config: BenchmarkConfig, stored_benchmark: Optional[Benchmark]
    ) -> Optional[Dict[str, Any]]:
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
                    f"Found git URL in task_config metadata: {git_data.get('url')}"
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
                    f"Found git URL in stored_benchmark metadata: {git_data.get('url')}"
                )
                return {
                    "url": git_data.get("url"),
                    "branch": git_data.get("branch"),
                    "commit": git_data.get("commit"),
                    "path": git_data.get("path"),
                }

        return None

    def _extract_pvc_name(
        self, task_config: BenchmarkConfig, stored_benchmark: Optional[Benchmark]
    ) -> Optional[str]:
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
                            f"Found PVC name in task_config metadata: {pvc_name}"
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
                            f"Found PVC name in stored_benchmark metadata: {pvc_name}"
                        )
                        return pvc_name

        return None

    def create_cr(
        self,
        benchmark_id: str,
        task_config: BenchmarkConfig,
        base_url: Optional[str] = None,
        limit: Optional[str] = None,
        stored_benchmark: Optional[Benchmark] = None,
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
        # Validate model candidate
        eval_candidate = task_config.eval_candidate
        if eval_candidate.type != "model":
            raise LMEvalConfigError("LMEval only supports model candidates for now")

        # Create model args
        model_args = self._create_model_args(eval_candidate.model, base_url)

        if (
            hasattr(stored_benchmark, "metadata")
            and stored_benchmark.metadata
            and "tokenizer" in stored_benchmark.metadata
        ):
            tokenizer_value = stored_benchmark.metadata.get("tokenizer")
            if isinstance(tokenizer_value, str) and tokenizer_value:
                logger.debug(f"Using custom tokenizer from metadata: {tokenizer_value}")
                model_args.append(ModelArg(name="tokenizer", value=tokenizer_value))

        # Add tokenized_requests parameter if present in metadata
        if (
            hasattr(stored_benchmark, "metadata")
            and stored_benchmark.metadata
            and "tokenized_requests" in stored_benchmark.metadata
        ):
            tokenized_requests_value = stored_benchmark.metadata.get("tokenized_requests")
            if isinstance(tokenized_requests_value, (bool, str)) and tokenized_requests_value is not None:
                value_str = str(tokenized_requests_value)
                logger.debug(f"Using tokenized_requests from metadata: {value_str}")
                model_args.append(ModelArg(name="tokenized_requests", value=value_str))

        # Collect environment variables
        env_vars = self._collect_env_vars(task_config, stored_benchmark)

        # Extract task name
        task_name = self._extract_task_name(benchmark_id)

        # Create pod config
        pod_config = self._create_pod_config(env_vars)

        # Generate UUID-based id
        import uuid

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
            logger.info(f"Setting up offline storage with PVC: {pvc_name}")
            if "offline" in cr_dict["spec"] and cr_dict["spec"]["offline"] is None:
                logger.warning("Removing null offline field from CR spec")
                del cr_dict["spec"]["offline"]

            cr_dict["spec"]["offline"] = {"storage": {"pvcName": pvc_name}}
            logger.debug(f"Added offline storage to CR with PVC: {pvc_name}")

        # Add custom tasks to CR if git source data is available
        if git_source_data:
            logger.info(f"Adding customTasks to CR with git data: {git_source_data}")

            custom_tasks_section = {"source": {"git": {}}}

            for key, value in git_source_data.items():
                if value is not None:
                    custom_tasks_section["source"]["git"][key] = value

            cr_dict["spec"]["taskList"]["customTasks"] = custom_tasks_section

            logger.debug(
                f"Added customTasks to CR: {json.dumps(custom_tasks_section, indent=2)}"
            )
        else:
            logger.warning("No git source data found for customTasks")

        logger.debug(f"Final LMEval Custom Resource: {json.dumps(cr_dict, indent=2)}")

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
        logger.debug(f"Using namespace from provider config: {config.namespace}")
        return config.namespace.strip()

    # Check from environment variable
    env_namespace = os.getenv('TRUSTYAI_LM_EVAL_NAMESPACE')  # noqa: Q000
    if env_namespace:
        logger.debug(f"Using namespace from environment variable: {env_namespace}")
        return env_namespace

    # Check from service account namespace file
    service_account_namespace_path = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
    if Path(service_account_namespace_path).exists():
        try:
            with open(service_account_namespace_path, 'r') as f:
                namespace = f.read().strip()
                if namespace:
                    logger.debug(f"Using namespace from service account: {namespace}")
                    return namespace
        except OSError as e:
            logger.warning(f"Failed to read namespace from service account: {e}")

    # Check for POD_NAMESPACE environment variable
    pod_namespace = os.getenv("POD_NAMESPACE")
    if pod_namespace:
        logger.debug(f"Using namespace from POD_NAMESPACE environment variable: {pod_namespace}")
        return pod_namespace

    # Check for NAMESPACE environment variable
    alt_namespace = os.getenv('NAMESPACE')  # noqa: Q000
    if alt_namespace:
        logger.debug(f"Using namespace from NAMESPACE environment variable: {alt_namespace}")
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
    def __init__(self, config: LMEvalEvalProviderConfig):
        self._config = config
        
        self._namespace = _resolve_namespace(self._config)
        
        logger.debug(
            f"LMEval provider initialized with namespace: {self._namespace}"
        )
        logger.debug(f"LMEval provider config values: {vars(self._config)}")
        self.benchmarks = {}
        self._jobs: List[Job] = []
        self._job_metadata = {}

        self._k8s_client = None
        self._k8s_custom_api = None
        if self.use_k8s:
            self._init_k8s_client()
            logger.debug(f"Initialized Kubernetes client with namespace: {self._namespace}")
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
                logger.error(f"Failed to load Kubernetes config: {e}")
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

    async def get_benchmark(self, benchmark_id: str) -> Optional[Benchmark]:
        """Get a specific benchmark by ID.

        Args:
            benchmark_id: The benchmark identifier

        Returns:
            Optional[Benchmark]: The benchmark if found, None otherwise
        """
        benchmark = self.benchmarks.get(benchmark_id)
        if benchmark:
            logger.debug(
                f"Retrieved benchmark {benchmark_id} with metadata: {benchmark.metadata}"
            )
            if "custom_task" in benchmark.metadata:
                logger.debug(
                    f"Benchmark {benchmark_id} has custom_task: {benchmark.metadata.get('custom_task')}"
                )
        return benchmark

    async def register_benchmark(self, benchmark: Benchmark) -> None:
        """Register a benchmark for evaluation.

        Args:
            benchmark: The benchmark to register
        """
        logger.info(f"Registering benchmark: {benchmark.identifier}")
        # FIXME: Remove debug logging
        if hasattr(benchmark, "metadata") and benchmark.metadata:
            logger.debug(f"Registering benchmark with metadata: {benchmark.metadata}")
            if "custom_task" in benchmark.metadata:
                logger.debug(
                    f"Benchmark has custom_task: {benchmark.metadata.get('custom_task')}"
                )
        self.benchmarks[benchmark.identifier] = benchmark

    def _get_job_id(self) -> str:
        """Generate a unique job ID.

        Returns:
            Unique job ID string
        """
        import uuid

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
                    f"Ensured offline storage in CR before deployment: {pvc_name}"
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
                logger.debug(f"Re-added offline storage in YAML: {pvc_name}")

            cr_yaml = yaml.dump(cr_dict, default_flow_style=False)
            cr = cr_dict
        except Exception as e:
            logger.error(f"Error fixing YAML: {e}")

        logger.debug(
            f"Deploying LMEvalJob Custom Resource to Kubernetes namespace: {self._namespace}"
        )
        logger.debug(f"Full Custom Resource being submitted: \n{cr_yaml}")

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
                f"Successfully deployed LMEval CR to Kubernetes: {response['metadata']['name']}"
            )

            self._job_metadata[job_id]["k8s_name"] = cr["metadata"]["name"]

        except ApiException as e:
            logger.error(f"Failed to deploy LMEval CR to Kubernetes: {e}")
            raise LMEvalConfigError(f"Failed to deploy LMEval CR: {e}") from e

    def _process_benchmark_config(
        self, benchmark_config: BenchmarkConfig
    ) -> Optional[str]:
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
        logger.debug(f"Using example limit from config: {config_limit}")
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
    ) -> Dict[str, str]:
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
        logger.info(f"Running evaluation for benchmark {benchmark_id}")

        config_limit = self._process_benchmark_config(benchmark_config)
        self._process_environment_vars(benchmark_config)

        if hasattr(benchmark_config, "metadata") and benchmark_config.metadata:
            logger.debug(
                f"Benchmark config metadata: {json.dumps(benchmark_config.metadata, indent=2)}"
            )
            if (
                "input" in benchmark_config.metadata
                and "storage" in benchmark_config.metadata.get("input", {})
            ):
                logger.debug(
                    f"Found storage config in metadata: {benchmark_config.metadata['input']['storage']}"
                )

        cr = self._cr_builder.create_cr(
            benchmark_id=benchmark_id,
            task_config=benchmark_config,
            base_url=getattr(self._config, "base_url", None),
            limit=config_limit or limit,
            stored_benchmark=stored_benchmark,
        )

        logger.debug(
            f"Generated LMEvalJob Custom Resource for benchmark {benchmark_id}"
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
                logger.debug(f"Ensured offline storage in CR with PVC: {pvc_name}")

        job_id = self._get_job_id()
        from datetime import datetime

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
        input_rows: List[Dict[str, Any]],
        scoring_functions: List[str],
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

        from llama_stack.apis.scoring import ScoringResult

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

    async def job_status(
        self, benchmark_id: str, job_id: str
    ) -> Optional[Dict[str, str]]:
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
            logger.warning(f"Job {job_id} not found")
            return None

        try:
            job_metadata = self._job_metadata.get(job_id, {})
            k8s_name = job_metadata.get("k8s_name")
            if not k8s_name:
                logger.warning(f"No K8s resource name found for job {job_id}")
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
                f"Job {job_id} status: state={state}, reason={reason}, message={message}"
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
            logger.error(f"Failed to get job status from Kubernetes: {e}")
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
            logger.warning(f"Job {job_id} not found")
            return

        try:
            job_metadata = self._job_metadata.get(job_id, {})
            k8s_name = job_metadata.get("k8s_name")
            if not k8s_name:
                logger.warning(f"No K8s resource name found for job {job_id}")
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
                f"Successfully cancelled job {job_id} (Kubernetes resource: {k8s_name})"
            )

        except ApiException as e:
            logger.error(f"Failed to cancel job in Kubernetes: {e}")
            raise LMEvalConfigError(f"Failed to cancel job: {e}") from e

    def _get_job_and_k8s_name(self, job_id: str) -> Tuple[Optional[Job], Optional[str]]:
        """Get job and Kubernetes resource name.

        Args:
            job_id: The job ID

        Returns:
            (job, k8s_name)
        """
        job = next((j for j in self._jobs if j.job_id == job_id), None)
        if not job:
            logger.warning(f"Job {job_id} not found")
            return None, None

        job_metadata = self._job_metadata.get(job_id, {})
        k8s_name = job_metadata.get("k8s_name")
        if not k8s_name:
            logger.warning(f"No K8s resource name found for job {job_id}")
            return job, None

        return job, k8s_name

    def _get_k8s_cr(self, k8s_name: str) -> Optional[Dict[str, Any]]:
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
            logger.error(f"Failed to get custom resource: {e}")
            return None

    def _parse_evaluation_results(
        self, results_str: str
    ) -> Tuple[Dict[str, ScoringResult], List[Dict[str, Any]], Dict[str, Any]]:
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
            logger.error(f"Failed to parse results JSON: {e}")
            raise ValueError(f"Invalid JSON in results field: {e}") from e

        from llama_stack.apis.scoring import ScoringResult

        # Extract scores
        scores = {}
        if "results" in results and isinstance(results["results"], dict):
            for task_name, metrics in results["results"].items():
                if not isinstance(metrics, dict):
                    continue

                for metric_key, metric_value in metrics.items():
                    if (
                        not isinstance(metric_value, (int, float))
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
            logger.warning(f"Job {job_id} is not complete yet (state: {state})")
            return EvaluateResponse(
                generations=[],
                scores={},
                metadata={"state": state, "message": message or "Job not complete"},
            )

        # Get results JSON
        results_str = status.get("results", "")
        if not results_str:
            logger.warning(f"No results found in job {job_id}")
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
            logger.error(f"Error retrieving job results: {e}")
            return EvaluateResponse(
                generations=[], scores={}, metadata={"error": str(e)}
            )

    async def shutdown(self) -> None:
        """Clean up resources when shutting down."""
        logger.info("Shutting down LMEval provider")
        if self._k8s_client:
            self._k8s_client.close()
            logger.debug("Closed Kubernetes client connection")
