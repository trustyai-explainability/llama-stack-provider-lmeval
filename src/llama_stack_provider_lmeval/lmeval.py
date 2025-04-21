from __future__ import annotations

import logging
import json
import yaml
from datetime import time
from typing import Any, Dict, List, Optional

from kubernetes import client as k8s_client
from kubernetes import config as k8s_config
from kubernetes.client.rest import ApiException
from pydantic import BaseModel, Field

from llama_stack.apis.benchmarks import Benchmark, ListBenchmarksResponse
from llama_stack.apis.common.job_types import Job, JobStatus
from llama_stack.apis.eval import BenchmarkConfig, Eval, EvaluateResponse
from llama_stack.providers.datatypes import BenchmarksProtocolPrivate
from .config import LMEvalEvalProviderConfig
from .errors import LMEvalConfigError

logger = logging.getLogger(__name__)


class ModelArg(BaseModel):
    """Model argument for the LMEval CR."""

    name: str
    value: str


class ContainerConfig(BaseModel):
    """Container configuration for the LMEval CR."""

    env: Optional[List[Dict[str, str]]] = None


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

    def model_dump(self, **kwargs):
        result = super().model_dump(**kwargs)

        if "taskList" in result:
            task_list = result["taskList"]
            if "customTasks" in task_list and task_list["customTasks"] is None:
                del task_list["customTasks"]

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

    def __init__(self, namespace: str = "default", service_account: Optional[str] = None):
        """Initialize the LMEvalCRBuilder.

        Args:
            namespace: The Kubernetes namespace to use
            service_account: Optional service account to use for the LMEval Custom Resource
        """
        self._namespace = namespace
        self._service_account = service_account
        self._config = None

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
        # Model information
        eval_candidate = task_config.eval_candidate
        if eval_candidate.type != "model":
            # FIXME: Support other candidate types?
            raise LMEvalConfigError("LMEval only supports model candidates for now")

        model_name = eval_candidate.model
        # FIXME: Unused
        sampling_params = eval_candidate.sampling_params

        # Create model args from the configuration
        model_args = [
            ModelArg(name="model", value=model_name),
        ]

        if base_url:
            base_url = base_url.rstrip('/')
            openai_base_url = f"{base_url}/v1/openai/v1/completions"
            model_args.append(ModelArg(name="base_url", value=openai_base_url))

        model_args.append(ModelArg(name="tokenizer", value="google/flan-t5-base"))
        # FIXME: batch_size is duplicated
        # model_args.append(ModelArg(name="batch_size", value="auto"))
        model_args.append(ModelArg(name="num_concurrent", value="3"))

        env_vars = []
        if hasattr(task_config, "env_vars") and task_config.env_vars:
            env_vars = task_config.env_vars

        # Get environment variables from metadata
        if hasattr(task_config, "metadata") and task_config.metadata:
            metadata_env = task_config.metadata.get("env")
            if metadata_env and isinstance(metadata_env, dict):
                logger.info(f"Found environment variables in metadata: {metadata_env}")
                for key, value in metadata_env.items():
                    env_vars.append({"name": key, "value": str(value)})
                    logger.info(f"Added environment variable from metadata: {key}={value}")

        # Get environment variables from stored benchmark metadata
        if (
            not env_vars and
            stored_benchmark and
            hasattr(stored_benchmark, "metadata") and
            stored_benchmark.metadata
        ):
            metadata_env = stored_benchmark.metadata.get("env")
            if metadata_env and isinstance(metadata_env, dict):
                logger.info(f"Found environment variables in stored benchmark metadata: {metadata_env}")
                for key, value in metadata_env.items():
                    env_vars.append({"name": key, "value": str(value)})
                    logger.info(f"Added environment variable from stored benchmark metadata: {key}={value}")

        # FIXME: Improve this
        if "::" in benchmark_id:
            task_name = benchmark_id.split("::")[-1]

        # Generate timestamp-based uid
        import time

        job_id = int(time.time() * 1000)

        pod_config = None
        if env_vars or self._service_account:
            # Add environment variables to the container config
            container_config = ContainerConfig(
                env=[{"name": e["name"], "value": e["value"]} for e in env_vars] if env_vars else None
            )
            # Add service account to the pod config
            pod_config = PodConfig(container=container_config, serviceAccountName=self._service_account)
            # FIXME: Remove debug logging
            if env_vars:
                logger.info(f"Setting pod environment variables: {json.dumps(env_vars, indent=2)}")

        custom_tasks = None

        if hasattr(task_config, "metadata") and task_config.metadata:
            custom_task_data = task_config.metadata.get("custom_task")
            if custom_task_data and isinstance(custom_task_data, dict):
                logger.info(f"Found custom_task in task_config metadata: {custom_task_data}")
                git_data = custom_task_data.get("git")
                if git_data and isinstance(git_data, dict):
                    git_url = git_data.get("url")
                    if git_url:
                        logger.info(
                            f"Setting up GitSource with URL: {git_url}, branch: {git_data.get('branch')}, commit: {git_data.get('commit')}, path: {git_data.get('path')}"
                        )

                        git_source = GitSource(
                            url=git_url,
                            branch=git_data.get("branch"),
                            commit=git_data.get("commit"),
                            path=git_data.get("path"),
                        )
                        custom_task_source = CustomTaskSource(git=git_source)
                        custom_tasks = CustomTasks(source=custom_task_source)
                        logger.info(f"Added custom tasks from git source: {git_url}")

        if (
            not custom_tasks
            and stored_benchmark
            and hasattr(stored_benchmark, "metadata")
            and stored_benchmark.metadata
        ):
            custom_task_data = stored_benchmark.metadata.get("custom_task")
            if custom_task_data and isinstance(custom_task_data, dict):
                logger.info(f"Found custom_task in stored benchmark metadata: {custom_task_data}")
                git_data = custom_task_data.get("git")
                if git_data and isinstance(git_data, dict):
                    git_url = git_data.get("url")
                    if git_url:
                        logger.info(f"Setting up GitSource from stored benchmark with URL: {git_url}")
                        git_source = GitSource(
                            url=git_url,
                            branch=git_data.get("branch"),
                            commit=git_data.get("commit"),
                            path=git_data.get("path"),
                        )
                        custom_task_source = CustomTaskSource(git=git_source)
                        custom_tasks = CustomTasks(source=custom_task_source)
                        logger.info(f"Added custom tasks from stored benchmark: {git_url}")

        task_list_params = {"taskNames": [task_name]}
        if custom_tasks:
            logger.info(f"Including customTasks in CR with source: {custom_tasks.source}")
            task_list_params["customTasks"] = custom_tasks
            logger.info(f"TaskList parameters: {task_list_params}")
            logger.info(f"customTasks object: {custom_tasks.model_dump()}")

        task_list = TaskList(**task_list_params)
        logger.info(f"Final TaskList object: {task_list.model_dump()}")

        spec_params = {
            "taskList": task_list,
            "modelArgs": model_args,
            "pod": pod_config,
        }

        if limit is not None:
            spec_params["limit"] = limit

        # Pass environment variables from metadata.env to benchmark_config
        if (
            hasattr(task_config, "metadata") and
            task_config.metadata and
            "env" in task_config.metadata
        ):
            env_data = task_config.metadata.get("env", {})
            if isinstance(env_data, dict):
                # Initialize
                if not hasattr(task_config, "env_vars") or task_config.env_vars is None:
                    task_config.env_vars = []

                for key, value in env_data.items():
                    task_config.env_vars.append({"name": key, "value": str(value)})
                logger.info(f"Added environment variables from metadata.env to task_config")

        # FIXME: Assert task_name is valid and non-empty
        cr = LMEvalCR(
            metadata=LMEvalMetadata(name=f"lmeval-llama-stack-job-{job_id}", namespace=self._namespace),
            spec=LMEvalSpec(**spec_params),
        )

        cr_dict = cr.model_dump()

        logger.info(
            f"Benchmark config metadata before CR creation: {getattr(task_config, 'metadata', {})}"
        )
        logger.info(f"Task list params before CR creation: {task_list_params}")
        logger.info(f"Final CR before manipulation: {json.dumps(cr_dict, indent=2)}")

        # Check if we have custom task data from the task_config
        git_source_data = None
        if hasattr(task_config, "metadata") and task_config.metadata and "custom_task" in task_config.metadata:
            custom_task_data = task_config.metadata.get("custom_task", {})
            git_data = custom_task_data.get("git", {})
            if git_data and "url" in git_data:
                logger.info(f"Found git URL in task_config metadata: {git_data.get('url')}")
                git_source_data = {
                    "url": git_data.get("url"),
                    "branch": git_data.get("branch"),
                    "commit": git_data.get("commit"),
                    "path": git_data.get("path"),
                }

        # Check in stored_benchmark
        if not git_source_data and stored_benchmark and hasattr(stored_benchmark, "metadata") and stored_benchmark.metadata:
            custom_task_data = stored_benchmark.metadata.get("custom_task", {})
            git_data = custom_task_data.get("git", {})
            if git_data and "url" in git_data:
                logger.info(f"Found git URL in stored_benchmark metadata: {git_data.get('url')}")
                git_source_data = {
                    "url": git_data.get("url"),
                    "branch": git_data.get("branch"),
                    "commit": git_data.get("commit"),
                    "path": git_data.get("path"),
                }

        # Add custom tasks to the CR (if present)
        if git_source_data:
            logger.info(f"Adding customTasks to CR with git data: {git_source_data}")

            custom_tasks_section = {
                "source": {
                    "git": {}
                }
            }

            for key, value in git_source_data.items():
                if value is not None:
                    custom_tasks_section["source"]["git"][key] = value

            cr_dict["spec"]["taskList"]["customTasks"] = custom_tasks_section

            logger.info(f"Added customTasks to CR: {json.dumps(custom_tasks_section, indent=2)}")
        else:
            logger.warning("No git source data found for customTasks")

        logger.info(f"Final CR after customTasks processing: {json.dumps(cr_dict, indent=2)}")

        # if "pod" in cr_dict.get("spec", {}) and "container" in cr_dict["spec"]["pod"]:
        #     container = cr_dict["spec"]["pod"]["container"]
        #     if "env" in container and container["env"]:
        #         logger.info(f"Environment variables in CR: {json.dumps(container['env'], indent=2)}")
        #     else:
        #         logger.info("No environment variables set in the CR")

        return cr_dict


class LMEval(Eval, BenchmarksProtocolPrivate):
    def __init__(self, config: LMEvalEvalProviderConfig):
        self._config = config
        logger.info(f"LMEval provider initialized with namespace: {getattr(self._config, 'namespace', 'default')}")
        logger.info(f"LMEval provider config values: {vars(self._config)}")
        self.benchmarks = {}
        self._jobs: List[Job] = []
        self._job_metadata = {}

        self._k8s_client = None
        self._k8s_custom_api = None
        self._namespace = getattr(self._config, "namespace", "default")
        logger.info(f"Initialized Kubernetes client with namespace: {self._namespace}")
        if self.use_k8s:
            self._init_k8s_client()
            self._cr_builder = LMEvalCRBuilder(
                namespace=self._namespace, service_account=getattr(self._config, "service_account", None)
            )
            self._cr_builder._config = self._config

    def _init_k8s_client(self):
        """Initialize the Kubernetes client."""
        # FIXME: Support in-cluster kubeconfig only?
        try:
            k8s_config.load_incluster_config()
            logger.info("Loaded Kubernetes config from within the cluster")
        except k8s_config.ConfigException:
            try:
                k8s_config.load_kube_config()
                logger.info("Loaded Kubernetes config from kubeconfig file")
            except k8s_config.ConfigException as e:
                logger.error(f"Failed to load Kubernetes config: {e}")
                raise LMEvalConfigError(f"Failed to initialize Kubernetes client: {e}")

        self._k8s_client = k8s_client.ApiClient()
        self._k8s_custom_api = k8s_client.CustomObjectsApi(self._k8s_client)

    @property
    def use_k8s(self) -> bool:
        """Check if K8s mode is enabled."""
        return getattr(self._config, "use_k8s", True)

    async def initialize(self):
        logger.info("Initializing Base LMEval")
        print("Initializing Base LMEval")

    async def _register_bundled_benchmarks(self):
        """Register bundled benchmarks from lm-evaluation-harness."""
        bundled_benchmarks = self._get_benchmarks()

        for benchmark in bundled_benchmarks:
            await self.register_benchmark(benchmark)
            logger.info(f"Registered bundled benchmark: {benchmark.identifier}")

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
            # FIXME: Remove debug logging
            logger.info(f"Retrieved benchmark {benchmark_id} with metadata: {benchmark.metadata}")
            if "custom_task" in benchmark.metadata:
                logger.info(f"Benchmark {benchmark_id} has custom_task: {benchmark.metadata.get('custom_task')}")
        return benchmark

    async def register_benchmark(self, benchmark: Benchmark) -> None:
        """Register a benchmark for evaluation.

        Args:
            benchmark: The benchmark to register
        """
        logger.info(f"Registering benchmark: {benchmark.identifier}")
        # FIXME: Remove debug logging
        if hasattr(benchmark, "metadata") and benchmark.metadata:
            logger.info(f"Registering benchmark with metadata: {benchmark.metadata}")
            if "custom_task" in benchmark.metadata:
                logger.info(f"Benchmark has custom_task: {benchmark.metadata.get('custom_task')}")
        self.benchmarks[benchmark.identifier] = benchmark

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
        if self.use_k8s:
            if not isinstance(benchmark_config, BenchmarkConfig):
                raise LMEvalConfigError("K8s mode requires BenchmarkConfig")

            # FIXME: Remove debug logging
            logger.info(f"Running evaluation for benchmark {benchmark_id}")

            stored_benchmark = await self.get_benchmark(benchmark_id)
            if stored_benchmark:
                logger.info(f"Retrieved benchmark from storage: {stored_benchmark.identifier}")
                logger.info(f"Stored benchmark metadata: {stored_benchmark.metadata}")
                if "custom_task" in stored_benchmark.metadata:
                    logger.info(f"Stored benchmark has custom_task: {stored_benchmark.metadata.get('custom_task')}")
                else:
                    logger.warning(f"Stored benchmark does NOT have custom_task in metadata")
            else:
                logger.warning(f"Benchmark {benchmark_id} not found in storage")

            if hasattr(benchmark_config, "metadata") and benchmark_config.metadata:
                logger.info(
                    f"Benchmark config from request contains metadata: {json.dumps(benchmark_config.metadata, indent=2)}"
                )

                if "env" in benchmark_config.metadata:
                    env_data = benchmark_config.metadata.get("env")
                    if isinstance(env_data, dict):
                        logger.info(f"Request includes environment variables: {json.dumps(env_data, indent=2)}")
                    else:
                        logger.warning("Request includes env section but it's not a valid dictionary")

                if "custom_task" in benchmark_config.metadata:
                    logger.info(f"Request includes custom_task: {benchmark_config.metadata.get('custom_task')}")
                    custom_task = benchmark_config.metadata.get("custom_task")
                    if isinstance(custom_task, dict) and "git" in custom_task:
                        git_data = custom_task.get("git")
                        if isinstance(git_data, dict) and "url" in git_data:
                            logger.info(f"Request includes custom_task with git URL: {git_data.get('url')}")
                        else:
                            logger.warning("Request includes custom_task with git section but missing required URL")
                    else:
                        logger.warning("Request includes custom_task but missing or invalid git section")
                else:
                    logger.info("No custom_task found in request metadata")
            else:
                logger.info("No metadata provided in request benchmark config")

            config_limit = None
            if hasattr(benchmark_config, "num_examples") and benchmark_config.num_examples is not None:
                config_limit = str(benchmark_config.num_examples)
                logger.info(f"Using example limit from config: {config_limit}")

            if (
                hasattr(benchmark_config, "metadata") and
                benchmark_config.metadata and
                "env" in benchmark_config.metadata
            ):
                env_data = benchmark_config.metadata.get("env", {})
                if isinstance(env_data, dict):
                    # Initialize
                    if not hasattr(benchmark_config, "env_vars") or benchmark_config.env_vars is None:
                        benchmark_config.env_vars = []

                    for key, value in env_data.items():
                        benchmark_config.env_vars.append({"name": key, "value": str(value)})
                    logger.info(f"Added environment variables from metadata.env to benchmark_config")

            cr = self._cr_builder.create_cr(
                benchmark_id=benchmark_id,
                task_config=benchmark_config,
                base_url=getattr(self._config, "base_url", None),
                limit=config_limit,
                stored_benchmark=stored_benchmark,
            )

            logger.info(f"Generated LMEval CR for benchmark {benchmark_id}")

            logger.info(f"CR structure: {json.dumps(cr, indent=2)}")

            # FIXME: Remove debug logging
            # if "pod" in cr.get("spec", {}) and "container" in cr["spec"]["pod"]:
            #     container = cr["spec"]["pod"]["container"]
            #     if "env" in container and container["env"]:
            #         logger.info(f"Environment variables in CR: {json.dumps(container['env'], indent=2)}")
            #     else:
            #         logger.info("No environment variables set in the CR")

            task_list = cr.get("spec", {}).get("taskList", {})
            if "customTasks" in task_list:
                logger.info(f"CR includes customTasks section: {json.dumps(task_list['customTasks'], indent=2)}")
            else:
                logger.warning("CR does not include customTasks section")

            # Print the full CR JSON for debugging
            logger.info(f"Full CR JSON: {json.dumps(cr, indent=2)}")

            logger.info(f"Full CR YAML: \n{yaml.dump(cr, default_flow_style=False)}")

            _job_id = f"lmeval-job-{len(self._jobs)}"

            _job = Job(job_id=_job_id, status=JobStatus.scheduled, metadata={"created_at": str(time())})
            self._jobs.append(_job)

            self._job_metadata[_job_id] = {}

            # Deploy LMEvalJob
            try:
                group = "trustyai.opendatahub.io"
                version = "v1alpha1"
                plural = "lmevaljobs"

                logger.info(f"Deploying LMEval CR to Kubernetes namespace: {self._namespace}")

                logger.info(f"Full CR being submitted to Kubernetes: \n{yaml.dump(cr, default_flow_style=False)}")

                response = self._k8s_custom_api.create_namespaced_custom_object(
                    group=group, version=version, namespace=self._namespace, plural=plural, body=cr
                )

                logger.info(f"Successfully deployed LMEval CR to Kubernetes: {response['metadata']['name']}")
                logger.info(f"CR creation response: {json.dumps(response, indent=2)}")

                # Store the k8s resource name in our metadata store
                self._job_metadata[_job_id]["k8s_name"] = cr["metadata"]["name"]

            except ApiException as e:
                logger.error(f"Failed to deploy LMEval CR to Kubernetes: {e}")
                _job.status = JobStatus.failed
                self._job_metadata[_job_id]["error"] = str(e)
                raise LMEvalConfigError(f"Failed to deploy LMEval CR: {e}")

            return {"job_id": _job_id}
        else:
            # TODO: Handle non-K8s evaluation
            raise NotImplementedError("Non-K8s evaluation not implemented yet")

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
        if self.use_k8s:
            # FIXME: Placeholder
            from llama_stack.apis.scoring import ScoringResult

            # FIXME: Placeholder
            generations = []
            for row in input_rows:
                generation = {**row, "generated_answer": "Placeholder answer from LMEval"}
                generations.append(generation)

            scores = {}
            for scoring_fn in scoring_functions:
                score_rows = [{"score": 0.5} for _ in input_rows]
                scores[scoring_fn] = ScoringResult(aggregated_results={"accuracy": 0.5}, score_rows=score_rows)

            return EvaluateResponse(generations=generations, scores=scores)
        else:
            raise NotImplementedError("Non-K8s evaluation not implemented yet")

    async def job_status(self, benchmark_id: str, job_id: str) -> Optional[Dict[str, str]]:
        """Get the status of a running evaluation job.

        Args:
            benchmark_id: The benchmark id
            job_id: The job id

        Returns:
            Dict with current status of the job
        """
        if self.use_k8s:
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
                    group=group, version=version, namespace=self._namespace, plural=plural, name=k8s_name
                )

                status = cr.get("status", {})
                state = status.get("state", "")
                reason = status.get("reason", "")
                message = status.get("message", "")

                logger.info(f"Job {job_id} status: state={state}, reason={reason}, message={message}")

                job_status = JobStatus.scheduled
                if state == "Complete":
                    if reason == "Failed":
                        job_status = JobStatus.failed
                    else:
                        job_status = JobStatus.completed
                elif state == "Running":
                    job_status = JobStatus.in_progress
                elif state == "Pending" or state == "Scheduled":
                    job_status = JobStatus.scheduled
                elif state == "Cancelled":
                    job_status = JobStatus.cancelled

                # Update the job status
                job.status = job_status

                return {"job_id": job_id, "status": job_status}

            except ApiException as e:
                logger.error(f"Failed to get job status from Kubernetes: {e}")
                return {"job_id": job_id, "status": JobStatus.scheduled}
        else:
            raise NotImplementedError("Non-K8s evaluation not implemented yet")

    async def job_cancel(self, benchmark_id: str, job_id: str) -> None:
        """Cancel a running evaluation job.

        Args:
            benchmark_id: The benchmark identifier
            job_id: The job identifier
        """
        if self.use_k8s:
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
                    group=group, version=version, namespace=self._namespace, plural=plural, name=k8s_name
                )

                # Update job status
                job.status = JobStatus.cancelled
                logger.info(f"Successfully cancelled job {job_id} (K8s resource: {k8s_name})")

            except ApiException as e:
                logger.error(f"Failed to cancel job in Kubernetes: {e}")
                raise LMEvalConfigError(f"Failed to cancel job: {e}")
        else:
            raise NotImplementedError("Non-K8s evaluation not implemented yet")

    async def job_result(self, benchmark_id: str, job_id: str) -> EvaluateResponse:
        """Get the results of a completed evaluation job.

        Args:
            benchmark_id: The benchmark id
            job_id: The job id

        Returns:
            EvaluateResponse: Results of the evaluation
        """
        if self.use_k8s:
            job = next((j for j in self._jobs if j.job_id == job_id), None)
            if not job:
                logger.warning(f"Job {job_id} not found")
                return EvaluateResponse(generations=[], scores={})

            try:
                # Get metadata from our separate store
                job_metadata = self._job_metadata.get(job_id, {})
                k8s_name = job_metadata.get("k8s_name")
                if not k8s_name:
                    logger.warning(f"No K8s resource name found for job {job_id}")
                    return EvaluateResponse(generations=[], scores={})

                group = "trustyai.opendatahub.io"
                version = "v1alpha1"
                plural = "lmevaljobs"
                cr = self._k8s_custom_api.get_namespaced_custom_object(
                    group=group, version=version, namespace=self._namespace, plural=plural, name=k8s_name
                )

                status = cr.get("status", {})

                state = status.get("state", "")
                if state != "Complete":
                    logger.warning(f"Job {job_id} is not complete yet (state: {state})")
                    return EvaluateResponse(
                        generations=[],
                        scores={},
                        metadata={"state": state, "message": status.get("message", "Job not complete")},
                    )

                # Extract the results JSON from the status
                results_str = status.get("results", "")
                if not results_str:
                    logger.warning(f"No results found in job {job_id}")
                    return EvaluateResponse(
                        generations=[],
                        scores={},
                        metadata={"state": state, "reason": status.get("reason", ""), "message": "No results found"},
                    )

                import json
                try:
                    results = json.loads(results_str)
                    logger.info(f"Successfully parsed results JSON for job {job_id}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse results JSON: {e}")
                    return EvaluateResponse(
                        generations=[], scores={}, metadata={"error": f"Invalid JSON in results field: {e}"}
                    )

                from llama_stack.apis.scoring import ScoringResult

                scores = {}
                if "results" in results and isinstance(results["results"], dict):
                    for task_name, metrics in results["results"].items():
                        if not isinstance(metrics, dict):
                            continue

                        for metric_key, metric_value in metrics.items():
                            if not isinstance(metric_value, (int, float)) or metric_key == "alias":
                                continue

                            if "," in metric_key:
                                metric_name = metric_key.split(",")[0]
                            else:
                                metric_name = metric_key

                            score_rows = [{"score": metric_value}]

                            scores[f"{task_name}:{metric_name}"] = ScoringResult(
                                aggregated_results={metric_name: metric_value}, score_rows=score_rows
                            )

                generations = []
                if "samples" in results and isinstance(results["samples"], list):
                    for sample in results["samples"]:
                        generation = {"generated_answer": sample.get("output", "")}
                        if "input" in sample:
                            generation["input"] = sample["input"]
                        generations.append(generation)

                metadata = {
                    "state": state,
                    "reason": status.get("reason", ""),
                    "message": status.get("message", ""),
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

                job.status = JobStatus.completed

                return EvaluateResponse(generations=generations, scores=scores, metadata=metadata)

            except Exception as e:
                logger.error(f"Error retrieving job results: {e}")
                return EvaluateResponse(generations=[], scores={}, metadata={"error": str(e)})
        else:
            return EvaluateResponse(
                generations=[], scores={}, metadata={"error": "Non-K8s implementation not available"}
            )

    async def shutdown(self) -> None:
        """Clean up resources when shutting down."""
        logger.info("Shutting down LMEval provider")
        if self._k8s_client:
            self._k8s_client.close()
            logger.info("Closed Kubernetes client connection")

async def get_adapter_impl(config: LMEvalEvalProviderConfig, _deps):
    return LMEval(config)
