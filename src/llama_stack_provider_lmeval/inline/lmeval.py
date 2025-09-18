"""LMEval Inline Eval Provider implementation for Llama Stack."""

import asyncio
import json
import logging
import os
import signal
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from llama_stack.apis.benchmarks import Benchmark, ListBenchmarksResponse
from llama_stack.apis.common.job_types import Job, JobStatus
from llama_stack.apis.datatypes import Api
from llama_stack.apis.eval import BenchmarkConfig, Eval, EvaluateResponse
from llama_stack.apis.files import OpenAIFileObject, OpenAIFilePurpose, UploadFile
from llama_stack.apis.scoring import ScoringResult
from llama_stack.providers.datatypes import BenchmarksProtocolPrivate

from ..config import LMEvalEvalProviderConfig
from ..errors import LMEvalConfigError

logger = logging.getLogger(__name__)


class LMEvalInline(Eval, BenchmarksProtocolPrivate):
    """LMEval inline provider implementation."""

    def __init__(
        self, config: LMEvalEvalProviderConfig, deps: dict[Api, Any] | None = None
    ):
        self.config: LMEvalEvalProviderConfig = config
        self.benchmarks: dict[str, Benchmark] = {}
        self._jobs: list[Job] = []
        self._job_metadata: dict[str, dict[str, str]] = {}
        self.files_api = deps.get(Api.files) if deps else None

    async def initialize(self):
        "Initialize the LMEval Inline provider"
        if not self.files_api:
            raise LMEvalConfigError("Files API is not initialized")

    async def list_benchmarks(self) -> ListBenchmarksResponse:
        """List all registered benchmarks."""
        return ListBenchmarksResponse(data=list(self.benchmarks.values()))

    async def get_benchmark(self, benchmark_id: str) -> Benchmark | None:
        """Get a specific benchmark by ID."""
        benchmark = self.benchmarks.get(benchmark_id)
        return benchmark

    async def register_benchmark(self, benchmark: Benchmark) -> None:
        """Register a benchmark for evaluation."""
        self.benchmarks[benchmark.identifier] = benchmark

    def _get_job_id(self) -> str:
        """Generate a unique job ID."""
        return str(uuid.uuid4())

    async def run_eval(
        self, benchmark_id: str, benchmark_config: BenchmarkConfig, limit="2"
    ) -> Job:
        if not isinstance(benchmark_config, BenchmarkConfig):
            raise LMEvalConfigError("LMEval requires BenchmarkConfig")

        stored_benchmark = await self.get_benchmark(benchmark_id)

        logger.info("Running evaluation for benchmark: %s", stored_benchmark)

        if (
            not hasattr(benchmark_config, "num_examples")
            or benchmark_config.num_examples is None
        ):
            config_limit = None
        else:
            config_limit = str(benchmark_config.num_examples)

        job_output_results_dir: Path = self.config.results_dir
        job_output_results_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique job ID - use the same ID for both file naming and job tracking
        job_id = self._get_job_id()
        job_uuid = job_id.replace("-", "")

        try:
            cmd = self.build_command(
                benchmark_id=benchmark_id,
                task_config=benchmark_config,
                limit=config_limit or limit,
                stored_benchmark=stored_benchmark,
                job_output_results_dir=job_output_results_dir,
                job_uuid=job_uuid,
            )

            logger.debug("Generated command for benchmark: %s", benchmark_id)
            job = Job(
                job_id=job_id,
                status=JobStatus.scheduled,
                metadata={"created_at": datetime.now().isoformat(), "process_id": None},
            )
            self._jobs.append(job)
            self._job_metadata[job_id] = {}

            env = os.environ.copy()
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            self._job_metadata[job_id]["process_id"] = str(process.pid)

            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)

            if process.returncode == 0:
                # Log successful completion
                logger.info("Evaluation completed successfully for job %s", job_id)
                # Check if the result file exists and process it
                result_files = list(
                    job_output_results_dir.glob(f"job_{job_uuid}_results_*.json")
                )
                if result_files:
                    # Use the most recent file if multiple exist
                    actual_result_file = max(
                        result_files, key=lambda f: f.stat().st_mtime
                    )
                    logger.info("Found results file: %s", actual_result_file)

                    # Parse results from the local file that lm_eval wrote to
                    try:
                        # Read and parse the results file
                        with open(actual_result_file, encoding="utf-8") as f:
                            results_data = json.load(f)

                        # Parse the results using the existing method
                        parsed_results = await self._parse_job_results_from_data(
                            results_data, job_id
                        )
                        # Store the parsed results in job metadata
                        self._job_metadata[job_id]["results"] = parsed_results

                        # Upload the original lm_eval results file to Files API
                        upload_job_result: OpenAIFileObject = await self._upload_file(
                            actual_result_file, OpenAIFilePurpose.ASSISTANTS
                        )

                        if upload_job_result:
                            self._job_metadata[job_id]["uploaded_file"] = (
                                upload_job_result.id
                            )
                            logger.info(
                                "Uploaded job result file %s to Files API with ID: %s",
                                actual_result_file,
                                upload_job_result.id,
                            )
                        else:
                            logger.warning(
                                "Failed to upload job result file %s to Files API",
                                actual_result_file,
                            )

                        job.status = JobStatus.completed
                    except Exception as e:
                        logger.error(
                            "Failed to process results file for job %s: %s", job_id, e
                        )
                        job.status = JobStatus.failed
                        self._job_metadata[job_id]["error"] = (
                            f"Failed to process results: {str(e)}"
                        )
                else:
                    logger.warning(
                        "No results files found for job %s in directory %s",
                        job_id,
                        job_output_results_dir,
                    )
                    job.status = JobStatus.failed
                    self._job_metadata[job_id]["error"] = "Results file not found"
            else:
                logger.error(
                    "LM-Eval process failed with return code %d", process.returncode
                )
                logger.error("stdout: %s", stdout.decode("utf-8") if stdout else "")
                logger.error("stderr: %s", stderr.decode("utf-8") if stderr else "")
                job.status = JobStatus.failed
                self._job_metadata[job_id]["error"] = f"""
                    Process failed with return code {process.returncode}
                """
        except Exception as e:
            job.status = JobStatus.failed
            self._job_metadata[job_id]["error"] = str(e)
            logger.error("Job %s failed with error: %s", job_id, e)
            # Only terminate if process is still running
            if "process" in locals() and process and process.returncode is None:
                try:
                    process.terminate()
                except Exception as term_e:
                    logger.warning("Failed to terminate process: %s", term_e)
        finally:
            # Clean up any remaining process
            if "process" in locals() and process and process.returncode is None:
                process.kill()
                await process.wait()
            # Clean up job file
            self._cleanup_job_files(job_output_results_dir, job_uuid)

        return job

    def _cleanup_job_files(self, results_dir: Path, job_uuid: str) -> None:
        """Clean up result files for a specific job.

        Args:
            results_dir: The results directory
            job_uuid: The job UUID to clean up files for
        """
        try:
            # Find and remove all job files
            job_files = list(results_dir.glob(f"job_{job_uuid}_results*.json"))
            for file_path in job_files:
                try:
                    file_path.unlink()
                    logger.debug("Deleted job result file: %s", file_path)
                except OSError as e:
                    logger.warning(
                        "Failed to delete job result file %s: %s", file_path, e
                    )
        except Exception as e:
            logger.warning("Error during job file cleanup for %s: %s", job_uuid, e)

    async def _upload_file(
        self, file: Path, purpose: OpenAIFilePurpose
    ) -> OpenAIFileObject | None:
        if self.files_api is None:
            logger.warning("Files API not available, cannot upload file %s", file)
            return None

        if file.exists():
            with open(file, "rb") as f:
                upload_file = await self.files_api.openai_upload_file(
                    file=UploadFile(file=f, filename=file.name), purpose=purpose
                )
                return upload_file
        else:
            logger.warning("File %s does not exist", file)
            return None

    async def _parse_job_results_from_data(
        self, results_data: dict, job_id: str
    ) -> EvaluateResponse:
        if not results_data:
            logger.warning("No results data for job %s", job_id)
            return EvaluateResponse(generations=[], scores={})
        try:
            # Extract generations and scores from lm_eval results
            generations: list[dict[str, Any]] = []
            scores: dict[str, ScoringResult] = {}

            if "results" in results_data:
                results = results_data["results"]

                # Extract scores for each task
                for task_name, task_results in results.items():
                    if isinstance(task_results, dict):
                        # Extract metric scores
                        for metric_name, metric_value in task_results.items():
                            if isinstance(metric_value, int | float):
                                score_key = f"{task_name}:{metric_name}"
                                scores[score_key] = ScoringResult(
                                    aggregated_results={
                                        metric_name: float(metric_value)
                                    },
                                    score_rows=[{"score": float(metric_value)}],
                                )

                # Extract generations if available (from samples)
                if "samples" in results_data:
                    samples = results_data["samples"]
                    for task_name, task_samples in samples.items():
                        if isinstance(task_samples, list):
                            for sample in task_samples:
                                if isinstance(sample, dict):
                                    generation = {
                                        "task": task_name,
                                        "input": sample.get("doc", {}),
                                        "output": sample.get("target", ""),
                                        "generated": sample.get("resps", []),
                                    }
                                    generations.append(generation)

            logger.info("Successfully parsed results from file for job %s", job_id)

            return EvaluateResponse(
                generations=generations,
                scores=scores,
                metadata={"job_id": job_id},
            )

        except Exception as e:
            logger.error(
                "Failed to parse job results from file for job %s: %s", job_id, e
            )
            return EvaluateResponse(generations=[], scores={})

    def _create_model_args(self, base_url: str, benchmark_config: BenchmarkConfig):
        model_args = {"model": None, "base_url": base_url}

        model_name = None
        if hasattr(benchmark_config, "model") and benchmark_config.model:
            model_name = benchmark_config.model
        elif (
            hasattr(benchmark_config, "eval_candidate")
            and benchmark_config.eval_candidate
        ):
            if (
                hasattr(benchmark_config.eval_candidate, "model")
                and benchmark_config.eval_candidate.model
            ):
                model_name = benchmark_config.eval_candidate.model

        # Set model name and default parameters if we have a model
        if model_name:
            model_args["model"] = model_name
            model_args["num_concurrent"] = "1"
            model_args["max_retries"] = "3"

        # Apply any custom model args
        if hasattr(benchmark_config, "model_args") and benchmark_config.model_args:
            for arg in benchmark_config.model_args:
                model_args[arg.name] = arg.value

        return model_args

    def _collect_lmeval_args(
        self, task_config: BenchmarkConfig, stored_benchmark: Benchmark | None
    ):
        lmeval_args = {}
        if hasattr(task_config, "lmeval_args") and task_config.lmeval_args:
            lmeval_args = task_config.lmeval_args

        if hasattr(task_config, "metadata") and task_config.metadata:
            metadata_lmeval_args = task_config.metadata.get("lmeval_args")
            if metadata_lmeval_args:
                for key, value in metadata_lmeval_args.items():
                    lmeval_args[key] = value

        # Check stored benchmark for additional lmeval args
        if (
            stored_benchmark
            and hasattr(stored_benchmark, "metadata")
            and stored_benchmark.metadata
        ):
            benchmark_lmeval_args = stored_benchmark.metadata.get("lmeval_args")
            if benchmark_lmeval_args:
                for key, value in benchmark_lmeval_args.items():
                    lmeval_args[key] = value

        return lmeval_args

    def build_command(
        self,
        task_config: BenchmarkConfig,
        benchmark_id: str,
        limit: str,
        stored_benchmark: Benchmark | None,
        job_output_results_dir: Path,
        job_uuid: str,
    ) -> list[str]:
        """Build lm_eval command with default args and user overrides."""

        logger.info(
            "BUILD_COMMAND: Starting to build command for benchmark: %s", benchmark_id
        )
        logger.info("BUILD_COMMAND: Task config type: %s", type(task_config))
        logger.info(
            "BUILD_COMMAND: Task config has metadata: %s",
            hasattr(task_config, "metadata"),
        )
        if hasattr(task_config, "metadata"):
            logger.info(
                "BUILD_COMMAND: Task config metadata content: %s", task_config.metadata
            )

        eval_candidate = task_config.eval_candidate
        if not eval_candidate.type == "model":
            raise LMEvalConfigError("LMEval only supports model candidates for now")

        # Create model args - use VLLM_URL environment variable for inference provider
        inference_url = os.environ.get("VLLM_URL", "http://localhost:8080/v1")
        openai_url = inference_url.replace("/v1", "/v1/completions")
        model_args = self._create_model_args(openai_url, task_config)

        if (
            stored_benchmark is not None
            and hasattr(stored_benchmark, "metadata")
            and stored_benchmark.metadata
            and "tokenizer" in stored_benchmark.metadata
        ):
            tokenizer_value = stored_benchmark.metadata.get("tokenizer")
            if isinstance(tokenizer_value, str) and tokenizer_value:
                logger.info(
                    "BUILD_COMMAND: Using custom tokenizer from metadata: %s",
                    tokenizer_value,
                )

            tokenized_requests = stored_benchmark.metadata.get("tokenized_requests")
            if isinstance(tokenized_requests, bool) and tokenized_requests:
                logger.info(
                    "BUILD_COMMAND: Using custom tokenized_requests from metadata: %s",
                    tokenized_requests,
                )

            model_args["tokenizer"] = tokenizer_value
            model_args["tokenized_requests"] = tokenized_requests

        lmeval_args = self._collect_lmeval_args(task_config, stored_benchmark)

        # Start building the command
        cmd = ["lm_eval"]

        # Add model type
        model_type = model_args.get("model_type", "local-completions")
        cmd.extend(["--model", model_type])

        # Build model_args string
        if model_args:
            model_args_list = []
            for key, value in model_args.items():
                if key != "model_type" and value is not None:
                    model_args_list.append(f"{key}={value}")

            if model_args_list:
                cmd.extend(["--model_args", ",".join(model_args_list)])

        # Extract task name from benchmark_id (remove provider prefix)
        # benchmark_id format: "inline::trustyai_lmeval::task_name"
        task_name = (
            benchmark_id.split("::")[-1] if "::" in benchmark_id else benchmark_id
        )
        cmd.extend(["--tasks", task_name])

        cmd.extend(["--limit", limit])

        cmd.extend(
            ["--output_path", f"{job_output_results_dir}/job_{job_uuid}_results.json"]
        )

        # Add lmeval_args
        if lmeval_args:
            for key, value in lmeval_args.items():
                if value is not None:
                    cmd.extend([key, value])

        logger.info(
            "BUILD_COMMAND: Generated command for benchmark %s: %s",
            benchmark_id,
            " ".join(cmd),
        )
        return cmd

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

        raise NotImplementedError(
            "Evaluate rows is not implemented, use run_eval instead"
        )

    async def job_cancel(self, benchmark_id: str, job_id: str) -> None:
        """Cancel a running evaluation job.

        Args:
            benchmark_id: The ID of the benchmark to run the evaluation on.
            job_id: The ID of the job to cancel.
        """
        job = next((j for j in self._jobs if j.job_id == job_id), None)
        if not job:
            logger.warning("Job %s not found", job_id)
            return

        if job.status in [JobStatus.completed, JobStatus.failed, JobStatus.cancelled]:
            logger.warning("Job %s is not running", job_id)
            return

        if job.status in [JobStatus.in_progress, JobStatus.scheduled]:
            process_id_str = self._job_metadata.get(job_id, {}).get("process_id")
            if process_id_str:
                process_id = int(process_id_str)
                logger.info("Attempting to cancel subprocess %s", process_id)

                try:
                    os.kill(process_id, signal.SIGTERM)
                    logger.info("Sent SIGTERM to process %s", process_id)

                    for _ in range(10):
                        try:
                            # Check if process is still running
                            os.kill(process_id, 0)
                            await asyncio.sleep(0.5)
                        except ProcessLookupError:
                            # Process has terminated
                            logger.info("Process %s terminated gracefully", process_id)
                            break
                    else:
                        # Process didn't terminate gracefully, force kill
                        try:
                            os.kill(process_id, signal.SIGKILL)
                            logger.info("Sent SIGKILL to process %s", process_id)
                        except ProcessLookupError:
                            logger.info("Process %s already terminated", process_id)

                except ProcessLookupError:
                    logger.warning(
                        "Process %s not found (may have already terminated)", process_id
                    )
                except OSError as e:
                    logger.error("Error terminating process %s: %s", process_id, e)

            job.status = JobStatus.cancelled

            # Clean up result files for cancelled job
            job_uuid = job_id.replace("-", "")
            self._cleanup_job_files(self.config.results_dir, job_uuid)

            logger.info("Successfully cancelled job %s", job_id)

    async def job_result(self, benchmark_id: str, job_id: str) -> EvaluateResponse:
        """Get the results of a completed evaluation job.

        Args:
            benchmark_id: The ID of the benchmark to run the evaluation on.
            job_id: The ID of the job to get the results of.
        """
        job = await self.job_status(benchmark_id, job_id)

        if job is None:
            logger.warning("Job %s not found", job_id)
            return EvaluateResponse(generations=[], scores={})

        if job.status == JobStatus.completed:
            # Get results from job metadata
            job_metadata = self._job_metadata.get(job_id, {})
            results = job_metadata.get("results")
            return results

    async def job_status(self, benchmark_id: str, job_id: str) -> Job | None:
        """Get the status of a running evaluation job.

        Args:
            benchmark_id: The ID of the benchmark to run the evaluation on.
            job_id: The ID of the job to get the status of.
        """
        job = next((j for j in self._jobs if j.job_id == job_id), None)

        if job is None:
            logger.warning("Job %s not found", job_id)
            return None

        return job

    async def shutdown(self) -> None:
        """Shutdown the LMEval Inline provider."""
        logger.info("Shutting down LMEval Inline provider")

        # Cancel all running jobs
        running_jobs = [
            job
            for job in self._jobs
            if job.status in [JobStatus.in_progress, JobStatus.scheduled]
        ]

        if running_jobs:
            logger.info("Cancelling %d running jobs", len(running_jobs))
            for job in running_jobs:
                try:
                    await self.job_cancel(benchmark_id="", job_id=job.job_id)
                except Exception as e:
                    logger.warning("Failed to cancel job %s: %s", job.job_id, e)
                await asyncio.sleep(0.1)  # Brief pause between cancellations

        # Clean up any remaining result files
        if self.config.results_dir.exists():
            try:
                # Clean up result files for all known jobs
                for job_id in list(self._job_metadata.keys()):
                    job_uuid = job_id.replace("-", "")
                    self._cleanup_job_files(self.config.results_dir, job_uuid)
            except Exception as e:
                logger.warning("Error during shutdown cleanup: %s", e)

        # Clear internal state
        self._jobs.clear()
        self._job_metadata.clear()
        self.benchmarks.clear()

        # Close files API connection if it exists and has cleanup methods
        if self.files_api and hasattr(self.files_api, "close"):
            try:
                await self.files_api.close()
                logger.debug("Closed Files API connection")
            except Exception as e:
                logger.warning("Failed to close Files API connection: %s", e)

        logger.info("LMEval Inline provider shutdown complete")
