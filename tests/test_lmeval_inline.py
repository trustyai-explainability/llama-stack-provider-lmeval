"""Comprehensive unit tests for LMEval inline provider."""
import subprocess
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from llama_stack.apis.benchmarks import Benchmark
from llama_stack.apis.common.job_types import JobStatus
from llama_stack.apis.datatypes import Api
from llama_stack.apis.eval import BenchmarkConfig

from src.llama_stack_provider_lmeval.config import LMEvalEvalProviderConfig
from src.llama_stack_provider_lmeval.errors import (
    LMEvalConfigError,
    LMEvalTaskNameError,
)
from src.llama_stack_provider_lmeval.inline.lmeval import LMEvalInline
from src.llama_stack_provider_lmeval.inline.provider import get_provider_spec


def create_mock_files_api():
    """Create a mock Files API with required methods."""
    mock_files_api = MagicMock()
    mock_files_api.openai_upload_file = AsyncMock()
    return mock_files_api


class TestLMEvalInlineProvider(unittest.TestCase):
    """Unit tests for the LMEvalInlineProvider."""

    def test_get_provider_spec(self):
        """Test that provider spec is correctly configured."""
        spec = get_provider_spec()
        assert spec is not None
        assert spec.api == Api.eval
        assert spec.pip_packages == ["lm-eval", "lm-eval[api]"]
        assert spec.config_class == "llama_stack_provider_lmeval.config.LMEvalEvalProviderConfig"
        assert spec.module == "llama_stack_provider_lmeval.inline"


class TestLMEvalInlineInitialization(unittest.IsolatedAsyncioTestCase):
    """Test LMEvalInline class initialization and basic functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = LMEvalEvalProviderConfig(use_k8s=False)
        self.provider = LMEvalInline(self.config, deps={Api.files: create_mock_files_api()})

    async def test_initialization(self):
        """Test that LMEvalInline initializes correctly."""
        assert self.provider.config == self.config
        assert isinstance(self.provider.benchmarks, dict)
        assert len(self.provider.benchmarks) == 0
        assert isinstance(self.provider._jobs, list)
        assert len(self.provider._jobs) == 0
        assert isinstance(self.provider._job_metadata, dict)
        assert len(self.provider._job_metadata) == 0

    async def test_initialize_method(self):
        """Test that initialize method completes without error."""
        await self.provider.initialize()
        # Should complete without raising any exceptions


class TestBenchmarkManagement(unittest.IsolatedAsyncioTestCase):
    """Test benchmark management functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = LMEvalEvalProviderConfig(use_k8s=False)
        self.provider = LMEvalInline(self.config, deps={Api.files: create_mock_files_api()})

        # Create test benchmark
        self.test_benchmark = Benchmark(
            identifier="lmeval::mmlu",
            dataset_id="trustyai_lmeval::mmlu",
            scoring_functions=[],
            provider_id="inline::trustyai_lmeval",
        )

    async def test_register_benchmark(self):
        """Test benchmark registration."""
        await self.provider.register_benchmark(self.test_benchmark)

        assert "lmeval::mmlu" in self.provider.benchmarks
        assert self.provider.benchmarks["lmeval::mmlu"] == self.test_benchmark

    async def test_get_benchmark_existing(self):
        """Test getting an existing benchmark."""
        await self.provider.register_benchmark(self.test_benchmark)

        result = await self.provider.get_benchmark("lmeval::mmlu")
        assert result == self.test_benchmark

    async def test_get_benchmark_nonexistent(self):
        """Test getting a non-existent benchmark."""
        result = await self.provider.get_benchmark("nonexistent::benchmark")
        assert result is None

    async def test_list_benchmarks_empty(self):
        """Test listing benchmarks when none are registered."""
        result = await self.provider.list_benchmarks()
        assert len(result.data) == 0

    async def test_list_benchmarks_with_data(self):
        """Test listing benchmarks when some are registered."""
        await self.provider.register_benchmark(self.test_benchmark)

        result = await self.provider.list_benchmarks()
        assert len(result.data) == 1
        assert result.data[0] == self.test_benchmark


class TestJobManagement(unittest.IsolatedAsyncioTestCase):
    """Test job management functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = LMEvalEvalProviderConfig(use_k8s=False)
        self.provider = LMEvalInline(self.config, deps={Api.files: create_mock_files_api()})

    def test_get_job_id(self):
        """Test job ID generation."""
        job_id1 = self.provider._get_job_id()
        job_id2 = self.provider._get_job_id()

        assert isinstance(job_id1, str)
        assert isinstance(job_id2, str)
        assert job_id1 != job_id2

class TestModelArgsCreation(unittest.IsolatedAsyncioTestCase):
    """Test model args creation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = LMEvalEvalProviderConfig(use_k8s=False)
        self.provider = LMEvalInline(self.config, deps={Api.files: create_mock_files_api()})

    async def test_create_model_args_from_config_model(self):
        """Test model args creation when model is in config."""
        benchmark_config = MagicMock()
        benchmark_config.model = "test-model"
        benchmark_config.eval_candidate = None
        benchmark_config.model_args = []

        model_args = self.provider._create_model_args("http://test", benchmark_config)

        expected = {
            "model": "test-model",
            "base_url": "http://test",
            "num_concurrent": "1",
            "max_retries": "3",
        }
        assert model_args == expected

    async def test_create_model_args_from_eval_candidate(self):
        """Test model args creation when model is in eval_candidate."""
        benchmark_config = MagicMock()
        benchmark_config.model = None

        eval_candidate = MagicMock()
        eval_candidate.model = "candidate-model"
        benchmark_config.eval_candidate = eval_candidate
        benchmark_config.model_args = []

        model_args = self.provider._create_model_args("http://test", benchmark_config)

        expected = {
            "model": "candidate-model",
            "base_url": "http://test",
            "num_concurrent": "1",
            "max_retries": "3",
        }
        assert model_args == expected

    async def test_create_model_args_with_additional_args(self):
        """Test model args creation with additional model_args."""
        benchmark_config = MagicMock()
        benchmark_config.model = "test-model"
        benchmark_config.eval_candidate = None

        # Mock model_args as a list of objects with name and value attributes
        model_arg1 = MagicMock()
        model_arg1.name = "temperature"
        model_arg1.value = "0.7"
        model_arg2 = MagicMock()
        model_arg2.name = "max_tokens"
        model_arg2.value = "100"

        benchmark_config.model_args = [model_arg1, model_arg2]

        model_args = self.provider._create_model_args("http://test", benchmark_config)

        expected = {
            "model": "test-model",
            "base_url": "http://test",
            "num_concurrent": "1",
            "max_retries": "3",
            "temperature": "0.7",
            "max_tokens": "100",
        }
        assert model_args == expected

    async def test_create_model_args_override_defaults(self):
        """Test that custom model_args can override defaults."""
        benchmark_config = MagicMock()
        benchmark_config.model = "test-model"
        benchmark_config.eval_candidate = None

        # Override default num_concurrent
        model_arg = MagicMock()
        model_arg.name = "num_concurrent"
        model_arg.value = "5"

        benchmark_config.model_args = [model_arg]

        model_args = self.provider._create_model_args("http://test", benchmark_config)

        expected = {
            "model": "test-model",
            "base_url": "http://test",
            "num_concurrent": "5",  # Should be overridden
            "max_retries": "3",
        }
        assert model_args == expected


class TestLMEvalArgsCollection(unittest.IsolatedAsyncioTestCase):
    """Test LMEval args collection functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = LMEvalEvalProviderConfig(use_k8s=False)
        self.provider = LMEvalInline(self.config, deps={Api.files: create_mock_files_api()})

    async def test_collect_lmeval_args_from_task_config(self):
        """Test collecting lmeval_args from task config."""
        task_config = MagicMock()
        task_config.lmeval_args = {"arg1": "value1", "arg2": "value2"}
        task_config.metadata = None

        stored_benchmark = None

        result = self.provider._collect_lmeval_args(task_config, stored_benchmark)

        expected = {"arg1": "value1", "arg2": "value2"}
        assert result == expected

    async def test_collect_lmeval_args_from_stored_benchmark(self):
        """Test collecting lmeval_args from stored benchmark metadata."""
        task_config = MagicMock()
        task_config.lmeval_args = None
        task_config.metadata = None

        stored_benchmark = MagicMock()
        stored_benchmark.metadata = {
            "lmeval_args": {"bench_arg1": "bench_value1", "bench_arg2": "bench_value2"}
        }

        result = self.provider._collect_lmeval_args(task_config, stored_benchmark)

        expected = {"bench_arg1": "bench_value1", "bench_arg2": "bench_value2"}
        assert result == expected

    async def test_collect_lmeval_args_combined(self):
        """Test collecting lmeval_args from multiple sources with precedence."""
        task_config = MagicMock()
        task_config.lmeval_args = {"arg1": "task_value1"}
        task_config.metadata = None

        stored_benchmark = MagicMock()
        stored_benchmark.metadata = {
            "lmeval_args": {"arg1": "bench_value1", "arg2": "bench_value2"}
        }

        result = self.provider._collect_lmeval_args(task_config, stored_benchmark)

        # Stored benchmark args should override task config args
        expected = {"arg1": "bench_value1", "arg2": "bench_value2"}
        assert result == expected


class TestCommandBuilding(unittest.IsolatedAsyncioTestCase):
    """Test command building functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = LMEvalEvalProviderConfig(use_k8s=False)
        self.provider = LMEvalInline(self.config, deps={Api.files: create_mock_files_api()})

    async def test_build_command_basic(self):
        """Test building basic command."""
        task_config = MagicMock()
        eval_candidate = MagicMock()
        eval_candidate.type = "model"
        eval_candidate.model = "test-model"
        task_config.eval_candidate = eval_candidate
        task_config.model = "test-model"
        task_config.model_args = []
        task_config.metadata = {}

        stored_benchmark = None

        cmd = self.provider.build_command(
            task_config=task_config,
            benchmark_id="lmeval::mmlu",
            limit="10",
            stored_benchmark=stored_benchmark,
            job_output_results_dir=Path("/tmp"),
            job_uuid="test_job_uuid"
        )
        cmd_str = " ".join(cmd)
        assert "lm_eval" in cmd_str
        assert "--model" in cmd_str
        assert "local-completions" in cmd_str
        assert "--model_args" in cmd_str
        assert "--tasks" in cmd_str
        assert "mmlu" in cmd_str
        assert "--limit" in cmd_str
        assert "10" in cmd_str

    async def test_build_command_with_tokenizer_metadata(self):
        """Test building command with tokenizer in stored benchmark metadata."""
        task_config = MagicMock()
        eval_candidate = MagicMock()
        eval_candidate.type = "model"
        eval_candidate.model = "test-model"
        task_config.eval_candidate = eval_candidate
        task_config.model = "test-model"
        task_config.model_args = []
        task_config.metadata = {}

        stored_benchmark = MagicMock()
        stored_benchmark.metadata = {
            "tokenizer": "custom-tokenizer",
            "tokenized_requests": True
        }

        cmd = self.provider.build_command(
            task_config=task_config,
            benchmark_id="lmeval::mmlu",
            limit="10",
            stored_benchmark=stored_benchmark,
            job_output_results_dir=Path("/tmp"),
            job_uuid="test_job_uuid"
        )

        cmd_str = " ".join(cmd)
        assert "tokenizer=custom-tokenizer" in cmd_str
        assert "tokenized_requests=True" in cmd_str

    async def test_build_command_with_lmeval_args(self):
        """Test building command with lmeval_args."""
        task_config = MagicMock()
        eval_candidate = MagicMock()
        eval_candidate.type = "model"
        eval_candidate.model = "test-model"
        task_config.eval_candidate = eval_candidate
        task_config.model = "test-model"
        task_config.model_args = []
        task_config.metadata = {}
        task_config.lmeval_args = {"output_path": "/tmp/results"}

        stored_benchmark = None

        cmd = self.provider.build_command(
            task_config=task_config,
            benchmark_id="lmeval::mmlu",
            limit="10",
            stored_benchmark=stored_benchmark,
            job_output_results_dir=Path("/tmp"),
            job_uuid="test_job_uuid"
        )
        cmd_str = " ".join(cmd)
        assert "output_path" in cmd_str
        assert "/tmp/results" in cmd_str

    async def test_build_command_invalid_eval_candidate(self):
        """Test building command with invalid eval candidate type."""
        task_config = MagicMock()
        eval_candidate = MagicMock()
        eval_candidate.type = "dataset"  # Invalid type
        task_config.eval_candidate = eval_candidate

        stored_benchmark = None

        with self.assertRaises(LMEvalConfigError):
            self.provider.build_command(
                task_config=task_config,
                benchmark_id="lmeval::mmlu",
                limit="10",
                stored_benchmark=stored_benchmark,
                job_output_results_dir=Path("/tmp"),
                job_uuid="test_job_uuid"
            )

if __name__ == "__main__":
    unittest.main()
