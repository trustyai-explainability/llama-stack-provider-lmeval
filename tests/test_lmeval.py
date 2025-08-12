"""Tests for the LMEval provider functionality."""

import unittest
from unittest.mock import patch, MagicMock
import os

from src.llama_stack_provider_lmeval.config import LMEvalEvalProviderConfig, TLSConfig
from src.llama_stack_provider_lmeval.lmeval import LMEvalCRBuilder


class TestLMEvalCRBuilder(unittest.TestCase):
    """Test the LMEvalCRBuilder."""

    def setUp(self):
        """Test fixtures."""
        self.namespace = "test-namespace"
        self.service_account = "test-service-account"
        self.builder = LMEvalCRBuilder(
            namespace=self.namespace, service_account=self.service_account
        )

        self.benchmark_config = MagicMock()
        self.benchmark_config.eval_candidate.type = "model"
        self.benchmark_config.eval_candidate.model = "test-model"
        self.benchmark_config.eval_candidate.sampling_params = {}
        self.benchmark_config.env_vars = []
        self.benchmark_config.metadata = {}
        self.benchmark_config.model = "test-model"  # Add this for _create_model_args

        self.stored_benchmark = MagicMock()
        self.stored_benchmark.metadata = {}

    @patch("src.llama_stack_provider_lmeval.lmeval.logger")
    def test_create_cr_without_tls(self, mock_logger):
        """Creating CR without no TLS configuration."""
        config = LMEvalEvalProviderConfig(
            namespace=self.namespace,
            service_account=self.service_account,
        )
        self.builder._config = config

        cr = self.builder.create_cr(
            benchmark_id="lmeval::mmlu",
            task_config=self.benchmark_config,
            base_url="http://my-model-url",
            limit="10",
            stored_benchmark=self.stored_benchmark,
        )

        model_args = cr.get("spec", {}).get("modelArgs", [])
        verify_cert_args = [
            arg for arg in model_args if arg.get("name") == "verify_certificate"
        ]

        self.assertEqual(
            len(verify_cert_args),
            0,
            "CR TLS configuration should be missing when not provided in the configuration",
        )

    @patch("src.llama_stack_provider_lmeval.lmeval.logger")
    def test_create_cr_with_tls_false(self, mock_logger):
        """Creating CR with TLS verification bypass."""
        config = LMEvalEvalProviderConfig(
            namespace=self.namespace,
            service_account=self.service_account,
        )
        self.builder._config = config

        cr = self.builder.create_cr(
            benchmark_id="lmeval::mmlu",
            task_config=self.benchmark_config,
            base_url="http://my-model-url",
            limit="10",
            stored_benchmark=self.stored_benchmark,
        )

        model_args = cr.get("spec", {}).get("modelArgs", [])
        verify_cert_args = [
            arg for arg in model_args if arg.get("name") == "verify_certificate"
        ]

        self.assertEqual(
            len(verify_cert_args),
            0,
            "CR TLS configuration should be missing when not provided in the configuration",
        )

    @patch("src.llama_stack_provider_lmeval.lmeval.logger")
    def test_create_cr_with_tls_certificate_path(self, mock_logger):
        """Creating CR with TLS certificate path."""
        config = LMEvalEvalProviderConfig(
            namespace=self.namespace,
            service_account=self.service_account,
        )
        self.builder._config = config

        cr = self.builder.create_cr(
            benchmark_id="lmeval::mmlu",
            task_config=self.benchmark_config,
            base_url="http://my-model-url",
            limit="10",
            stored_benchmark=self.stored_benchmark,
        )

        model_args = cr.get("spec", {}).get("modelArgs", [])
        verify_cert_args = [
            arg for arg in model_args if arg.get("name") == "verify_certificate"
        ]

        self.assertEqual(
            len(verify_cert_args),
            0,
            "TLS configuration should be missing when not provided in the configuration",
        )

    @patch("src.llama_stack_provider_lmeval.lmeval.logger")
    @patch.dict(os.environ, {"TRUSTYAI_LMEVAL_TLS": "true"})
    def test_create_cr_with_env_tls_true(self, mock_logger):
        """Creating CR with TLS verification enabled via environment variable."""
        config = LMEvalEvalProviderConfig(
            namespace=self.namespace,
            service_account=self.service_account,
        )
        self.builder._config = config

        cr = self.builder.create_cr(
            benchmark_id="lmeval::mmlu",
            task_config=self.benchmark_config,
            base_url="http://my-model-url",
            limit="10",
            stored_benchmark=self.stored_benchmark,
        )

        model_args = cr.get("spec", {}).get("modelArgs", [])
        verify_cert_args = [
            arg for arg in model_args if arg.get("name") == "verify_certificate"
        ]

        self.assertEqual(
            len(verify_cert_args),
            1,
            "CR TLS configuration should be present when TRUSTYAI_LMEVAL_TLS is True",
        )
        self.assertEqual(
            verify_cert_args[0].get("value"),
            "True",
            "TLS configuration value should be 'True'",
        )

    @patch("src.llama_stack_provider_lmeval.lmeval.logger")
    @patch.dict(os.environ, {
        "TRUSTYAI_LMEVAL_TLS": "true",
        "TRUSTYAI_LMEVAL_CERT_FILE": "custom-ca.pem",
        "TRUSTYAI_LMEVAL_CERT_SECRET": "vllm-ca-bundle"
    })
    def test_create_cr_with_env_tls_certificate(self, mock_logger):
        """Creating CR with TLS certificate path via environment variables."""
        config = LMEvalEvalProviderConfig(
            namespace=self.namespace,
            service_account=self.service_account,
        )
        self.builder._config = config

        cr = self.builder.create_cr(
            benchmark_id="lmeval::mmlu",
            task_config=self.benchmark_config,
            base_url="http://my-model-url",
            limit="10",
            stored_benchmark=self.stored_benchmark,
        )

        model_args = cr.get("spec", {}).get("modelArgs", [])
        verify_cert_args = [
            arg for arg in model_args if arg.get("name") == "verify_certificate"
        ]

        self.assertEqual(
            len(verify_cert_args),
            1,
            "CR TLS configuration should be present when TLS certificate is configured",
        )
        self.assertEqual(
            verify_cert_args[0].get("value"),
            "/etc/ssl/certs/custom-ca.pem",
            "TLS configuration value should be the full mounted certificate path",
        )

        # Check that volumes and volume mounts are created
        pod_config = cr.get("spec", {}).get("pod", {})
        self.assertIsNotNone(pod_config.get("volumes"), "Pod should have volumes for TLS certificate")
        self.assertIsNotNone(pod_config.get("container", {}).get("volumeMounts"), "Container should have volume mounts for TLS certificate")

    @patch("src.llama_stack_provider_lmeval.lmeval.logger")
    def test_create_cr_with_provider_config_tls_true(self, mock_logger):
        """Creating CR with TLS verification enabled via provider config (backward compatibility)."""
        config = LMEvalEvalProviderConfig(
            namespace=self.namespace,
            service_account=self.service_account,
            tls=TLSConfig(enable=True),
        )
        self.builder._config = config

        cr = self.builder.create_cr(
            benchmark_id="lmeval::mmlu",
            task_config=self.benchmark_config,
            base_url="http://my-model-url",
            limit="10",
            stored_benchmark=self.stored_benchmark,
        )

        model_args = cr.get("spec", {}).get("modelArgs", [])
        verify_cert_args = [
            arg for arg in model_args if arg.get("name") == "verify_certificate"
        ]

        self.assertEqual(
            len(verify_cert_args),
            1,
            "CR TLS configuration should be present when provider config tls is enabled",
        )
        self.assertEqual(
            verify_cert_args[0].get("value"),
            "True",
            "TLS configuration value should be 'True'",
        )

    @patch("src.llama_stack_provider_lmeval.lmeval.logger")
    def test_create_cr_with_provider_config_tls_certificate(self, mock_logger):
        """Creating CR with TLS certificate path via provider config (backward compatibility)."""
        config = LMEvalEvalProviderConfig(
            namespace=self.namespace,
            service_account=self.service_account,
            tls=TLSConfig(
                enable=True,
                cert_file="custom-ca.pem",
                cert_secret="vllm-ca-bundle"
            ),
        )
        self.builder._config = config

        cr = self.builder.create_cr(
            benchmark_id="lmeval::mmlu",
            task_config=self.benchmark_config,
            base_url="http://my-model-url",
            limit="10",
            stored_benchmark=self.stored_benchmark,
        )

        model_args = cr.get("spec", {}).get("modelArgs", [])
        verify_cert_args = [
            arg for arg in model_args if arg.get("name") == "verify_certificate"
        ]

        self.assertEqual(
            len(verify_cert_args),
            1,
            "CR TLS configuration should be present when provider config tls is set",
        )
        self.assertEqual(
            verify_cert_args[0].get("value"),
            "/etc/ssl/certs/custom-ca.pem",
            "TLS configuration value should be the full mounted certificate path",
        )

        # Check that volumes and volume mounts are created
        pod_config = cr.get("spec", {}).get("pod", {})
        self.assertIsNotNone(pod_config.get("volumes"), "Pod should have volumes for TLS certificate")
        self.assertIsNotNone(pod_config.get("container", {}).get("volumeMounts"), "Container should have volume mounts for TLS certificate")

    @patch("src.llama_stack_provider_lmeval.lmeval.logger")
    def test_create_cr_without_tokenizer(self, mock_logger):
        """Creating CR without tokenizer specified."""
        config = LMEvalEvalProviderConfig(
            namespace=self.namespace,
            service_account=self.service_account,
        )
        self.builder._config = config

        self.benchmark_config.metadata = {}

        cr = self.builder.create_cr(
            benchmark_id="lmeval::mmlu",
            task_config=self.benchmark_config,
            base_url="http://my-model-url",
            limit="10",
            stored_benchmark=self.stored_benchmark,
        )

        model_args = cr.get("spec", {}).get("modelArgs", [])
        tokenizer_args = [arg for arg in model_args if arg.get("name") == "tokenizer"]

        self.assertEqual(
            len(tokenizer_args),
            0,
            "Tokenizer should not be present when not specified in metadata",
        )

    @patch("src.llama_stack_provider_lmeval.lmeval.logger")
    def test_create_cr_with_custom_tokenizer(self, mock_logger):
        """Creating CR with custom tokenizer specified in metadata."""
        config = LMEvalEvalProviderConfig(
            namespace=self.namespace,
            service_account=self.service_account,
        )
        self.builder._config = config

        tokenizer = "google/flan-t5-base"
        self.stored_benchmark.metadata = {"tokenizer": tokenizer}

        cr = self.builder.create_cr(
            benchmark_id="lmeval::mmlu",
            task_config=self.benchmark_config,
            base_url="http://my-model-url",
            limit="10",
            stored_benchmark=self.stored_benchmark,
        )

        model_args = cr.get("spec", {}).get("modelArgs", [])
        tokenizer_args = [arg for arg in model_args if arg.get("name") == "tokenizer"]

        self.assertEqual(
            len(tokenizer_args),
            1,
            "Tokenizer should be present when specified in the request's metadata",
        )
        self.assertEqual(
            tokenizer_args[0].get("value"),
            tokenizer,
            "Tokenizer value should match the value specified in the request's metadata",
        )

    @patch("src.llama_stack_provider_lmeval.lmeval.logger")
    def test_create_cr_with_tokenized_requests(self, mock_logger):
        """Creating CR with tokenized_requests specified in metadata."""
        config = LMEvalEvalProviderConfig(
            namespace=self.namespace,
            service_account=self.service_account,
        )
        self.builder._config = config

        self.stored_benchmark.metadata = {"tokenized_requests": False}

        cr = self.builder.create_cr(
            benchmark_id="lmeval::mmlu",
            task_config=self.benchmark_config,
            base_url="http://my-model-url",
            limit="10",
            stored_benchmark=self.stored_benchmark,
        )

        model_args = cr.get("spec", {}).get("modelArgs", [])
        tokenized_requests_args = [arg for arg in model_args if arg.get("name") == "tokenized_requests"]

        self.assertEqual(
            len(tokenized_requests_args),
            1,
            "tokenized_requests should be present when specified in the request's metadata",
        )
        self.assertEqual(
            tokenized_requests_args[0].get("value"),
            "False",
            "tokenized_requests value should match the value specified in the request's metadata",
        )


if __name__ == "__main__":
    unittest.main()
