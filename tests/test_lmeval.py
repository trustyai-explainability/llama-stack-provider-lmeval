import unittest
from unittest.mock import patch, MagicMock

from src.llama_stack_provider_lmeval.config import LMEvalEvalProviderConfig
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

        self.stored_benchmark = MagicMock()
        self.stored_benchmark.metadata = {}

    @patch("src.llama_stack_provider_lmeval.lmeval.logger")
    def test_create_cr_without_tls(self, mock_logger):
        """Creating CR without no TLS configuration."""
        config = LMEvalEvalProviderConfig(
            namespace=self.namespace,
            service_account=self.service_account,
            tls=None,
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
            tls=False,
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
            "CR TLS configuration should be present when passed to the configuration",
        )
        self.assertEqual(
            verify_cert_args[0].get("value"),
            "False",
            "TLS configuration value should be 'False'",
        )

    @patch("src.llama_stack_provider_lmeval.lmeval.logger")
    def test_create_cr_with_tls_certificate_path(self, mock_logger):
        """Creating CR with TLS certificate path."""
        cert_path = "/path/to/certificate.crt"
        config = LMEvalEvalProviderConfig(
            namespace=self.namespace,
            service_account=self.service_account,
            tls=cert_path,
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
            "TLS configuration should be present when tls is passed to the configuration",
        )
        self.assertEqual(
            verify_cert_args[0].get("value"),
            cert_path,
            "TLS configuration value should be a certificate path",
        )

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
