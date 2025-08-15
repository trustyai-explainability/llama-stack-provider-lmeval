"""Tests for base URL handling functionality in the LMEval provider."""

import unittest
from unittest.mock import MagicMock

from src.llama_stack_provider_lmeval.config import TLSConfig, LMEvalConfigError
from src.llama_stack_provider_lmeval.lmeval import LMEvalCRBuilder, ModelArg

BASE_URL = "http://example.com"


class TestBaseUrlHandling(unittest.TestCase):
    """Test the base URL handling logic in LMEvalCRBuilder."""

    def setUp(self):
        """Test fixtures."""
        self.namespace = "test-namespace"
        self.service_account = "test-service-account"
        self.builder = LMEvalCRBuilder(
            namespace=self.namespace, service_account=self.service_account
        )

        # Mock config to avoid TLS handling in these tests
        self.builder._config = MagicMock()
        self.builder._config.tls = None  # Explicitly disable TLS

        # Mock benchmark config for _create_model_args tests
        self.mock_benchmark_config = MagicMock()
        self.mock_benchmark_config.model = "test-model"

    # Tests for the static _build_openai_url method
    def test_build_openai_url_regular_url(self):
        """Test building OpenAI URL from regular base URL."""
        result = LMEvalCRBuilder._build_openai_url(BASE_URL)
        self.assertEqual(result, f"{BASE_URL}/v1/completions")

    def test_build_openai_url_with_v1_ending(self):
        """Test building OpenAI URL when base URL ends with /v1."""
        base_url = f"{BASE_URL}/v1"
        result = LMEvalCRBuilder._build_openai_url(base_url)
        self.assertEqual(result, f"{BASE_URL}/v1/completions")

    def test_build_openai_url_with_v1_slash_ending(self):
        """Test building OpenAI URL when base URL ends with /v1/."""
        base_url = f"{BASE_URL}/v1/"
        result = LMEvalCRBuilder._build_openai_url(base_url)
        self.assertEqual(result, f"{BASE_URL}/v1/completions")

    def test_build_openai_url_with_multiple_trailing_slashes(self):
        """Test building OpenAI URL with multiple trailing slashes."""
        base_url = f"{BASE_URL}///"
        result = LMEvalCRBuilder._build_openai_url(base_url)
        self.assertEqual(result, f"{BASE_URL}/v1/completions")

    def test_build_openai_url_with_v1_and_multiple_slashes(self):
        """Test building OpenAI URL when base URL ends with /v1/// (multiple slashes)."""
        base_url = f"{BASE_URL}/v1///"
        result = LMEvalCRBuilder._build_openai_url(base_url)
        self.assertEqual(result, f"{BASE_URL}/v1/completions")

    def test_build_openai_url_complex_path_with_v1(self):
        """Test building OpenAI URL with complex path ending in v1."""
        base_url = "https://api.example.com:8080/some/path/v1"
        result = LMEvalCRBuilder._build_openai_url(base_url)
        self.assertEqual(
            result, "https://api.example.com:8080/some/path/v1/completions"
        )

    def test_build_openai_url_v1_in_middle(self):
        """Test building OpenAI URL when v1 appears in middle but not at end."""
        base_url = f"{BASE_URL}/v1/something/else"
        result = LMEvalCRBuilder._build_openai_url(base_url)
        self.assertEqual(result, f"{BASE_URL}/v1/something/else/v1/completions")

    def test_build_openai_url_case_sensitivity(self):
        """Test that the v1 check is case sensitive."""
        base_url = f"{BASE_URL}/V1"
        result = LMEvalCRBuilder._build_openai_url(base_url)
        # Should treat V1 as different from v1
        self.assertEqual(result, f"{BASE_URL}/V1/v1/completions")

    def test_build_openai_url_with_v1_and_query_params(self):
        """Test building OpenAI URL when base URL ends with /v1 and has query parameters."""
        base_url = f"{BASE_URL}/v1?param=value&other=test"
        result = LMEvalCRBuilder._build_openai_url(base_url)
        # Query parameters should be preserved but v1 not detected at end
        self.assertEqual(result, f"{BASE_URL}/v1?param=value&other=test/v1/completions")

    def test_build_openai_url_with_v1_and_fragment(self):
        """Test building OpenAI URL when base URL ends with /v1 and has fragment."""
        base_url = f"{BASE_URL}/v1#section"
        result = LMEvalCRBuilder._build_openai_url(base_url)
        # Fragment should be preserved but v1 not detected at end
        self.assertEqual(result, f"{BASE_URL}/v1#section/v1/completions")

    def test_build_openai_url_with_v1_query_params_and_fragment(self):
        """Test building OpenAI URL when base URL ends with /v1 and has both query params and fragment."""
        base_url = f"{BASE_URL}/v1?param=value#section"
        result = LMEvalCRBuilder._build_openai_url(base_url)
        # Both query params and fragment should be preserved but v1 not detected at end
        self.assertEqual(result, f"{BASE_URL}/v1?param=value#section/v1/completions")

    # Tests for the _create_model_args method integration
    def test_create_model_args_without_base_url(self):
        """Test _create_model_args without base_url."""
        model_name = "test-model"
        result = self.builder._create_model_args("", self.mock_benchmark_config)

        # Should contain model name and base_url
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].name, "base_url")
        self.assertEqual(result[0].value, "/v1/completions")
        self.assertEqual(result[1].name, "model")
        self.assertEqual(result[1].value, model_name)

    def test_create_model_args_with_regular_base_url(self):
        """Test _create_model_args with regular base_url."""
        model_name = "test-model"
        result = self.builder._create_model_args(BASE_URL, self.mock_benchmark_config)

        # Should contain model name and base_url
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].name, "base_url")
        self.assertEqual(result[0].value, f"{BASE_URL}/v1/completions")
        self.assertEqual(result[1].name, "model")
        self.assertEqual(result[1].value, model_name)

    def test_create_model_args_with_base_url_ending_with_v1(self):
        """Test _create_model_args with base_url ending with /v1."""
        model_name = "test-model"
        base_url = "https://api.example.com/v1"
        result = self.builder._create_model_args(base_url, self.mock_benchmark_config)

        # Should contain model name and base_url
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].name, "base_url")
        self.assertEqual(result[0].value, f"{base_url}/completions")
        self.assertEqual(result[1].name, "model")
        self.assertEqual(result[1].value, model_name)

    def test_create_model_args_with_base_url_ending_with_slash_v1(self):
        """Test _create_model_args with base_url ending with /v1/."""
        model_name = "test-model"
        base_url = "https://api.example.com/v1/"
        result = self.builder._create_model_args(base_url, self.mock_benchmark_config)

        # Should contain model name and base_url
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].name, "base_url")
        self.assertEqual(result[0].value, f"{base_url.rstrip('/')}/completions")
        self.assertEqual(result[1].name, "model")
        self.assertEqual(result[1].value, model_name)

    def test_create_model_args_with_complex_base_url_ending_with_v1(self):
        """Test _create_model_args with complex base_url ending with /v1."""
        model_name = "test-model"
        base_url = "https://api.example.com/path/to/service/v1"
        result = self.builder._create_model_args(base_url, self.mock_benchmark_config)

        # Should contain model name and base_url
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].name, "base_url")
        self.assertEqual(result[0].value, f"{base_url}/completions")
        self.assertEqual(result[1].name, "model")
        self.assertEqual(result[1].value, model_name)

    def test_create_model_args_with_base_url_containing_v1_but_not_ending(self):
        """Test _create_model_args with base_url containing /v1 but not ending with it."""
        model_name = "test-model"
        base_url = "https://api.example.com/v1/endpoint"
        result = self.builder._create_model_args(base_url, self.mock_benchmark_config)

        # Should contain model name and base_url
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].name, "base_url")
        self.assertEqual(result[0].value, f"{base_url}/v1/completions")
        self.assertEqual(result[1].name, "model")
        self.assertEqual(result[1].value, model_name)

    def test_create_model_args_with_empty_base_url(self):
        """Test _create_model_args with empty base_url."""
        model_name = "test-model"
        result = self.builder._create_model_args("", self.mock_benchmark_config)

        # Should contain model name and base_url
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].name, "base_url")
        self.assertEqual(result[0].value, "/v1/completions")
        self.assertEqual(result[1].name, "model")
        self.assertEqual(result[1].value, model_name)

    def test_create_model_args_with_trailing_slashes(self):
        """Test _create_model_args with base_url containing trailing slashes."""
        model_name = "test-model"
        base_url = "https://api.example.com/"
        result = self.builder._create_model_args(base_url, self.mock_benchmark_config)

        # Should contain model name and base_url
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].name, "base_url")
        self.assertEqual(result[0].value, f"{base_url.rstrip('/')}/v1/completions")
        self.assertEqual(result[1].name, "model")
        self.assertEqual(result[1].value, model_name)

    def test_create_model_args_with_v1and_trailing_slashes(self):
        """Test _create_model_args with base_url ending with /v1/."""
        model_name = "test-model"
        base_url = "https://api.example.com/v1/"
        result = self.builder._create_model_args(base_url, self.mock_benchmark_config)

        # Should contain model name and base_url
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].name, "base_url")
        self.assertEqual(result[0].value, f"{base_url.rstrip('/')}/completions")
        self.assertEqual(result[1].name, "model")
        self.assertEqual(result[1].value, model_name)

    def test_create_model_args_with_none_base_url(self):
        """Test _create_model_args with None base_url."""
        model_name = "test-model"
        result = self.builder._create_model_args(None, self.mock_benchmark_config)

        # Should contain model name and base_url
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].name, "base_url")
        self.assertEqual(result[0].value, "")  # None becomes empty string
        self.assertEqual(result[1].name, "model")
        self.assertEqual(result[1].value, model_name)

    def test_model_arg_type_consistency(self):
        """Test that all model args are of type ModelArg."""
        _model_name = "test-model"
        result = self.builder._create_model_args(BASE_URL, self.mock_benchmark_config)

        for arg in result:
            self.assertIsInstance(arg, ModelArg)
            self.assertIsInstance(arg.name, str)
            self.assertIsInstance(arg.value, str)

    def test_create_model_args_fallback_to_provider_tls_when_benchmark_tls_none(self):
        """Test _create_model_args falls back to provider config TLS when benchmark_tls is None."""
        model_name = "test-model"

        # Set provider config tls to True
        self.builder._config.tls = TLSConfig(enable=True)

        # Pass benchmark_tls as None
        result = self.builder._create_model_args(BASE_URL, self.mock_benchmark_config)

        # Should have model, base_url, and verify_certificate args
        self.assertEqual(len(result), 3)

        # Check verify_certificate arg
        verify_cert_arg = next(
            (arg for arg in result if arg.name == "verify_certificate"), None
        )
        self.assertIsNotNone(verify_cert_arg)
        self.assertEqual(verify_cert_arg.value, "True")


if __name__ == "__main__":
    unittest.main() 