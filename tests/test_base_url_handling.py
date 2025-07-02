import unittest
from unittest.mock import MagicMock

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
        self.builder._config.tls = None

    # Tests for the static _build_openai_url method
    def test_build_openai_url_regular_url(self):
        """Test building OpenAI URL from regular base URL."""
        result = LMEvalCRBuilder._build_openai_url(BASE_URL)
        self.assertEqual(result, f"{BASE_URL}/v1/openai/v1/completions")

    def test_build_openai_url_with_v1_ending(self):
        """Test building OpenAI URL when base URL ends with /v1."""
        base_url = f"{BASE_URL}/v1"
        result = LMEvalCRBuilder._build_openai_url(base_url)
        self.assertEqual(result, f"{BASE_URL}/v1/openai/v1/completions")

    def test_build_openai_url_with_v1_slash_ending(self):
        """Test building OpenAI URL when base URL ends with /v1/."""
        base_url = f"{BASE_URL}/v1/"
        result = LMEvalCRBuilder._build_openai_url(base_url)
        self.assertEqual(result, f"{BASE_URL}/v1/openai/v1/completions")

    def test_build_openai_url_with_multiple_trailing_slashes(self):
        """Test building OpenAI URL with multiple trailing slashes."""
        base_url = f"{BASE_URL}///"
        result = LMEvalCRBuilder._build_openai_url(base_url)
        self.assertEqual(result, f"{BASE_URL}/v1/openai/v1/completions")

    def test_build_openai_url_with_v1_and_multiple_slashes(self):
        """Test building OpenAI URL when base URL ends with /v1/// (multiple slashes)."""
        base_url = f"{BASE_URL}/v1///"
        result = LMEvalCRBuilder._build_openai_url(base_url)
        self.assertEqual(result, f"{BASE_URL}/v1/openai/v1/completions")

    def test_build_openai_url_complex_path_with_v1(self):
        """Test building OpenAI URL with complex path ending in v1."""
        base_url = "https://api.example.com:8080/some/path/v1"
        result = LMEvalCRBuilder._build_openai_url(base_url)
        self.assertEqual(result, "https://api.example.com:8080/some/path/v1/openai/v1/completions")

    def test_build_openai_url_v1_in_middle(self):
        """Test building OpenAI URL when v1 appears in middle but not at end."""
        base_url = f"{BASE_URL}/v1/something/else"
        result = LMEvalCRBuilder._build_openai_url(base_url)
        self.assertEqual(result, f"{BASE_URL}/v1/something/else/v1/openai/v1/completions")

    def test_build_openai_url_case_sensitivity(self):
        """Test that the v1 check is case sensitive."""
        base_url = f"{BASE_URL}/V1"
        result = LMEvalCRBuilder._build_openai_url(base_url)
        # Should treat V1 as different from v1
        self.assertEqual(result, f"{BASE_URL}/V1/v1/openai/v1/completions")

    def test_build_openai_url_with_v1_and_query_params(self):
        """Test building OpenAI URL when base URL ends with /v1 and has query parameters."""
        base_url = f"{BASE_URL}/v1?param=value&other=test"
        result = LMEvalCRBuilder._build_openai_url(base_url)
        # Query parameters should be preserved but v1 not detected at end
        self.assertEqual(result, f"{BASE_URL}/v1?param=value&other=test/v1/openai/v1/completions")

    def test_build_openai_url_with_v1_and_fragment(self):
        """Test building OpenAI URL when base URL ends with /v1 and has fragment."""
        base_url = f"{BASE_URL}/v1#section"
        result = LMEvalCRBuilder._build_openai_url(base_url)
        # Fragment should be preserved but v1 not detected at end
        self.assertEqual(result, f"{BASE_URL}/v1#section/v1/openai/v1/completions")

    def test_build_openai_url_with_v1_query_params_and_fragment(self):
        """Test building OpenAI URL when base URL ends with /v1 and has both query params and fragment."""
        base_url = f"{BASE_URL}/v1?param=value#section"
        result = LMEvalCRBuilder._build_openai_url(base_url)
        # Both query params and fragment should be preserved but v1 not detected at end
        self.assertEqual(result, f"{BASE_URL}/v1?param=value#section/v1/openai/v1/completions")

    # Tests for the _create_model_args method integration
    def test_create_model_args_without_base_url(self):
        """Test creating model args without base_url (None)."""
        model_name = "test-model"
        
        result = self.builder._create_model_args(model_name, None)
        
        # Should have model and num_concurrent args only
        self.assertEqual(len(result), 2)
        
        # Check model arg
        model_arg = next((arg for arg in result if arg.name == "model"), None)
        self.assertIsNotNone(model_arg)
        self.assertEqual(model_arg.value, model_name)
        
        # Check num_concurrent arg
        concurrent_arg = next((arg for arg in result if arg.name == "num_concurrent"), None)
        self.assertIsNotNone(concurrent_arg)
        self.assertEqual(concurrent_arg.value, "1")
        
        # Should not have base_url arg
        base_url_arg = next((arg for arg in result if arg.name == "base_url"), None)
        self.assertIsNone(base_url_arg)

    def test_create_model_args_with_regular_base_url(self):
        """Test creating model args with regular base_url (not ending with /v1)."""
        model_name = "test-model"
        
        result = self.builder._create_model_args(model_name, BASE_URL)
        
        # Should have model, base_url, and num_concurrent args
        self.assertEqual(len(result), 3)
        
        # Check base_url arg
        base_url_arg = next((arg for arg in result if arg.name == "base_url"), None)
        self.assertIsNotNone(base_url_arg)
        self.assertEqual(base_url_arg.value, f"{BASE_URL}/v1/openai/v1/completions")

    def test_create_model_args_with_base_url_ending_with_v1(self):
        """Test creating model args with base_url ending with /v1."""
        model_name = "test-model"
        base_url = f"{BASE_URL}/v1"
        
        result = self.builder._create_model_args(model_name, base_url)
        
        # Should have model, base_url, and num_concurrent args
        self.assertEqual(len(result), 3)
        
        # Check base_url arg
        base_url_arg = next((arg for arg in result if arg.name == "base_url"), None)
        self.assertIsNotNone(base_url_arg)
        self.assertEqual(base_url_arg.value, f"{BASE_URL}/v1/openai/v1/completions")

    def test_create_model_args_with_base_url_ending_with_slash_v1(self):
        """Test creating model args with base_url ending with /v1/ (trailing slash)."""
        model_name = "test-model"
        base_url = f"{BASE_URL}/v1/"
        
        result = self.builder._create_model_args(model_name, base_url)
        
        # Should have model, base_url, and num_concurrent args
        self.assertEqual(len(result), 3)
        
        # Check base_url arg - trailing slash should be stripped first, then v1 detected
        base_url_arg = next((arg for arg in result if arg.name == "base_url"), None)
        self.assertIsNotNone(base_url_arg)
        self.assertEqual(base_url_arg.value, f"{BASE_URL}/v1/openai/v1/completions")

    def test_create_model_args_with_complex_base_url_ending_with_v1(self):
        """Test creating model args with complex base_url ending with /v1."""
        model_name = "test-model"
        base_url = "https://api.example.com:8080/some/path/v1"
        
        result = self.builder._create_model_args(model_name, base_url)
        
        # Should have model, base_url, and num_concurrent args
        self.assertEqual(len(result), 3)
        
        # Check base_url arg
        base_url_arg = next((arg for arg in result if arg.name == "base_url"), None)
        self.assertIsNotNone(base_url_arg)
        self.assertEqual(base_url_arg.value, "https://api.example.com:8080/some/path/v1/openai/v1/completions")

    def test_create_model_args_with_base_url_containing_v1_but_not_ending(self):
        """Test creating model args with base_url containing v1 but not ending with it."""
        model_name = "test-model"
        base_url = f"{BASE_URL}/v1/something/else"
        
        result = self.builder._create_model_args(model_name, base_url)
        
        # Should have model, base_url, and num_concurrent args
        self.assertEqual(len(result), 3)
        
        # Check base_url arg - should get full /v1/openai/v1/completions since it doesn't end with v1
        base_url_arg = next((arg for arg in result if arg.name == "base_url"), None)
        self.assertIsNotNone(base_url_arg)
        self.assertEqual(base_url_arg.value, f"{BASE_URL}/v1/something/else/v1/openai/v1/completions")

    def test_create_model_args_with_empty_base_url(self):
        """Test creating model args with empty base_url."""
        model_name = "test-model"
        base_url = ""
        
        result = self.builder._create_model_args(model_name, base_url)
        
        # Should have model and num_concurrent args only (empty string is falsy)
        self.assertEqual(len(result), 2)
        
        # Should not have base_url arg
        base_url_arg = next((arg for arg in result if arg.name == "base_url"), None)
        self.assertIsNone(base_url_arg)

    def test_create_model_args_with_trailing_slashes(self):
        """Test creating model args with base_url having multiple trailing slashes."""
        model_name = "test-model"
        base_url = f"{BASE_URL}///"
        
        result = self.builder._create_model_args(model_name, base_url)
        
        # Should have model, base_url, and num_concurrent args
        self.assertEqual(len(result), 3)
        
        # Check base_url arg - trailing slashes should be stripped
        base_url_arg = next((arg for arg in result if arg.name == "base_url"), None)
        self.assertIsNotNone(base_url_arg)
        self.assertEqual(base_url_arg.value, f"{BASE_URL}/v1/openai/v1/completions")

    def test_create_model_args_with_v1_and_trailing_slashes(self):
        """Test creating model args with base_url ending with v1 and trailing slashes."""
        model_name = "test-model"
        base_url = f"{BASE_URL}/v1///"
        
        result = self.builder._create_model_args(model_name, base_url)
        
        # Should have model, base_url, and num_concurrent args
        self.assertEqual(len(result), 3)
        
        # Check base_url arg - trailing slashes should be stripped first, then v1 detected
        base_url_arg = next((arg for arg in result if arg.name == "base_url"), None)
        self.assertIsNotNone(base_url_arg)
        self.assertEqual(base_url_arg.value, f"{BASE_URL}/v1/openai/v1/completions")

    def test_model_arg_type_consistency(self):
        """Test that all returned objects are ModelArg instances."""
        model_name = "test-model"
        base_url = f"{BASE_URL}/v1"
        
        result = self.builder._create_model_args(model_name, base_url)
        
        # All results should be ModelArg instances
        for arg in result:
            self.assertIsInstance(arg, ModelArg)
            self.assertIsInstance(arg.name, str)
            self.assertIsInstance(arg.value, str)


if __name__ == "__main__":
    unittest.main() 