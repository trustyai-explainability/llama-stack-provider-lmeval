import json
import unittest
from unittest.mock import patch, MagicMock
import sys
sys.path.insert(0, 'src')

from llama_stack_provider_lmeval.lmeval import LMEvalCRBuilder
from llama_stack_provider_lmeval.config import LMEvalBenchmarkConfig
from llama_stack.apis.eval import EvalCandidate
from llama_stack.apis.benchmarks import Benchmark


class TestEnvironmentVariables(unittest.TestCase):
    """Test environment variable handling in CR creation."""

    def setUp(self):
        """Set up test fixtures."""
        self.cr_builder = LMEvalCRBuilder(namespace="test-namespace")
        self.cr_builder._config = MagicMock()
        self.cr_builder._config.tls = None

    def test_simple_env_var_handling(self):
        """Test handling of simple string environment variables."""
        env_vars = [
            {"name": "SIMPLE_VAR", "value": "simple_value"},
            {"name": "ANOTHER_VAR", "value": "another_value"}
        ]
        
        pod_config = self.cr_builder._create_pod_config(env_vars)
        
        self.assertIsNotNone(pod_config)
        self.assertIsNotNone(pod_config.container.env)
        self.assertEqual(len(pod_config.container.env), 2)
        
        # Check first environment variable
        self.assertEqual(pod_config.container.env[0]["name"], "SIMPLE_VAR")
        self.assertEqual(pod_config.container.env[0]["value"], "simple_value")
        
        # Check second environment variable
        self.assertEqual(pod_config.container.env[1]["name"], "ANOTHER_VAR")
        self.assertEqual(pod_config.container.env[1]["value"], "another_value")

    def test_valuefrom_dict_structure(self):
        """Test handling of valueFrom structure when passed as dict."""
        env_vars = [
            {
                "name": "OPENAI_API_KEY",
                "value": {
                    "valueFrom": {
                        "secretKeyRef": {
                            "name": "user-one-token",
                            "key": "token"
                        }
                    }
                }
            }
        ]
        
        pod_config = self.cr_builder._create_pod_config(env_vars)
        
        self.assertIsNotNone(pod_config)
        self.assertIsNotNone(pod_config.container.env)
        self.assertEqual(len(pod_config.container.env), 1)
        
        env_var = pod_config.container.env[0]
        self.assertEqual(env_var["name"], "OPENAI_API_KEY")
        self.assertIn("valueFrom", env_var)
        self.assertEqual(env_var["valueFrom"]["secretKeyRef"]["name"], "user-one-token")
        self.assertEqual(env_var["valueFrom"]["secretKeyRef"]["key"], "token")
        # Should not have a "value" field
        self.assertNotIn("value", env_var)

    def test_valuefrom_stringified_structure(self):
        """Test handling of valueFrom structure when passed as stringified dict."""
        # Simulate the problematic case from the user's YAML
        stringified_value = "{'valueFrom': {'secretKeyRef': {'name': 'user-one-token', 'key': 'token'}}}"
        env_vars = [
            {
                "name": "OPENAI_API_KEY",
                "value": stringified_value
            }
        ]
        
        pod_config = self.cr_builder._create_pod_config(env_vars)
        
        self.assertIsNotNone(pod_config)
        self.assertIsNotNone(pod_config.container.env)
        self.assertEqual(len(pod_config.container.env), 1)
        
        env_var = pod_config.container.env[0]
        self.assertEqual(env_var["name"], "OPENAI_API_KEY")
        self.assertIn("valueFrom", env_var)
        self.assertEqual(env_var["valueFrom"]["secretKeyRef"]["name"], "user-one-token")
        self.assertEqual(env_var["valueFrom"]["secretKeyRef"]["key"], "token")
        # Should not have a "value" field
        self.assertNotIn("value", env_var)

    def test_malformed_stringified_structure_fallback(self):
        """Test handling of malformed stringified structure falls back to simple value."""
        env_vars = [
            {
                "name": "MALFORMED_VAR",
                "value": "{'invalid': 'structure"  # Missing closing brace
            }
        ]
        
        pod_config = self.cr_builder._create_pod_config(env_vars)
        
        self.assertIsNotNone(pod_config)
        self.assertIsNotNone(pod_config.container.env)
        self.assertEqual(len(pod_config.container.env), 1)
        
        env_var = pod_config.container.env[0]
        self.assertEqual(env_var["name"], "MALFORMED_VAR")
        self.assertEqual(env_var["value"], "{'invalid': 'structure")
        # Should not have a "valueFrom" field
        self.assertNotIn("valueFrom", env_var)

    def test_mixed_env_vars(self):
        """Test handling of mixed simple and valueFrom environment variables."""
        env_vars = [
            {"name": "SIMPLE_VAR", "value": "simple_value"},
            {
                "name": "SECRET_VAR",
                "value": {
                    "valueFrom": {
                        "secretKeyRef": {
                            "name": "my-secret",
                            "key": "secret-key"
                        }
                    }
                }
            },
            {"name": "ANOTHER_SIMPLE", "value": "another_simple_value"}
        ]
        
        pod_config = self.cr_builder._create_pod_config(env_vars)
        
        self.assertIsNotNone(pod_config)
        self.assertIsNotNone(pod_config.container.env)
        self.assertEqual(len(pod_config.container.env), 3)
        
        # Check simple variable
        simple_var = pod_config.container.env[0]
        self.assertEqual(simple_var["name"], "SIMPLE_VAR")
        self.assertEqual(simple_var["value"], "simple_value")
        self.assertNotIn("valueFrom", simple_var)
        
        # Check secret variable
        secret_var = pod_config.container.env[1]
        self.assertEqual(secret_var["name"], "SECRET_VAR")
        self.assertIn("valueFrom", secret_var)
        self.assertEqual(secret_var["valueFrom"]["secretKeyRef"]["name"], "my-secret")
        self.assertEqual(secret_var["valueFrom"]["secretKeyRef"]["key"], "secret-key")
        self.assertNotIn("value", secret_var)
        
        # Check another simple variable
        another_simple = pod_config.container.env[2]
        self.assertEqual(another_simple["name"], "ANOTHER_SIMPLE")
        self.assertEqual(another_simple["value"], "another_simple_value")
        self.assertNotIn("valueFrom", another_simple)

    def test_empty_env_vars(self):
        """Test handling of empty environment variables list."""
        pod_config = self.cr_builder._create_pod_config([])
        
        # Should return None if no env vars and no service account
        self.assertIsNone(pod_config)

    def test_no_env_vars_with_service_account(self):
        """Test handling of no env vars but with service account."""
        self.cr_builder._service_account = "test-service-account"
        
        pod_config = self.cr_builder._create_pod_config([])
        
        self.assertIsNotNone(pod_config)
        self.assertEqual(pod_config.serviceAccountName, "test-service-account")
        self.assertIsNone(pod_config.container.env)

    def test_configmap_valuefrom(self):
        """Test handling of ConfigMap valueFrom structure."""
        env_vars = [
            {
                "name": "CONFIG_VAR",
                "value": {
                    "valueFrom": {
                        "configMapKeyRef": {
                            "name": "my-config",
                            "key": "config-key"
                        }
                    }
                }
            }
        ]
        
        pod_config = self.cr_builder._create_pod_config(env_vars)
        
        self.assertIsNotNone(pod_config)
        self.assertIsNotNone(pod_config.container.env)
        self.assertEqual(len(pod_config.container.env), 1)
        
        env_var = pod_config.container.env[0]
        self.assertEqual(env_var["name"], "CONFIG_VAR")
        self.assertIn("valueFrom", env_var)
        self.assertEqual(env_var["valueFrom"]["configMapKeyRef"]["name"], "my-config")
        self.assertEqual(env_var["valueFrom"]["configMapKeyRef"]["key"], "config-key")
        self.assertNotIn("value", env_var)

    def test_custom_secret_structure(self):
        """Test handling of custom secret structure with name/value/secret fields."""
        env_vars = [
            {
                "name": "OPENAI_API_KEY",
                "value": "",  # Empty value when using secret
                "secret": {
                    "name": "user-one-token",
                    "key": "token"
                }
            }
        ]
        
        pod_config = self.cr_builder._create_pod_config(env_vars)
        
        self.assertIsNotNone(pod_config)
        self.assertIsNotNone(pod_config.container.env)
        self.assertEqual(len(pod_config.container.env), 1)
        
        env_var = pod_config.container.env[0]
        self.assertEqual(env_var["name"], "OPENAI_API_KEY")
        self.assertIn("valueFrom", env_var)
        self.assertEqual(env_var["valueFrom"]["secretKeyRef"]["name"], "user-one-token")
        self.assertEqual(env_var["valueFrom"]["secretKeyRef"]["key"], "token")
        # Should not have a "value" field
        self.assertNotIn("value", env_var)

    def test_custom_secret_structure_without_value_field(self):
        """Test handling of custom secret structure without value field."""
        env_vars = [
            {
                "name": "API_TOKEN",
                "secret": {
                    "name": "api-secret",
                    "key": "api-token"
                }
            }
        ]
        
        pod_config = self.cr_builder._create_pod_config(env_vars)
        
        self.assertIsNotNone(pod_config)
        self.assertIsNotNone(pod_config.container.env)
        self.assertEqual(len(pod_config.container.env), 1)
        
        env_var = pod_config.container.env[0]
        self.assertEqual(env_var["name"], "API_TOKEN")
        self.assertIn("valueFrom", env_var)
        self.assertEqual(env_var["valueFrom"]["secretKeyRef"]["name"], "api-secret")
        self.assertEqual(env_var["valueFrom"]["secretKeyRef"]["key"], "api-token")
        self.assertNotIn("value", env_var)

    def test_invalid_custom_secret_structure_fallback(self):
        """Test handling of invalid custom secret structure falls back to value."""
        env_vars = [
            {
                "name": "INVALID_SECRET_VAR",
                "value": "fallback_value",
                "secret": {
                    "name": "secret-name"
                    # Missing "key" field
                }
            }
        ]
        
        pod_config = self.cr_builder._create_pod_config(env_vars)
        
        self.assertIsNotNone(pod_config)
        self.assertIsNotNone(pod_config.container.env)
        self.assertEqual(len(pod_config.container.env), 1)
        
        env_var = pod_config.container.env[0]
        self.assertEqual(env_var["name"], "INVALID_SECRET_VAR")
        self.assertEqual(env_var["value"], "fallback_value")
        self.assertNotIn("valueFrom", env_var)

    def test_mixed_simple_and_custom_secret_vars(self):
        """Test handling of mixed simple values and custom secret structures."""
        env_vars = [
            {"name": "SIMPLE_VAR", "value": "simple_value"},
            {
                "name": "SECRET_VAR",
                "value": "",
                "secret": {
                    "name": "my-secret",
                    "key": "secret-key"
                }
            },
            {"name": "ANOTHER_SIMPLE", "value": "another_value"}
        ]
        
        pod_config = self.cr_builder._create_pod_config(env_vars)
        
        self.assertIsNotNone(pod_config)
        self.assertIsNotNone(pod_config.container.env)
        self.assertEqual(len(pod_config.container.env), 3)
        
        # Check simple variable
        simple_var = pod_config.container.env[0]
        self.assertEqual(simple_var["name"], "SIMPLE_VAR")
        self.assertEqual(simple_var["value"], "simple_value")
        self.assertNotIn("valueFrom", simple_var)
        
        # Check custom secret variable
        secret_var = pod_config.container.env[1]
        self.assertEqual(secret_var["name"], "SECRET_VAR")
        self.assertIn("valueFrom", secret_var)
        self.assertEqual(secret_var["valueFrom"]["secretKeyRef"]["name"], "my-secret")
        self.assertEqual(secret_var["valueFrom"]["secretKeyRef"]["key"], "secret-key")
        self.assertNotIn("value", secret_var)
        
        # Check another simple variable
        another_simple = pod_config.container.env[2]
        self.assertEqual(another_simple["name"], "ANOTHER_SIMPLE")
        self.assertEqual(another_simple["value"], "another_value")
        self.assertNotIn("valueFrom", another_simple)

    def test_collect_env_vars_from_metadata_dict(self):
        """Test collecting environment variables from metadata dictionary format."""
        # Create a mock task config with metadata
        task_config = MagicMock()
        task_config.env_vars = None
        task_config.metadata = {
            "env": {
                "DK_BENCH_DATASET_PATH": "/opt/app-root/src/hf_home/upload-files/example-dk-bench-input-bmo.jsonl",
                "JUDGE_MODEL_URL": "http://vllm-server:8000/v1/chat/completions",
                "JUDGE_MODEL_NAME": "MODEL",
                "JUDGE_API_KEY": {"secret": {"name": "user-one-token", "key": "token"}}
            }
        }
        
        # Collect environment variables
        env_vars = self.cr_builder._collect_env_vars(task_config, None)
        
        # Should have 4 environment variables
        self.assertEqual(len(env_vars), 4)
        
        # Check simple environment variables
        simple_vars = {env["name"]: env for env in env_vars if "value" in env}
        self.assertIn("DK_BENCH_DATASET_PATH", simple_vars)
        self.assertEqual(simple_vars["DK_BENCH_DATASET_PATH"]["value"], "/opt/app-root/src/hf_home/upload-files/example-dk-bench-input-bmo.jsonl")
        self.assertIn("JUDGE_MODEL_URL", simple_vars)
        self.assertEqual(simple_vars["JUDGE_MODEL_URL"]["value"], "http://vllm-server:8000/v1/chat/completions")
        self.assertIn("JUDGE_MODEL_NAME", simple_vars)
        self.assertEqual(simple_vars["JUDGE_MODEL_NAME"]["value"], "MODEL")
        
        # Check secret environment variable
        secret_vars = {env["name"]: env for env in env_vars if "secret" in env}
        self.assertIn("JUDGE_API_KEY", secret_vars)
        self.assertEqual(secret_vars["JUDGE_API_KEY"]["secret"]["name"], "user-one-token")
        self.assertEqual(secret_vars["JUDGE_API_KEY"]["secret"]["key"], "token")

    def test_full_integration_metadata_to_cr(self):
        """Test full integration from metadata dictionary to final CR environment variables."""
        # Create a mock task config with metadata
        task_config = MagicMock()
        task_config.env_vars = None
        task_config.metadata = {
            "env": {
                "SIMPLE_VAR": "simple_value",
                "SECRET_VAR": {"secret": {"name": "my-secret", "key": "secret-key"}}
            }
        }
        
        # Collect environment variables
        env_vars = self.cr_builder._collect_env_vars(task_config, None)
        
        # Create pod config
        pod_config = self.cr_builder._create_pod_config(env_vars)
        
        self.assertIsNotNone(pod_config)
        self.assertIsNotNone(pod_config.container.env)
        self.assertEqual(len(pod_config.container.env), 2)
        
        # Find variables by name
        env_by_name = {env["name"]: env for env in pod_config.container.env}
        
        # Check simple variable
        self.assertIn("SIMPLE_VAR", env_by_name)
        simple_var = env_by_name["SIMPLE_VAR"]
        self.assertEqual(simple_var["value"], "simple_value")
        self.assertNotIn("valueFrom", simple_var)
        
        # Check secret variable
        self.assertIn("SECRET_VAR", env_by_name)
        secret_var = env_by_name["SECRET_VAR"]
        self.assertIn("valueFrom", secret_var)
        self.assertEqual(secret_var["valueFrom"]["secretKeyRef"]["name"], "my-secret")
        self.assertEqual(secret_var["valueFrom"]["secretKeyRef"]["key"], "secret-key")
        self.assertNotIn("value", secret_var)

    def test_logging_debug_output(self):
        """Test that debug logging is called with environment variable names only (no sensitive data)."""
        env_vars = [
            {"name": "TEST_VAR", "value": "test_value"},
            {"name": "SECRET_VAR", "secret": {"name": "my-secret", "key": "secret-key"}}
        ]
        
        with patch('llama_stack_provider_lmeval.lmeval.logger') as mock_logger:
            pod_config = self.cr_builder._create_pod_config(env_vars)
            
            # Verify that debug logging was called
            mock_logger.debug.assert_called()
            
            # Check that the debug message includes only variable names, not values or secrets
            debug_call_args = mock_logger.debug.call_args[0][0]
            self.assertIn("Setting pod environment variables:", debug_call_args)
            self.assertIn("TEST_VAR", debug_call_args)
            self.assertIn("SECRET_VAR", debug_call_args)
            # Ensure no sensitive data is logged
            self.assertNotIn("test_value", debug_call_args)
            self.assertNotIn("my-secret", debug_call_args)
            self.assertNotIn("secret-key", debug_call_args)
            self.assertNotIn("valueFrom", debug_call_args)


if __name__ == '__main__':
    unittest.main() 