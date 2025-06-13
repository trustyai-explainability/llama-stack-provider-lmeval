import os
import unittest
from pathlib import Path
from unittest.mock import patch, mock_open
import sys
sys.path.insert(0, 'src')

from llama_stack_provider_lmeval.config import LMEvalEvalProviderConfig
from llama_stack_provider_lmeval.errors import LMEvalConfigError
from llama_stack_provider_lmeval.lmeval import _resolve_namespace


class TestNamespaceResolution(unittest.TestCase):
    """Test the namespace resolution."""

    def setUp(self):
        """Set up test fixtures."""
        env_vars_to_clear = [
            'TRUSTYAI_LM_EVAL_NAMESPACE',
            'POD_NAMESPACE', 
            'NAMESPACE'
        ]
        for var in env_vars_to_clear:
            if var in os.environ:
                del os.environ[var]

    def tearDown(self):
        """Clean up after each test."""
        env_vars_to_clear = [
            'TRUSTYAI_LM_EVAL_NAMESPACE',
            'POD_NAMESPACE',
            'NAMESPACE'
        ]
        for var in env_vars_to_clear:
            if var in os.environ:
                del os.environ[var]

    def test_namespace_from_provider_config(self):
        """Test namespace resolution from provider config."""
        config = LMEvalEvalProviderConfig(namespace="test-namespace")
        
        with patch('llama_stack_provider_lmeval.lmeval.logger') as mock_logger:
            namespace = _resolve_namespace(config)
            
        self.assertEqual(namespace, "test-namespace")
        mock_logger.debug.assert_called_with("Using namespace from provider config: test-namespace")

    def test_namespace_respects_any_config_value(self):
        """Test that any namespace value in provider config is respected."""
        config = LMEvalEvalProviderConfig(namespace="default")
        
        with patch('llama_stack_provider_lmeval.lmeval.logger') as mock_logger:
            namespace = _resolve_namespace(config)
            
        self.assertEqual(namespace, "default")
        mock_logger.debug.assert_called_with("Using namespace from provider config: default")
        
        config_test = LMEvalEvalProviderConfig(namespace="test")
        
        with patch('llama_stack_provider_lmeval.lmeval.logger') as mock_logger:
            namespace = _resolve_namespace(config_test)
            
        self.assertEqual(namespace, "test")
        mock_logger.debug.assert_called_with("Using namespace from provider config: test")

    def test_namespace_from_trustyai_env_var(self):
        """Test namespace resolution from TRUSTYAI_LM_EVAL_NAMESPACE environment variable."""
        config = LMEvalEvalProviderConfig()
        os.environ['TRUSTYAI_LM_EVAL_NAMESPACE'] = 'trustyai-namespace'
        
        with patch('llama_stack_provider_lmeval.lmeval.logger') as mock_logger:
            namespace = _resolve_namespace(config)
            
        self.assertEqual(namespace, "trustyai-namespace")
        mock_logger.debug.assert_called_with("Using namespace from environment variable: trustyai-namespace")

    @patch('llama_stack_provider_lmeval.lmeval.Path')
    def test_namespace_from_service_account_file(self, mock_path):
        """Test namespace resolution from service account file."""
        config = LMEvalEvalProviderConfig()
        
        mock_path_instance = mock_path.return_value
        mock_path_instance.exists.return_value = True
        
        with patch('builtins.open', mock_open(read_data='service-account-namespace')):
            with patch('llama_stack_provider_lmeval.lmeval.logger') as mock_logger:
                namespace = _resolve_namespace(config)
        
        self.assertEqual(namespace, "service-account-namespace")
        mock_logger.debug.assert_called_with("Using namespace from service account: service-account-namespace")

    @patch('llama_stack_provider_lmeval.lmeval.Path')
    def test_namespace_from_empty_service_account_file(self, mock_path):
        """Test namespace resolution when service account file is empty or whitespace."""

        config = LMEvalEvalProviderConfig()
        os.environ['TRUSTYAI_LM_EVAL_NAMESPACE'] = 'trustyai-namespace'
        
        mock_path_instance = mock_path.return_value
        mock_path_instance.exists.return_value = True
        
        with patch('builtins.open', mock_open(read_data='')):
            with patch('llama_stack_provider_lmeval.lmeval.logger') as mock_logger:
                namespace = _resolve_namespace(config)
        
        self.assertEqual(namespace, "trustyai-namespace")
        mock_logger.debug.assert_called_with("Using namespace from environment variable: trustyai-namespace")

    @patch('llama_stack_provider_lmeval.lmeval.Path')
    def test_namespace_from_whitespace_service_account_file(self, mock_path):
        """Test namespace resolution when service account file contains only whitespace."""
        
        config = LMEvalEvalProviderConfig()
        os.environ['POD_NAMESPACE'] = 'pod-namespace'
        
        mock_path_instance = mock_path.return_value
        mock_path_instance.exists.return_value = True
        
        with patch('builtins.open', mock_open(read_data='   \n\t  ')):
            with patch('llama_stack_provider_lmeval.lmeval.logger') as mock_logger:
                namespace = _resolve_namespace(config)
        
        self.assertEqual(namespace, "pod-namespace")
        mock_logger.debug.assert_called_with("Using namespace from POD_NAMESPACE environment variable: pod-namespace")

    @patch('llama_stack_provider_lmeval.lmeval.Path')
    def test_namespace_from_pod_namespace_env_var(self, mock_path):
        """Test namespace resolution from POD_NAMESPACE environment variable."""
        config = LMEvalEvalProviderConfig()
        os.environ['POD_NAMESPACE'] = 'pod-namespace'
        
        mock_path_instance = mock_path.return_value
        mock_path_instance.exists.return_value = False
        
        with patch('llama_stack_provider_lmeval.lmeval.logger') as mock_logger:
            namespace = _resolve_namespace(config)
            
        self.assertEqual(namespace, "pod-namespace")
        mock_logger.debug.assert_called_with("Using namespace from POD_NAMESPACE environment variable: pod-namespace")

    @patch('llama_stack_provider_lmeval.lmeval.Path')
    def test_namespace_from_namespace_env_var(self, mock_path):
        """Test namespace resolution from NAMESPACE environment variable."""
        config = LMEvalEvalProviderConfig()
        os.environ['NAMESPACE'] = 'generic-namespace'
        
        mock_path_instance = mock_path.return_value
        mock_path_instance.exists.return_value = False
        
        with patch('llama_stack_provider_lmeval.lmeval.logger') as mock_logger:
            namespace = _resolve_namespace(config)
            
        self.assertEqual(namespace, "generic-namespace")
        mock_logger.debug.assert_called_with("Using namespace from NAMESPACE environment variable: generic-namespace")

    @patch('llama_stack_provider_lmeval.lmeval.Path')
    def test_namespace_resolution_priority(self, mock_path):
        """Test that namespace resolution follows right order."""
        config = LMEvalEvalProviderConfig(namespace="config-namespace")
        os.environ['TRUSTYAI_LM_EVAL_NAMESPACE'] = 'trustyai-namespace'
        os.environ['POD_NAMESPACE'] = 'pod-namespace'
        
        mock_path_instance = mock_path.return_value
        mock_path_instance.exists.return_value = True
        
        with patch('builtins.open', mock_open(read_data='service-account-namespace')):
            with patch('llama_stack_provider_lmeval.lmeval.logger') as mock_logger:
                namespace = _resolve_namespace(config)
        
        self.assertEqual(namespace, "config-namespace")
        mock_logger.debug.assert_called_with("Using namespace from provider config: config-namespace")

    @patch('llama_stack_provider_lmeval.lmeval.Path')
    def test_namespace_resolution_failure(self, mock_path):
        """Test that function raises exception when no namespace is found."""
        config = LMEvalEvalProviderConfig()
        
        mock_path_instance = mock_path.return_value
        mock_path_instance.exists.return_value = False
        
        with self.assertRaises(LMEvalConfigError) as context:
            _resolve_namespace(config)
        
        error_msg = str(context.exception)
        self.assertIn("Unable to determine namespace", error_msg)
        self.assertIn("Set 'namespace' in your run.yaml provider config", error_msg)
        self.assertIn("Set TRUSTYAI_LM_EVAL_NAMESPACE environment variable", error_msg)


if __name__ == '__main__':
    unittest.main() 