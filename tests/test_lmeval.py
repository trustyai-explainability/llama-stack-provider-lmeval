"""Tests for the LMEval provider functionality."""

import unittest
from unittest.mock import patch, MagicMock
import os

from src.llama_stack_provider_lmeval.config import LMEvalEvalProviderConfig, TLSConfig
from src.llama_stack_provider_lmeval.lmeval import LMEvalCRBuilder, _get_tls_config_from_env


class TestTLSConfigFromEnv(unittest.TestCase):
    """Test the _get_tls_config_from_env function."""

    def setUp(self):
        """Test fixtures."""
        # Clear any existing environment variables
        self.env_vars_to_clear = [
            "TRUSTYAI_LMEVAL_TLS",
            "TRUSTYAI_LMEVAL_CERT_FILE",
            "TRUSTYAI_LMEVAL_CERT_SECRET"
        ]
        self.original_env = {}
        for var in self.env_vars_to_clear:
            if var in os.environ:
                self.original_env[var] = os.environ[var]
                del os.environ[var]

    def tearDown(self):
        """Restore original environment variables."""
        for var in self.env_vars_to_clear:
            if var in self.original_env:
                os.environ[var] = self.original_env[var]
            elif var in os.environ:
                del os.environ[var]

    def test_no_tls_environment_variables(self):
        """Test when no TLS environment variables are set."""
        result = _get_tls_config_from_env()
        self.assertIsNone(result)

    def test_tls_disabled_via_environment(self):
        """Test when TRUSTYAI_LMEVAL_TLS is explicitly set to false."""
        os.environ["TRUSTYAI_LMEVAL_TLS"] = "false"
        result = _get_tls_config_from_env()
        self.assertIsNone(result)

    def test_tls_enabled_with_both_cert_variables(self):
        """Test when TLS is enabled and both certificate variables are set."""
        os.environ["TRUSTYAI_LMEVAL_TLS"] = "true"
        os.environ["TRUSTYAI_LMEVAL_CERT_FILE"] = "test-cert.pem"
        os.environ["TRUSTYAI_LMEVAL_CERT_SECRET"] = "test-secret"

        result = _get_tls_config_from_env()
        expected_path = "/etc/ssl/certs/test-cert.pem"
        self.assertEqual(result, expected_path)

    def test_tls_enabled_with_only_cert_file(self):
        """Test when TLS is enabled but only TRUSTYAI_LMEVAL_CERT_FILE is set."""
        os.environ["TRUSTYAI_LMEVAL_TLS"] = "true"
        os.environ["TRUSTYAI_LMEVAL_CERT_FILE"] = "test-cert.pem"
        # TRUSTYAI_LMEVAL_CERT_SECRET is not set

        result = _get_tls_config_from_env()
        self.assertIsNone(result)

    def test_tls_enabled_with_only_cert_secret(self):
        """Test when TLS is enabled but only TRUSTYAI_LMEVAL_CERT_SECRET is set."""
        os.environ["TRUSTYAI_LMEVAL_TLS"] = "true"
        os.environ["TRUSTYAI_LMEVAL_CERT_SECRET"] = "test-secret"
        # TRUSTYAI_LMEVAL_CERT_FILE is not set

        result = _get_tls_config_from_env()
        self.assertIsNone(result)

    def test_tls_enabled_with_no_cert_variables(self):
        """Test when TLS is enabled but no certificate variables are set."""
        os.environ["TRUSTYAI_LMEVAL_TLS"] = "true"
        # Neither TRUSTYAI_LMEVAL_CERT_FILE nor TRUSTYAI_LMEVAL_CERT_SECRET are set

        result = _get_tls_config_from_env()
        self.assertTrue(result)

    def test_fallback_to_provider_config_when_cert_file_missing(self):
        """Test fallback to provider config when only cert_file is missing."""
        os.environ["TRUSTYAI_LMEVAL_TLS"] = "true"
        os.environ["TRUSTYAI_LMEVAL_CERT_SECRET"] = "test-secret"
        # TRUSTYAI_LMEVAL_CERT_FILE is not set

        provider_config = MagicMock()
        provider_config.tls = MagicMock()
        provider_config.tls.enable = True
        provider_config.tls.cert_file = "provider-cert.pem"
        provider_config.tls.cert_secret = "provider-secret"

        result = _get_tls_config_from_env(provider_config)
        expected_path = "/etc/ssl/certs/provider-cert.pem"
        self.assertEqual(result, expected_path)

    def test_fallback_to_provider_config_when_cert_secret_missing(self):
        """Test fallback to provider config when only cert_secret is missing."""
        os.environ["TRUSTYAI_LMEVAL_TLS"] = "true"
        os.environ["TRUSTYAI_LMEVAL_CERT_FILE"] = "test-cert.pem"
        # TRUSTYAI_LMEVAL_CERT_SECRET is not set

        provider_config = MagicMock()
        provider_config.tls = MagicMock()
        provider_config.tls.enable = True
        provider_config.tls.cert_file = "provider-cert.pem"
        provider_config.tls.cert_secret = "provider-secret"

        result = _get_tls_config_from_env(provider_config)
        expected_path = "/etc/ssl/certs/provider-cert.pem"
        self.assertEqual(result, expected_path)

    def test_fallback_to_provider_config_tls_true_when_cert_file_missing(self):
        """Test fallback to provider config TLS=True when cert_file is missing."""
        os.environ["TRUSTYAI_LMEVAL_TLS"] = "true"
        os.environ["TRUSTYAI_LMEVAL_CERT_SECRET"] = "test-secret"
        # TRUSTYAI_LMEVAL_CERT_FILE is not set

        provider_config = MagicMock()
        provider_config.tls = MagicMock()
        provider_config.tls.enable = True
        provider_config.tls.cert_file = None
        provider_config.tls.cert_secret = None

        result = _get_tls_config_from_env(provider_config)
        self.assertTrue(result)

    def test_fallback_to_provider_config_tls_true_when_cert_secret_missing(self):
        """Test fallback to provider config TLS=True when cert_secret is missing."""
        os.environ["TRUSTYAI_LMEVAL_TLS"] = "true"
        os.environ["TRUSTYAI_LMEVAL_CERT_FILE"] = "test-cert.pem"
        # TRUSTYAI_LMEVAL_CERT_SECRET is not set

        provider_config = MagicMock()
        provider_config.tls = MagicMock()
        provider_config.tls.enable = True
        provider_config.tls.cert_file = None
        provider_config.tls.cert_secret = None

        result = _get_tls_config_from_env(provider_config)
        self.assertTrue(result)

    def test_no_fallback_when_provider_config_tls_disabled(self):
        """Test no fallback when provider config TLS is disabled."""
        os.environ["TRUSTYAI_LMEVAL_TLS"] = "true"
        os.environ["TRUSTYAI_LMEVAL_CERT_FILE"] = "test-cert.pem"
        # TRUSTYAI_LMEVAL_CERT_SECRET is not set

        provider_config = MagicMock()
        provider_config.tls = MagicMock()
        provider_config.tls.enable = False

        result = _get_tls_config_from_env(provider_config)
        self.assertIsNone(result)

    def test_no_fallback_when_provider_config_tls_none(self):
        """Test no fallback when provider config TLS is None."""
        os.environ["TRUSTYAI_LMEVAL_TLS"] = "true"
        os.environ["TRUSTYAI_LMEVAL_CERT_FILE"] = "test-cert.pem"
        # TRUSTYAI_LMEVAL_CERT_SECRET is not set

        provider_config = MagicMock()
        provider_config.tls = None

        result = _get_tls_config_from_env(provider_config)
        self.assertIsNone(result)

    def test_no_fallback_when_provider_config_has_no_tls_attr(self):
        """Test no fallback when provider config has no tls attribute."""
        os.environ["TRUSTYAI_LMEVAL_TLS"] = "true"
        os.environ["TRUSTYAI_LMEVAL_CERT_FILE"] = "test-cert.pem"
        # TRUSTYAI_LMEVAL_CERT_SECRET is not set

        # Use a regular object instead of MagicMock to avoid automatic attribute creation
        class MockProviderConfig:
            pass

        provider_config = MockProviderConfig()

        result = _get_tls_config_from_env(provider_config)
        self.assertIsNone(result)

    def test_provider_config_fallback_when_env_tls_disabled(self):
        """Test provider config fallback when environment TLS is disabled."""
        os.environ["TRUSTYAI_LMEVAL_TLS"] = "false"
        # Environment TLS is disabled

        provider_config = MagicMock()
        provider_config.tls = MagicMock()
        provider_config.tls.enable = True
        provider_config.tls.cert_file = "provider-cert.pem"
        provider_config.tls.cert_secret = "provider-secret"

        result = _get_tls_config_from_env(provider_config)
        expected_path = "/etc/ssl/certs/provider-cert.pem"
        self.assertEqual(result, expected_path)

    def test_provider_config_fallback_when_env_tls_disabled_no_certs(self):
        """Test provider config fallback when environment TLS is disabled and no certs in provider config."""
        os.environ["TRUSTYAI_LMEVAL_TLS"] = "false"
        # Environment TLS is disabled

        provider_config = MagicMock()
        provider_config.tls = MagicMock()
        provider_config.tls.enable = True
        provider_config.tls.cert_file = None
        provider_config.tls.cert_secret = None

        result = _get_tls_config_from_env(provider_config)
        self.assertTrue(result)

    def test_provider_config_fallback_when_env_tls_disabled_incomplete_certs(self):
        """Test provider config fallback when environment TLS is disabled and incomplete certs in provider config."""
        os.environ["TRUSTYAI_LMEVAL_TLS"] = "false"
        # Environment TLS is disabled

        provider_config = MagicMock()
        provider_config.tls = MagicMock()
        provider_config.tls.enable = True
        provider_config.tls.cert_file = "provider-cert.pem"
        provider_config.tls.cert_secret = None  # Incomplete

        result = _get_tls_config_from_env(provider_config)
        self.assertTrue(result)

    def test_case_insensitive_tls_enabled(self):
        """Test that TLS enabled is case insensitive."""
        os.environ["TRUSTYAI_LMEVAL_TLS"] = "TRUE"
        os.environ["TRUSTYAI_LMEVAL_CERT_FILE"] = "test-cert.pem"
        os.environ["TRUSTYAI_LMEVAL_CERT_SECRET"] = "test-secret"

        result = _get_tls_config_from_env()
        expected_path = "/etc/ssl/certs/test-cert.pem"
        self.assertEqual(result, expected_path)

    def test_case_insensitive_tls_disabled(self):
        """Test that TLS disabled is case insensitive."""
        os.environ["TRUSTYAI_LMEVAL_TLS"] = "FALSE"

        result = _get_tls_config_from_env()
        self.assertIsNone(result)

    @patch("src.llama_stack_provider_lmeval.lmeval.logger")
    def test_tls_volume_config_with_empty_cert_file(self, mock_logger):
        """Test TLS volume config when cert_file is empty string."""
        os.environ["TRUSTYAI_LMEVAL_TLS"] = "true"
        os.environ["TRUSTYAI_LMEVAL_CERT_FILE"] = ""
        os.environ["TRUSTYAI_LMEVAL_CERT_SECRET"] = "test-secret"

        from src.llama_stack_provider_lmeval.lmeval import _create_tls_volume_config
        volume_mounts, volumes = _create_tls_volume_config()

        # Should return None, None due to validation failure
        self.assertIsNone(volume_mounts)
        self.assertIsNone(volumes)

        # Should log warning about validation failure
        mock_logger.warning.assert_called()
        warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
        self.assertTrue(any("TLS configuration validation failed" in call for call in warning_calls))

    @patch("src.llama_stack_provider_lmeval.lmeval.logger")
    def test_tls_volume_config_with_empty_cert_secret(self, mock_logger):
        """Test TLS volume config when cert_secret is empty string."""
        os.environ["TRUSTYAI_LMEVAL_TLS"] = "true"
        os.environ["TRUSTYAI_LMEVAL_CERT_FILE"] = "test-cert.pem"
        os.environ["TRUSTYAI_LMEVAL_CERT_SECRET"] = ""

        from src.llama_stack_provider_lmeval.lmeval import _create_tls_volume_config
        volume_mounts, volumes = _create_tls_volume_config()

        # Should return None, None due to validation failure
        self.assertIsNone(volume_mounts)
        self.assertIsNone(volumes)

        # Should log warning about validation failure
        mock_logger.warning.assert_called()
        warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
        self.assertTrue(any("TLS configuration validation failed" in call for call in warning_calls))

    @patch("src.llama_stack_provider_lmeval.lmeval.logger")
    def test_tls_volume_config_with_path_traversal_in_cert_file(self, mock_logger):
        """Test TLS volume config when cert_file contains path traversal characters."""
        os.environ["TRUSTYAI_LMEVAL_TLS"] = "true"
        os.environ["TRUSTYAI_LMEVAL_CERT_FILE"] = "../malicious.pem"
        os.environ["TRUSTYAI_LMEVAL_CERT_SECRET"] = "test-secret"

        from src.llama_stack_provider_lmeval.lmeval import _create_tls_volume_config
        volume_mounts, volumes = _create_tls_volume_config()

        # Should return None, None due to validation failure
        self.assertIsNone(volume_mounts)
        self.assertIsNone(volumes)

        # Should log warning about validation failure
        mock_logger.warning.assert_called()
        warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
        self.assertTrue(any("TLS configuration validation failed" in call for call in warning_calls))

    @patch("src.llama_stack_provider_lmeval.lmeval.logger")
    def test_tls_volume_config_with_unsafe_characters_in_cert_file(self, mock_logger):
        """Test TLS volume config when cert_file contains unsafe characters."""
        os.environ["TRUSTYAI_LMEVAL_TLS"] = "true"
        os.environ["TRUSTYAI_LMEVAL_CERT_FILE"] = "test@cert.pem"
        os.environ["TRUSTYAI_LMEVAL_CERT_SECRET"] = "test-secret"

        from src.llama_stack_provider_lmeval.lmeval import _create_tls_volume_config
        volume_mounts, volumes = _create_tls_volume_config()

        # Should return None, None due to validation failure
        self.assertIsNone(volume_mounts)
        self.assertIsNone(volumes)

        # Should log warning about validation failure
        mock_logger.warning.assert_called()
        warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
        self.assertTrue(any("TLS configuration validation failed" in call for call in warning_calls))

    @patch("src.llama_stack_provider_lmeval.lmeval.logger")
    def test_tls_volume_config_with_valid_cert_file(self, mock_logger):
        """Test TLS volume config with valid certificate file name."""
        os.environ["TRUSTYAI_LMEVAL_TLS"] = "true"
        os.environ["TRUSTYAI_LMEVAL_CERT_FILE"] = "test-cert.pem"
        os.environ["TRUSTYAI_LMEVAL_CERT_SECRET"] = "test-secret"

        from src.llama_stack_provider_lmeval.lmeval import _create_tls_volume_config
        volume_mounts, volumes = _create_tls_volume_config()

        # Should return valid volume configuration
        self.assertIsNotNone(volume_mounts)
        self.assertIsNotNone(volumes)
        self.assertEqual(len(volume_mounts), 1)
        self.assertEqual(len(volumes), 1)

        # Verify volume mount configuration
        volume_mount = volume_mounts[0]
        self.assertEqual(volume_mount["name"], "tls-cert")
        self.assertEqual(volume_mount["mountPath"], "/etc/ssl/certs")
        self.assertTrue(volume_mount["readOnly"])

        # Verify volume configuration
        volume = volumes[0]
        self.assertEqual(volume["name"], "tls-cert")
        self.assertEqual(volume["secret"]["secretName"], "test-secret")
        self.assertEqual(len(volume["secret"]["items"]), 1)
        self.assertEqual(volume["secret"]["items"][0]["key"], "test-cert.pem")
        self.assertEqual(volume["secret"]["items"][0]["path"], "test-cert.pem")

        # Should log info about successful creation
        mock_logger.info.assert_called_once()
        info_call_args = mock_logger.info.call_args[0]
        self.assertIn("Created TLS volume config", info_call_args[0])

    @patch("src.llama_stack_provider_lmeval.lmeval.logger")
    def test_tls_volume_config_with_valid_subdirectory_path(self, mock_logger):
        """Test TLS volume config with valid subdirectory path in cert_file."""
        os.environ["TRUSTYAI_LMEVAL_TLS"] = "true"
        os.environ["TRUSTYAI_LMEVAL_CERT_FILE"] = "certs/ca-bundle.pem"
        os.environ["TRUSTYAI_LMEVAL_CERT_SECRET"] = "test-secret"

        from src.llama_stack_provider_lmeval.lmeval import _create_tls_volume_config
        volume_mounts, volumes = _create_tls_volume_config()

        # Should return valid volume configuration
        self.assertIsNotNone(volume_mounts)
        self.assertIsNotNone(volumes)
        self.assertEqual(len(volume_mounts), 1)
        self.assertEqual(len(volumes), 1)

        # Verify volume configuration uses the correct subdirectory path
        volume = volumes[0]
        self.assertEqual(volume["secret"]["items"][0]["key"], "certs/ca-bundle.pem")
        self.assertEqual(volume["secret"]["items"][0]["path"], "certs/ca-bundle.pem")

        # Should log info about successful creation
        mock_logger.info.assert_called_once()

    @patch("src.llama_stack_provider_lmeval.lmeval.logger")
    def test_tls_volume_config_with_whitespace_only_cert_file(self, mock_logger):
        """Test TLS volume config when cert_file contains only whitespace."""
        os.environ["TRUSTYAI_LMEVAL_TLS"] = "true"
        os.environ["TRUSTYAI_LMEVAL_CERT_FILE"] = "   "
        os.environ["TRUSTYAI_LMEVAL_CERT_SECRET"] = "test-secret"

        from src.llama_stack_provider_lmeval.lmeval import _create_tls_volume_config
        volume_mounts, volumes = _create_tls_volume_config()

        # Should return None, None due to validation failure
        self.assertIsNone(volume_mounts)
        self.assertIsNone(volumes)

        # Should log warning about validation failure
        mock_logger.warning.assert_called()
        warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
        self.assertTrue(any("TLS configuration validation failed" in call for call in warning_calls))

    @patch("src.llama_stack_provider_lmeval.lmeval.logger")
    def test_tls_volume_config_with_valid_special_characters(self, mock_logger):
        """Test TLS volume config with valid certificate file name containing dots, hyphens, and underscores."""
        os.environ["TRUSTYAI_LMEVAL_TLS"] = "true"
        os.environ["TRUSTYAI_LMEVAL_CERT_FILE"] = "test-cert-v1.2.pem"
        os.environ["TRUSTYAI_LMEVAL_CERT_SECRET"] = "test-secret"

        from src.llama_stack_provider_lmeval.lmeval import _create_tls_volume_config
        volume_mounts, volumes = _create_tls_volume_config()

        # Should return valid volume configuration
        self.assertIsNotNone(volume_mounts)
        self.assertIsNotNone(volumes)
        self.assertEqual(len(volume_mounts), 1)
        self.assertEqual(len(volumes), 1)

        # Verify volume configuration uses the correct certificate file name
        volume = volumes[0]
        self.assertEqual(volume["secret"]["items"][0]["key"], "test-cert-v1.2.pem")
        self.assertEqual(volume["secret"]["items"][0]["path"], "test-cert-v1.2.pem")

        # Should log info about successful creation
        mock_logger.info.assert_called_once()

    @patch("src.llama_stack_provider_lmeval.lmeval.logger")
    def test_tls_volume_config_with_no_certificates(self, mock_logger):
        """Test TLS volume config when TLS is enabled but no certificates specified (verify=True case)."""
        os.environ["TRUSTYAI_LMEVAL_TLS"] = "true"
        # Neither TRUSTYAI_LMEVAL_CERT_FILE nor TRUSTYAI_LMEVAL_CERT_SECRET are set

        from src.llama_stack_provider_lmeval.lmeval import _create_tls_volume_config
        volume_mounts, volumes = _create_tls_volume_config()

        # Should return None, None since no volumes are needed for verify=True
        self.assertIsNone(volume_mounts)
        self.assertIsNone(volumes)

        # Should log debug about no certificates specified
        mock_logger.debug.assert_called()
        debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
        self.assertTrue(any("no certificates specified, no volumes created" in call for call in debug_calls))

    @patch("src.llama_stack_provider_lmeval.lmeval.logger")
    def test_logging_when_only_cert_file_set(self, mock_logger):
        """Test that appropriate error logging occurs when only cert_file is set."""
        os.environ["TRUSTYAI_LMEVAL_TLS"] = "true"
        os.environ["TRUSTYAI_LMEVAL_CERT_FILE"] = "test-cert.pem"
        # TRUSTYAI_LMEVAL_CERT_SECRET is not set

        result = _get_tls_config_from_env()

        # Verify error was logged for incomplete environment variables
        mock_logger.error.assert_called()
        error_calls = [call[0][0] for call in mock_logger.error.call_args_list]
        self.assertTrue(any("Invalid TLS configuration:" in call for call in error_calls))
        self.assertTrue(any("is set but" in call for call in error_calls))
        self.assertTrue(any("is missing" in call for call in error_calls))

        # Verify result is None (no fallback)
        self.assertIsNone(result)

    @patch("src.llama_stack_provider_lmeval.lmeval.logger")
    def test_logging_when_only_cert_secret_set(self, mock_logger):
        """Test that appropriate error logging occurs when only cert_secret is set."""
        os.environ["TRUSTYAI_LMEVAL_TLS"] = "true"
        os.environ["TRUSTYAI_LMEVAL_CERT_SECRET"] = "test-secret"
        # TRUSTYAI_LMEVAL_CERT_FILE is not set

        result = _get_tls_config_from_env()

        # Verify error was logged for incomplete environment variables
        mock_logger.error.assert_called()
        error_calls = [call[0][0] for call in mock_logger.error.call_args_list]
        self.assertTrue(any("Invalid TLS configuration:" in call for call in error_calls))
        self.assertTrue(any("is set but" in call for call in error_calls))
        self.assertTrue(any("is missing" in call for call in error_calls))

        # Verify result is None (no fallback)
        self.assertIsNone(result)

    @patch("src.llama_stack_provider_lmeval.lmeval.logger")
    def test_logging_when_fallback_to_provider_config_successful(self, mock_logger):
        """Test that warning logging occurs when falling back to provider config."""
        os.environ["TRUSTYAI_LMEVAL_TLS"] = "true"
        os.environ["TRUSTYAI_LMEVAL_CERT_FILE"] = "test-cert.pem"
        # TRUSTYAI_LMEVAL_CERT_SECRET is not set

        provider_config = MagicMock()
        provider_config.tls = MagicMock()
        provider_config.tls.enable = True
        provider_config.tls.cert_file = "provider-cert.pem"
        provider_config.tls.cert_secret = "provider-secret"

        result = _get_tls_config_from_env(provider_config)

        # Verify error was logged for incomplete environment variables
        mock_logger.error.assert_called_once()
        # Verify warning was logged for successful fallback
        mock_logger.warning.assert_called_once()
        warning_call_args = mock_logger.warning.call_args[0]
        self.assertIn("Falling back to provider config TLS due to incomplete environment variables", warning_call_args[0])

        # Verify result is the provider config path
        expected_path = "/etc/ssl/certs/provider-cert.pem"
        self.assertEqual(result, expected_path)

    @patch("src.llama_stack_provider_lmeval.lmeval.logger")
    def test_logging_when_fallback_to_provider_config_tls_true(self, mock_logger):
        """Test that warning logging occurs when falling back to provider config TLS=True."""
        os.environ["TRUSTYAI_LMEVAL_TLS"] = "true"
        os.environ["TRUSTYAI_LMEVAL_CERT_FILE"] = "test-cert.pem"
        # TRUSTYAI_LMEVAL_CERT_SECRET is not set

        provider_config = MagicMock()
        provider_config.tls = MagicMock()
        provider_config.tls.enable = True
        provider_config.tls.cert_file = None
        provider_config.tls.cert_secret = None

        result = _get_tls_config_from_env(provider_config)

        # Verify error was logged for incomplete environment variables
        mock_logger.error.assert_called_once()
        # Verify warning was logged for successful fallback
        mock_logger.warning.assert_called_once()
        warning_call_args = mock_logger.warning.call_args[0]
        self.assertIn("Falling back to provider config TLS (verify=True) due to incomplete environment variables", warning_call_args[0])

        # Verify result is True
        self.assertTrue(result)

    @patch("src.llama_stack_provider_lmeval.lmeval.logger")
    def test_logging_when_fallback_not_possible(self, mock_logger):
        """Test that error logging occurs when fallback to provider config is not possible."""
        os.environ["TRUSTYAI_LMEVAL_TLS"] = "true"
        os.environ["TRUSTYAI_LMEVAL_CERT_FILE"] = "test-cert.pem"
        # TRUSTYAI_LMEVAL_CERT_SECRET is not set

        provider_config = MagicMock()
        provider_config.tls = None  # No TLS config

        result = _get_tls_config_from_env(provider_config)

        # Verify error was logged for incomplete environment variables
        mock_logger.error.assert_called()
        # Verify additional error was logged for fallback failure
        error_calls = [call[0][0] for call in mock_logger.error.call_args_list]
        self.assertTrue(any("Cannot fall back to provider config TLS" in call for call in error_calls))

        # Verify result is None
        self.assertIsNone(result)

    @patch("src.llama_stack_provider_lmeval.lmeval.logger")
    def test_logging_when_using_environment_variables(self, mock_logger):
        """Test that debug logging occurs when using complete environment variables."""
        os.environ["TRUSTYAI_LMEVAL_TLS"] = "true"
        os.environ["TRUSTYAI_LMEVAL_CERT_FILE"] = "test-cert.pem"
        os.environ["TRUSTYAI_LMEVAL_CERT_SECRET"] = "test-secret"

        result = _get_tls_config_from_env()

        # Verify debug was logged
        mock_logger.debug.assert_called_once()
        debug_call_args = mock_logger.debug.call_args[0]
        self.assertIn("Using TLS configuration from environment variables", debug_call_args[0])

        # Verify result is correct
        expected_path = "/etc/ssl/certs/test-cert.pem"
        self.assertEqual(result, expected_path)

    @patch("src.llama_stack_provider_lmeval.lmeval.logger")
    def test_logging_when_no_cert_variables_set(self, mock_logger):
        """Test that debug logging occurs when no certificate variables are set."""
        os.environ["TRUSTYAI_LMEVAL_TLS"] = "true"
        # Neither TRUSTYAI_LMEVAL_CERT_FILE nor TRUSTYAI_LMEVAL_CERT_SECRET are set

        result = _get_tls_config_from_env()

        # Verify debug was logged
        mock_logger.debug.assert_called_once()
        debug_call_args = mock_logger.debug.call_args[0]
        self.assertIn("No TLS certificate files specified, using verify=True", debug_call_args[0])

        # Verify result is True
        self.assertTrue(result)


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

    def test_create_cr_with_provider_config_tls_missing_cert_file(self):
        """Test TLS enabled but cert_file missing, should raise validation error."""
        from src.llama_stack_provider_lmeval.config import LMEvalConfigError

        # This should raise LMEvalConfigError when creating the config
        with self.assertRaises(LMEvalConfigError) as excinfo:
            config = LMEvalEvalProviderConfig(
                namespace=self.namespace,
                service_account=self.service_account,
                tls=TLSConfig(
                    enable=True,
                    cert_file=None,
                    cert_secret="vllm-ca-bundle"
                ),
            )

        # Verify the error message
        self.assertIn("Both cert_file and cert_secret must be set when TLS is enabled and certificates are specified", str(excinfo.exception))

    def test_create_cr_with_provider_config_tls_missing_cert_secret(self):
        """Test TLS enabled but cert_secret missing, should raise validation error."""
        from src.llama_stack_provider_lmeval.config import LMEvalConfigError

        # This should raise LMEvalConfigError when creating the config
        with self.assertRaises(LMEvalConfigError) as excinfo:
            config = LMEvalEvalProviderConfig(
                namespace=self.namespace,
                service_account=self.service_account,
                tls=TLSConfig(
                    enable=True,
                    cert_file="custom-ca.pem",
                    cert_secret=None
                ),
            )

        # Verify the error message
        self.assertIn("Both cert_file and cert_secret must be set when TLS is enabled and certificates are specified", str(excinfo.exception))

    def test_create_cr_with_provider_config_tls_empty_strings(self):
        """Test TLS enabled but both certificates are empty strings, should raise validation error."""
        from src.llama_stack_provider_lmeval.config import LMEvalConfigError

        # This should raise LMEvalConfigError when creating the config
        with self.assertRaises(LMEvalConfigError) as excinfo:
            config = LMEvalEvalProviderConfig(
                namespace=self.namespace,
                service_account=self.service_account,
                tls=TLSConfig(
                    enable=True,
                    cert_file="",
                    cert_secret=""
                ),
            )

        # Verify the error message
        self.assertIn("cert_file and cert_secret cannot be empty strings", str(excinfo.exception))

    def test_create_cr_with_provider_config_tls_whitespace_only_cert_secret(self):
        """Test TLS enabled but cert_secret is only whitespace, should raise validation error."""
        from src.llama_stack_provider_lmeval.config import LMEvalConfigError

        # This should raise LMEvalConfigError when creating the config
        with self.assertRaises(LMEvalConfigError) as excinfo:
            config = LMEvalEvalProviderConfig(
                namespace=self.namespace,
                service_account=self.service_account,
                tls=TLSConfig(
                    enable=True,
                    cert_file="test-cert.pem",
                    cert_secret="   "  # Only whitespace
                ),
            )

        # Verify the error message
        self.assertIn("cert_file and cert_secret cannot be empty strings", str(excinfo.exception))

    def test_create_cr_with_provider_config_tls_whitespace_only_cert_file(self):
        """Test TLS enabled but cert_file is only whitespace, should raise validation error."""
        from src.llama_stack_provider_lmeval.config import LMEvalConfigError

        # This should raise LMEvalConfigError when creating the config
        with self.assertRaises(LMEvalConfigError) as excinfo:
            config = LMEvalEvalProviderConfig(
                namespace=self.namespace,
                service_account=self.service_account,
                tls=TLSConfig(
                    enable=True,
                    cert_file="   ",  # Only whitespace
                    cert_secret="test-secret"
                ),
            )

        # Verify the error message
        self.assertIn("cert_file and cert_secret cannot be empty strings", str(excinfo.exception))

    def test_create_cr_with_provider_config_tls_unsafe_characters(self):
        """Test TLS enabled but cert_file contains unsafe characters, should raise validation error."""
        from src.llama_stack_provider_lmeval.config import LMEvalConfigError

        # This should raise LMEvalConfigError when creating the config
        with self.assertRaises(LMEvalConfigError) as excinfo:
            config = LMEvalEvalProviderConfig(
                namespace=self.namespace,
                service_account=self.service_account,
                tls=TLSConfig(
                    enable=True,
                    cert_file="../malicious.pem",
                    cert_secret="vllm-ca-bundle"
                ),
            )

        # Verify the error message - should now mention directory traversal specifically
        self.assertIn("must not contain directory traversal", str(excinfo.exception))

    def test_create_cr_with_provider_config_tls_valid_subdirectory_path(self):
        """Test TLS enabled with valid subdirectory path in cert_file, should work correctly."""
        # This should work without raising validation errors
        config = LMEvalEvalProviderConfig(
            namespace=self.namespace,
            service_account=self.service_account,
            tls=TLSConfig(
                enable=True,
                cert_file="certs/ca-bundle.pem",
                cert_secret="vllm-ca-bundle"
            ),
        )
        self.builder._config = config

        # Should not raise any exceptions
        cr = self.builder.create_cr(
            benchmark_id="lmeval::mmlu",
            task_config=self.benchmark_config,
            base_url="http://my-model-url",
            limit="10",
            stored_benchmark=self.stored_benchmark,
        )

        # Should have verify_certificate set to the full path
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
            "/etc/ssl/certs/certs/ca-bundle.pem",
            "TLS configuration value should contain the full subdirectory path",
        )

    def test_create_cr_with_provider_config_tls_valid_dotted_filename(self):
        """Test TLS enabled with valid dotted filename in cert_file, should work correctly."""
        # This should work without raising validation errors
        config = LMEvalEvalProviderConfig(
            namespace=self.namespace,
            service_account=self.service_account,
            tls=TLSConfig(
                enable=True,
                cert_file="ca.bundle.pem",
                cert_secret="vllm-ca-bundle"
            ),
        )
        self.builder._config = config

        # Should not raise any exceptions
        cr = self.builder.create_cr(
            benchmark_id="lmeval::mmlu",
            task_config=self.benchmark_config,
            base_url="http://my-model-url",
            limit="10",
            stored_benchmark=self.stored_benchmark,
        )

        # Should have verify_certificate set to the full path
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
            "/etc/ssl/certs/ca.bundle.pem",
            "TLS configuration value should contain the full filename with dots",
        )

    def test_create_cr_with_provider_config_tls_unsafe_characters_still_blocked(self):
        """Test TLS enabled but cert_file contains unsafe characters (other than slashes), should raise validation error."""
        from src.llama_stack_provider_lmeval.config import LMEvalConfigError

        # This should raise LMEvalConfigError when creating the config
        with self.assertRaises(LMEvalConfigError) as excinfo:
            config = LMEvalEvalProviderConfig(
                namespace=self.namespace,
                service_account=self.service_account,
                tls=TLSConfig(
                    enable=True,
                    cert_file="cert@file.pem",  # Contains @ which is unsafe
                    cert_secret="vllm-ca-bundle"
                ),
            )

        # Verify the error message - should mention potentially unsafe characters
        self.assertIn("contains potentially unsafe characters", str(excinfo.exception))

    @patch("src.llama_stack_provider_lmeval.lmeval.logger")
    def test_create_cr_with_provider_config_tls_no_certificates(self, mock_logger):
        """Test TLS enabled but no certificates specified (verify=True case), should work correctly."""
        # This should work without raising validation errors
        config = LMEvalEvalProviderConfig(
            namespace=self.namespace,
            service_account=self.service_account,
            tls=TLSConfig(
                enable=True,
                cert_file=None,
                cert_secret=None
            ),
        )
        self.builder._config = config

        # Should not raise any exceptions
        cr = self.builder.create_cr(
            benchmark_id="lmeval::mmlu",
            task_config=self.benchmark_config,
            base_url="http://my-model-url",
            limit="10",
            stored_benchmark=self.stored_benchmark,
        )

        # Should have verify_certificate set to True
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
            "TLS configuration value should be 'True' for verify=True case",
        )

        # No volumes should be created for verify=True case
        pod_config = cr.get("spec", {}).get("pod", {})
        self.assertIsNone(pod_config.get("volumes"), "Pod should not have volumes for verify=True case")
        self.assertIsNone(pod_config.get("container", {}).get("volumeMounts"), "Container should not have volume mounts for verify=True case")

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
