"""Tests for OpenRouter model registry functionality."""

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from providers.registries.openrouter import OpenRouterModelRegistry
from providers.shared import ModelCapabilities, ProviderType


class TestOpenRouterModelRegistry:
    """Test cases for OpenRouter model registry."""

    def test_registry_initialization(self):
        """Test registry initializes with default config."""
        registry = OpenRouterModelRegistry()

        # Should load models from default location
        assert len(registry.list_models()) > 0
        assert len(registry.list_aliases()) > 0

    def test_custom_config_path(self):
        """Test registry with custom config path."""
        # Create temporary config
        config_data = {
            "models": [
                {
                    "model_name": "test/model-1",
                    "aliases": ["test1", "t1"],
                    "context_window": 4096,
                    "max_output_tokens": 2048,
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            registry = OpenRouterModelRegistry(config_path=temp_path)
            assert len(registry.list_models()) == 1
            assert "test/model-1" in registry.list_models()
            assert "test1" in registry.list_aliases()
            assert "t1" in registry.list_aliases()
        finally:
            os.unlink(temp_path)

    def test_environment_variable_override(self):
        """Test OPENROUTER_MODELS_CONFIG_PATH environment variable."""
        # Create custom config
        config_data = {
            "models": [
                {"model_name": "env/model", "aliases": ["envtest"], "context_window": 8192, "max_output_tokens": 4096}
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            # Set environment variable
            original_env = os.environ.get("OPENROUTER_MODELS_CONFIG_PATH")
            os.environ["OPENROUTER_MODELS_CONFIG_PATH"] = temp_path

            # Create registry without explicit path
            registry = OpenRouterModelRegistry()

            # Should load from environment path
            assert "env/model" in registry.list_models()
            assert "envtest" in registry.list_aliases()

        finally:
            # Restore environment
            if original_env is not None:
                os.environ["OPENROUTER_MODELS_CONFIG_PATH"] = original_env
            else:
                del os.environ["OPENROUTER_MODELS_CONFIG_PATH"]
            os.unlink(temp_path)

    def test_alias_resolution(self):
        """Test alias resolution functionality."""
        registry = OpenRouterModelRegistry()

        # Test various aliases
        test_cases = [
            ("opus", "anthropic/claude-opus-4.1"),
            ("OPUS", "anthropic/claude-opus-4.1"),  # Case insensitive
            ("sonnet", "anthropic/claude-sonnet-4.5"),
            ("o3", "openai/o3"),
            ("deepseek", "deepseek/deepseek-r1-0528"),
            ("mistral", "mistralai/mistral-large-2411"),
        ]

        for alias, expected_model in test_cases:
            config = registry.resolve(alias)
            assert config is not None, f"Failed to resolve alias '{alias}'"
            assert config.model_name == expected_model

    def test_direct_model_name_lookup(self):
        """Test looking up models by their full name."""
        registry = OpenRouterModelRegistry()

        # Should be able to look up by full model name
        config = registry.resolve("anthropic/claude-opus-4.1")
        assert config is not None
        assert config.model_name == "anthropic/claude-opus-4.1"

        config = registry.resolve("openai/o3")
        assert config is not None
        assert config.model_name == "openai/o3"

    def test_unknown_model_resolution(self):
        """Test resolution of unknown models."""
        registry = OpenRouterModelRegistry()

        # Unknown aliases should return None
        assert registry.resolve("unknown-alias") is None
        assert registry.resolve("") is None
        assert registry.resolve("non-existent") is None

    def test_model_capabilities_conversion(self):
        """Test that registry returns ModelCapabilities directly."""
        registry = OpenRouterModelRegistry()

        config = registry.resolve("opus")
        assert config is not None

        # Registry now returns ModelCapabilities objects directly
        assert config.provider == ProviderType.OPENROUTER
        assert config.model_name == "anthropic/claude-opus-4.1"
        assert config.friendly_name == "OpenRouter (anthropic/claude-opus-4.1)"
        assert config.context_window == 200000
        assert not config.supports_extended_thinking

    def test_duplicate_alias_detection(self):
        """Test that duplicate aliases are detected."""
        config_data = {
            "models": [
                {"model_name": "test/model-1", "aliases": ["dupe"], "context_window": 4096, "max_output_tokens": 2048},
                {
                    "model_name": "test/model-2",
                    "aliases": ["DUPE"],  # Same alias, different case
                    "context_window": 8192,
                    "max_output_tokens": 2048,
                },
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Duplicate alias"):
                OpenRouterModelRegistry(config_path=temp_path)
        finally:
            os.unlink(temp_path)

    def test_backwards_compatibility_max_tokens(self):
        """Test that legacy max_tokens field maps to max_output_tokens."""
        config_data = {
            "models": [
                {
                    "model_name": "test/old-model",
                    "aliases": ["old"],
                    "max_tokens": 16384,  # Old field name should cause error
                    "supports_extended_thinking": False,
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(ValueError, match="max_output_tokens"):
                    OpenRouterModelRegistry(config_path=temp_path)
        finally:
            os.unlink(temp_path)

    def test_missing_config_file(self):
        """Test behavior with missing config file."""
        # Use a non-existent path
        with patch.dict("os.environ", {}, clear=True):
            registry = OpenRouterModelRegistry(config_path="/non/existent/path.json")

        # Should initialize with empty maps
        assert len(registry.list_models()) == 0
        assert len(registry.list_aliases()) == 0
        assert registry.resolve("anything") is None

    def test_invalid_json_config(self):
        """Test handling of invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json }")
            temp_path = f.name

        try:
            registry = OpenRouterModelRegistry(config_path=temp_path)
            # Should handle gracefully and initialize empty
            assert len(registry.list_models()) == 0
            assert len(registry.list_aliases()) == 0
        finally:
            os.unlink(temp_path)

    def test_model_with_all_capabilities(self):
        """Test model with all capability flags."""
        from providers.shared import TemperatureConstraint

        caps = ModelCapabilities(
            provider=ProviderType.OPENROUTER,
            model_name="test/full-featured",
            friendly_name="OpenRouter (test/full-featured)",
            aliases=["full"],
            context_window=128000,
            max_output_tokens=8192,
            supports_extended_thinking=True,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            description="Fully featured test model",
            temperature_constraint=TemperatureConstraint.create("range"),
        )
        assert caps.context_window == 128000
        assert caps.supports_extended_thinking
        assert caps.supports_system_prompts
        assert caps.supports_streaming
        assert caps.supports_function_calling
        # Note: supports_json_mode is not in ModelCapabilities yet
