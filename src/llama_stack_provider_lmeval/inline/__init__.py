"""LMEval Inline Eval Llama Stack provider."""

import logging

from llama_stack.apis.datatypes import Api
from llama_stack.providers.datatypes import ProviderSpec

from llama_stack_provider_lmeval.config import LMEvalEvalProviderConfig

from .lmeval import LMEvalInline

logger = logging.getLogger(__name__)


async def get_provider_impl(
    config: LMEvalEvalProviderConfig,
    deps: dict[Api, ProviderSpec] | None = None,
) -> LMEvalInline:
    """Get an inline Eval  implementation from the configuration.

    Args:
        config: LMEvalEvalProviderConfig
        deps: Optional[dict[Api, Any]] = None - can be ProviderSpec or API instances

    Returns:
        Configured LMEval Inline implementation

    Raises:
        Exception: If configuration is invalid
    """
    try:
        if deps is None:
            deps = {}

        # Extract base_url from config if available
        base_url = None
        if hasattr(config, "model_args") and config.model_args:
            for arg in config.model_args:
                if arg.get("name") == "base_url":
                    base_url = arg.get("value")
                    logger.debug("Using base_url from config: %s", base_url)
                    break

        return LMEvalInline(config=config, deps=deps)
    except Exception as e:
        raise RuntimeError(f"Failed to create LMEval implementation: {str(e)}") from e


__all__ = [
    "get_provider_impl",
    "LMEvalInline",
]
