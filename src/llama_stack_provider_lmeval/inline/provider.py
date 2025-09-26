"""LMEval Inline Eval Llama Stack provider specification."""

from llama_stack.providers.datatypes import Api, InlineProviderSpec, ProviderSpec


def get_provider_spec() -> ProviderSpec:
    return InlineProviderSpec(
        api=Api.eval,
        provider_type="inline::trustyai_lmeval",
        pip_packages=["lm-eval", "lm-eval[api]"],
        config_class="llama_stack_provider_lmeval.config.LMEvalEvalProviderConfig",
        module="llama_stack_provider_lmeval.inline",
    )
