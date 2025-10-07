from llama_stack.providers.datatypes import (
    Api,
    ProviderSpec,
    RemoteProviderSpec,
)


def get_provider_spec() -> ProviderSpec:
    return RemoteProviderSpec(
        api=Api.eval,
        provider_type="remote::trustyai_lmeval",
        adapter_type="lmeval",
        pip_packages=["kubernetes"],
        config_class="llama_stack_provider_lmeval.config.LMEvalEvalProviderConfig",
        module="llama_stack_provider_lmeval",
    )
