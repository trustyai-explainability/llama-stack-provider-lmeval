from llama_stack.providers.datatypes import (
    ProviderSpec,
    Api,
    AdapterSpec,
    remote_provider_spec,
)

def get_provider_spec() -> ProviderSpec:
    return remote_provider_spec(
        api=Api.eval,
        adapter=AdapterSpec(
            name="trustyai_lmeval",
            pip_packages=["kubernetes"],
            config_class="config.LMEvalBenchmarkConfig",
            module="lmeval",
        ),
    )
