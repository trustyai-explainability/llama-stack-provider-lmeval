# TrustyAI LM-Eval as and Out-of-Tree Llama Stack Provider

[![PyPI version](https://img.shields.io/pypi/v/llama_stack_provider_lmeval.svg)](https://pypi.org/project/llama-stack-provider-lmeval/) [![pre-commit.ci](https://github.com/trustyai-explainability/llama-stack-provider-lmeval/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/trustyai-explainability/llama-stack-provider-lmeval/actions/workflows/pre-commit.yml) [![Bandit](https://github.com/trustyai-explainability/llama-stack-provider-lmeval/actions/workflows/security.yml/badge.svg)](https://github.com/trustyai-explainability/llama-stack-provider-lmeval/actions/workflows/security.yml?label="Bandit") ![Llama compatibility](https://img.shields.io/badge/%F0%9F%A6%99-0.2.16-blue)


## About
This repository implements [TrustyAI's LM-Eval](https://trustyai-explainability.github.io/trustyai-site/main/lm-eval-tutorial.html) as an out-of-tree Llama Stack remote provider.

## Use
### Prerequsites
* Admin access to an OpenShift cluster with RHOAI installed
* Login to your OpenShift cluster with `oc login --token=<TOKEN> --server=<SERVER>`
* Installation of `uv`
* Installation of `oc` cli tool
* Installation of `llama stack` cli tool

1. Clone this repository
    ```
    git clone https://github.com/trustyai-explainability/llama-stack-provider-lmeval.git
    ```

2. Set `llama-stack-provider-lmeval` as your working directory.
    ```
    cd llama-stack-provider-lmeval
    ```

3. Deploy `microsoft/Phi-3-mini-4k-instruct` on vLLM Serving Runtime

    a. Create a namespace with a name of your choice
    ```bash
    TEST_NS=<NAMESPACE>
    oc create ns $TEST_NS
    oc get ns $TEST_NS
    ```

    b. Deploy the model via vLLM
    ```bash
    oc apply -k demos/resources/kustomization.yaml
    ```

4. Before continuing, preform a sanity check to make sure the model was sucessfully deployed
    ```bash
    oc get pods | grep "predictor"
    ```

    Expected output:
    ```
    phi-3-predictor-00002-deployment-794fb6b4b-clhj7   3/3     Running   0          5h55m
    ```

5. Create and activate a virtual enviornment
    ```
    uv venv .llamastack-venv
    ```

    ```
    source .llamastack-venv/bin/activate
    ```

6. Install the required libraries
    ```
    uv pip install -e .
    ```

7. Define the following ennvironment variables
    ```
    export VLLM_URL=https://$(oc get $(oc get ksvc -o name | grep predictor) --template={{.status.url}})/v1/completions

    export TRUSTYAI_LM_EVAL_NAMESPACE=$(oc project | cut -d '"' -f2)
    ```

8. Start the llama stack server in a virtual enviornment
    ```
    llama stack run run.yaml --image-type venv
    ```

    Expected output:
    ```
    INFO:     Application startup complete.
    INFO:     Uvicorn running on http://['::', '0.0.0.0']:8321 (Press CTRL+C to quit)
    ```

9. Navigate to `demos/` to run the demo notebooks

## TLS Support
This provider supports TLS for secure communication with model inference endpoints. TLS configuration is controlled through environment variables:

### Environment Variables
- `TRUSTYAI_LMEVAL_TLS`: Set to `true` to enable TLS support
- `TRUSTYAI_LMEVAL_CERT_FILE`: Name of the certificate file in the secret (e.g., `custom-ca.pem`)
- `TRUSTYAI_LMEVAL_CERT_SECRET`: Name of the Kubernetes secret containing the TLS certificate

### Structured Configuration
The provider also supports structured TLS configuration through the `TLSConfig` class:

```python
from llama_stack_provider_lmeval.config import TLSConfig

tls_config = TLSConfig(
    enable=True,                    # Enable TLS support
    cert_file="custom-ca.pem",      # Certificate filename in secret
    cert_secret="vllm-ca-bundle"   # Kubernetes secret name
)
```

**Note**: When using structured configuration, both `cert_file` and `cert_secret` must be provided together, or neither should be provided (for simple TLS verification).

### TLS Configuration Modes

The provider supports two TLS configuration modes:

1. **Environment Variables**: Set TLS configuration via environment variables for runtime flexibility
2. **Provider Config**: Set TLS configuration via the `TLSConfig` object for code-based configuration

### Priority Order

TLS configuration follows this priority order:
1. **Environment Variables** (highest priority)
2. **Provider Config** (`TLSConfig` object)
3. **No TLS** (default)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed information on:

- Development setup and prerequisites
- Pre-commit hooks and code quality standards
- Running tests and development workflow
- Troubleshooting common issues

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
