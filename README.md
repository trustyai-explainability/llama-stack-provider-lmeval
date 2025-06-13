# TrustyAI LM-Eval as and Out-of-Tree Llama Stack Provider

[![PyPI version](https://img.shields.io/pypi/v/llama_stack_provider_lmeval.svg)](https://pypi.org/project/llama-stack-provider-lmeval/)

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
