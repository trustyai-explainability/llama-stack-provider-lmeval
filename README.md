# TrustyAI LM-Eval as and Out-of-Tree Llama Stack Provider

This repository implements [TrustyAI's LM-Eval](https://trustyai-explainability.github.io/trustyai-site/main/lm-eval-tutorial.html) as an out-of-tree Llama Stack remote provider.

It also includes an end-to-end instructions demonstratring how one can use LM-Eval on LLama Stack to run benchmark evaluations over [DK-Bench](https://github.com/instructlab/instructlab/blob/main/src/instructlab/model/evaluate.py#L30) on a deployed [Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) model via OpenShift.

## Use
### Prerequsites
* Admin access to an OpenShift cluster with RHOAI installed
* Installation of `uv`
* Installation of `oc` cli tool
* Installation of `llama stack` cli tool

1. Clone this repository
    ```
    git clone https://github.com/trustyai-explainability/llama-stack-provider-lmeval.git
    ```

2. Set `llama-stack-provider-lmeval/demo` as your working directory.
    ```
    cd llama-stack-provider-lmeval/demo
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
    oc apply -k resources/kustomization.yaml
    ```

4. Before continuing, preform a sanity check to make sure the model was sucessfully deployed
    ```bash
    oc get pods | grep "predictor"
    ```

    Expected output:
    ```
    phi-3-predictor-00002-deployment-794fb6b4b-clhj7   3/3     Running   0          5h55m
    ```

5. Retrive the model route
    ```
    VLLM_URL=$(oc get $(oc get ksvc -o name | grep predictor) --template={{.status.url}})
    ```
6. Create and activate a virtual enviornment
    ```
    uv venv .llamastack-venv
    ```

    ```
    source .llamastack-venv/bin/activate
    ```

7. Install the required libraries
    ```
    uv pip install -e .
    ```

8. In the `run.yaml`, make the following changes:

    a. Replace the `remote::vllm` url
    ```
    providers:
        inference:
        - provider_id: vllm-0
            provider_type: remote::vllm
            config:
            url: ${env.VLLM_URL:https://phi-3-predictor-llama-test.apps.rosa.p2i7w2k6p6w7t7e.3emk.p3.openshiftapps.com/v1/completions}
    ```

    b. Replace the `remote::lmeval` base_url and namespace
    ```
    - provider_id: lmeval-1
        provider_type: remote::lmeval
        config:
            use_k8s: True
            base_url: https://vllm-test.apps.rosa.p2i7w2k6p6w7t7e.3emk.p3.openshiftapps.com/v1/completions
            namespace: "llama-test"
    ```

9. Start the llama stack server in a virtual enviornment
    ```
    llama stack run ../run.yaml --image-type venv
    ```

    Expected output:
    ```
    INFO:     Application startup complete.
    INFO:     Uvicorn running on http://['::', '0.0.0.0']:8321 (Press CTRL+C to quit)
    ```

10. Navigate to `demo.ipynb` to run evaluation
