# llama-stack-provider-lmeval
Llama Stack Remote Eval Provider for TrustyAI LM-Eval

## About
This repository implements [TrustyAI's LM-Eval](https://trustyai-explainability.github.io/trustyai-site/main/lm-eval-tutorial.html) as an out-of-tree Llama Stack remote provider.

It also includes an end-to-end instructions demonstratring how one can use LM-Eval on LLama Stack to run benchmark evaluations over ARC-Easy on a deployed [Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) model via Red Hat OpenShift AI (RHOAI).

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

2. Add a devFlag to the `trustyai` component in the `DataScienceCluster` to enable LM-Eval jobs to download models from HuggingFace
   ```
   demo/scripts/setup.sh
   ```

3. Navigate to [00-getting_started_with_lmeval.ipynb](demo/00-getting_started_with_lmeval.ipynb) to learn how to use LM-Eval to run evaluations on your model!

