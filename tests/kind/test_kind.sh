#!/bin/bash

NAMESPACE="test"
BASE_PATH="tests/kind/manifests"
VLLM_EMULATOR="vllm_emulator.yaml"
LLAMA_STACK_DISTRIBUTION="llama_stack_distribution.yaml"


wait_for_pods() {
    local namespace=$1
    local label_selector=$2
    local timeout=${3:-300}
    local description=$4

    echo "Waiting for $description to be ready..."
    if ! kubectl wait --for=condition=ready pod -l "$label_selector" -n "$namespace" --timeout="${timeout}s"; then
        echo "ERROR: $description failed to become ready within ${timeout} seconds"
        kubectl get pods -l "$label_selector" -n "$namespace"
        kubectl describe pods -l "$label_selector" -n "$namespace"
        return 1
    fi
    echo "$description is ready"
    return 0
}

# Ensure that the TrustyAI operator controller is ready
echo "Waiting for TrustyAI operator controller to be ready..."
if ! kubectl wait --for=condition=ready pod -l control-plane=controller-manager -n system --timeout=300s; then
    echo "ERROR: TrustyAI operator controller failed to become ready within 300 seconds"
    kubectl get pods -n system
    kubectl logs -n system $(kubectl get pods -n system | grep trustyai-service-operator-controller-manager | awk '{print $1}') --tail=100
    exit 1
fi
echo "TrustyAI operator controller is ready"

# Create a namespace for testing
kubectl create namespace "$NAMESPACE"

# Deploy the vLLM emulator
kubectl apply -f ${BASE_PATH}/${VLLM_EMULATOR} -n "$NAMESPACE"
wait_for_pods "$NAMESPACE" "app=vllm-emulator" 300 "vLLM emulator"

# Wait a moment for the vLLM emulator
sleep 60

# Deploy the LlamaStackDistribution
envsubst < ${BASE_PATH}/${LLAMA_STACK_DISTRIBUTION} | kubectl apply -f - -n "$NAMESPACE"

# Check if the LlamaStackDistribution pods are ready
if ! wait_for_pods "test" "app.kubernetes.io/instance=llamastack-custom-distribution" 60 "LlamaStackDistribution"; then
    echo "LlamaStackDistribution failed to start. Collecting debugging information..."
    echo "Pod status:"
    kubectl get pods -l app.kubernetes.io/instance=llamastack-custom-distribution -n "$NAMESPACE"
    echo "Pod description:"
    kubectl describe pods -l app.kubernetes.io/instance=llamastack-custom-distribution -n "$NAMESPACE"
    echo "Pod logs:"
    kubectl logs -l app.kubernetes.io/instance=llamastack-custom-distribution -n "$NAMESPACE" --tail=50 --all-containers=true || echo "No logs available"
    exit 1
fi
