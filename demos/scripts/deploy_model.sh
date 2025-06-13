echo " "
# Create a namespace for the demo
TEST_NS=model-namespace
echo "Creating namepace ${TEST_NS}..."
oc create ns $TEST_NS

echo " "
# Create service account and token
echo "Creating service account and token..."
oc apply -f demo/manifests/service-account.yaml -n $TEST_NS
TOKEN=$(oc create token user-one -n $TEST_NS)

echo " "
# Deploy the model container and wait for it to be ready
echo "Creating model container..."
oc apply -f demo/manifests/phi3-model-container.yaml -n $TEST_NS
MODEL_CONTAINER=$(oc get pods -o name -n $TEST_NS | grep minio)
sleep 100
oc wait --for=condition=Ready=true $MODEL_CONTAINER

echo " "
# Deploy the serving runtime and inference service
echo "Deploying model as inference service..."
oc apply -f demo/manifests/phi3-serving-runtime.yaml -n $TEST_NS
oc apply -f demo/manifests/phi3-isvc.yaml -n $TEST_NS
MODEL_POD=$(oc get pods -o name -n $TEST_NS | grep predictor)
sleep 150
oc wait --for=condition=Ready=true $MODEL_POD

echo " "
# Sanity check inference service
echo "Validating inference service..."
LLM_ROUTE=$(oc get $(oc get ksvc -o name | grep predictor) --template={{.status.url}} -n $TEST_NS)
MODEL=$(curl -sk $LLM_ROUTE/v1/models -H "Authorization: Bearer ${TOKEN}" | jq --raw-output ".data[0].root")
# Add if
echo "Model ${MODEL} deployed successfully. You can query it via: ${LLM_ROUTE}"
