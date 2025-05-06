#!/bin/bash

# Patch the default-dsc to set lmes-allow-online=True and lmes-allow-code-execution=True in the trustyai-service-operator-config CM
if oc get DataScienceCluster default-dsc &> /dev/null; then
  echo "Patching DataScienceCluster"
  oc patch dsc default-dsc --type=merge \
  -p '{
    "spec": {
      "components": {
        "trustyai": {
          "devFlags": {
            "manifests": [
              {
                "contextDir": "config",
                "sourcePath": "",
                "uri": "https://github.com/christinaexyou/trustyai-service-operator/tarball/llamastack-ootp-lmeval"
              }
            ]
          }
        }
      }
    }
  }'
else
    echo "Creating DataScienceCluster..."
    oc apply -f manifests/dsc.yaml -n redhat-ods-applications
fi
sleep 10

# Patch for ROSA clusters
echo "Patching cluster..."
export sa_issuer="$(oc get authentication cluster -o jsonpath --template='{ .spec.serviceAccountIssuer }' -n openshift-authentication)"
export dsci_audience="$(oc get DSCInitialization default-dsci -o jsonpath='{.spec.serviceMesh.auth.audiences[0]}')"
if [[ "z$sa_issuer" != "z" ]] && [[ "$sa_issuer" != "$dsci_audience" ]]
then
  echo “DSCI is updated”
  oc patch DSCInitialization default-dsci --type='json' -p="[{'op': 'replace', 'path': '/spec/serviceMesh/auth/audiences/0', 'value': '$sa_issuer'}]"
# fi
