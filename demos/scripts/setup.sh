#!/bin/bash

# Patch for ROSA clusters
echo "Patching cluster..."
export sa_issuer="$(oc get authentication cluster -o jsonpath --template='{ .spec.serviceAccountIssuer }' -n openshift-authentication)"
export dsci_audience="$(oc get DSCInitialization default-dsci -o jsonpath='{.spec.serviceMesh.auth.audiences[0]}')"
if [[ "z$sa_issuer" != "z" ]] && [[ "$sa_issuer" != "$dsci_audience" ]]
then
  echo “DSCI is updated”
  oc patch DSCInitialization default-dsci --type='json' -p="[{'op': 'replace', 'path': '/spec/serviceMesh/auth/audiences/0', 'value': '$sa_issuer'}]"
# fi
