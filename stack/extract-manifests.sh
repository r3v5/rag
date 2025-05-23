#!/bin/zsh
# Extracts the manifests from an existing installed llamastack+llamastack-distribution

oc eksporter secret hf-token-secret --drop data.HF_TOKEN  > base/secret.yaml 
oc eksporter configmap llama3-chat-template               > base/config-map.yaml
oc eksporter deployment vllm                              > base/deployment.yaml
oc eksporter svc vllm --drop spec.clusterIPs              > base/service.yaml
oc eksporter route vllm --drop spec.host                  > base/route.yaml
oc eksporter LlamaStackDistribution                       > base/llama-stack-distribution.yaml

