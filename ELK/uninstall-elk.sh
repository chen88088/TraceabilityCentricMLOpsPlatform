#!/bin/bash

set -e

echo "🧹 卸載 Kibana（含 pre-install hooks 清除）"
helm uninstall kibana -n elk --no-hooks || true

kubectl delete serviceaccount pre-install-kibana-kibana -n elk --ignore-not-found
kubectl delete role pre-install-kibana-kibana -n elk --ignore-not-found
kubectl delete rolebinding pre-install-kibana-kibana -n elk --ignore-not-found
kubectl delete configmap kibana-kibana-helm-scripts -n elk --ignore-not-found
kubectl delete job pre-install-kibana-kibana -n elk --ignore-not-found
kubectl delete job post-delete-kibana-kibana -n elk --ignore-not-found
kubectl delete role post-delete-kibana-kibana -n elk --ignore-not-found
kubectl delete rolebinding post-delete-kibana-kibana -n elk --ignore-not-found
kubectl delete serviceaccount post-delete-kibana-kibana -n elk --ignore-not-found

echo "🧹 卸載 Logstash"
helm uninstall logstash -n elk || true

echo "🧹 卸載 Elasticsearch"
helm uninstall elasticsearch -n elk || true

echo "🧹 移除 PV / PVC（如需）"
kubectl delete -f elasticsearch-pv.yaml --ignore-not-found
kubectl delete -f logstash-pv.yaml --ignore-not-found

echo "🧹 移除 Secret / Config"
kubectl delete secret elk-kibana-credentials -n elk --ignore-not-found

echo "🧹 清除 Namespace（如無其他資源）"
kubectl delete namespace elk --ignore-not-found

echo "✅ ELK Stack 已全部卸載完成"
