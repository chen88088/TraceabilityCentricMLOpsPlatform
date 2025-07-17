#!/bin/bash

set -e

echo "✅ 加入 Elastic Helm Repo..."
helm repo add elastic https://helm.elastic.co || true
helm repo update

echo "✅ 建立 NAMESPACE..."
kubectl apply -f elk-namespace.yaml

echo "✅ 建立 Elasticsearch PV..."
kubectl apply -f elasticsearch-pv.yaml

echo "✅ 安裝 Elasticsearch..."
helm upgrade --install elasticsearch elastic/elasticsearch \
  -n elk -f elasticsearch-values.yaml

echo "✅ 建立 Logstash PV..."
kubectl apply -f logstash-pv.yaml

echo "✅ 安裝 Logstash..."
helm upgrade --install logstash elastic/logstash \
  -n elk -f logstash-values.yaml

echo "✅ 建立 Kibana credentials Secret..."
kubectl create secret generic elk-kibana-credentials \
  -n elk \
  --from-literal=username=kibana_system \
  --from-literal=password=<請填入密碼> \
  --dry-run=client -o yaml | kubectl apply -f -

echo "✅ 安裝 Kibana..."
helm upgrade --install kibana elastic/kibana \
  -n elk -f kibana-values.yaml

echo "🎉 ELK Stack 安裝完成！請稍候幾分鐘等待全部 Pod 就緒"
echo "🌐 Kibana 入口：http://<NodeIP>:30601"
