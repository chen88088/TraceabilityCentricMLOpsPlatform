#!/bin/bash

set -e
echo "✅ 建立 namespace"
kubectl create namespace kafka


echo "✅ 建立 Zookeeper PV"
kubectl apply -f zookeeper-pv.yaml

echo "✅ 建立 Zookeeper StatefulSet"
kubectl apply -f zookeeper-statefulset.yaml

echo "✅ 建立 Kafka PV"
kubectl apply -f kafka-pv.yaml

echo "✅ 建立 Kafka StatefulSet"
kubectl apply -f kafka-statefulset.yaml

echo "⏳ 等待 Kafka Ready（可能需幾分鐘）"
kubectl rollout status statefulset/kafka -n elk

echo "✅ Kafka & Zookeeper 已完成部署"
echo "🔍 Kafka Broker 服務：kafka-headless.elk.svc:9092"
