#!/bin/bash

set -e

echo "🗑 移除 Kafka StatefulSet"
kubectl delete -f kafka-statefulset.yaml --ignore-not-found

echo "🗑 移除 Kafka PV"
kubectl delete -f kafka-pv.yaml --ignore-not-found

echo "🗑 移除 Zookeeper StatefulSet"
kubectl delete -f zookeeper-statefulset.yaml --ignore-not-found

echo "🗑 移除 Zookeeper PV"
kubectl delete -f zookeeper-pv.yaml --ignore-not-found

kubectl delete namespace kafka

echo "✅ Kafka & Zookeeper 已完全解除安裝"
