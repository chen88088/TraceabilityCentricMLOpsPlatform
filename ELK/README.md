# ELK

### 如果還沒加 Elastic Helm repo：

```
$ helm repo add elastic https://helm.elastic.co
```

```
$ helm repo update
```

### NAMESPACE

```
$ kubectl apply -f elk-namespace.yaml
```

### PV

```
$ kubectl apply -f elasticsearch-pv.yaml
```


### 安裝 elasticsearch（使用 Helm）

```
$ helm upgrade --install elasticsearch elastic/elasticsearch -n elk -f elasticsearch-values.yaml
```

---
# Logstash

### PV

```
$ kubectl apply -f logstash-pv.yaml
```

### 安裝 Logstash（使用 Helm）

```
$ helm upgrade --install logstash elastic/logstash -n elk -f logstash-values.yaml
```




# Kibana
<!-- 
```
$ kubectl delete serviceaccount pre-install-kibana-kibana -n elk --ignore-not-found
kubectl delete role pre-install-kibana-kibana -n elk --ignore-not-found
kubectl delete rolebinding pre-install-kibana-kibana -n elk --ignore-not-found
kubectl delete configmap kibana-kibana-helm-scripts -n elk --ignore-not-found
kubectl delete job pre-install-kibana-kibana -n elk --ignore-not-found
helm uninstall kibana -n elk --no-hooks || true

``` -->

 ### (OPTIONAL) 先建立一個 Secret 供 Kibana 使用連接 Elasticsearch：

```
$ kubectl create secret generic elk-kibana-credentials \
  -n elk \
  --from-literal=username=kibana_system \
  --from-literal=password=<你的 elastic 密碼>

```
### 安裝 Kibana（使用 Helm）

```
$ helm upgrade --install kibana elastic/kibana -n elk -f kibana-rendered.yaml
```