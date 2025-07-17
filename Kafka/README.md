# Kafka

# local test create topic and test nofunction or not

```
$ bin/kafka-topics.sh --create \
  --bootstrap-server 10.52.52.142:30092 \
  --replication-factor 1 \
  --partitions 1 \
  --topic test-topic
```

```
$ bin/kafka-topics.sh --list --bootstrap-server 10.52.52.142:30092
```

producer (one terminal)

```
$ bin/kafka-console-producer.sh --broker-list 10.52.52.142:30092 --topic test-topic
```


consumer (one terminal)

```
$ bin/kafka-console-producer.sh --broker-list 10.52.52.142:30092 --topic test-topic
```