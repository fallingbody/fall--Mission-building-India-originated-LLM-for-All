"""
Kafka-based streaming data pipeline for distributed training.
"""
import json
import pickle
import asyncio
from confluent_kafka import Producer, Consumer, KafkaException
import torch
from typing import List, Iterator
from .dataset import FALLDataset

class KafkaDataStream:
    def __init__(self, brokers: List[str], topic: str, group_id: str):
        self.brokers = brokers
        self.topic = topic
        self.group_id = group_id

        self.producer = Producer({
            'bootstrap.servers': ','.join(brokers),
            'acks': 'all',
            'compression.type': 'lz4',
            'batch.size': 1048576,
        })

        self.consumer = Consumer({
            'bootstrap.servers': ','.join(brokers),
            'group.id': group_id,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': True,
            'fetch.max.bytes': 104857600,
        })
        self.consumer.subscribe([topic])

    def publish_batch(self, batch: Dict[str, torch.Tensor]):
        """Publish a batch to Kafka."""
        data = {
            'input_ids': batch['input_ids'].numpy().tolist(),
            'labels': batch['labels'].numpy().tolist(),
        }
        self.producer.produce(
            self.topic,
            key=str(hash(str(data['input_ids'][:10]))),
            value=json.dumps(data),
            callback=self._delivery_callback,
        )
        self.producer.flush()

    def consume_batches(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Consume batches from Kafka."""
        while True:
            msg = self.consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                raise KafkaException(msg.error())
            data = json.loads(msg.value())
            yield {
                'input_ids': torch.tensor(data['input_ids'], dtype=torch.long),
                'labels': torch.tensor(data['labels'], dtype=torch.long),
            }

    def _delivery_callback(self, err, msg):
        if err:
            print(f'Message delivery failed: {err}')

    def close(self):
        self.consumer.close()