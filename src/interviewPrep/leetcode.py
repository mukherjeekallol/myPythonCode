# from collections import OrderedDict

# class LRU:
#     def __init__(self,capacity):
#         self.capacity = capacity
#         self.len = 0
#         self.cache = OrderedDict()

    
#     def get(self,key):
#         if key in self.cache:
#             self.cache.move_to_end(key)
#             return self.cache[key]
#         return -1
    
#     def put(self,key,value):
#         if key in self.cache:
#             self.cache.move_to_end(key)
#         self.cache[key]=value

#         if self.len==self.capacity:
#             self.cache.popitem(last=False)
       

# lru = LRU(3)
# lru.put(1, "A")
# lru.put(2, "B")
# lru.put(3, "C")

# print(lru.get(1))


# def flatDict(d,pref):
#     flat_d={}
#     for key in d:
#         val = d[key]
#         new_key=pref + key
#         if type(val)==dict:
#             flat_d.update(flatDict(val,new_key+"."))
#         else:            
#             flat_d[new_key]=val
#     return flat_d


# d1 = {
#     'k1':'v1',
#     'k2':'v2',
#     'k3':{
#         'k31':'v31',
#         'k32':'v32'
#     }
# }

# fd = flatDict(d1,'')

# print(fd)


# def flat(input):
#     flatList=[]
#     for item in input:
#         if type(item)==list:
#             flatList.extend(flat(item))
#         else:
#             flatList.append(item)
#     return flatList

# l = [1,2,3,[45,46,[123,980],12,[123,32]]]

# fl = flat(l)

# print(fl)

from confluent_kafka import avro
from confluent_kafka.avro import AvroProducer
import csv

value_schema_str = """
{
   "type": "record",
   "name": "myrecord",
   "fields": [
     {
       "name": "name",
       "type": "string"
     },
     {
       "name": "favorite_color",
       "type": ["string", "null"]
     },
     {
       "name": "favorite_number",
       "type": ["int", "null"]
     }
   ]
}
"""

value_schema = avro.loads(value_schema_str)

avro_producer = AvroProducer({
    'bootstrap.servers': 'localhost:9092',
    'schema.registry.url': 'http://localhost:8081'
    }, default_value_schema=value_schema)

from google.cloud import bigquery

client = bigquery.Client()
query = "SELECT * FROM `your_project.your_dataset.your_table`"
query_job = client.query(query)
rows = query_job.result()

for row in rows:
    avro_producer.produce(topic='my_topic', value=dict(row))
    avro_producer.flush()

from confluent_kafka import Consumer, KafkaError, SerializingProducer
from confluent_kafka.serialization import StringDeserializer
from confluent_kafka.avro import AvroDeserializer
from confluent_kafka.avro.serializer import SerializerError

settings = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'mygroup',
    'client.id': 'client-1',
    'key.deserializer': StringDeserializer('utf_8'),
    'value.deserializer': AvroDeserializer(value_schema_str),
    'enable.auto.commit': True,
    'session.timeout.ms': 6000,
    'default.topic.config': {'auto.offset.reset': 'smallest'}
}

c = Consumer(settings)

c.subscribe(['my_topic'])

with open('output.txt','w') as f:
    while True:
        msg = c.poll(0.1)
        if msg is None:
            continue
        elif not msg.error():
            f.write('{}\n'.format(msg.value().get('name')))
        elif msg.error().code() == KafkaError._PARTITION_EOF:
            print('End of partition reached {0}/{1}'
                  .format(msg.topic(), msg.partition()))
        else:
            print('Error occured: {0}'.format(msg.error().str()))

c.close()
