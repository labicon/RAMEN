import redis
import pickle
import numpy as np

# Redis connection
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=False)

# create data
array = np.zeros((320,640)).astype(np.uint16)

# serialization and send 
array_pickled = pickle.dumps(array)
redis_client.set('test_data', array_pickled)

# receive and deserialization
array_received = pickle.loads(redis_client.get('test_data'))
print(array_received.shape)


