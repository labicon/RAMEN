import redis
import pickle
import numpy as np

# Redis connection
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=False)
array = np.zeros(1724720).astype(np.float32)
theta_i = np.array([1, 2, 3, 4, 5]).astype(np.float32)
uncertainty_j = np.array([1, 2, 3, 4, 5]).astype(np.float32)*2
while True:
    # send data
    # send data

    array = array.flatten()
    msg = {'theta_i':theta_i, 'uncertainty_i':uncertainty_j}
    pickled_data = pickle.dumps(msg)
    redis_client.set('agent_i', pickled_data )


    # get data
    agent_j = redis_client.get('agent_j')
    if agent_j:
        agent_j = pickle.loads(agent_j)
        array_1 = agent_j['theta_j']
        array_2 = agent_j['uncertainty_j']

        print(f'theta_j = {array_1}, uncertainty_j= {array_2}')