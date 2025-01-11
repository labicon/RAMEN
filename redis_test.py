import redis
import pickle
import numpy as np
import datetime


# Redis connection
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=False)




while True:
    # send data
    # send data
    theta_i =  np.random.rand(1633360).astype(np.float32)
    uncertainty_i  =  np.random.rand(1633360).astype(np.float32)

    msg = {'theta_i':theta_i, 'uncertainty_i':uncertainty_i}
    pickled_data = pickle.dumps(msg)
    redis_client.set('agent_i', pickled_data )


    # get data
    agent_j = redis_client.get('agent_j')
    if agent_j:
        print(f"{datetime.datetime.now()}: receive agent j!")
        agent_j = pickle.loads(agent_j)
        array_1 = agent_j['theta_j']
        array_2 = agent_j['uncertainty_j']

        print(f'theta_j = {array_1.shape}, uncertainty_j= {array_2.shape}')