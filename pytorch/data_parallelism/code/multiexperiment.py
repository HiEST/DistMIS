import json
import argparse
import numpy as np
import os
import ray
import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", "-g", type=str, help="number of gpus")
    args, _ = parser.parse_known_args()

    if int(args.gpu) > 4:
        ray.init(address='auto', _node_ip_address=os.environ["ip_head"].split(":")[0],
                 _redis_password=os.environ["redis_password"])
    else:
        ray.init(address='auto', _redis_password='5241590000000000')

    # number of experiments to train
    for i in range(24):
        print('experiment ' + str(i))
        with open('../configs/'+str(args.gpu)+'gpu/config.json', 'r+') as f:
            config = json.load(f)
            config['lr']= config['lr'] + np.random.uniform(0, 0.00000009)
            f.seek(0)
            json.dump(config, f)
            f.truncate()            
        train.main(args.gpu)
    ray.shutdown()
