import numpy as np
import torch
from torch.utils import data

from pusher_policy_model import PusherPolicyModel
from pusher_goal import pusher_env


if __name__ == "__main__":
    model = PusherPolicyModel()
    model.train(num_epochs=20)

    ## evaluate model on 100 episodes
    env = PusherEnv()
    num_episodes = 100
    avg_L2_dist = 0


    for i in range(num_episodes):
        done = False
        obs = env.reset()
        while not done:
            action = model.infer(obs)
            obs, reward, done, info = env.step(action)

        dist = np.linalg.norm(obs[3:6] - obs[6:9])
        avg_L2_dist += dist / num_episodes

    print("Average L2 distance, 100 trials:", avg_L2_dist)