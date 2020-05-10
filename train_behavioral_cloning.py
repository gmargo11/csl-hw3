import numpy as np
import torch
from torch.utils import data

from pusher_policy_model import PusherPolicyModel
from pusher_goal import PusherEnv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


if __name__ == "__main__":

    model = PusherPolicyModel()
    num_epochs = 20
    train_losses, valid_losses = model.train(num_epochs=num_epochs)
    
    plt.figure()
    plt.plot(range(num_epochs+1), train_losses)
    plt.plot(range(num_epochs+1), valid_losses)
    plt.ylabel("Loss (MSE)")
    plt.xlabel("Epoch")
    plt.title("Behavioral Cloning Training")
    plt.legend(["Training Loss", "Validation Loss"])
    plt.ylim(0, train_losses[1] * 2.0)
    plt.show()
    plt.savefig("behavioral_cloning_training.png")
    
    

    ## evaluate model on 100 episodes
    env = PusherEnv()
    num_episodes = 100
    avg_L2_dist = 0
    avg_reward = 0

    frame = 0

    for i in range(num_episodes):
        done = False
        obs = env.reset()
        total_reward = 0
        while not done:
            action = model.infer(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if i < 10:
                    rgb = env.render()
                    im = Image.fromarray(rgb)
                    im.save('imgs/{}{:04d}.png'.format("bc", frame))
                    frame += 1

        dist = np.linalg.norm(obs[3:6] - obs[6:9])
        avg_L2_dist += dist / num_episodes
        avg_reward += total_reward / num_episodes

    print("Average L2 distance, 100 trials:", avg_L2_dist)
    print("Average reward, 100 trials", avg_reward)
