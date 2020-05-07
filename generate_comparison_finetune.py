from ppo_fine_tuning import train_ppo_fine_tune
from ppo_fine_tuning_joint import train_ppo_fine_tune_joint
from ppo_from_scratch import train_ppo_from_scratch

import numpy as np

from a2c_ppo_acktr.arguments import get_args

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import gym
from pusher_goal import PusherEnv


def generate_comparison_finetune():
    
    # modify default args
    args = get_args()
    args.env_name = 'PusherEnv-v0'
    args.num_processes = 1
    args.num_steps=1000
    args.num_env_steps=301000
    args.cuda = False

    args.algo = "ppo"
    scratch_rewards, scratch_times = train_ppo_from_scratch(args)
    
    args.algo = "ppo_fine_tune"
    vanilla_rewards, vanilla_times = train_ppo_fine_tune(args)

    args.algo = "ppo_fine_tune_jointloss"
    joint_rewards, joint_times = train_ppo_fine_tune_joint(args)


    plt.figure()
    plt.plot(scratch_times, scratch_rewards)
    plt.plot(vanilla_times, vanilla_rewards)
    plt.plot(joint_times, joint_rewards)
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Episode Reward (10 episodes)")
    plt.legend(["PPO from scratch", "PPO with expert, vanilla", "PPO with expert, joint loss"])

    plt.show()
    plt.savefig("ppo_comparison.png")



if __name__ == "__main__":
    # register env
    np.random.seed(1)
    gym.register(id='PusherEnv-v0',
         entry_point='pusher_goal:PusherEnv',        
         kwargs={})

    generate_comparison_finetune()
