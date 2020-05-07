from ppo_fine_tuning import train_ppo_fine_tune
from ppo_from_scratch import train_ppo_from_scratch

from a2c_ppo_acktr.arguments import get_args

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import gym
from pusher_goal import PusherEnv
gym.register(id='PusherEnv-v0',
         entry_point='pusher_goal:PusherEnv',        
         kwargs={})


def generate_comparison_finetune():
    # modify default args
    args = get_args()
    args.env_name = 'PusherEnv-v0'
    args.num_processes = 1
    args.num_steps=1000
    args.num_env_steps=40000
    args.cuda = True

    args.algo = "ppo"
    scratch_rewards, scratch_times = train_ppo_from_scratch(args)
    
    args.algo = "ppo_fine_tune"
    vanilla_rewards, vanilla_times = train_ppo_fine_tune(args)

    args.algo = "ppo_fine_tune_jointloss"
    #joint_rewards, joint_times = train_ppo_joint_loss(args)


    plt.figure()
    plt.plot(scratch_rewards, scratch_times)
    plt.plot(vanilla_rewards, vanilla_times)
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Episode Reward (10 episodes)")
    plt.legend(["PPO from scratch", "PPO with expert, vanilla"]) #, "PPO with expert, joint loss"])

    plt.show()
    plt.savefig("ppo_comparison.png")



