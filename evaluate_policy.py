import os
import numpy as np
import torch
import gym

from a2c_ppo_acktr.model import Policy, MLPBase
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs

from pusher_goal import PusherEnv
from pusher_policy_model import PusherPolicyModel

from PIL import Image


def evaluate_policy(args):
    with torch.no_grad():
        # load model
        save_path = os.path.join(args.save_dir, args.algo)
        save_file = os.path.join(save_path, args.env_name + ".pt")

        actor_critic = torch.load(save_file)[0]
        
        model = PusherPolicyModel()

        model.net.fc1.weight.data.copy_(actor_critic.base.actor[0].weight.data)
        model.net.fc1.bias.data.copy_(actor_critic.base.actor[0].bias.data)
        model.net.fc2.weight.data.copy_(actor_critic.base.actor[2].weight.data)
        model.net.fc2.bias.data.copy_(actor_critic.base.actor[2].bias.data)
        model.net.fc3.weight.data.copy_(actor_critic.dist.fc_mean.weight.data)
        model.net.fc3.bias.data.copy_(actor_critic.dist.fc_mean.bias.data)

        #device = torch.device("cuda:0" if args.cuda else "cpu")
        # make env
        env = PusherEnv()
        #envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
        #                 args.gamma, args.log_dir, device, True)

        # do episodes
        num_episodes = 100
        avg_L2_dist = 0

        frame = 0        

        for i in range(num_episodes):
            done = False
            obs = env.reset()

            while not done:
                #value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(torch.tensor(obs).float(), None, None, deterministic=True)
                action = model.infer(obs)
                #print(action.numpy()[0,1,0])
                obs, reward, done, info = env.step(action)
                if i < 10:
                    rgb = env.render()
                    im = Image.fromarray(rgb)
                    im.save('imgs/{}{:04d}.png'.format(args.algo, frame))
                    frame += 1

            print(obs)
            dist = np.linalg.norm(obs[3:6] - obs[6:9])
            print(dist)
            avg_L2_dist += dist / num_episodes

        print("Average L2 distance, 100 trials:", avg_L2_dist)\


if __name__ == "__main__":
    gym.register(id='PusherEnv-v0',
         entry_point='pusher_goal:PusherEnv',        
         kwargs={})

    # modify default args
    args = get_args()
    args.env_name = 'PusherEnv-v0'
    args.num_processes = 1
    args.num_steps=1000
    args.cuda = False
    
    args.algo = "ppo"
    evaluate_policy(args)

    args.algo = "ppo_fine_tune"
    evaluate_policy(args)

    args.algo = "ppo_fine_tune_joint"
    evaluate_policy(args)
