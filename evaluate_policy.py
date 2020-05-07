from a2c_ppo_acktr.model import Policy, MLPBase
from a2c_ppo_acktr.arguments import get_args

from pusher_goal import PusherEnv



def evaluate_policy(args):

    # load model
    save_path = os.path.join(args.save_dir, args.algo)
    save_file = os.path.join(save_path, args.env_name + ".pt")

    actor_critic = torch.load(save_file)[0]

    # make env
    env = PusherEnv()

    # do episodes
    num_episodes = 100
    avg_L2_dist = 0


    for i in range(num_episodes):
        done = False
        obs = env.reset()
        while not done:
            action = model.act(obs, None, None, deterministic=False)
            obs, reward, done, info = env.step(action)

        dist = np.linalg.norm(obs[3:6] - obs[6:9])
        avg_L2_dist += dist / num_episodes

    print("Average L2 distance, 100 trials:", avg_L2_dist)\


if __name__ == "__main__":
    # modify default args
    args = get_args()
    args.env_name = 'PusherEnv-v0'
    args.num_processes = 1
    args.num_steps=1000
    args.cuda = True

    args.algo = "ppo"
    evaluate_policy(args)

    args.algo = "ppo_fine_tune"
    evaluate_policy(args)

    args.algo = "ppo_fine_tune_joint"
    evaluate_policy(args)