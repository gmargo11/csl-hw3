# csl-hw3

## Setup
Dependencies:
- python 3.6+
- airobot
- a2c_ppo_acktr: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
- pytorch
- baselines


## Training a Policy with Behavioral Cloning

You can train a policy in the pushing environment using behavioral cloning by running: `python3 train_behavioral_cloning.py`. The policy network is defined in `pusher_policy_model.py`. The expert data is given in the `expert_data.npz`.

## Fine-Tuning the Policy with PPO

Running `generate_comparison_finetune.py` will execute three PPO runs: one with no pre-training, one pre-trained with a vanilla loss function, and one pre-trained with a joint loss function that biases exploration towards the neighborhood of expert behavior.

The parameters of the trained networks are saved in the `trained_models` folder. To evaluate these models, run `evaluate_policy.py`. This will evaluate model performance across 100 episodes and save frames of the first 10 episodes in the `imgs` folder. Run `bash make_videos.sh` to generate videos from these frames.
