ffmpeg -i imgs/ppo%04d.png -f segment -strftime 1 videos/ppo%Y-%m-%d_%H-%M-%S.webm
ffmpeg -i imgs/ppo_fine_tune%04d.png -f segment -strftime 1 videos/ppo_fine_tune%Y-%m-%d_%H-%M-%S.webm
ffmpeg -i imgs/ppo_fine_tune_joint%04d.png -f segment -strftime 1 videos/ppo_fine_tune_joint%Y-%m-%d_%H-%M-%S.webm