# Overview
An implementation of [Data-Efficient Hierarchical Reinforcement Learning](https://arxiv.org/pdf/1805.08296.pdf) (HIRO) in PyTorch.
![Demonstration](media/demo.gif)

# Installation
1. Follow installation of [OpenAI Gym Mujoco Installation](https://github.com/openai/mujoco-py)
```
1. Obtain a 30-day free trial on the MuJoCo website or free license if you are a student. The license key will arrive in an email with your username and password.
2. Download the MuJoCo version 2.0 binaries for Linux or OSX.
3. Unzip the downloaded mujoco200 directory into ~/.mujoco/mujoco200, and place your license key (the mjkey.txt file from your email) at ~/.mujoco/mjkey.txt.
```
2. Install Dependencies
```
pip install -r requirements.txt
```

# Run
For `HIRO`,
```
python main.py --train
```

For `TD3`,
```
python main.py --train --td3
```
# Evaluate Trained Model
Passing `--eval` argument will read the most updated model parameters and start playing. The goal is to get to the position (0, 16), which is top left corner.

For `HIRO`,
```
python main.py --eval
```

For `TD3`,
```
python main.py --eval --td3
```


# Trainining result
Blue is HIRO and orange is TD3

## Succss Rate
<img src="media/Success_Rate.svg" alt="Success_Rate" width="400"/>

## Reward
<img src="media/reward_Reward.svg" alt="reward_Reward" width="400"/>

## Intrinsic Reward
<img src="media/reward_Intrinsic_Reward.svg" alt="reward_Intrinsic_Reward" width="400"/>

## Losses
Higher Controller Actor <br>
<img src="media/loss_actor_loss_high.svg" alt="loss_actor_loss_high" width="400"/>

Higher Controller Critic<br>
<img src="media/loss_critic_loss_high.svg" alt="loss_critic_loss_high" width="400"/>

Lower Controller Actor<br>
<img src="media/loss_actor_loss_low.svg" alt="loss_actor_loss_low" width="400"/>

Lower Controller Critic<br>
<img src="media/loss_critic_loss_low.svg" alt="loss_critic_loss_low" width="400"/>

TD3 Controller Actor<br>
<img src="media/loss_actor_loss_td3.svg" alt="loss_actor_loss_td3" width="400"/>

TD3 Controller Critic<br>
<img src="media/loss_critic_loss_td3.svg" alt="loss_critic_loss_td3" width="400"/>

