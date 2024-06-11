import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback


config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 10000,
    "env_name": "CartPole-v1",
}
run = wandb.init(
    project="examples-wandb",
    notes="v0.1 Minimal PPO on CartPole-v1",
    config=config,
    sync_tensorboard=True,  # Send ep_rew_mean++
    monitor_gym=True,  # not working?
)


def make_env():
    env = gym.make(config["env_name"], render_mode="rgb_array")
    env = Monitor(env)  # record stats such as returns
    return env


env = DummyVecEnv([make_env])
env = VecVideoRecorder(
    env,
    f"videos/{run.id}",
    record_video_trigger=lambda x: x % 2000 == 0,
    video_length=200,
)
model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_freq=1000,
        model_save_path=f"models/{run.id}",
        verbose=2,
    ),
)
run.finish()