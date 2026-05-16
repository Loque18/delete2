from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from gym_envs.epuck_env_ppo import EpuckEnv

def make_env():
    env = EpuckEnv(render_mode=None)
    env = Monitor(env)
    return env

def train():
    # ==================================================
    # VALIDATE ENV
    # ==================================================

    test_env = EpuckEnv(render_mode=None)
    check_env(test_env, warn=True)
    test_env.close()

    print("Env OK")

    # ==================================================
    # TRAIN ENV
    # ==================================================

    env = DummyVecEnv([make_env])

    # ==================================================
    # EVAL ENV
    # ==================================================

    eval_env = DummyVecEnv([make_env])

    # ==================================================
    # CALLBACKS
    # ==================================================

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path="./models/checkpoints/",
        name_prefix="ppo_epuck",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best/",
        log_path="./logs/eval/",
        eval_freq=5_000,
        deterministic=True,
        render=False,
    )

    # ==================================================
    # MODEL
    # ==================================================

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./logs/tensorboard/",
        device="cpu"
    )

    # ==================================================
    # TRAIN
    # ==================================================
    model.learn(
        total_timesteps=1_000_000,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )

        # ==================================================
    # SAVE FINAL MODEL
    # ==================================================

    model.save("./models/ppo_epuck_final")

    env.close()
    eval_env.close()

    print("Training finished. Model saved at ./models/ppo_epuck_final")