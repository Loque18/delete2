# ——————————————————————————————————————————————————
# IMPORTS
# ——————————————————————————————————————————————————
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

print('importing from delete2.epuck_env')
from delete2.gym_envs.epuck_env import EpuckEnv



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
        name_prefix="sac_epuck",
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

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=2_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        verbose=1,
        tensorboard_log="./logs/tensorboard/",
        device="cuda"
    )

    # ==================================================
    # TRAIN
    # ==================================================

    model.learn(
        total_timesteps=200_000,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    # ==================================================
    # SAVE FINAL MODEL
    # ==================================================

    model.save("./models/sac_epuck_final")

    env.close()
    eval_env.close()

    print("Training finished. Model saved at ./models/sac_epuck_final")