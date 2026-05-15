import os
import cv2
from stable_baselines3 import SAC, PPO

from gym_envs.epuck_env import EpuckEnv


def record_video():
    model_path = "./models/ppo_epuck_final.zip"
    output_path = "./videos/ppo_epuck_final.mp4"

    os.makedirs("./videos", exist_ok=True)

    env = EpuckEnv(render_mode="rgb_array")
    model = PPO.load(model_path, device="cuda")  # en tu VM

    obs, info = env.reset()

    frame = env.render()    
    h, w, _ = frame.shape

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (w, h),
    )

    done = False
    step = 0
    max_steps = 2000

    print('recording video...')

    while not done and step < max_steps:
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        frame = env.render()
        writer.write(frame)

        if terminated or truncated:
            print("EPISODE END")
            print("terminated:", terminated)
            print("truncated:", truncated)
            print("info:", info)

        step += 1

    writer.release()
    env.close()

    print(f"Video guardado en: {output_path}")


if __name__ == "__main__":
    record_video()