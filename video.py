import os
import cv2
from stable_baselines3 import SAC

from delete2.gym_envs.epuck_env import EpuckEnv


def record_video():
    model_path = "./models/sac_epuck_final.zip"
    output_path = "./videos/sac_epuck_eval.mp4"

    os.makedirs("./videos", exist_ok=True)

    env = EpuckEnv(render_mode="rgb_array")
    model = SAC.load(model_path, device="cuda")  # en tu VM

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
    max_steps = 1000

    while not done and step < max_steps:
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        frame = env.render()
        writer.write(frame)

        step += 1

    writer.release()
    env.close()

    print(f"Video guardado en: {output_path}")


if __name__ == "__main__":
    record_video()