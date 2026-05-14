from epuck_env import EpuckEnv
from gymnasium.utils.env_checker import check_env


if __name__ == "__main__":

    # ==================================================
    # VALIDACIÓN SIN RENDER
    # ==================================================

    test_env = EpuckEnv(render_mode=None)
    check_env(test_env.unwrapped)
    test_env.close()

    print("check_env OK")

    # ==================================================
    # EJECUCIÓN VISUAL
    # ==================================================

    env = EpuckEnv(render_mode="human")

    obs, info = env.reset(seed=42)

    print("obs shape:", obs.shape)
    print("initial obs:", obs)

    for _ in range(1000):
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        print(reward, terminated, truncated, info)

        if terminated or truncated:
            obs, info = env.reset()

    env.close()