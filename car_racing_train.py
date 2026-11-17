import gymnasium as gym

from car_racing_helper import make_env, load_model


def main():
    env = make_env(16)

    model = load_model(env)

    model.learn(total_timesteps=1_000_000,
                tb_log_name="first_run", progress_bar=True, reset_num_timesteps=False)

    model.save("carracing.zip")


if __name__ == "__main__":
    main()
