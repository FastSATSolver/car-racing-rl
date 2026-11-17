import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from car_racing_helper import make_env, load_model


episode_num = 0


def report_end(locals_, globals_):
    global episode_num
    if not locals_["done"]:
        return

    info = locals_["info"]
    episode_num += 1

    if info.get("TimeLimit.truncated", False):
        reason = "TRUNCATED (time limit reached)"
    else:
        reason = "TERMINATED (env reached a terminal state)"

    ep = info.get("episode", {})
    print(f"Episode {episode_num}: {reason} | "
          f"reward={ep.get('r', float('nan')):.2f}, length={ep.get('l', '?')}, lap_finished={info.get("lap_finished", False)}")


def main():
    env = make_env(n_envs=10, render_mode="human")

    model = load_model(env)

    mean_reward, std_reward = evaluate_policy(
        model,
        model.get_env(),
        n_eval_episodes=10,
        deterministic=True,
        render=True,
        callback=report_end
    )

    print(f"mean_reward = {mean_reward:.2f} +/- {std_reward:.2f}")

    # vec_env = model.get_env()
    # obs = vec_env.reset()
    # for i in range(10000):
    #     actions, _states = model.predict(obs, deterministic=True)
    #     obs, rewards, dones, info = vec_env.step(actions)
    #     vec_env.render("human")
    #     # VecEnv resets automatically
    #     # if done:
    #     #   obs = vec_env.reset()
    #     if dones[0]:
    #         print("Died:", 'lap_finished:', info[0]['lap_finished'] if 'lap_finished' in info[0] else None,
    #               'TimeLimit.truncated:', info[0]['TimeLimit.truncated'] if 'TimeLimit.truncated' in info[0] else None)


if __name__ == "__main__":
    main()
