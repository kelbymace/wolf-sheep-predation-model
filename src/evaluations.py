import numpy as np
import pandas as pd
import random
from .model import WolfSheepModel

def evaluate_one_episode(model, max_steps=200):
    grass_eaten = 0
    wolf_close_steps = 0
    nearest_wolf_dists = []

    for _ in range(max_steps):
        if not model.sheep:
            break

        sheep = model.sheep[0] if len(model.sheep) == 1 else None

        # Track local danger before step
        if sheep is not None and model.wolves:
            dists = [
                model.neighbor_distance(sheep.x, sheep.y, wolf.x, wolf.y)
                for wolf in model.wolves if wolf.alive
            ]
            if dists:
                nearest = min(dists)
                nearest_wolf_dists.append(nearest)
                if nearest <= 1:
                    wolf_close_steps += 1

        prev_energy = sheep.energy if sheep is not None else None

        result = model.go()

        # In rl-training mode, go() returns (reward, done)
        if model.model_version == "rl-training":
            _, done = result
            if done:
                break
        else:
            if result is False:
                break

        # Approximate grass eaten via positive energy change
        if sheep is not None and model.sheep and model.sheep[0].alive:
            if prev_energy is not None and model.sheep[0].energy is not None:
                if model.sheep[0].energy > prev_energy:
                    grass_eaten += 1

    sheep_alive = len(model.sheep) > 0
    final_energy = model.sheep[0].energy if sheep_alive and model.sheep[0].energy is not None else None

    return {
        "episode_length": model.ticks,
        "survived_to_end": sheep_alive and model.ticks >= max_steps,
        "final_energy": final_energy,
        "grass_eaten_events": grass_eaten,
        "wolf_close_steps": wolf_close_steps,
        "avg_nearest_wolf_dist": np.mean(nearest_wolf_dists) if nearest_wolf_dists else None,
    }

def evaluate_policy(policy_name, n_episodes=100, max_steps=200, model_kwargs=None, policy_net=None, seed_base=0):
    if model_kwargs is None:
        model_kwargs = {}

    results = []

    for ep in range(n_episodes):
        seed = seed_base + ep # Make sure each policy is tested against the same starting configurations
        random.seed(seed)
        np.random.seed(seed)

        kwargs = dict(model_kwargs)
        kwargs["sheep_strategy"] = policy_name

        if policy_name == "rl":
            kwargs["policy_net"] = policy_net

        model = WolfSheepModel(**kwargs)
        model.collect_log_probs = False
        model.setup()

        metrics = evaluate_one_episode(model, max_steps=max_steps)
        metrics["policy"] = policy_name
        metrics["episode"] = ep
        results.append(metrics)

    return pd.DataFrame(results)

def compare_policies(policy_names, n_episodes=500, max_steps=200, model_kwargs=None, policy_net=None):
    dfs = []

    for policy in policy_names:
        df = evaluate_policy(
            policy_name=policy,
            n_episodes=n_episodes,
            max_steps=max_steps,
            model_kwargs=model_kwargs,
            policy_net=policy_net,
        )
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

def summarize_policy_results(df):
    summary = df.groupby("policy").agg({
        "episode_length": ["mean", "std"],
        "survived_to_end": "mean",
        "grass_eaten_events": "mean",
        "wolf_close_steps": "mean",
        "avg_nearest_wolf_dist": "mean",
        "final_energy": "mean",
    })
    return summary