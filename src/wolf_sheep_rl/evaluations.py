import numpy as np
import pandas as pd
import random
from .model import WolfSheepModel

def evaluate_one_episode(model, max_steps=200):

    # Shouldn't be in rl-training mode
    if model.model_version == "rl-training":
        raise ValueError("Policies should not be evaluated in rl-training mode")

    initial_sheep = len(model.sheep)

    for _ in range(max_steps):
        if not model.sheep:
            break
        model.go()

    sheep_survived = len(model.sheep)
    percent_survived = sheep_survived / initial_sheep if initial_sheep > 0 else None
    mean_final_energy = np.mean([s.energy for s in model.sheep]) if model.sheep else None

    return {
        "episode_length": model.ticks,
        "percent_survived": percent_survived,
        "starvation_deaths": model.starvation_deaths,
        "wolf_attack_deaths": model.wolf_attack_deaths,
        "mean_final_energy": mean_final_energy
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
        "percent_survived": ["mean", "std"],
        "starvation_deaths": ["mean", "std"],
        "wolf_attack_deaths": ["mean", "std"],
        "mean_final_energy": ["mean", "std"]
    })
    return summary