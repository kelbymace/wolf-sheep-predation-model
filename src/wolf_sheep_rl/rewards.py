def get_reward(model, sheep, grass_eaten=False):
    """
    RL reward:
    - base reward for surviving the step
    - strong death penalty
    - negative shaping for nearby wolves
    - optional small positive shaping for energy gain
    """

    # If sheep died, return death penalty immediately
    if sheep is None or not sheep.alive:
        return model.rl_death_penalty

    reward = model.rl_alive_reward

    # Penalize nearby wolves by ring
    wolves_d1 = model.count_wolves_at_distance(sheep, 1)
    wolves_d2 = model.count_wolves_at_distance(sheep, 2)

    reward -= model.rl_wolf_d1_penalty * wolves_d1
    reward -= model.rl_wolf_d2_penalty * wolves_d2

    # Optional shaping for positive energy gain
    if grass_eaten:
        reward += model.rl_grass_bonus
    return reward