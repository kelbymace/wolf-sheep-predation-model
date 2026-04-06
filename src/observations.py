
def get_sheep_observation(sheep):
    """
    RL observation for a sheep:
    - wolf presence in each cell of local Moore neighborhood
    - grass presence in each cell of local Moore neighborhood
    - sheep's current energy (normalized)
    """
    model = sheep.model
    radius = model.sheep_sight_radius

    wolf_obs = []
    grass_obs = []

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            x = (sheep.x + dx) % model.width
            y = (sheep.y + dy) % model.height

            wolf_present = any(
                w.alive and w.x == x and w.y == y
                for w in model.wolves
            )
            wolf_obs.append(1.0 if wolf_present else 0.0)

            if model.enable_grass:
                patch = model.get_patch(x, y)
                grass_present = 1.0 if patch.color == "green" else 0.0
            else:
                grass_present = 0.0

            grass_obs.append(grass_present)

    # Normalize energy so it is on a similar scale to the 0/1 inputs
    if sheep.energy is None:
        energy_obs = [0.0]
    else:
        energy_obs = [sheep.energy / max(1, model.sheep_gain_from_food * 2)]

    return wolf_obs + grass_obs + energy_obs