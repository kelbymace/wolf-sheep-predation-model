import random
import numpy as np
from .patch import Patch
from .sheep import Sheep
from .wolf import Wolf
from .rewards import get_reward
from .observations import get_sheep_observation


class WolfSheepModel:
    def __init__(
        self,
        width=50,
        height=50,
        initial_number_sheep=100,
        initial_number_wolves=50,
        sheep_gain_from_food=4,
        wolf_gain_from_food=20,
        sheep_reproduce=4.0,
        wolf_reproduce=5.0,
        grass_regrowth_time=30,
        model_version="sheep-wolves-grass",
        show_energy=False,
        max_sheep=3000,
        sheep_strategy="random",
        wolf_strategy="random",
        wolf_sight_radius=1,
        sheep_sight_radius=1,
        policy_net=None,
        enable_grass=True,
        rl_death_penalty=-100.0,
        rl_alive_reward=1.0,
        rl_grass_bonus=0.0, # best at 0.01
        rl_wolf_d1_penalty=0.0, # best at 0.4
        rl_wolf_d2_penalty=0.0 # best at 0.15
    ):
        self.width = width
        self.height = height

        self.initial_number_sheep = initial_number_sheep
        self.initial_number_wolves = initial_number_wolves
        self.sheep_gain_from_food = sheep_gain_from_food
        self.wolf_gain_from_food = wolf_gain_from_food
        self.sheep_reproduce = sheep_reproduce
        self.wolf_reproduce = wolf_reproduce
        self.grass_regrowth_time = grass_regrowth_time
        self.model_version = model_version
        self.show_energy = show_energy
        self.max_sheep = max_sheep

        self.sheep_strategy = sheep_strategy
        self.wolf_strategy = wolf_strategy
        self.sheep_sight_radius = sheep_sight_radius
        self.wolf_sight_radius = wolf_sight_radius

        self.enable_grass = enable_grass
        self.policy_net = policy_net

        # RL Rewards & penalties
        self.rl_death_penalty = rl_death_penalty
        self.rl_alive_reward = rl_alive_reward
        self.rl_grass_bonus = rl_grass_bonus
        self.rl_wolf_d1_penalty = rl_wolf_d1_penalty
        self.rl_wolf_d2_penalty = rl_wolf_d2_penalty

        self.collect_log_probs = False
        self.current_episode_log_probs = []

        self.patches = []
        self.sheep = []
        self.wolves = []
        self.new_sheep = []
        self.new_wolves = []
        self.ticks = 0

        if self.model_version not in ["sheep-wolves-grass", "sheep-wolves", "rl-training"]:
            raise ValueError(f"Unknown model version: {self.model_version}.\nShould be one of: 'sheep-wolves-grass', 'sheep-wolves', 'rl-training'")

    def get_patch(self, x, y):
        return self.patches[y][x]

    def setup(self):
        self.ticks = 0
        self.sheep = []
        self.wolves = []
        self.new_sheep = []
        self.new_wolves = []
        self.patches = []
        self.current_episode_log_probs = [] # record for rl training

        for y in range(self.height):
            row = []
            for x in range(self.width):
                patch = Patch(x, y)

                if self.model_version == "sheep-wolves-grass":
                    patch.color = random.choice(["green", "brown"])
                    if patch.color == "green":
                        patch.countdown = self.grass_regrowth_time
                    else:
                        patch.countdown = random.randrange(self.grass_regrowth_time)
                else:
                    patch.color = "green"

                row.append(patch)
            self.patches.append(row)
        
        if self.model_version == "rl-training" and not self.enable_grass:
            sheep_energy = None
        else:
            sheep_energy = random.randrange(max(1, 2 * self.sheep_gain_from_food))
        wolf_energy = None if self.model_version == "rl-training" else random.randrange(max(1, 2 * self.wolf_gain_from_food))

        for _ in range(self.initial_number_sheep):
            x = random.randrange(self.width)
            y = random.randrange(self.height)
            if self.model_version in ["sheep-wolves-grass", "rl-training"]:
                energy = random.randrange(2 * self.sheep_gain_from_food)
            self.sheep.append(Sheep(self, x, y, energy))

        for _ in range(self.initial_number_wolves):
            x = random.randrange(self.width)
            y = random.randrange(self.height)
            if self.model_version == "sheep-wolves-grass":
                energy = random.randrange(2 * self.wolf_gain_from_food)
            self.wolves.append(Wolf(self, x, y, energy))

        self.display_labels()

    def grow_grass(self):
        for row in self.patches:
            for patch in row:
                if patch.color == "brown":
                    if patch.countdown <= 0:
                        patch.color = "green"
                        patch.countdown = self.grass_regrowth_time
                    else:
                        patch.countdown -= 1

    def grass(self):
        if self.model_version in ["sheep-wolves-grass", "rl-training"]:
            return [
                patch
                for row in self.patches
                for patch in row
                if patch.color == "green"
            ]
        return 0

    def display_labels(self):
        for s in self.sheep:
            s.label = ""
        for w in self.wolves:
            w.label = ""

        if self.show_energy:
            for w in self.wolves:
                w.label = str(round(w.energy))
            if self.model_version == "sheep-wolves-grass":
                for s in self.sheep:
                    s.label = str(round(s.energy))
                    
    def moore_directions(self, include_stay=False):
        directions = [
            (-1, -1), (0, -1), (1, -1),
            (-1,  0),          (1,  0),
            (-1,  1), (0,  1), (1,  1),
        ]
        if include_stay:
            directions.append((0, 0))
        return directions

    def wrap_delta(self, source, target, size):
        """
        Smallest signed wrapped displacement from source to target.
        """
        delta = target - source
        if delta > size / 2:
            delta -= size
        elif delta < -size / 2:
            delta += size
        return delta

    def get_patch(self, x, y):
        return self.patches[y][x]
    
    def sheep_at(self, x, y):
        return [s for s in self.sheep if s.alive and s.x == x and s.y == y]
    
    def get_animals_in_neighborhood(self, x, y, radius, animal_type, exclude=None):
        """
        Return animals of the requested type within a Moore neighborhood
        of the given radius around (x, y).

        animal_type should be either "sheep" or "wolves".
        """
        if animal_type == "sheep":
            animals = self.sheep
        elif animal_type == "wolves":
            animals = self.wolves
        else:
            raise ValueError(f"Unknown animal_type: {animal_type}")

        found = []
        for animal in animals:
            if not animal.alive:
                continue
            if exclude is not None and animal is exclude:
                continue

            dx = self.wrap_delta(x, animal.x, self.width)
            dy = self.wrap_delta(y, animal.y, self.height)

            if max(abs(dx), abs(dy)) <= radius:
                found.append(animal)

        return found
    
    def neighbor_distance(self, x1, y1, x2, y2):
        """
        Moore distance on a toroidal grid.
        """
        dx = self.wrap_delta(x1, x2, self.width)
        dy = self.wrap_delta(y1, y2, self.height)
        return max(abs(dx), abs(dy))
    
    def count_wolves_at_distance(self, sheep, distance):
        """
        Count living wolves at the given distance from the sheep.
        """
        count = 0
        for wolf in self.wolves:
            if not wolf.alive:
                continue
            if self.neighbor_distance(sheep.x, sheep.y, wolf.x, wolf.y) == distance:
                count += 1
        return count

    def count_sheep(self):
        return len(self.sheep)

    def count_wolves(self):
        return len(self.wolves)

    def patch_array(self):
        """
        Convert patches into a numeric grid for plotting:
        green grass = 1
        brown/no grass = 0
        """
        arr = np.zeros((self.height, self.width))
        for y in range(self.height):
            for x in range(self.width):
                arr[y, x] = 1 if self.patches[y][x].color == "green" else 0
        return arr
    
    def go(self):
        if self.model_version != "rl-training":
            if not self.sheep and not self.wolves:
                return False

            if not self.wolves and self.sheep:
                # print("The sheep have inherited the earth")
                return False
        else:
            if len(self.sheep) > 1:
                raise ValueError("Too mnay sheep confuses the shepherd!\n\nrl-training mode expects exactly one sheep or zero after predation.")
        
        if self.model_version == "sheep-wolves-grass":
            self.new_sheep = []
            self.new_wolves = []

        grass_eaten = False
        sheep_before = self.sheep[0] if (self.model_version == "rl-training" and self.sheep) else None
        sheep_energy_before = sheep_before.energy if sheep_before is not None else None

        for sheep in self.sheep[:]:
            if sheep.alive:
                sheep.step()

        self.sheep = [s for s in self.sheep if s.alive]

        for wolf in self.wolves[:]:
            if wolf.alive:
                wolf.step()

        self.wolves = [w for w in self.wolves if w.alive]
        self.sheep = [s for s in self.sheep if s.alive]

        if self.model_version == "sheep-wolves-grass":
            self.sheep.extend(self.new_sheep)
            self.wolves.extend(self.new_wolves)

        if self.model_version in ["sheep-wolves-grass", "rl-training"] and self.enable_grass:
            self.grow_grass()

        self.ticks += 1
        self.display_labels()

        # Return rewards for RL training
        if self.model_version == "rl-training":
            if not self.sheep:
                # print("Our sheep did not survive training.")
                return self.rl_death_penalty, True # reward, done
            sheep_after = self.sheep[0]
            if sheep_energy_before is not None and sheep_after.energy is not None and sheep_after.energy > sheep_energy_before:
                grass_eaten = True
            reward = self.get_reward(sheep_after, grass_eaten=grass_eaten)
            done = False
            return reward, done

        return True