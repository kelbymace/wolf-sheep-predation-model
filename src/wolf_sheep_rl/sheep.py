import random
import math
import numpy as np
from .animal import Animal
from .policy import choose_action
from .observations import get_sheep_observation


action_to_move = {
    0: (0, 1),     # N
    1: (1, 1),     # NE
    2: (1, 0),     # E
    3: (1, -1),    # SE
    4: (0, -1),    # S
    5: (-1, -1),   # SW
    6: (-1, 0),    # W
    7: (-1, 1),    # NW
}

move_to_action = {v: k for k, v in action_to_move.items()}

class Sheep(Animal):

    def move(self):
        strategy = self.model.sheep_strategy

        if strategy == "rl":
            self.move_rl()
        elif strategy == "random":
            self.move_random()
        elif strategy == "avoid_wolves":
            self.move_avoid_wolves()
        elif strategy == "flock":
            self.move_flock()
        else:
            raise ValueError(f"The sheep haven't been taught this strategy: {strategy}")
    
    def get_avoid_wolves_action(self):
        """
        Look in a Moore neighborhood of radius = sheep_sight_radius.
        If wolves are found, move in the opposite direction of the average wolf position.
        If no wolves are found, move randomly.
        """
        radius = self.model.wolf_sight_radius
        nearby_wolves = self.model.get_animals_in_neighborhood(self.x, self.y, radius, animal_type="wolves")

        if not nearby_wolves:
            return random.choice(list(action_to_move.keys()))

        # Compute average relative direction toward wolves
        dx_total = 0
        dy_total = 0

        for wolf in nearby_wolves:
            dx = self.model.wrap_delta(self.x, wolf.x, self.model.width)
            dy = self.model.wrap_delta(self.y, wolf.y, self.model.height)
            dx_total += dx
            dy_total += dy

        # Move opposite the wolves
        move_dx = int(-np.sign(dx_total))
        move_dy = int(-np.sign(dy_total))

        # If exact cancellation happens, fall back to random
        if move_dx == 0 and move_dy == 0:
            return random.choice(list(action_to_move.keys()))

        return move_to_action[(move_dx, move_dy)]
        
    def apply_action(self, action):

        dx, dy = action_to_move[action]
        self.x = (self.x + dx) % self.model.width
        self.y = (self.y + dy) % self.model.height
        self.heading = (math.degrees(math.atan2(dy, dx)) % 360) if not (dx == 0 and dy == 0) else self.heading
    
    def move_rl(self):
        obs = get_sheep_observation(self)
        if self.model.model_version != "rl-training":
            greedy = True
        else:
            greedy = False
        action, log_prob = choose_action(self.model.policy_net, obs, greedy=greedy)
        self.apply_action(action)

        if self.model.collect_log_probs:
            self.model.current_episode_log_probs.append(log_prob)
    
    def move_random(self):

        self.heading += random.randint(0, 49)
        self.heading -= random.randint(0, 49)

        radians = math.radians(self.heading)
        dx = round(math.cos(radians))
        dy = round(math.sin(radians))

        self.x = (self.x + dx) % self.model.width
        self.y = (self.y + dy) % self.model.height

    def move_avoid_wolves(self):

        action = self.get_avoid_wolves_action()
        self.apply_action(action)
    
    def move_flock(self):
        """Flocking strategy"""
        radius = self.model.sheep_sight_radius
        nearby_sheep = self.model.get_animals_in_neighborhood(self.x, self.y, radius, animal_type="sheep", exclude=self)

        if not nearby_sheep:
            self.move_random()
            return

        dx_total = 0
        dy_total = 0

        for sheep in nearby_sheep:
            dx = self.model.wrap_delta(self.x, sheep.x, self.model.width)
            dy = self.model.wrap_delta(self.y, sheep.y, self.model.height)
            dx_total += dx
            dy_total += dy

        move_dx = int(np.sign(dx_total))
        move_dy = int(np.sign(dy_total))

        # If direction cancels out, just stay or move randomly
        if move_dx == 0 and move_dy == 0:
            self.move_random()
            return

        new_x = (self.x + move_dx) % self.model.width
        new_y = (self.y + move_dy) % self.model.height

        # Prefer not to step onto a patch that already has sheep
        if not self.model.sheep_at(new_x, new_y):
            self.x = new_x
            self.y = new_y
            return

        # Fallback: look for an adjacent empty cell that is next to sheep
        candidates = []
        for dx, dy in self.model.moore_directions(include_stay=False):
            cx = (self.x + dx) % self.model.width
            cy = (self.y + dy) % self.model.height

            if self.model.sheep_at(cx, cy):
                continue

            neighbors = self.model.get_animals_in_neighborhood(cx, cy, 1, animal_type="sheep", exclude=self)
            if neighbors:
                candidates.append((cx, cy))

        if candidates:
            self.x, self.y = random.choice(candidates)
        else:
            self.move_random()

    def eat_grass(self):
        patch = self.model.get_patch(self.x, self.y)
        if patch.color == "green":
            patch.color = "brown"
            self.energy += self.model.sheep_gain_from_food

    def reproduce(self):
        if random.random() * 100 < self.model.sheep_reproduce:
            self.energy /= 2
            child = Sheep(self.model, self.x, self.y, self.energy, self.animal_type)
            child.heading = self.heading + random.uniform(0, 360)
            child.move()
            self.model.new_sheep.append(child)
            self.model.sheep_births += 1

    def step(self):
        self.move()

        if self.model.model_version in ["sheep-wolves-grass", "rl-training"]:
            self.energy -= 1
            self.eat_grass()
            self.death()
        if self.model.model_version == "sheep-wolves-grass":
            self.reproduce()