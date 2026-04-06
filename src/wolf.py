import random
import math
from .animal import Animal

class Wolf(Animal):

    def move(self):
        strategy = self.model.wolf_strategy

        if strategy == "random":
            self.move_random()
        elif strategy == "seek_sheep":
            self.move_seek_sheep()
        else:
            raise ValueError(f"Unknown wolf strategy: {strategy}")

    def move_random(self):
        self.heading += random.randint(0, 49)
        self.heading -= random.randint(0, 49)

        radians = math.radians(self.heading)
        dx = round(math.cos(radians))
        dy = round(math.sin(radians))

        self.x = (self.x + dx) % self.model.width
        self.y = (self.y + dy) % self.model.height
    
    def move_seek_sheep(self):
        """
        Look in a Moore neighborhood of radius = sheep_detection_radius.
        If sheep are found, move toward the average sheep position.
        If no sheep are found, move randomly.
        """
        radius = self.model.wolf_sight_radius
        # nearby_sheep = self.model.get_sheep_in_neighborhood(self.x, self.y, radius)
        nearby_sheep = self.model.get_animals_in_neighborhood(self.x, self.y, radius, animal_type="sheep")

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

        if move_dx == 0 and move_dy == 0:
            self.move_random()
            return

        self.x = (self.x + move_dx) % self.model.width
        self.y = (self.y + move_dy) % self.model.height

    def eat_sheep(self):
        sheep_here = [
            s for s in self.model.sheep
            if s.alive and s.x == self.x and s.y == self.y
        ]
        if sheep_here:
            prey = random.choice(sheep_here)
            prey.die()
            if self.model.model_version == "sheep-wolves-grass":
                self.energy += self.model.wolf_gain_from_food

    def reproduce(self):
        if random.random() * 100 < self.model.wolf_reproduce:
            self.energy /= 2
            child = Wolf(self.model, self.x, self.y, self.energy)
            child.heading = self.heading + random.uniform(0, 360)
            child.move()
            self.model.new_wolves.append(child)

    def step(self):
        self.move()
        if self.model.model_version == "sheep-wolves-grass":
            self.energy -= 1
            self.eat_sheep()
            self.death()
            if self.alive:
                self.reproduce()
        else:
            self.eat_sheep()