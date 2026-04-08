import random

class Animal:
    def __init__(self, model, x, y, energy):
        self.model = model
        self.x = x
        self.y = y
        self.energy = energy
        self.heading = random.uniform(0, 360)
        self.alive = True
        self.label = ""

    def die(self):
        self.alive = False

    def death(self):
        # energy None is used for RL training - deaths just come from predation
        if self.model.model_version in ["sheep-wolves-grass", "rl-training"] and self.energy < 0:
            self.die()