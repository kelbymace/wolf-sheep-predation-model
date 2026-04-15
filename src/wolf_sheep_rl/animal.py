import random

class Animal:
    def __init__(self, model, x, y, energy, animal_type):
        self.model = model
        self.x = x
        self.y = y
        self.energy = energy
        self.heading = random.uniform(0, 360)
        self.alive = True
        self.label = ""
        self.animal_type = animal_type

    def die(self):
        self.alive = False

    def death(self):
        # energy None is used for RL training - deaths just come from predation
        if self.model.model_version in ["sheep-wolves-grass", "rl-training"] and self.energy < 0:
            self.die()
            # Track starvation deaths in sheep for policy evaluation
            if self.animal_type == "sheep":
                self.model.starvation_deaths += 1