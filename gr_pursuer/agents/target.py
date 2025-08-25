from .base import BaseAgent
from ..astar import astar

import random
import numpy as np
import matplotlib.pyplot as plt


class AstarTarget(BaseAgent):

    def __init__(self, env) -> None:

        super().__init__(env.target)
        self.env = env
        self.goal = env.goal
        self.goals = env.goals

        self.path = None

        self.index = 0
        self.enable_hidden_cost = env.enable_hidden_cost
        if self.enable_hidden_cost:
            self.hidden_cost = env.hidden_cost
        else:
            self.hidden_cost = np.ones((env.grid_size, env.grid_size), dtype=np.float32)

    def compute_action(self, obs):
        pos = self.agent.state.pos
        dir = self.agent.state.dir


        if self.path is None or self.index >= len(self.path) - 1:
            
            self.path = astar((pos, dir), self.goal, self.env, self.hidden_cost)
            self.index = 0


        self.index += 1
        return self.path[self.index][0]

