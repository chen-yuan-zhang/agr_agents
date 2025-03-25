from .base import BaseAgent
from ..astar import astar2d

import random
import numpy as np
from multigrid.core.constants import DIR_TO_VEC
from multigrid.core.actions import Action

# MODES
EVADE = 0
MOVE2GOAL = 1

class Target(BaseAgent):

    def __init__(self, env) -> None:

        super().__init__(env.target)
        self.env = env
        self.agent.name = "Target"
        self.goal = env.goal
        self.goals = env.goals
        self.evasion_goal = None
        self.path = None
        self.agent.can_overlap = False
        self.index = 0

        self.hidden_cost = env.hidden_cost

        self.mode = MOVE2GOAL


    def compute_action(self, obs):
        pos = tuple(obs["pos"])
        dir = obs["dir"]

        # if self.path is not None and pos + [int(dir)] in self.path:
        #     index = self.path.index(pos + [int(dir)])
        #     if index == len(self.path)-1:
        #         self.path = None

        #         if self.mode == EVADE:
        #             self.mode = MOVE2GOAL

        # if self.mode == EVADE:
        #     # Choose a random position from the grid and move to that position
        #     if self.evasion_goal is None:
        #         self.evasion_goal = random.choice(np.argwhere(grid!=2))
        #         self.path = astar2d(pos, self.evasion_goal, env, cost)

        #     if self.path is None or len(self.path)<=1:
        #         self.mode = MOVE2GOAL
        #         self.evasion_goal = None
        #         self.path = None

        # if self.mode == MOVE2GOAL:
        #     if self.path is None or pos not in self.path:
        #         self.path = astar2d((pos, dir), self.goal, self.env, cost)  
        #         index = 0  



        if self.path is None:
            self.path = astar2d((pos, dir), self.goal, self.env, self.hidden_cost)


        self.index += 1
        return self.path[self.index][0]

