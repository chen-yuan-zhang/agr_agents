from .base import BaseAgent
from ..astar import astar2d

import random
import numpy as np
from multigrid.core.constants import DIR_TO_VEC
from multigrid.core.actions import Action

# MODES
EVADE = 0
MOVE2GOAL = 1

class Evader(BaseAgent):

    def __init__(self, agent, goal) -> None:

        super().__init__(agent)
    
        self.goal = goal
        self.evade_goal = None
        self.path = None

        self.mode = EVADE

    def compute_action(self, obs):

        grid = obs["grid"][:, :, 0]
        pos = list(obs["pos"])
        dir = np.array(obs["dir"])
        dir_vec = DIR_TO_VEC[dir]
        cost = (grid==2).astype(int)*1000

        if self.path is not None and pos in self.path:
            index = self.path.index(pos)
            if index == len(self.path)-1:
                self.path = None

                if self.mode == EVADE:
                    self.mode = MOVE2GOAL


        if self.mode == EVADE:
            # Choose a random position from the grid and move to that position
            if self.evade_goal is None:
                self.evade_goal = random.choice(np.argwhere(grid==2))
                self.path = astar2d(pos, self.evade_goal, cost)

            if len(self.path)==1:
                self.mode = MOVE2GOAL
                self.evade_goal = None
                self.path = None

        if self.mode == MOVE2GOAL:
            if self.path is None or pos not in self.path:
                self.path = astar2d(pos, self.goal, cost)
                
        if pos in self.path:
            index = self.path.index(pos)
        else:
            print(f"Error: pos {pos} not in path {self.path}")
        # self.path = self.path[index+1:]

        if not self.path:
            return Action.right

        next_pos = np.array(self.path[index+1])
        dir_vec_ = next_pos - np.array(pos)

        if (dir_vec_==dir_vec).all():
            action = Action.forward
            self.path = self.path[index+1:]
        else:
            n_dir = len(DIR_TO_VEC)
            dir_vec_opt = DIR_TO_VEC[(dir+1)%n_dir]

            if (dir_vec_==dir_vec_opt).all():
                action = Action.right
            else:
                action = Action.left

        if (next_pos == self.goal).all():
            self.agent.state.terminated = True

        return action

