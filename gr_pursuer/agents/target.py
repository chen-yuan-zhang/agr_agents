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

        if env.hidden_cost is not None:
            self.hidden_cost = env.hidden_cost

        self.mode = MOVE2GOAL


    def compute_action(self, obs):

        grid = obs["grid"][:, :, 0]
        pos = list(obs["pos"])
        dir = np.array(obs["dir"])
        dir_vec = DIR_TO_VEC[dir]
            
        cost = self.hidden_cost if hasattr(self, "hidden_cost") else None

        if self.path is not None and pos + [int(dir)] in self.path:
            index = self.path.index(pos + [int(dir)])
            if index == len(self.path)-1:
                self.path = None

                if self.mode == EVADE:
                    self.mode = MOVE2GOAL

        if self.mode == EVADE:
            # Choose a random position from the grid and move to that position
            if self.evasion_goal is None:
                self.evasion_goal = random.choice(np.argwhere(grid!=2))
                self.path = astar2d(pos, self.evasion_goal, env, cost)

            if self.path is None or len(self.path)<=1:
                self.mode = MOVE2GOAL
                self.evasion_goal = None
                self.path = None

        if self.mode == MOVE2GOAL:
            if self.path is None or pos not in self.path:
                self.path = astar2d((pos, dir), self.goal, self.env, cost)  
                index = 0  


                
        if pos + [int(dir)] in self.path:
            index = self.path.index(pos + [int(dir)])
        else:
            print(f"Error: pos {pos} not in path {self.path}")

        if len(self.path) <= index+1:
            print("Target: Path not processed")
            return None

        next_pos = self.path[index+1][:2]
        next_dir = self.path[index+1][2]

        if next_pos!=pos:
            action = Action.forward
        else:
            if next_dir == (dir+1)%4:
                action = Action.right
            else:
                action = Action.left

        return action

