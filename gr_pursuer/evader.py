from .astar import astar2d

import numpy as np
from multigrid.core.constants import DIR_TO_VEC
from multigrid.core.actions import Action

class Evader:

    def __init__(self, agent, goal) -> None:

        self.agent = agent
        self.goal = goal
        self.path = None

    def compute_action(self, obs):

        grid = obs["grid"][:, :, 0]
        pos = list(obs["pos"])
        dir = np.array(obs["dir"])
        dir_vec = DIR_TO_VEC[dir]
        cost = (grid==2).astype(int)*1000

        if self.path is None or pos not in self.path:
            self.path = astar2d(pos, self.goal, cost)
            
        index = self.path.index(pos)
        self.path = self.path[index+1:]
        next_pos = np.array(self.path[0])
        dir_vec_ = next_pos - np.array(pos)

        if (dir_vec_==dir_vec).all():
            action = Action.forward
        else:
            n_dir = len(DIR_TO_VEC)
            dir_vec_ = DIR_TO_VEC[(dir+1)%n_dir]

            if (dir_vec_==dir_vec).all():
                action = Action.left
            else:
                action = Action.right

        return action

