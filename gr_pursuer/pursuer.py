from .astar import astar2d

import numpy as np
from multigrid.core.constants import DIR_TO_VEC
from multigrid.core.actions import Action

class Pursuer:

    def __init__(self, agent):
        self.agent = agent

    def compute_action(self, observations):

        obs = observations[0]
        grid = obs["grid"][:, :, 0]
        pos = list(obs["pos"])
        dir = np.array(obs["dir"])
        dir_vec = DIR_TO_VEC[dir]
        cost = (grid==2).astype(int)*1000

        if tuple(pos) == observations[1]["pos"]:
            self.agent.state.terminated == True
            return Action.left

        path = astar2d(pos, observations[1]["pos"], cost)
        next_pos = np.array(path[1])     
        dir_vec_ = next_pos - np.array(pos)

        if (dir_vec_==dir_vec).all():
            action = Action.forward
        else:
            n_dir = len(DIR_TO_VEC)
            dir_vec_ = DIR_TO_VEC[(dir+1)%n_dir]

            if (dir_vec_==dir_vec).all():
                action = Action.right
            else:
                action = Action.left

        return action

