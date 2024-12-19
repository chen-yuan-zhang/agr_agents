from .base import BaseAgent
from ..astar import astar2d

import numpy as np
from multigrid.core.constants import DIR_TO_VEC
from multigrid.core.actions import Action

# MODES
TRACK = 0
MOVE2GOAL = 1

class Pursuer(BaseAgent):

    def __init__(self, agent, goals):

        super().__init__(agent)

        self.goals = goals
        self.start = None
        self.prob_dict = None
        self.agent.can_overlap = True

        self.step = -1
        self.evader_observations = []

        self.infer_goal = None
        self.mode = TRACK

    def compute_gr(self, evader_pos, cost):
        current_dis = len(astar2d(self.start, evader_pos, cost)) - 1

        probs = []
        for goal in self.goals:
            opt_cost = len(astar2d(self.start, goal, cost)) - 1
            real_cost = current_dis + len(astar2d(evader_pos, goal, cost)) - 1
            prob = np.exp(-(real_cost - opt_cost))/(1+np.exp(-(real_cost - opt_cost)))
            probs.append(prob)

        total = sum(probs)
        normalized_probs = [p / total for p in probs]

        largest_index = np.argmax(normalized_probs)
        infer_goal = self.goals[largest_index]

        probs = { g:p for g, p in zip(self.goals, normalized_probs)}

        return infer_goal, probs

    def compute_action(self, obs):

        self.step += 1
        grid = obs["grid"][:, :, 0]
        pos = list(obs["pos"])
        dir = np.array(obs["dir"])
        dir_vec = DIR_TO_VEC[dir]
        cost = (grid==-1).astype(int)*1000

        if self.start is None:
            self.start = pos

        # STACK OBSERVARIONS
        if "evader_pos" in obs:
            evader_pos = obs["evader_pos"]
            evader_dir = obs["evader_dir"]
            self.infer_goal, self.prob_dict = self.compute_gr(evader_pos, cost)

            self.evader_observations.append((self.step, evader_pos, evader_dir))
            # print(self.evader_observations)

            if len(self.evader_observations) > 3 and max((self.prob_dict).values())>0.8:
                self.mode = MOVE2GOAL

        path = None
        dir_vec_ = None
        # EXE MODE BEHAVIOUR
        if self.mode == TRACK:
            last_evader_obs = self.evader_observations[-1]
            step, evader_pos, evader_dir = last_evader_obs
            dir_vec_ = DIR_TO_VEC[evader_dir]
            path = astar2d(pos, evader_pos, cost)

        elif self.mode == MOVE2GOAL:
            path = astar2d(pos, self.infer_goal, cost)

        if len(path)<2 or path is None:
            return Action.right

        elif len(path)<2 and dir_vec_ is not None:
            n_dir = len(DIR_TO_VEC)
            dir_vec_curr = DIR_TO_VEC[(dir+1)%n_dir]

            if (dir_vec_==dir_vec_curr).all():
                action = Action.right
            else:
                action = Action.left

        else:            

            next_pos = np.array(path[1])     
            dir_vec_ = next_pos - np.array(pos)

            if (dir_vec_==dir_vec).all():
                action = Action.forward
            else:
                n_dir = len(DIR_TO_VEC)
                dir_vec_curr = DIR_TO_VEC[(dir+1)%n_dir]

                if (dir_vec_==dir_vec_curr).all():
                    action = Action.right
                else:
                    action = Action.left

 
        
        return action

