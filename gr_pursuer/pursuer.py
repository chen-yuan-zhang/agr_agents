from .astar import astar2d

import numpy as np
from multigrid.core.constants import DIR_TO_VEC
from multigrid.core.actions import Action

class Pursuer:

    def __init__(self, agent, goals):
        self.agent = agent
        self.goals = goals
        self.start = None

    def compute_action(self, observations):

        obs = observations[0]
        grid = obs["grid"][:, :, 0]
        pos = list(obs["pos"])
        dir = np.array(obs["dir"])
        dir_vec = DIR_TO_VEC[dir]
        cost = (grid==2).astype(int)*1000

        if self.start is None:
            self.start = pos

        if tuple(pos) == observations[1]["pos"]:
            self.agent.state.terminated == True
            return Action.left


        current_pos = observations[1]["pos"]

        current_dis = len(astar2d(self.start, current_pos, cost)) - 1

        probs = []
        for goal in self.goals:
            opt_cost = len(astar2d(self.start, goal, cost)) - 1
            real_cost = current_dis + len(astar2d(current_pos, goal, cost)) - 1
            prob = np.exp(-(real_cost - opt_cost))/(1+np.exp(-(real_cost - opt_cost)))
            probs.append(prob)


        total = sum(probs)
        normalized_probs = [p / total for p in probs]

        largest_index = np.argmax(normalized_probs)
        infer_goal = self.goals[largest_index]
        print(infer_goal)
        

        path = astar2d(pos, infer_goal, cost)

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

        prob_dict = { g:p for g, p in zip(self.goals,normalized_probs)}
        print(prob_dict)
        return action, prob_dict

