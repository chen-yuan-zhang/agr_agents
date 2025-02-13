from .base import BaseAgent
from ..astar import astar2d
import matplotlib.pyplot as plt

import math
import numpy as np
from multigrid.core.constants import DIR_TO_VEC
from multigrid.core.actions import Action
from multigrid.utils.obs import gen_obs_grid_encoding
from multigrid.core.constants import Type
import random
import os

# MODES
TRACK = 0
MOVE2GOAL = 1
BETA = 1

class Observer(BaseAgent):

    def __init__(self, env):

        super().__init__(env.observer)

        self.agent.name = "Observer"
        self.goals = env.goals
        self.goal_costs = None
        self.start = None
        self.prob_dict = None
        self.agent.can_overlap = True

        self.step = -1
        self.target_observations = []

        self.infer_goal = None
        self.mode = TRACK

        self.agent.reported_goal = None

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
    
    def compute_target_paths(self, target_pos, cost):
        paths = []
        costs = []

        for goal in self.goals:
            path = astar2d(target_pos, goal, cost)
            costs.append(len(path))

        return paths, costs

    def compute_action(self, obs):

        self.step += 1
        grid = obs["grid"][:, :, 0]
        pos = list(obs["pos"])
        dir = np.array(obs["dir"])
        dir_vec = DIR_TO_VEC[dir]

        # cost = (grid==-1).astype(int)*1000
        cost = np.zeros((grid.shape))

        if self.start is None:
            self.start = pos

        # STACK OBSERVATIONS
        if "target_pos" in obs:
            target_pos = obs["target_pos"]
            target_dir = obs["target_dir"]
            self.infer_goal, self.prob_dict = self.compute_gr(target_pos, cost)
            target_paths, target_costs = self.compute_target_paths(target_pos, cost)

            self.target_observations.append((self.step, target_pos, target_dir, target_paths, target_costs))

            if len(self.target_observations) > 3 and max((self.prob_dict).values())>0.8:
                max_idx = np.argmax((self.prob_dict).values())
                self.agent.reported_goal = self.goals[max_idx]
                
                return Action.done


        path = None
        dir_vec_ = None
        # EXE MODE BEHAVIOUR
        if self.mode == TRACK:
            if len(self.target_observations) > 0:
                last_target_obs = self.target_observations[-1]
                step, target_pos, target_dir, target_paths, target_costs = last_target_obs
                dir_vec_ = DIR_TO_VEC[target_dir]
                path = astar2d(pos, target_pos, cost)
            else:
                print("Target: Lost Track")
                return Action.right

        elif self.mode == MOVE2GOAL:
            path = astar2d(pos, self.infer_goal, cost)

        
        if len(path)<=1 and dir_vec_ is not None:
            n_dir = len(DIR_TO_VEC)
            dir_vec_curr = DIR_TO_VEC[(dir+1)%n_dir]

            if (dir_vec_==dir_vec_curr).all():
                action = Action.right
            else:
                action = Action.left

        elif len(path)<2 or path is None:
            print("Target: Path not processed")
            return Action.right


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

class BeliefUpdateObserver(BaseAgent):
    def __init__(self, env, init_actor_belief = None, init_goal_belief = None):
        super().__init__(env.observer)

        self.env = env
        self.agent.name = "BeliefUpdateObserver"
        self.agent.can_overlap = True

        self.goals = env.goals
        self.step = -1

        if init_actor_belief:
            self.actor_belief = init_actor_belief
        else:
            self.actor_belief = set_uniform_prob(env.base_grid)
            print(self.actor_belief)
            
        if init_goal_belief:
            self.goal_belief = init_goal_belief
        else:
            self.goal_belief = { g:1/len(self.goals) for g in self.goals}

        
    def compute_action(self, obs):
        self.update_belief(obs) # update the belief based on observation, not consider transition dynamics yet

        self.step += 1

        # assume goal directed behavior

        new_actor_belief = np.zeros_like(self.actor_belief)
        for goal in self.goals:
            goal_belief = self.goal_belief[goal]
            for cell in np.argwhere(self.actor_belief > 0):
                pos, dir = cell[:2], cell[2]
                prob = self.actor_belief[tuple(cell)]
                successors = get_successor(self.env, pos, dir)

                tran_probs = {}
                for succ in successors:
                    pos, dir = succ
                    path = astar2d(succ, goal, self.env)
                    if path:
                        tran_probs[(pos[0], pos[1], dir)] = math.exp( - BETA * (1 + len(path)))
                    else:
                        tran_probs[(pos[0], pos[1], dir)] = 0
                
                total_prob = sum(tran_probs.values())
                if total_prob > 0:
                    for succ in tran_probs:
                        tran_probs[succ] /= total_prob


                for succ in successors:
                    new_actor_belief[(succ[0][0],succ[0][1],succ[1])] += goal_belief*prob*tran_probs[(succ[0][0],succ[0][1],succ[1])]

        self.actor_belief = new_actor_belief
        self.render_and_save(f'belief_update_2/actor_belief_step_{self.step}.png', obs)

        # select action
        return random.choice([Action.right, Action.left, Action.forward])

    def render_and_save(self, filename, obs):
        """
        Render the environment and save the visualization.
        
        Parameters:
        filename (str): The name of the file to save the visualization.
        obs (dict): The observation dictionary containing the observer and goal positions.
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        belief_sum = np.sum(self.actor_belief, axis=2)
        
        plt.imshow(belief_sum, cmap='coolwarm', interpolation='nearest', vmin=0, vmax=0.1)
        plt.colorbar()
        plt.title('Actor Belief Heatmap')

        # Overlay obstacles
        obstacles = np.where(self.env.base_grid != 0)
        plt.scatter(obstacles[1], obstacles[0], c='black', marker='s', label='Obstacle')

        # Overlay observer position
        observer_pos = obs["pos"]
        plt.scatter(observer_pos[1], observer_pos[0], c='blue', marker='o', label='Observer')

        # Overlay goal positions
        for goal in self.goals:
            plt.scatter(goal[1], goal[0], c='green', marker='*', label='Goal')

        # Overlay target position if observed
        if "target_pos" in obs:
            target_pos = obs["target_pos"]
            plt.scatter(target_pos[1], target_pos[0], c='red', marker='x', label='Target')

        plt.savefig(filename)
        plt.close()

    def update_belief(self, obs):
        """
        Update the belief of the observer based on the observed FoV.
        
        Parameters:
        FoV (np.array): The field of view of the observer.
        pos (tuple): The position of the actor or None.
        """
        # Update the belief of the observer based on the observed FoV
        if "target_pos" in obs:
            target_pos = obs["target_pos"]
            target_dir = obs["target_dir"] # 0-4 denote east south west north respectively
            self.actor_belief = np.zeros_like(self.actor_belief)
            self.actor_belief[tuple(target_pos)][target_dir] = 1.0

        else:

            obs_shape = self.agent.observation_space['image'].shape[:-1]
            vis_mask = np.zeros_like(obs_shape, dtype=bool)
            vis_mask = (self.env.gen_obs()[0]['image'][..., 0] !=  Type.unseen.to_index()) # 0 denotes the observer
  

            highlight_mask = np.zeros((self.env.width, self.env.height), dtype=bool)


            # of the agent's view area
            f_vec = self.agent.state.dir.to_vec()
            r_vec = np.array((-f_vec[1], f_vec[0]))
            top_left = (
                self.agent.state.pos
                + f_vec * (self.agent.view_size - 1)
                - r_vec * (self.agent.view_size // 2)
            )

            # For each cell in the visibility mask
            for vis_j in range(0, self.agent.view_size):
                for vis_i in range(0, self.agent.view_size):
                    # If this cell is not visible, don't highlight it
                    if not vis_mask[vis_i, vis_j]:
                        continue

                    # Compute the world coordinates of this cell
                    abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                    if abs_i < 0 or abs_i >= self.env.width:
                        continue
                    if abs_j < 0 or abs_j >= self.env.height:
                        continue

                    # Mark this cell to be highlighted
                    highlight_mask[abs_i, abs_j] = True
            
    
            add_probs = 0
            for cell in np.argwhere(highlight_mask == True):
                add_probs += sum(self.actor_belief[tuple(cell)])
                self.actor_belief[tuple(cell)] = 0

            for cell in np.argwhere((highlight_mask == False) & (self.env.base_grid == 0)):
                self.actor_belief[tuple(cell)] *= 1/(1-add_probs)

def set_uniform_prob(grid, dir=4):
    """
    Set a uniform probability for all free cells in the grid.
    
    Parameters:
    grid (np.array): The grid to be analyzed.
    
    Returns:
    np.array: A grid with uniform probabilities for all free cells.
    """
    free_cells = np.argwhere(grid == 0)
    num_free_cells = len(free_cells)
    uniform_prob = 1.0 / (num_free_cells * dir) if num_free_cells > 0 else 0

    prob_grid = np.zeros((*grid.shape, dir), dtype=float)
    for cell in free_cells:
        prob_grid[tuple(cell)] = uniform_prob

    return prob_grid


def get_successor(env, pos, dir):
    """
    Generate the next position and direction given the current position and direction.
    
    Parameters:
    env (object): The environment object containing the grid and other relevant information.
    pos (tuple): The current position (x, y).
    dir (int): The current direction (0: east, 1: south, 2: west, 3: north).
    
    Returns:
    list: A list of tuples representing the next position and direction.
    """
    successors = []
    x, y = pos

    # Define the direction vectors for east, south, west, and north
    DIR_TO_VEC_tmp = {
        0: (1, 0),  # east
        1: (0, 1),  # south
        2: (-1, 0), # west
        3: (0, -1)  # north
    }

    # Define the possible actions: forward, turn right, turn left
    actions = [
        (0, 0),  # forward
        (1, 1),  # turn right
        (2, -1)  # turn left
    ]

    for action, turn in actions:
        if action == 0:  # forward
            dx, dy = DIR_TO_VEC_tmp[dir]
            new_pos = (x + dx, y + dy)
            new_dir = dir
        else:  # turn right or left
            new_pos = pos
            new_dir = (dir + turn) % 4

        # Check if the new position is within the grid bounds and not an obstacle
        if 0 <= new_pos[0] < env.width and 0 <= new_pos[1] < env.height and env.base_grid[new_pos[0], new_pos[1]] == 0:
            successors.append((new_pos, new_dir))

    return successors