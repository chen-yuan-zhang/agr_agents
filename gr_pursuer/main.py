from .agents.target import Target
from .agents.observer import Observer

import argparse
import numpy as np
import pandas as pd
from time import sleep
from .astar import astar2d
from multigrid.envs.goal_prediction import GREnv

def get_data(env):
   base_grid = env.base_grid
   goals = env.goals
   target_goal = env.goal

   cost = []
   for goal in goals:
         cost.append(len(astar2d(env.observer.pos, goal, base_grid)) - 1)

   data = {"observer_pos": env.observer.pos, "target_pos": env.target.pos, 
            "observer_dir": env.observer.dir, "target_dir": env.target.dir,
            "goals": goals, "target_goal": target_goal, "cost": cost, "base_grid": base_grid.tolist(),
            "hidden_cost": env.hidden_cost.tolist()}
   return data


def run(base_grid=None, goals=None, hidden_cost=None):
   env = GREnv(size=32, base_grid=base_grid, 
               see_through_walls=[False, True],
               agent_view_size=[5, 3],
                     goals=goals, hidden_cost=hidden_cost, 
                     enable_hidden_cost=hidden_cost is not None, 
                     render_mode='human')
   observations, infos = env.reset()

   # Green
   observer = Observer(env)
   # Red
   target = Target(env)
   agents = [observer, target]

   data = [get_data(env)]

   while not env.is_done():  
      step = {}
      actions = {}
      for i, agent in enumerate(agents):
         action = agent.compute_action(observations[i])
         if action is not None:
            actions[agent.agent.index] = action

      observations, rewards, terminations, truncations, infos = env.step(actions)

      probs = observer.prob_dict
      if probs is not None:
         probs = " ".join([f"{env.POS2COLOR[k]}: {v:.2f}   " for k, v in probs.items()])
      else:
         probs = " None "

      env.mission = probs

      data.append(get_data(env))

      sleep(0.3)

   return data


def main(scenarios=None):

   if scenarios is not None:
      scenarios = pd.read_csv(scenarios)

      for idx, scenario in scenarios.iterrows():

         print(f"Scenario {idx}")
         base_grid = np.array(eval(scenario["base_grid"]))
         goals = eval(scenario["goals"])
         hidden_cost = np.array(eval(scenario["hidden_cost"]))

         run(base_grid=base_grid, goals=goals, hidden_cost=hidden_cost)
   else:

      while True:
         run()
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main file for running the scenario")
    parser.add_argument("--scenarios", type=str, default=None, help="csv file to run as scenarios")

    args = parser.parse_args()
    main(args.scenarios)