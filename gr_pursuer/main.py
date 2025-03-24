from .agents.target import Target
from .agents.observer import Observer

import argparse
import numpy as np
import pandas as pd
from time import sleep
from .astar import astar2d
from multigrid.envs.goal_prediction import GREnv

def get_data(env, step, action):
   base_grid = env.base_grid
   goals = env.goals
   target_goal = env.goal

   cost = []
   for goal in goals:
         cost.append(len(astar2d(env.observer.pos, goal, base_grid)) - 1)

   data = {
            "step": step,
            "observer_pos": env.observer.pos, "target_pos": env.target.pos, 
            "observer_dir": env.observer.dir, "target_dir": env.target.dir,
            "action": action, 
            "base_grid": base_grid.tolist()}
   
   return data


def run(base_grid=None, goals=None, hidden_cost=None, hidden_cost_type=None):
   env = GREnv(size=32, base_grid=base_grid, 
               see_through_walls=[False, True],
               agent_view_size=[5, 3],
               goals=goals, hidden_cost=hidden_cost, 
               hidden_cost_type=hidden_cost_type, 
               render_mode='human'
   )
   observations, infos = env.reset()

   # Green
   observer = Observer(env)
   # Red
   target = Target(env)
   agents = [observer, target]

   data = []

   step = 1
   while not env.is_done():  
      actions = {}
      for i, agent in enumerate(agents):
         action = agent.compute_action(observations[i])
         if action is not None:
            actions[agent.agent.index] = action

      data.append(get_data(env, step, actions.get(1, None)))

      observations, rewards, terminations, truncations, infos = env.step(actions)

      probs = observer.prob_dict
      if probs is not None:
         probs = " ".join([f"{env.POS2COLOR[k]}: {v:.2f}   " for k, v in probs.items()])
      else:
         probs = " None "

      env.mission = probs
      step+=1

      sleep(0.1)

   return data


def main(scenarios_file=None, hidden_cost_type=None):


   dataset = pd.DataFrame()
   if scenarios_file is not None:

      scenarios = pd.read_csv(scenarios_file)

      for idx, scenario in scenarios.iterrows():

         print(f"Scenario {idx}")
         base_grid = np.array(eval(scenario["base_grid"]))
         goals = eval(scenario["goals"])
         hidden_cost = np.array(eval(scenario["hidden_cost"]))

         data = run(base_grid=base_grid, goals=goals, hidden_cost=hidden_cost)
         data = pd.DataFrame(data)
         data["scenario"] = scenario["scenario"]
         data["layout"] = scenario["layout"]

         dataset = pd.concat([dataset, data], ignore_index=True)

      # Get the name of the file
      name = scenarios_file.split("/")[-1]
      params = "_".join(name.split("_")[1:])
      file_name = f"gr_pursuer/data/dataset_{params}"
      dataset.to_csv(file_name, index=False)
      
   else:

      while True:
         run(hidden_cost_type=hidden_cost_type)
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main file for running the scenario")
    parser.add_argument("--scenarios", type=str, default=None, help="csv file to run as scenarios")
    parser.add_argument("--hidden_cost_type", type=str, default=None, help="csv file to run as scenarios")

    args = parser.parse_args()
    main(args.scenarios, args.hidden_cost_type)