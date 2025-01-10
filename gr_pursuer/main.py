from .agents.target import Target
from .agents.pursuer import Pursuer

import argparse
import numpy as np
import pandas as pd
from time import sleep
from multigrid.envs.goal_prediction import GREnv


def run(base_grid=None, goals=None, hidden_cost=None):
   env = GREnv(size=32, agent_view_size=5, base_grid=base_grid, 
                     goals=goals, render_mode='human')
   observations, infos = env.reset()

   # Green
   pursuer = Pursuer(env.observer, env.goals)
   # Red
   enable_hidden_cost = hidden_cost is not None
   target = Target(env, hidden_cost, enable_hidden_cost=enable_hidden_cost)

   while not env.is_done():  

      actions = {
         pursuer.agent.index: pursuer.compute_action(observations[0]),
         target.agent.index: target.compute_action(observations[1])
      }
      # print(actions,  pursuer.agent.index, target.agent.index)
      observations, rewards, terminations, truncations, infos = env.step(actions)

      probs = pursuer.prob_dict
      if probs is not None:
         probs = " ".join([f"{env.POS2COLOR[k]}: {v:.2f}   " for k, v in probs.items()])
      else:
         probs = " None "
         sleep(10)

      env.mission = probs

      sleep(0.3)


def main(dataset=None):

   if dataset is not None:
      scenarios = pd.read_csv(dataset)

      for idx, scenario in scenarios.iterrows():

         base_grid = np.array(eval(scenario["base_grid"]))
         goals = eval(scenario["goals"])
         hidden_cost = np.array(eval(scenario["hidden_cost"]))

         run(base_grid=base_grid, goals=goals, hidden_cost=hidden_cost)
   else:

      while True:
         run()
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main file for running the scenario")
    parser.add_argument("--dataset", type=str, default=None, help="csv file to run as dataset")

    args = parser.parse_args()
    main(args.dataset)