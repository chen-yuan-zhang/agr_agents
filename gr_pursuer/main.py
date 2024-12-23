from .agents.target import Target
from .agents.pursuer import Pursuer

import numpy as np
import pandas as pd
from time import sleep
from multigrid.envs.pursuer import PursuerEnv


scenarios = pd.read_csv("scenarios.csv")

for idx, scenario in scenarios.iterrows():

   base_grid = np.array(eval(scenario["base_grid"]))
   goals = eval(scenario["goals"])
   env = PursuerEnv(size=32, agent_view_size=5, base_grid=base_grid, 
                    goals=goals, render_mode='human')
   env.reset()

   observations, infos = env.reset()

   pursuer = Pursuer(env.observer, env.goals)
   target = Target(env.target, env.goal)

   while not env.is_done():  

      actions = {
         pursuer.agent.index: pursuer.compute_action(observations[0]),
         target.agent.index: target.compute_action(observations[1])
      }
      observations, rewards, terminations, truncations, infos = env.step(actions)

      probs = pursuer.prob_dict
      probs = " ".join([f"{env.POS2COLOR[k]}: {v:.2f}   " for k, v in probs.items()])
      env.mission = probs

      sleep(0.3)