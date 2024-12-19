from .agents.evader import Evader
from .agents.pursuer import Pursuer
from multigrid.envs.evader_pursuer import PursuerEnv
from time import sleep


env = PursuerEnv(size=16, agent_view_size=5, render_mode='human')
while True:
   env.reset()

   observations, infos = env.reset()
   # observations = [{"grid": env.grid.state, 
   #                  "observation": observations[i], 
   #                  "pos": agent.pos, "dir": agent.dir} 
   #                for i, agent in enumerate(env.agents)]

   pursuer = Pursuer(env.pursuer, env.goals)
   evader = Evader(env.evader, env.goal)

   while not env.is_done():  

      actions = {
         pursuer.agent.index: pursuer.compute_action(observations[0]),
         evader.agent.index: evader.compute_action(observations[1])
      }
      observations, rewards, terminations, truncations, infos = env.step(actions)

      # observations = [{"grid": env.grid.state, "pos": agent.pos, "dir": agent.dir} 
      #                for agent in env.agents]
      
      # print(pursuer.prob_dict)

      probs = pursuer.prob_dict
      probs = " ".join([f"{env.POS2COLOR[k]}: {v:.2f}   " for k, v in probs.items()])
      env.mission = probs

      # if pursuer.agent.state.terminated or evader.agent.state.terminated:
      #    break

      sleep(0.3)