from .evader import Evader
from .pursuer import Pursuer
from multigrid.envs.evader_pursuer import EvaderPursuerEnv
from time import sleep

env = EvaderPursuerEnv(size=16, render_mode='human')


while True:
   env.reset()

   observations, infos = env.reset()
   observations = [{"grid": env.grid.state, "pos": agent.pos, "dir": agent.dir} 
                  for agent in env.agents]

   pursuer = Pursuer(env.agents[0])
   evader = Evader(env.agents[1], env.goal)
   

   while not env.is_done():  

      actions = {
         pursuer.agent.index: pursuer.compute_action(observations),
         evader.agent.index: evader.compute_action(observations)
      }
      observations, rewards, terminations, truncations, infos = env.step(actions)

      observations = [{"grid": env.grid.state, "pos": agent.pos, "dir": agent.dir} 
                     for agent in env.agents]
      
      if pursuer.agent.state.terminated or evader.agent.state.terminated or observations[0]["pos"] == observations[1]["pos"]:
         break
      
      sleep(0.3)