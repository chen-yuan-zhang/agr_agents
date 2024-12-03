from .evader import Evader
from multigrid.envs.evader_pursuer import EvaderPursuerEnv


env = EvaderPursuerEnv(size=16, render_mode='human')
env.reset()

observations, infos = env.reset()
observations = [{"grid": env.grid.state, "pos": agent.pos, "dir": agent.dir} 
                for agent in env.agents]

pursuer = env.agents[0]
evader = Evader(env.agents[1], env.goal)


while not env.is_done():  

   actions = {evader.agent.index: evader.compute_action(observations[1])}#, pursuer.index: None}
   observations, rewards, terminations, truncations, infos = env.step(actions)

   observations = [{"grid": env.grid.state, "pos": agent.pos, "dir": agent.dir} 
                   for agent in env.agents]