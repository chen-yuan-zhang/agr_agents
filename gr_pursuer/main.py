# import gymnasium as gym
# import multigrid.envs

# env = gym.make('MultiGrid-Empty-5x5-v0', agents=2, render_mode='human')



# env.close()

from multigrid.envs.evader_pursuer import EvaderPursuerEnv


env = EvaderPursuerEnv(size=16, render_mode='human')
env.reset()

observations, infos = env.reset()
print(env.agents)

pursuer = env.agents[0]
evader = env.agents[1]
while not env.is_done():
   # this is where you would insert your policy / policies
#    actions = {agent.index: agent.action_space.sample() for agent in env.agents}
   
   actions = {evader.index: evader.action_space.sample()}#, pursuer.index: None}
   observations, rewards, terminations, truncations, infos = env.step(actions)