import gym
import numpy as np
import torch
from ddpg_brain import DDPGAgent
from ddpg_agent import run_episode_ddpg

gamma = 0.99   
critic_lr = 3e-4    
actor_lr = 1e-4     
batch_size = 128
buffer_size = 1000000
tau = 0.001
update_per_step = 1
seed = 5      
env = gym.make("BipedalWalker-v3")

Q_ddpg = DDPGAgent(env, gamma, tau, buffer_size, batch_size, critic_lr, actor_lr, update_per_step, seed)

Q_ddpg.actor.load_state_dict(torch.load('Trained Agent/checkpoint1.pth'))

# env = gym.wrappers.Monitor(env, "trained_agent/trained_results", force=True)
obs = env.reset()
cum_reward = 0
done = False
while not done:
  env.render()
  action = Q_ddpg.get_action(obs)
  obs, reward, done, info = env.step(action)
  cum_reward += reward
env.close()
print(f'cumulative reward = {cum_reward:.2f}')

#==================================================
n_runs = 10
ddpg_return = np.mean([run_episode_ddpg(Q_ddpg, env) for _ in range(n_runs)])
print(f'Average DDPG cumulative reward over {n_runs} runs = {ddpg_return:.2f}')