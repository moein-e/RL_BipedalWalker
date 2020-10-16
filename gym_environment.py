import gym

env = gym.make("BipedalWalker-v3")
observation = env.reset()
for _ in range(500):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)

  if done:
    observation = env.reset()
env.close()

# env = gym.make("BipedalWalker-v3")
# observation = env.reset()
# actions = []
# return_ = 0
# for _ in range(500):
#   env.render()
#   action = env.action_space.sample()
#   observation, reward, done, info = env.step(action)
#   actions.append(action)
#   return_+=reward
#   if done:
#     observation = env.reset()
# env.close()