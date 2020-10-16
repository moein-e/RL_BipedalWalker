import numpy as np

class GaussianNoise:
    def __init__(self, num_actions, mean=0.0):
        self.name = 'GaussianNoise'
        self.mean = mean
        self.size = num_actions
        
    def sample(self, std_dev=2.):
        x = np.random.normal(self.mean, std_dev, self.size)
        return x
    

class OUNoise:   # originally taken from: https://keras.io/examples/rl/ddpg_pendulum/
    def __init__(self, num_actions, mean=0.0, theta=1.5, dt=1e-2, x_initial=None):
        self.name = 'OUNoise'
        self.mean = mean * np.ones(num_actions)
        self.num_actions = num_actions
        self.theta = theta
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def sample(self, std_dev):
        std_dev = std_dev * np.ones(self.num_actions)
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)
 