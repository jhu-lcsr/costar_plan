import numpy as np
import matplotlib.pyplot as plt

class OUProcess(object):

    # mu is the desired mean of the noise
    # theta controls how aggressively the noise tries to revert to the mean
    # sigma defines the possible delta of the noise at each time step.  In conjuction with theta it defines the range of the noise.
    def __init__(self, mu, theta, sigma):

        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.x = 0.0
        self.reset()

    def reset(self):
        self.x = self.mu
        return self.x

    def step(self):
        self.x = self.x + self.theta * (self.mu - self.x) + self.sigma * np.random.randn(1)
        return self.x



if __name__ == "__main__":
    ou = OUProcess(0.0, 0.6, 0.05)
    history = []
    for i in range(150):
        history.append(ou.step())

    plt.figure(7)
    plt.plot(history)
    plt.show()