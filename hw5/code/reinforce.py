import torch
from torch import optim
from torch import nn
import gym
import numpy as np
import wandb

gym.logger.set_level(40)


class BanditEnv(gym.Env):
    '''
    Toy env to test your implementation
    The state is fixed (bandit setup)
    Action space: gym.spaces.Discrete(10)
    Note that the action takes integer values
    '''

    def __init__(self):
        self.action_space = gym.spaces.Discrete(10)
        self.observation_space = gym.spaces.Box(low=np.array([-1]), high=np.array([1]), dtype=np.float32)
        self.state = np.array([0])

    def reset(self):
        return np.array([0])

    def step(self, action):
        assert int(action) in self.action_space

        done = True
        s = np.array([0])
        r = float(-(action - 7) ** 2)
        info = {}
        return s, r, done, info


class Reinforce:
    def __init__(self, policy, env, optimizer):
        self.policy = policy
        self.env = env
        self.optimizer = optimizer

    @staticmethod
    def compute_expected_cost(trajectory, gamma, baseline):
        """
        Compute the expected cost of this episode for gradient backprop
        DO NOT change its method signature
        :param trajectory: a list of 3-tuple of (reward: Float, policy_output_probs: torch.Tensor, action: Int)
        NOTE: policy_output_probs will have a grad_fn, i.e., it's able to backpropagate gradients from your computed cost
        :param gamma: gamma
        :param baseline: a simple running mean baseline to be subtracted from the total discounted returns
        :return: a 2-tuple of torch.tensor([cost]) of this episode that allows backprop and updated baseline
        """
        cost = 0.0

        ### YOUR CODE HERE ###
        rewards, policy_output_probs, actions = list(zip(*trajectory))
        rewards = torch.FloatTensor(rewards)
        actions = torch.LongTensor(actions)
        policy_output_probs = torch.log(torch.stack(policy_output_probs))

        GTs = torch.zeros(len(rewards))
        for k in reversed(range(len(rewards))):
            if k != len(rewards)-1:
                GTs[k] = rewards[k] + gamma * GTs[k+1]
            else:
                GTs[k] = rewards[k]

        mu = torch.mean(GTs)
        rho = torch.sqrt(torch.mean((GTs - mu) ** 2))
        GTs = (GTs - baseline) / rho
        p = 0.99
        baseline = p * baseline + (1-p) * mu

        cost = -torch.sum((GTs * torch.gather(policy_output_probs, 1, actions.unsqueeze(1)).squeeze()))
        # #### -------------- ###
        return cost, baseline

    def train(self, num_episodes, gamma):
        """
        train the policy using REINFORCE for specified number of episodes
        :param num_episodes: number of episodes to train for
        :param gamma: gamma
        :return: self
        """

        baseline = 0
        for episode_i in range(num_episodes):
            ### YOUR CODE HERE ###
            trajectory = self.generate_episode()
            self.optimizer.zero_grad()
            rewards, _, _ = list(zip(*trajectory))
            rewards = torch.FloatTensor(rewards)
            cost, baseline = self.compute_expected_cost(trajectory, gamma, baseline)
            cost.backward()
            self.optimizer.step()
            wandb.log({'reward': torch.sum(rewards)})
            ### -------------- ###

        # torch.save(policy.state_dict(), "mypolicy.pth")
        return self

    def generate_episode(self):
        """
        run the environment for 1 episode
        NOTE: do not limit the number
        :return: whatever you need for training
        """

        ### YOUR CODE HERE AND REMOVE `pass` below ###
        initial_s = self.env.reset()
        current_s, next_s = initial_s, None
        trajectory = []
        done = False

        while not done:
            policy_output_probs = self.policy(current_s)
            action = np.random.choice(self.env.action_space.n, p=policy_output_probs.detach().numpy())
            next_s, reward, done, _ = self.env.step(action)
            trajectory.append((reward, policy_output_probs, action))
            current_s = next_s

        return trajectory


# Do NOT change the name of the class.
# This class should contain your policy model architecture.
# Please make sure we can load your model with:
# policy = MyPolicy()
# policy.load_state_dict(torch.load("mypolicy.pth"))
# This means you must give default values to all parameters you may wish to set, such as output size.
class MyPolicy(nn.Module):
    def __init__(self):
        super(MyPolicy, self).__init__()
        ### YOUR CODE HERE AND REMOVE `pass` below ###
        self.policy = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        ### YOUR CODE HERE AND REMOVE `pass` below ###
        x = torch.FloatTensor(x)
        return self.policy(x)


def weights_initialization(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0)


if __name__ == '__main__':
    # define and train your policy here
    ### YOUR CODE HERE AND REMOVE `pass` below ###
    wandb.init(project="10701_hw5", reinit=True)
    # env = BanditEnv()
    env = gym.make('LunarLander-v2')
    policy = MyPolicy()
    policy.apply(weights_initialization)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    R = Reinforce(env=env, policy=policy, optimizer=optimizer)
    R.train(num_episodes=1000, gamma=0.99)
