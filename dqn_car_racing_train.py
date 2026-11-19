# Adapted from https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from car_racing_helper import make_env

import math
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, n_actions, in_channels=4) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 9 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, x):
        # x: uint8 (B, H, W, C) -> float (B, C, H, W)
        x = x.permute(0, 3, 1, 2).float() / 255.0
        return self.fc(self.conv(x))


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(
            *map(lambda x: torch.tensor(x, device=device) if x is not None else x, args)))
        # for t in zip(*args):
        #     self.memory.append(Transition(*t))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


steps_done = 0


BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 25000
TAU = 0.005
LR = 1e-4
MAX_BUFFER = 100000
MAX_STEPS = 1000000

device = "cuda"


def plot_durations(show_result=False):
    plt.figure(1)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_t.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


def select_action(state, q_net, env):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return q_net(state).max(1).indices
    else:
        return torch.tensor([env.action_space.sample()], device=device, dtype=torch.long)


def optimize_model(memory, optimizer, q_net, target_net):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack(
        ([s for s in batch.next_state if s is not None]))
    state_batch = torch.stack((batch.state))
    action_batch = torch.stack(
        (batch.action)).unsqueeze(1)
    reward_batch = torch.stack((batch.reward))

    # print(state_batch.shape, action_batch.shape, reward_batch.shape)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = q_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(
            non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values,
                     expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(q_net.parameters(), 100)
    optimizer.step()


episode_durations = []
episode_rewards = []


def main():
    env = make_env(1, render_mode="human")
    n_actions = env.action_space.n

    obs = env.reset()
    n_observations = obs.shape[-1]
    print(n_observations, n_actions)

    q_net = QNetwork(n_actions, in_channels=n_observations).to(device)
    target_net = QNetwork(n_actions, in_channels=n_observations).to(device)
    target_net.load_state_dict(q_net.state_dict())

    memory = ReplayMemory(MAX_BUFFER)

    optimizer = optim.AdamW(q_net.parameters(), lr=LR, amsgrad=True)

    pbar = tqdm(total=MAX_STEPS)

    step_count = 0
    while step_count < MAX_STEPS:
        obs = env.reset()
        states = obs
        for t in (count()):
            states_tensor = torch.as_tensor(np.array(states), device=device)
            actions = select_action(states_tensor, q_net, env)
            actions = actions.cpu().numpy()
            obs, rewards, dones, info = env.step(actions)

            # print(any(dones))
            # print(rewards)
            # env.render("human")

            truncated = info[0].get("TimeLimit.truncated", False)
            next_states = [obs[0] if not dones[0] or truncated else None]

            memory.push(states[0], actions[0], next_states[0], rewards[0])

            states = next_states

            optimize_model(memory, optimizer, q_net, target_net)

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = q_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * \
                    TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            pbar.update(1)
            step_count += 1

            if dones[0]:
                episode_durations.append(t + 1)
                episode_rewards.append(info[0]["episode"]["r"])
                print()
                print(info[0]["episode"]["r"], info[0]
                      ["episode"]["l"], truncated)
                plot_durations()
                break

    pbar.close()
    print('Complete')
    plot_durations(show_result=True)
    plt.savefig("carracing_dqn_scratch.png")
    torch.save(q_net.state_dict(), "carracing_dqn_scratch.pt")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
