import random
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from typing import List
from phermes import HyperHeuristic, Problem

class PPOHH(HyperHeuristic):
    def __init__(self, features: List[str], heuristics: List[str]):
        super().__init__(features, heuristics)
        self.state_size = len(features)
        self.action_size = len(heuristics)
        self.memory = deque(maxlen=5000)
        self.gamma = 0.99
        self.epsilon = 0.2  # PPO clip parameter
        self.learning_rate = 0.0003
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_model = self._build_policy_model().to(self.device)
        self.value_model = self._build_value_model().to(self.device)
        self.policy_optimizer = optim.Adam(self.policy_model.parameters(), lr=self.learning_rate)
        self.value_optimizer = optim.Adam(self.value_model.parameters(), lr=self.learning_rate)
        self.losses = []  # List to store losses for plotting

    def _build_policy_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size),
            nn.Softmax(dim=-1)
        )
        return model

    def _build_value_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        return model

    def train(self, filename: str) -> None:
        data = pd.read_csv(filename, header=0)
        columns = ["INSTANCE", "BEST", "ORACLE"] + self._heuristics
        self.X = data.drop(columns, axis=1).values
        self.y = data["BEST"].values
        for i in range(len(self._heuristics)):
            self.y[self.y == self._heuristics[i]] = i
        self.y = self.y.astype("int")
        self._train_agent()

    def _train_agent(self, batch_size=32, epochs=500, max_steps=100):
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        line, = ax.plot(self.losses, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss over Epochs")
        plt.legend()

        for e in range(epochs):
            epoch_loss = 0  # Track loss for the epoch
            for instance, features in zip(self.X, self.y):
                features = np.array(features)
                if features.size != self.state_size:
                    print(f"Skipping instance with features of size {features.size}, expected {self.state_size}")
                    continue
                state = np.reshape(features, [1, self.state_size])
                state = torch.FloatTensor(state).to(self.device)
                trajectory = []

                for time in range(max_steps):
                    action, action_prob = self._act(state)
                    next_state, reward = self._apply_heuristic(instance, action)
                    done = reward <= 0
                    reward = reward if not done else -10
                    next_state = np.reshape(next_state, [1, self.state_size])
                    next_state = torch.FloatTensor(next_state).to(self.device)
                    trajectory.append((state, action, action_prob, reward, next_state, done))
                    state = next_state
                    if done:
                        break

                self.memory.extend(trajectory)
                if len(self.memory) > batch_size:
                    loss = self._replay(batch_size)
                    epoch_loss += loss  # Accumulate loss

            self.losses.append(epoch_loss)  # Record the epoch loss

            # Update plot
            line.set_ydata(self.losses)
            line.set_xdata(range(len(self.losses)))
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.01)  # Pause to update the plot
        
        plt.ioff()  # Turn off interactive mode
        plt.show()

    def _replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, actions, action_probs, rewards, next_states, dones = zip(*minibatch)

        states = torch.cat(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        action_probs = torch.cat(action_probs).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        next_states = torch.cat(next_states).to(self.device)
        dones = torch.tensor(dones).to(self.device)

        # Compute advantages
        values = self.value_model(states).squeeze()
        next_values = self.value_model(next_states).squeeze()
        advantages = rewards + self.gamma * next_values * (1 - dones) - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Normalize

        # Update policy
        for _ in range(10):  # 10 epochs of optimization
            new_action_probs = self.policy_model(states)
            new_action_probs = new_action_probs.gather(1, actions.unsqueeze(-1)).squeeze()
            ratios = new_action_probs / action_probs
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

        # Update value function
        value_targets = rewards + self.gamma * next_values * (1 - dones)
        value_loss = nn.MSELoss()(values, value_targets.detach())

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        return policy_loss.item() + value_loss.item()

    def _act(self, state):
        with torch.no_grad():
            action_probs = self.policy_model(state)
        action = np.random.choice(self.action_size, p=action_probs.cpu().numpy().flatten())
        return action, action_probs[:, action]

    def _apply_heuristic(self, instance, action):
        # Define this function based on your environment's specifics
        # It should return next_state and reward based on the action taken
        next_state = instance  # Example placeholder
        reward = 1  # Example placeholder
        return next_state, reward

    def getHeuristic(self, problem: Problem) -> str:
        state = pd.DataFrame()
        for i in range(len(self._features)):
            state[self._features[i]] = [problem.getFeature(self._features[i])]
        state = torch.FloatTensor(state.values).to(self.device)
        with torch.no_grad():
            action_probs = self.policy_model(state)
        return self._heuristics[torch.argmax(action_probs[0]).item()]
