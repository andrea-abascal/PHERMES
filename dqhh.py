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


class DQHH(HyperHeuristic):
    def __init__(self, features: List[str], heuristics: List[str]):
        super().__init__(features, heuristics)
        self.state_size = len(features)
        self.action_size = len(heuristics)
        self.memory = deque(maxlen=5000)
        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001  # Adjusted learning rate for stability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.losses = []  # List to store losses for plotting

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
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
                for time in range(max_steps):
                    action = self._act(state)
                    next_state, reward = self._apply_heuristic(instance, action)
                    done = reward <= 0
                    reward = reward if not done else -10
                    next_state = np.reshape(next_state, [1, self.state_size])
                    next_state = torch.FloatTensor(next_state).to(self.device)
                    self.memory.append((state, action, reward, next_state, done))
                    state = next_state
                    if done:
                        break
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

            # print(f"Epoch {e+1}/{epochs}, Loss: {epoch_loss:.4f}")
        
        plt.ioff()  # Turn off interactive mode
        plt.show()

    def _replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        total_loss = 0
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                with torch.no_grad():
                    target = reward + self.gamma * torch.max(self.model(next_state)[0]).item()
            target_f = self.model(state)
            target_f[0][action] = target
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(state)
            loss = self.criterion(output, target_f)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return total_loss / batch_size

    def _act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def getHeuristic(self, problem: Problem) -> str:
        state = pd.DataFrame()
        for i in range(len(self._features)):
            state[self._features[i]] = [problem.getFeature(self._features[i])]
        state = torch.FloatTensor(state.values).to(self.device)
        with torch.no_grad():
            prediction = self.model(state)
        return self._heuristics[torch.argmax(prediction[0]).item()]
