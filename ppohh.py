
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

class PPOActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(PPOActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, action_size),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, 1)
        )

    def forward(self, state):
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value

class PPOHH(HyperHeuristic):
    def __init__(self, features: List[str], heuristics: List[str]):
        super().__init__(features, heuristics)
        self.state_size = len(features)
        self.action_size = len(heuristics)
        self.memory = deque(maxlen=5000)
        self.gamma = 0.85 # 0.99
        self.epsilon = 0.2
        self.learning_rate = 0.001  # Adjusted learning rate for stability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {self.device}")
        self.model = PPOActorCritic(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.losses = []  # List to store losses for plotting

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def _act(self, state):
        #state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action_probs, _ = self.model(state)
        action = torch.multinomial(action_probs, 1).item()
        return action

    def _compute_advantage(self, rewards, values, next_values, dones):
        advantages = []
        returns = []
        advantage = 0
        for i in reversed(range(len(rewards))):
            if dones[i]:
                advantage = 0
            td_error = rewards[i] + self.gamma * next_values[i] * (1 - dones[i]) - values[i]
            advantage = td_error + self.gamma * advantage
            returns.insert(0, advantage + values[i])
            advantages.insert(0, advantage)
        return advantages, returns

    def _update_model(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        _, next_values = self.model(next_states)
        _, values = self.model(states)

        advantages, returns = self._compute_advantage(rewards, values, next_values, dones)

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        old_action_probs, _ = self.model(states)
        old_action_probs = old_action_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1).detach()

        for _ in range(10):  # Update the model multiple times
            action_probs, values = self.model(states)
            action_probs = action_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)

            ratios = action_probs / old_action_probs
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (returns - values).pow(2).mean()

            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item()

    def train(self, filename: str) -> None:
        data = pd.read_csv(filename, header=0)
        columns = ["INSTANCE", "BEST", "ORACLE"] + self._heuristics
        self.X = data.drop(columns, axis=1).values
        self.y = data["BEST"].values
        for i in range(len(self._heuristics)):
            self.y[self.y == self._heuristics[i]] = i
        self.y = self.y.astype("int")
        self._train_agent()
        plt.close()

    def _apply_heuristic(self, instance, action):
        next_state = instance
        reward = 1
        return next_state, reward

    def _train_agent(self, batch_size=32, epochs=3, max_steps=5): # 8 40
        plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot(self.losses, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss over Epochs")
        plt.legend()
        done = False

        for e in range(epochs):
            epoch_loss = 0

            for instance, features in zip(self.X, self.y):
                features = np.array([features, 0, 0])
                if features.size != self.state_size:
                    print(f"Skipping instance with features of size {features.size}, expected {self.state_size}")
                    continue
                state = np.reshape(features, [1, self.state_size])
                state = torch.FloatTensor(state).to(self.device)

                episode_states = []
                episode_actions = []
                episode_rewards = []
                episode_next_states = []
                episode_dones = []

                for time in range(max_steps):
                    action = self._act(state[0])
                    next_state, reward = self._apply_heuristic(instance, action)
                    done = reward <= 0
                    reward = reward if not done else -10
                    next_state = np.reshape(next_state, [1, self.state_size])
                    next_state = torch.FloatTensor(next_state).to(self.device)

                    episode_states.append(state.cpu().numpy())
                    episode_actions.append(action)
                    episode_rewards.append(reward)
                    episode_next_states.append(next_state.cpu().numpy())
                    episode_dones.append(done)

                    state = next_state
                    if done:
                        print(f"episode: {e}/{epochs}, score: {time}, e: {self.epsilon:.2}")
                        break
                print(len(episode_states), batch_size)
                if len(episode_states) >= batch_size:
                    loss = self._update_model(
                        np.vstack(episode_states), 
                        episode_actions, 
                        episode_rewards, 
                        np.vstack(episode_next_states), 
                        episode_dones
                    )
                    print(loss)
                    epoch_loss += loss

            self.losses.append(epoch_loss)

            line.set_ydata(self.losses)
            line.set_xdata(range(len(self.losses)))
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.01)

            print(f"Epoch {e+1}/{epochs}, Loss: {epoch_loss:.4f}")


        plt.ioff()
        plt.show()


    def getHeuristic(self, problem: Problem) -> str:
        state = pd.DataFrame()
        
        for i in range(len(self._features)):
            state[self._features[i]] = [problem.getFeature(self._features[i])]
        #print(state.values[0])
        # Ensure the state tensor is on the same device as the model
        state_tensor = torch.FloatTensor(state.values[0]).to(self.device)
        with torch.no_grad():
            prediction = self.model(state_tensor)
        return self._heuristics[torch.argmax(prediction[0]).item()]