import torch

import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

import numpy as np

from homework2 import Hw2Env
import time
from tqdm import tqdm
import os
import sys
from torchinfo import summary
from torchvision.transforms.functional import rgb_to_grayscale
import argparse
import DQN_Models
import matplotlib.pyplot as plt

# Argparse
parser = argparse.ArgumentParser(description='DQN')
parser.add_argument('--test', help='Test the model', action='store_true')
parser.add_argument('--load', type=str, help='Load the model')
parser.add_argument('--gpu', type=str, help='Use GPU', default="cuda:0")
parser.add_argument('--episodes', type=int, help='Number of episodes', default=1000)
parser.add_argument('--num_of_frames', type=int, help='Number of frames', default=4)
parser.add_argument('--learning_type', type=str, help='Learning type', default="epsilon", choices=["epsilon", "boltzmann"])
parser.add_argument('--high_level', help='High level',action='store_true')
parser.add_argument('--model', type=str, help='Model', default="residual", choices=["residual", "example", "mlp"])
parser.add_argument('--episode_length', type=int, help='Episode length', default=50)
parser.add_argument('--tau', type=float, help='Tau', default=1.0)
args = parser.parse_args()
TEST = args.test
LOAD = args.load
GPU = args.gpu
HIGH_LEVEL = args.high_level
MODEL = args.model
EPISODE_LENGTH = args.episode_length

GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.999 # decay epsilon by 0.999 every EPSILON_DECAY_ITER
EPSILON_DECAY_ITER = 100 # decay epsilon every 100 updates
MIN_EPSILON = 0.1 # minimum epsilon
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
UPDATE_FREQ = 4 # update the network every 4 steps
TARGET_NETWORK_UPDATE_FREQ = 1000 # update the target network every 1000 steps
BUFFER_LENGTH = 10000
N_ACTIONS = 8
TAU = args.tau
M = args.episodes
RENDER_MODE = "offscreen" #gui
BETA = 1.0
NUM_OF_FRAMES = args.num_of_frames
LEARNING_TYPE = args.learning_type

def restart_script():
    print("#"*50)
    print("Training is stuck, restarting the script to free up the memory.")
    python = sys.executable  # Get the Python interpreter path
    script = os.path.abspath(__file__)  # Get the absolute path of the current script
    torch.cuda.empty_cache()  # Empty the CUDA cache
    os.execle(python, python, script, *sys.argv[1:], os.environ)  # Replace the current process with a new one


class Transition:
    def __init__(self, state, action, reward, next_state, is_terminal, is_truncated):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.is_terminal = is_terminal
        self.is_truncated = is_truncated


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return np.random.choice(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        if MODEL == "residual":
            self.model = DQN_Models.ResidualCNN(NUM_OF_FRAMES, N_ACTIONS)
        elif MODEL == "example":
            self.model = DQN_Models.ExampleCNN(NUM_OF_FRAMES, N_ACTIONS)
        elif MODEL == "mlp":
            self.model = DQN_Models.MLP(NUM_OF_FRAMES, N_ACTIONS)

    def forward(self, x):
        return self.model(x)

class DQNAgent:
    def __init__(self):
        self.memory = ReplayMemory(BUFFER_LENGTH)
        #"mps" if torch.backends.mps.is_available() else
        self.device = torch.device(GPU if torch.cuda.is_available() else "cpu")
        self.Q = DQN().to(self.device)
        self.Q_hat = DQN().to(self.device)
        self.Q_hat.load_state_dict(self.Q.state_dict())
        self.Q_hat.eval()
        self.optimizer = optim.AdamW(self.Q.parameters(), lr=LEARNING_RATE, amsgrad=True)
        
        self.episode = 0
        self.step = 1 # start at one to avoid division by zero

        self.env = Hw2Env(n_actions=N_ACTIONS, render_mode=RENDER_MODE, max_timesteps=EPISODE_LENGTH * NUM_OF_FRAMES)

        self.criterion = nn.MSELoss()

    def preprocess_sequence(self, sequence):
        if not HIGH_LEVEL:
            sequence = rgb_to_grayscale(sequence)
        elif sequence.shape[0] > NUM_OF_FRAMES:
            sequence = sequence.unsqueeze(0)

        if sequence.shape[0] < NUM_OF_FRAMES:
            sequence = torch.cat([sequence for _ in range(NUM_OF_FRAMES)], dim=0)
        else:
            sequence = sequence.squeeze(1)

        return sequence.to(self.device)

    def select_action(self, state, train=True):
        prob = np.random.random()
        if LEARNING_TYPE == "epsilon":
            epsilon = max(MIN_EPSILON, EPSILON * (EPSILON_DECAY ** (self.step // EPSILON_DECAY_ITER)))
            if not train:
                epsilon = 0.0
            if prob < epsilon:
                action = np.random.randint(N_ACTIONS)
            else:
                with torch.no_grad():
                    if train:
                        q_values = self.Q(state)  
                    else:
                        q_values = self.Q_hat(state)
                    action = q_values.argmax().item()

        elif LEARNING_TYPE == "boltzmann":
            with torch.no_grad():
                if train:
                    q_values = self.Q(state)   
                    # q_values = q_values - q_values.max()
                    p_as = torch.exp(BETA * q_values)/torch.sum(torch.exp(BETA * q_values))
                    action = p_as.multinomial(1).item()
                else:
                    q_values = self.Q_hat(state)
                    action = q_values.argmax().item()

        return action

    def execute_action(self, action):
        state, reward, is_terminal, is_truncated = self.env.step(action, high_level=HIGH_LEVEL)
        return state, reward, is_terminal, is_truncated
    
    def calculate_reward(self, samples):
        targets = []
        for sample in samples:
            if not sample.is_truncated and not sample.is_terminal:
                target = sample.reward + GAMMA * self.Q_hat(sample.next_state).max().item()
            else:
                target = sample.reward
            targets.append(target)
        return torch.tensor(targets, dtype=torch.float32).view(-1, 1).to(self.device)
    
    def train(self):
        cum_loss = 0.0
        rewards = []
        while self.episode < M:
            self.env.reset()
            if HIGH_LEVEL:
                state = self.env.high_level_state()
            else:
                state = self.env.state()
            state = self.preprocess_sequence(state)
            start = time.time()
            
            cum_reward = 0.0
            while not self.env.is_terminal() and not self.env.is_truncated():
                for _ in range(UPDATE_FREQ):
                    action = self.select_action(state)
                    reward = 0.0
                    next_state = torch.tensor([])
                    for _ in range(NUM_OF_FRAMES):
                        frame_state, frame_reward, is_terminal, is_truncated = self.execute_action(action)
                        reward += frame_reward
                        next_state = torch.cat([next_state, frame_state.unsqueeze(0)], dim=0)
                        if is_terminal or is_truncated:
                            break
                    cum_reward += reward
                    next_state = self.preprocess_sequence(next_state)
                    self.memory.push(state, action, reward, next_state, is_terminal, is_truncated)
                    state = next_state
                    if is_terminal or is_truncated:
                        break
                
                if len(self.memory) < BATCH_SIZE:
                    continue

                cum_loss = self.batch_train(cum_loss)
                self.optimizer.step()
                self.step += 1

                if self.step % TARGET_NETWORK_UPDATE_FREQ == 0:
                    self.update_network()

            end = time.time()
            self.log_training(cum_loss, cum_reward, start, end)
            rewards.append(cum_reward)
            self.episode += 1
            
            if self.episode % 200 == 0:
                self.save(f"dqn-{LEARNING_TYPE}-model:{MODEL}-high:{HIGH_LEVEL}-frame:{NUM_OF_FRAMES}-length:{EPISODE_LENGTH}.pth")

        plt.plot(rewards)
        plt.savefig(f"dqn-{LEARNING_TYPE}-model:{MODEL}-high:{HIGH_LEVEL}-frame:{NUM_OF_FRAMES}-length:{EPISODE_LENGTH}.png")


    def batch_train(self, cum_loss):
        batch = self.memory.sample(batch_size=BATCH_SIZE)
        target = self.calculate_reward(batch)
        states = torch.stack([sample.state for sample in batch])
        mask = torch.tensor([sample.action for sample in batch], dtype=torch.long).view(-1, 1).to(self.device)
        q_values = self.Q(states)
        result_tensor = q_values[torch.arange(mask.size(0)), mask.flatten()].reshape(mask.shape)
        loss = self.criterion(result_tensor, target)
        cum_loss += loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        return cum_loss


    def update_network(self):
        print("Updating target network at step", self.step)
        Q_state_dict = self.Q.state_dict()
        Q_hat_state_dict = self.Q_hat.state_dict()
        for key in Q_state_dict:
            Q_hat_state_dict[key] = TAU * Q_state_dict[key] + (1 - TAU) * Q_hat_state_dict[key]
        self.Q_hat.load_state_dict(Q_hat_state_dict)
        self.Q_hat.eval()
        # self.validity_check()


    def log_training(self, cum_loss, cum_reward, start, end):
        if LEARNING_TYPE == "epsilon":
            print(f"Episode={self.episode}, Reward = {cum_reward:.2f}, loss = {cum_loss/self.step:.5f}, Time={(end-start):.2f}, Steps={self.step}, Epsilon={EPSILON * (EPSILON_DECAY ** (self.step // EPSILON_DECAY_ITER)):.3f}")
        elif LEARNING_TYPE == "boltzmann":
            print(f"Episode={self.episode}, Reward = {cum_reward:.2f}, loss = {cum_loss/self.step:.6f}, Time={(end-start):.2f}, Steps={self.step}")

    def validity_check(self):
        print("#"*50)
        print("Testing the model")
        working = self.test(None)
        print("#"*50)
        if not working:
            restart_script()
         

    def save(self, path):
        torch.save(self.Q.state_dict(), path)
        path = path.split(".")
        path = path[0] + "_hat." + path[1]
        torch.save(self.Q_hat.state_dict(), path)

    def load(self, path):
        self.Q.load_state_dict(torch.load(path, map_location=torch.device(self.device)))
        self.Q_hat.load_state_dict(self.Q.state_dict())
        self.Q_hat.eval()
        if TEST:
            self.episode = 0
        else:
            self.episode = int(path.split("_")[1].split(".")[0])
    
    def test(self, path):
        if path:
            self.load(path)
        for episode in range(10):
            self.env.reset()
            if HIGH_LEVEL:
                state = self.env.high_level_state()
            else:
                state = self.env.state()
            state = self.preprocess_sequence(state)
            cum_reward = 0.0
            actions = []
            while not self.env.is_terminal() and not self.env.is_truncated():
                next_state = torch.tensor([])
                reward = 0.0
                action = self.select_action(state, train=False)
                actions.append(action)
                for _ in range(NUM_OF_FRAMES):
                    frame_state, frame_reward, is_terminated, is_truncated = self.execute_action(action)
                    reward += frame_reward
                    next_state = torch.cat([next_state, frame_state.unsqueeze(0)], dim=0)
                    if is_terminated or is_truncated:
                        break
                next_state = self.preprocess_sequence(next_state)
                state = next_state
                cum_reward += reward
            print(f"Episode={episode}, reward={cum_reward}")
            # return false if all actions same in the array
            return not np.all(np.isclose(actions, actions[0]))
        return True
 

if __name__ == "__main__":
    if TEST:
        RENDER_MODE = "gui"
        agent = DQNAgent()
        agent.test(LOAD)
    else:
        agent = DQNAgent()
        agent.train()
        agent.save(f"dqn-{LEARNING_TYPE}-model:{MODEL}-high:{HIGH_LEVEL}-frame:{NUM_OF_FRAMES}-length:{EPISODE_LENGTH}.pth")