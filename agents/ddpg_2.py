import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import csv
import collections
import random

# --- DDPG Network Definitions ---

class DDPG_Actor(nn.Module):
    def __init__(self, state_dim, emb_dim, action_dim):
        super(DDPG_Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, emb_dim)
        self.layer_2 = nn.Linear(emb_dim, emb_dim)
        self.layer_3 = nn.Linear(emb_dim, action_dim)

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        # --- KEY: No output activation ---
        # We output unbounded "logits" which the env.step() will squash
        # using a sigmoid. This matches the PPO setup.
        a = self.layer_3(s)
        return a

class DDPG_Critic(nn.Module):
    def __init__(self, state_dim, emb_dim, action_dim):
        super(DDPG_Critic, self).__init__()
        # Q-network takes (state + action) as input
        self.layer_1 = nn.Linear(state_dim + action_dim, emb_dim)
        self.layer_2 = nn.Linear(emb_dim, emb_dim)
        self.layer_3 = nn.Linear(emb_dim, 1) # Outputs a single Q-value

    def forward(self, s, a):
        # Concatenate state and action
        sa = torch.cat([s, a], 1)
        q = F.relu(self.layer_1(sa))
        q = F.relu(self.layer_2(q))
        q = self.layer_3(q) # No activation on final Q-value
        return q

# --- Replay Buffer ---

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device):
        self.buffer = collections.deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.buffer, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.buffer)

# --- DDPG Agent Class ---

class DDPG(object):
    def __init__(self, args, env):
        self.device = args.device
        self.path = args.path
        self.log_interval = args.log_interval
        self.est_interval = args.est_interval
        self.q_alpha = args.q_alpha
        self.gamma = args.gamma
        self.max_episode = args.max_episode
        self.env = env
        self.env_name = args.env_name

        state_dim = np.prod(self.env.observation_space.shape)
        action_dim = np.prod(self.env.action_space.shape)
        
        # --- DDPG Hyperparameters ---
        self.batch_size = args.mini_batch # Use same name as PPO for 'mini_batch'
        self.tau = args.tau
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.buffer_size = int(args.buffer_size)
        self.start_steps = args.start_steps
        self.exploration_noise = args.exploration_noise

        # --- Initialize Networks ---
        self.actor = DDPG_Actor(state_dim, args.emb_dim, action_dim).to(self.device)
        self.actor_target = DDPG_Actor(state_dim, args.emb_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = DDPG_Critic(state_dim, args.emb_dim, action_dim).to(self.device)
        self.critic_target = DDPG_Critic(state_dim, args.emb_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # --- Optimizers ---
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        
        # --- Replay Buffer & Writer ---
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, self.device)
        self.writer = SummaryWriter(log_dir=args.path)
        self.total_steps = 0

    def select_action(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(self.device)
        self.actor.eval() # Set actor to evaluation mode
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()
        self.actor.train() # Set actor back to training mode
        
        if add_noise:
            # Add Gaussian noise for exploration
            noise = np.random.normal(0, self.exploration_noise, size=action.shape)
            action += noise
            
        # --- IMPORTANT ---
        # We do NOT clip or squash the action here.
        # The environment's step function will apply the sigmoid.
        return action

    def train(self):
        disc_epi_rewards = []
        csv_path = os.path.join(self.path, 'ddpg_rewards.csv')
        
        with open(csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Episode', 'Reward', 'Total_Steps'])

            for i_episode in range(1, self.max_episode + 1):
                state = self.env.reset()
                episode_reward = 0
                disc_epi_reward = 0
                disc_factor = 0.99 # Matches your PPO discount factor
                
                while True:
                    self.total_steps += 1
                    
                    # Select action:
                    # Use random actions for the first `start_steps` to fill buffer
                    if self.total_steps < self.start_steps:
                        action = self.env.action_space.sample()
                    else:
                        action = self.select_action(state, add_noise=True)
                    
                    next_state, reward, done, _ = self.env.step(action)
                    self.memory.add(state, action, reward, next_state, done)
                    
                    # Perform gradient update
                    if len(self.memory) > self.batch_size and self.total_steps > self.start_steps:
                        self.update()
                    
                    state = next_state
                    episode_reward += reward
                    disc_epi_reward += disc_factor * reward
                    disc_factor *= self.gamma
                    
                    if done:
                        break
                
                disc_epi_rewards.append(disc_epi_reward)
                self.writer.add_scalar('disc_reward/raw_reward', disc_epi_reward, i_episode)
                csv_writer.writerow([i_episode, disc_epi_reward, self.total_steps])
                
                if i_episode % self.log_interval == 0:
                    lb = max(0, len(disc_epi_rewards) - self.est_interval)
                    disc_a_reward = np.mean(disc_epi_rewards[lb:])
                    disc_q_reward = np.percentile(disc_epi_rewards[lb:], self.q_alpha * 100)
                    
                    self.writer.add_scalar('disc_reward/aver_reward', disc_a_reward, i_episode)
                    self.writer.add_scalar('disc_reward/quantile_reward', disc_q_reward, i_episode)
                    print(f'Epi:{i_episode:05d} | Steps:{self.total_steps:07d} | disc_a_r:{disc_a_reward:.03f} | disc_q_r:{disc_q_reward:.03f}')

            # --- Save the final model ---
            self.save_model()
            print(f"Training complete. Model saved in {self.path}")

    def update(self):
        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample()
        
        # --- 1. Compute Target Q-Values ---
        with torch.no_grad():
            # Select next actions from target actor
            next_actions = self.actor_target(next_states)
            # Compute Q-value of next state-action pair
            target_q = self.critic_target(next_states, next_actions)
            # Compute the final target Q
            y = rewards + (self.gamma * (1 - dones) * target_q)

        # --- 2. Update Critic ---
        # Get current Q-values
        current_q = self.critic(states, actions)
        # Compute critic loss (MSE)
        critic_loss = F.mse_loss(current_q, y)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # --- 3. Update Actor ---
        # Compute actor loss (policy gradient)
        # We want to maximize Q(s, mu(s))
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()
        
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # --- 4. Soft-update Target Networks ---
        self.soft_update(self.critic, self.critic_target)
        self.soft_update(self.actor, self.actor_target)
        
        # Log losses
        self.writer.add_scalar('loss/actor_loss', actor_loss.item(), self.total_steps)
        self.writer.add_scalar('loss/critic_loss', critic_loss.item(), self.total_steps)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters: θ_target = τ*θ_local + (1 - τ)*θ_target"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save_model(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        torch.save(self.actor.state_dict(), os.path.join(self.path, 'ddpg_actor.pt'))
        torch.save(self.critic.state_dict(), os.path.join(self.path, 'ddpg_critic.pt'))

    def load_model(self):
        self.actor.load_state_dict(torch.load(os.path.join(self.path, 'ddpg_actor.pt')))
        self.critic.load_state_dict(torch.load(os.path.join(self.path, 'ddpg_critic.pt')))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
