import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import csv
import os

# --- 1. Replay Buffer ---
# DDPG is off-policy, so it requires a replay buffer to store and sample experiences.
class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim):
        self.mem_size = max_size
        self.mem_counter = 0
        self.state_memory = np.zeros((self.mem_size, state_dim), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, action_dim), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.next_state_memory = np.zeros((self.mem_size, state_dim), dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.terminal_memory[index] = done
        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.next_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, next_states, dones

# --- 2. Actor and Critic Networks ---
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.q_value = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_val = self.q_value(x)
        return q_val

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.action_out = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Use tanh to bound the actions between -1 and 1
        actions = torch.tanh(self.action_out(x))
        return actions

# --- 3. DDPG Agent ---
class DDPG:
    def __init__(self, args, env):
        self.env = env
        self.state_dim = np.prod(env.observation_space.shape)
        self.action_dim = np.prod(env.action_space.shape)
        self.max_action = env.action_space.high # Used for scaling the tanh output
        self.min_action = env.action_space.low

        # Hyperparameters
        self.gamma = args.gamma # Discount factor
        self.tau = args.tau # Soft update parameter
        self.batch_size = args.batch_size
        self.max_episode = args.max_episode
        self.noise_std = args.noise_std # Standard deviation for action noise
        
        self.device = args.device

        # Initialize networks
        self.actor = ActorNetwork(self.state_dim, self.action_dim).to(self.device)
        self.critic = CriticNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_actor = ActorNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_critic = CriticNetwork(self.state_dim, self.action_dim).to(self.device)

        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(args.buffer_size, self.state_dim, self.action_dim)

        # Copy initial weights to target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # For logging
        self.log_interval = args.log_interval
        self.path = args.path


    def choose_action(self, state, is_training=True):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        # Get deterministic action from the actor network
        action = self.actor(state).cpu().detach().numpy()

        if is_training:
            # Add Gaussian noise for exploration
            noise = np.random.normal(0, self.noise_std, size=self.action_dim)
            action = action + noise
        
        # Clip the action to be within the valid action space range [-1, 1] from tanh
        action = np.clip(action, -1, 1)

        # --- IMPORTANT: Scale the action from [-1, 1] to the environment's action range ---
        # This is a critical step. The actor outputs values in [-1, 1]. We need to scale them
        # to the actual ranges defined in your RSMA environment.
        # (action + 1) / 2 scales it to [0, 1]. Then we scale it to [min, max].
        scaled_action = self.min_action + (action + 1.0) / 2.0 * (self.max_action - self.min_action)
        
        return scaled_action

    def update(self):
        if self.memory.mem_counter < self.batch_size:
            return # Don't update if the buffer doesn't have enough samples

        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample_buffer(self.batch_size)
        
        # Convert to PyTorch tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # --- Critic Update ---
        # Get target Q-value
        with torch.no_grad():
            target_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, target_actions)
            target_q = rewards + self.gamma * (1 - dones) * target_q.squeeze()

        # Get current Q-value
        current_q = self.critic(states, actions).squeeze()
        
        # Calculate critic loss and update
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor Update ---
        # Calculate actor loss and update
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # --- Soft Update Target Networks ---
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def train(self):
        # Create CSV file for logging
        csv_path = os.path.join(self.path, 'ddpg_rewards.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Episode', 'Reward'])

            all_rewards = []
            for i_episode in range(self.max_episode + 1):
                state = self.env.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    action = self.choose_action(state, is_training=True)
                    next_state, reward, done, _ = self.env.step(action)
                    
                    # Store transition in replay buffer
                    self.memory.store_transition(state, action, reward, next_state, done)
                    
                    # Update networks
                    self.update()
                    
                    state = next_state
                    episode_reward += reward

                all_rewards.append(episode_reward)
                csv_writer.writerow([i_episode, episode_reward])
                
                if i_episode % self.log_interval == 0 and i_episode != 0:
                    avg_reward = np.mean(all_rewards[-self.log_interval:])
                    print(f'Epi:{i_episode:05d} || Avg Reward: {avg_reward:.03f}')

