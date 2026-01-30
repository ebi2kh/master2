import gym
from gym import spaces
import numpy as np
import scipy.stats as stats
from scipy.special import erfinv
import math
import torch



""" Main Code Starts Here: """

# **********
# CONSTANTS
# **********
N_RIS_ELEMENTS = 16# 16, 32, 64, ...


# Path-loss average channel power gain at reference distance d0 = 1m
L0 = 0.00001 # Unity channel gain(path-loss average channel gain at reference distance 1m) and its unit is in watts(the dBm is -20dBm)
ERROR_PROB = 0.00001 # This is the error probability

# These are the fixed LoS components of the Rician Fading channels. They are constant throughout the simulation.
# We create a dictionary to hold these for each RIS.
H_TILDA_USER_RIS_DICT = {}
H_TILDA_RIS_BS_DICT = {}


def get_h_tilda(ris_idx):
    """ Helper to generate or retrieve constant LoS components for a given RIS """
    if ris_idx not in H_TILDA_USER_RIS_DICT:
        H_TILDA_USER_RIS_DICT[ris_idx] = np.exp(1j * np.random.random(size = (N_RIS_ELEMENTS,1)) * 2 * np.pi)
        H_TILDA_RIS_BS_DICT[ris_idx] = np.exp(1j * np.random.random(size = (N_RIS_ELEMENTS,1)) * 2 * np.pi)
    return H_TILDA_USER_RIS_DICT[ris_idx], H_TILDA_RIS_BS_DICT[ris_idx]


def calculate_ris_component_gain(distance_user_ris, distance_ris_bs, phase_shift, ris_idx, K_1=8, K_2=8):
    """ NEW: This function calculates the channel gain component for a SINGLE RIS path. """
    H_TILDA_USER_RIS, H_TILDA_RIS_BS = get_h_tilda(ris_idx)
    
    # --- User -> RIS ---
    delta_user_ris = np.sqrt(L0 * (distance_user_ris ** -2))
    h_bar_user_ris = np.random.normal(loc=0, scale=np.sqrt(2)/2, size=(len(phase_shift), 2)).view(np.complex128)
    h_hat_user_ris = np.sqrt(K_1 / (K_1 + 1)) * H_TILDA_USER_RIS + np.sqrt(1 / (K_1 + 1)) * h_bar_user_ris
    h_user_ris = h_hat_user_ris * delta_user_ris

    # --- RIS -> BS ---
    delta_ris_bs = np.sqrt(L0 * (distance_ris_bs ** -2))
    h_bar_ris_bs = np.random.normal(loc=0, scale=np.sqrt(2)/2, size=(len(phase_shift), 2)).view(np.complex128)
    h_hat_ris_bs = np.sqrt(K_2 / (K_2 + 1)) * H_TILDA_RIS_BS + np.sqrt(1 / (K_2 + 1)) * h_bar_ris_bs
    h_ris_bs = h_hat_ris_bs * delta_ris_bs

    phase_shift_diag = np.diag(np.cos(phase_shift) + 1j * np.sin(phase_shift))
    
    return h_user_ris.conj().T @ phase_shift_diag @ h_ris_bs


def compute_all_SINRs(decoding_orders, powers, channel_gains):
    """ This code calculates all SINRS based on Successive Interference Cancellation (SIC) """
    M = len(powers)
    SINRS = np.zeros(M)
    noise_variance = db_to_linear(-174)
    
    desired_signal = np.array(powers) * np.abs(channel_gains).reshape(-1)**2
    sorted_indices = np.argsort(decoding_orders)
    
    for i, idx in enumerate(sorted_indices):
        interference = np.sum(np.array(powers)[sorted_indices[i+1:]] * np.abs(np.array(channel_gains)[sorted_indices[i+1:]])**2) if i < M - 1 else 0
        
        decoding_err = np.random.uniform()
        if decoding_err <= ERROR_PROB:
            SINRS[sorted_indices[i:]] = 0 # All subsequent messages fail
            break
        SINRS[idx] = desired_signal[idx] / (interference + noise_variance)
    
    return SINRS


def db_to_linear(rho):
    return 10 ** (rho/10)


def channel_desperation(sinr):
    v = 1 - (1 + sinr) ** -2
    return v


def inverse_Q(epsilon = ERROR_PROB):
    return np.sqrt(2) * erfinv(2 * epsilon)


def rate(sinr):
    packet_length = 250
    inverse_Q_err = inverse_Q(epsilon = ERROR_PROB)
    v = channel_desperation(sinr)
    
    if 1 + sinr <= 0 or v < 0: return 0
        
    log_part = np.log2(1 + sinr)
    sqrt_part = np.sqrt(v / packet_length)
    calculated_rate = 180000 * (log_part - inverse_Q_err * sqrt_part)
    return max(0, calculated_rate)

    
def age_of_information(user_idx, current_time, generation_times, r_user):
    r_min = 250 / 1e-3

    if r_user >= r_min:
        aoi = 1
        generation_times[user_idx] = current_time
    else:
        aoi = current_time - generation_times[user_idx] + 1
    return aoi


class RSMA(gym.Env):
    """
    This class defines the environment for the wireless communications model with MULTIPLE MOBILE RISs.
    """
    def __init__(self, n_users=2, n_ris=2):
        self.epilength = 10
        self.n_users = n_users
        self.n_ris = n_ris

        # --- Define 3D positions for BS and Users ---
        self.bs_position = np.array([1000, 500, 20])
        self.user_positions = [np.array([np.random.uniform(0, 500), np.random.uniform(0, 500), 0]) for _ in range(n_users)]

        # --- NEW: Initialize multiple RIS positions randomly ---
        self.initial_ris_positions = [np.array([np.random.uniform(400, 600), np.random.uniform(400, 600), 20]) for _ in range(self.n_ris)]
        self.ris_positions = [pos.copy() for pos in self.initial_ris_positions]
        
        # --- UPDATED: Action and State space definitions for MULTI-RIS ---
        self.n_message_parts = (self.n_users - 1) * 2 + 1
        self.n_ris_elements = N_RIS_ELEMENTS

        # 1. Action Space Definition
        power_low = np.zeros(self.n_message_parts)
        power_high = np.ones(self.n_message_parts) * 0.63
        
        decoding_low = np.zeros(self.n_message_parts)
        decoding_high = np.ones(self.n_message_parts)
        
        # Phase shifts for ALL RISs
        phase_shift_low = np.zeros(self.n_ris * self.n_ris_elements)
        phase_shift_high = np.ones(self.n_ris * self.n_ris_elements) * 2 * np.pi
        
        # Y and Z coordinates for ALL RISs
        self.ris_y_range = [50, 600]
        self.ris_z_range = [5, 70]
        ris_pos_low = np.tile([self.ris_y_range[0], self.ris_z_range[0]], self.n_ris)
        ris_pos_high = np.tile([self.ris_y_range[1], self.ris_z_range[1]], self.n_ris)
        
        low = np.concatenate([power_low, decoding_low, phase_shift_low, ris_pos_low])
        high = np.concatenate([power_high, decoding_high, phase_shift_high, ris_pos_high])
        self.action_space = spaces.Box(low, high, dtype=np.float32)

        # 2. Observation Space Definition
        aoi_low = np.ones(self.n_users)
        aoi_high = np.ones(self.n_users) * 10
        
        # Concatenate AoIs and ALL RIS positions for the state vector
        obs_low = np.concatenate([aoi_low, ris_pos_low])
        obs_high = np.concatenate([aoi_high, ris_pos_high])
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float64)

        # Initialize state variables
        self.reset()

    def step(self, action):
        # --- UPDATED: Decode action for MULTI-RIS ---
        action = torch.sigmoid(torch.tensor(action))

        # Decode powers
        p_offset = 0
        action_powers = action[p_offset : p_offset + self.n_message_parts]
        powers = [0]* self.n_message_parts
        for i in range(0, len(action_powers), 2):
            if i != len(action_powers) - 1:
                powers[i] = action_powers[i] * 0.63
                powers[i+1] = (1 - action_powers[i]) * 0.63
            else:
                powers[i] = action_powers[i] * 0.63
        
        # Decode decoding orders
        d_offset = p_offset + self.n_message_parts
        decoding_orders = action[d_offset : d_offset + self.n_message_parts]

        # Decode phase shifts for each RIS
        ph_offset = d_offset + self.n_message_parts
        action_phases = action[ph_offset : ph_offset + self.n_ris * self.n_ris_elements]
        all_phase_shifts = np.array(action_phases).reshape(self.n_ris, self.n_ris_elements) * 2 * np.pi

        # Decode and update positions for each RIS
        pos_offset = ph_offset + self.n_ris * self.n_ris_elements
        action_ris_pos = action[pos_offset:]
        for r in range(self.n_ris):
            self.ris_positions[r][1] = action_ris_pos[r*2] * (self.ris_y_range[1] - self.ris_y_range[0]) + self.ris_y_range[0]
            self.ris_positions[r][2] = action_ris_pos[r*2+1] * (self.ris_z_range[1] - self.ris_z_range[0]) + self.ris_z_range[0]

        # --- UPDATED: Channel gain calculation for MULTI-RIS ---
        channel_gains = []
        for c in range(self.n_message_parts):
            user_idx = c // 2 if c < self.n_message_parts - 1 else self.n_users - 1
            user_pos = self.user_positions[user_idx]
            
            # Direct path gain
            dist_ub = np.linalg.norm(user_pos - self.bs_position)
            total_gain = np.sqrt(L0 * (dist_ub**-2.7)) * np.random.normal(0, np.sqrt(2)/2, (1,2)).view(np.complex128)
            
            # Sum of gains from all RIS paths
            for r in range(self.n_ris):
                dist_ur = np.linalg.norm(user_pos - self.ris_positions[r])
                dist_rb = np.linalg.norm(self.ris_positions[r] - self.bs_position)
                total_gain += calculate_ris_component_gain(dist_ur, dist_rb, all_phase_shifts[r], r)
            channel_gains.append(total_gain)
        
        # --- Simulation core (logic remains the same) ---
        SINRS = list(compute_all_SINRs(decoding_orders, powers, channel_gains))
        
        RATES = []
        for k in range(0, len(SINRS), 2):
            if k < len(SINRS) - 1:
                rate1 = rate(SINRS[k])
                rate2 = rate(SINRS[k+1]) if SINRS[k] > 0 else 0 # Simplified check
                RATES.append(rate1 + rate2)
            else:
                RATES.append(rate(SINRS[k]))

        self.step_count += 1
        self.aoi = np.array([age_of_information(i, self.step_count, self.generation_times, r_user) for i, r_user in enumerate(RATES)])
        reward = -1 * np.sum(self.aoi)
        
        # --- UPDATED: State includes all RIS positions ---
        ris_positions_flat = np.array(self.ris_positions)[:, 1:].flatten()
        state = np.concatenate([self.aoi, ris_positions_flat])
        done = self.step_count >= self.epilength
        
        return state, reward, done, {}

    def reset(self):
        self.step_count = 1
        self.aoi = np.ones(self.n_users)
        self.generation_times = np.ones(self.n_users)
        
        # --- UPDATED: Reset all RIS positions ---
        self.ris_positions = [pos.copy() for pos in self.initial_ris_positions]
        
        # --- UPDATED: Initial state includes all RIS positions ---
        initial_aoi = np.ones(self.n_users)
        initial_ris_pos_flat = np.array(self.ris_positions)[:, 1:].flatten()
        state = np.concatenate([initial_aoi, initial_ris_pos_flat])
        
        return state

    def close(self):
        pass

    def render(self, mode=None):
        pass
