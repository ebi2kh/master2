import gym
from gym import spaces
import numpy as np
import scipy.stats as stats
from scipy.special import erfinv
import math
import torch

####################################
# START
####################################

# def error_prob(sinr, rate, packet_length):
#     """ This function calculates the error probability """
#     v = channel_desperation(sinr)

#     x = 0.5 * ((np.log2(1 + sinr) - rate) / np.sqrt(v / packet_length))

#     return Q(x)

#####################################
# END
####################################  


################################################
# ------------- Utility Methods -------------- # 
################################################

# **********
# BEGIN
# **********

# def path_loss(d, d0=1, PL_d0=0, n=2):
#     # DONE
#     """
#     Calculate the path loss using the Log-Distance Path Loss Model.
    
#     Parameters:
#     d (float): Distance at which path loss is calculated.
#     d0 (float): Reference distance (default is 1 meter).
#     PL_d0 (float): Path loss at reference distance d0 (default is -20 dB).
#     n (float): Path loss exponent (default is 2 for free space).
    
#     Returns:
#     float: Path loss at distance d.
#     """
#     return PL_d0 + 10 * n * np.log10(d / d0)

# def transmit_signal(k, n_parts, n_packets, powers):
#     # DONE
#     """
#     This function calculates the transmit signal for user-k 
#     Parameters:
#     powers: a vector of all allocated powers to each user
#     k: user id 
#     n_packets: n_packets
#     n_parts: number message parts
#     """
#     # The last user(User K)'s message is not splitted
#     K = len(powers) - 1 # This is the number of users in the environment
    
#     if k == K:
#         return np.sqrt(powers[k]) * n_packets
#     else:
#         start = k * n_parts

#         return np.sum(np.sqrt(powers[start: start + n_parts]) * n_packets) 

# **********
# END
# **********




""" Main Code Starts Here: """

# **********
# CONSTANTS
# **********
N_RIS_ELEMENTS = 16 # 16, 32, 64, ...

# DISTANCE_USER_BS = 300 # 300m = 300 meters






# Path-loss average channel power gain at reference distance d0 = 1m
L0 = 0.00001 # Unity channel gain(path-loss average channel gain at reference distance 1m) and its unit is in watts(the dBm is -20dBm)
ERROR_PROB = 0.00001 # This is the error probability

H_TILDA_USER_RIS = np.exp(1j * np.random.random(size = (N_RIS_ELEMENTS,1)) * 2 * np.pi)
H_TILDA_RIS_BS = np.exp(1j * np.random.random(size = (N_RIS_ELEMENTS,1)) * 2 * np.pi)





def channel_gain(distance_user_bs, phase_shift, K_1=8, K_2=8):
    """ This function calculates the channel gains of user-k ---> RIS and RIS ---> BS """
    # DONE

    # Consider that there is no direct path and
    # There's scattering(rayleigh fading) between user_k and BS
    # h_direct_nlos = h_direct_nlos
    h_user_bs =  np.sqrt(L0 * (distance_user_bs** -2.7 )) * np.random.normal(loc = 0, scale = np.sqrt(2) / 2, size=(1,2)).view(np.complex128) 

    # Add direct link to the equation
    complete_channel_gain = h_user_bs 

    return complete_channel_gain
 

def SINR(k, decoding_orders, powers, distance_user_ris, distance_ris_bs, distance_user_bs):
    # DONE
    """ This function calculates the SINR for user-k  
    powers ----> is a vector of all allocated powers to each user
    Transceiver is considered MISO(Multiple-Input Single-Output)
    """
    
    n_user_messages = len(decoding_orders)

    # Calculate the channel gain for user-k
    h = channel_gain(distance_user_ris, distance_ris_bs, distance_user_bs)
    # print("CHANNEL GAIN", h)

    # Scalar receive filter for user i (for simplicity, assume d_i = 1 for all users)
    d = np.ones(3)  # Assume the base station applies the same filter to all users messages
    
    # Initialize interference to zero
    interference = 0


    # Calculate the desired signal power for user k
    desired_signal = powers[k] * abs(h)**2

    # Calculate the interference from other users that their decoding orders are higher than the decoding order of the current stream
    for i in range(n_user_messages):
        if i != k and len(decoding_orders) != 1:
            # if decoding_orders[i] >= decoding_orders[k]:
                interference += powers[i] * abs(h)**2
    # print(k)    
    
    # print(decoding_orders)
    # decoding_orders.pop()
    


    ############################################
    # Generate AWGN noise (zero mean, 1 standard deviation)
    noise_variance =  db_to_linear(-174)
    ############################################
    # print("Desired Singal:" , desired_signal)

    # Calculate SINR for user i 
    sinr = desired_signal / (interference + noise_variance)

    return sinr


def db_to_linear(rho):
    return 10 ** (rho/10) # use the formula 10^(rho/10)


def channel_desperation(sinr):
    # DONE: Formula is checked
    """ This function calculates the channel desperation or V for each user-k """
    v = 1 - (1 + sinr) ** -2
    return v


# def Q(x):
#     # DONE
#     """ This function calculates the tail distribution of the standard normal distribution """
#     return stats.norm.sf(x)


def inverse_Q(epsilon = ERROR_PROB):
    # DONE
    """ This function calculates the inverse of Q-function which is tail distribution function of the standard normal distribution """
    
    return np.sqrt(2) * erfinv(2 * epsilon)


def rate(sinr):
    # DONE
    """ This function calculates rate for each user-k with URLLC condition"""
    packet_length = 250 # Candidate values could be 250bits, 500bits, 1500bits and 2500bits 
    inverse_Q_err = inverse_Q(epsilon = ERROR_PROB)
    v = channel_desperation(sinr)
    # print('First part: ', np.log2(1 + sinr))
    # print('Second part: ', inverse_Q_err * np.sqrt(v / packet_length))

    return 180000 * (np.log2(1 + sinr) - inverse_Q_err * np.sqrt(v / packet_length))

    
def age_of_information(t, r_user, aoi_default = 1):
    # DONE
    """ This function calculates the AoI of user-k at timeslot t
  
    Parameters:
    t: the timestep(seconds)
    r_user: Current rate of user-k
    packet_length: finite-blocklength size. Candidate values could be 250bits, 500bits, 1500bits and 2500bits 

    """
    packet_length = 250 # Candidate values could be 250bits, 500bits, 1500bits and 2500bits 
    r_min = packet_length / 1e-3 # delay is considered 1ms
    
    if t == 0 :
        aoi = aoi_default
    elif r_user >= r_min:
        aoi = 1
    else:
        aoi = age_of_information(t - 1, r_user) + 1

    return aoi


class RSMA_NO_RIS(gym.Env):
    """
    This class defines the environment for wireless communications model
    """
    def __init__(self, n_users=2):
        self.epilength = 20

        self.USER_RIS_DISTANCES = []
        self.USER_BS_DISTANCES = []
        
        # Base-station position is (0, 500), RIS position is (500, 500) and the users positions
        # are uniformly choosen(uniform distribution)
        for i in range(n_users):
            distance_x_i = np.random.uniform(0, 500)
            distance_y_i = np.random.uniform(0, 500)
            distance_user_ris = np.sqrt((distance_x_i - 500)**2 + (distance_y_i - 500)**2)
            distance_user_bs = np.sqrt((distance_x_i - 0)**2 + (distance_y_i - 500)**2)
            if i != n_users - 1:
                self.USER_RIS_DISTANCES.append(distance_user_ris)
                self.USER_RIS_DISTANCES.append(distance_user_ris)
                
                self.USER_BS_DISTANCES.append(distance_user_bs)
                self.USER_BS_DISTANCES.append(distance_user_bs)
            else:
                self.USER_RIS_DISTANCES.append(distance_user_ris)
                self.USER_BS_DISTANCES.append(distance_user_bs)

        #############################################
        # DONE: Section 1: Build the Action vector
        #############################################

        # Number of users and message parts
        self.n_users = n_users
        self.n_message_parts = (self.n_users - 1) * 2 + 1  # 1 users with 2 parts, 1 user with 1 part
        self.n_ris_elements = N_RIS_ELEMENTS  # Number of RIS reflective elements
        
        # Action Space Definition
        
        # Transmit powers: Continuous, one value per message part
        self.transmit_power_low = np.zeros(self.n_message_parts)
        self.transmit_power_high = np.ones(self.n_message_parts) * 0.63  # or define appropriate max power in watts
        
        # Decoding orders: Continuous between (0, 1) for each message part
        self.decoding_order_low = np.zeros(self.n_message_parts)
        self.decoding_order_high = np.ones(self.n_message_parts)
        
        # Phase shifts: Continuous between [0, 2π) for each RIS element
        # self.phase_shift_low = np.zeros(self.n_ris_elements)
        # self.phase_shift_high = np.ones(self.n_ris_elements) * 2 * np.pi
        
        # Combine continuous actions into a single action space
       

        # Combine all action spaces
        low = np.concatenate([self.transmit_power_low, self.decoding_order_low])
        high = np.concatenate([self.transmit_power_high, self.decoding_order_high])
        
        self.action_space = spaces.Box(low, high, dtype=np.float32)

        ######################################################
        # DONE: Section 2: Build the State(observation) vector
        ######################################################

        # Observation Space: AoI for each user
        # AoI is typically a non-negative integer, but we can define it as continuous for flexibility
        aoi_low = np.ones((self.n_users, ))  # Lower bound for AoI (0 for each user)
        aoi_high = np.ones((self.n_users, )) * 10  # Upper bound for AoI (no practical upper limit) 
        self.observation_space = spaces.Box(low=aoi_low, high=aoi_high, dtype=np.float64)

        # self.order = np.arange(n)
        self.step_count = None
        self.aoi = np.ones(self.n_users)
       
        # self.std_buf = None

    def step(self, action):
        # DONE: Finish the step function
        ############################
        # My Code
        ############################
        
        # Apply transformations
        # action = torch.cat([
        #     # softmax * pmax
        #     torch.softmax(sampled_action[:3], dim=0) * 0.1,  # First 3 dimensions ReLU (0 to pmax = 100mw = 0.1w)
        #     torch.softmax(sampled_action[3:6], dim = 0),             # Next 3 dimensions Sigmoid (0 to 1)
        #     torch.sigmoid(sampled_action[6:]) * 2 * torch.pi  # Last 16 dimensions (0 to 2π)
        # ], dim=-1)

        action = torch.tensor(action)

        # powers = torch.softmax(action[:self.n_message_parts], dim=0) * 0.1 # First 3 dimensions ReLU (0 to pmax = 100mw = 0.1w)
        # decoding_orders = torch.softmax(action[self.n_message_parts: 2* self.n_message_parts], dim = 0) # 3 different decoding orders

        # phase_shifts = torch.sigmoid(action[2* self.n_message_parts:]) * 2 * torch.pi  # Last 16 dimensions (0 to 2π)

        # powers = torch.softmax(action[:self.n_message_parts], dim=0) * 0.1 # First 3 dimensions ReLU (0 to pmax = 100mw = 0.1w)
        # decoding_orders = torch.softmax(action[self.n_message_parts: 2* self.n_message_parts], dim = 0) # 3 different decoding orders

        # phase_shifts = torch.sigmoid(action[2* self.n_message_parts:]) * 2 * torch.pi  # Last 16 dimensions (0 to 2π)


        # for i in range(0, len(action[:self.n_message_parts]), 2):
        #     if i != len(action[:self.n_message_parts]) - 1:
        #         if powers[i] + powers[i+1] > 0.63:
        # powers = []
        # for i in range(self.n_message_parts-1, 0, 2):
        #     powers.append(0.001 + (0.63 - 0.001)*torch.softmax(action[i:i+2], dim = 1))
        # powers.append(0.001 + (0.63 - 0.001)* action[-1])

        # powers = torch.tensor(powers)

        # powers = torch.clamp(action[:self.n_message_parts], 0.001, 0.63) # First 3 dimensions ReLU (0 to pmax = 100mw = 0.1w)

        action_powers = torch.clamp(action[:self.n_message_parts], 0.001, 0.63)
        powers = []
        for i in range(0, self.n_message_parts-1,2):
            # if i == self.n_message_parts
            # print(action_powers[i:i+2])
            # print('Hello', i)
            
            if torch.sum(action_powers[i:i+2], dim=-1) > 0.63:
                scale = 0.63 / torch.sum(action_powers[i:i+2], dim=-1)
                powers += list(scale * action_powers[i:i+2])
            else:
                powers += list(action_powers[i:i+2])

            # print('END', i)
        # print(powers)
        
        powers.append( action_powers[self.n_message_parts-1])
        powers = list(powers)


        # powers = torch.clamp(action[self.n_message_parts-1:self.n_message_parts], 0.001, 0.63) # First 3 dimensions ReLU (0 to pmax = 100mw = 0.1w)
        decoding_orders = torch.clamp(action[self.n_message_parts: ], 0, 1) # 3 different decoding orders

        # phase_shifts = torch.clamp(action[2* self.n_message_parts:],0, 2*np.pi)

        # print(powers)


        # powers = action[:self.n_message_parts]  
        # decoding_orders = action[self.n_message_parts: 2*self.n_message_parts]
        # phase_shifts = action[2*self.n_message_parts:]

        n_messages = len(decoding_orders)

        # Calculate SINRS for each user messages
        decoding_orders_real = list(decoding_orders)
        decoding_orders_copy = list(decoding_orders)
        SINRS = [0] * n_messages
        for _ in range(n_messages):
            min_idx_copy = np.argmin(decoding_orders_copy)
            min_idx = decoding_orders_real.index(decoding_orders_copy[min_idx_copy])


            SINRS[min_idx] = SINR(min_idx_copy, decoding_orders = decoding_orders_copy, powers = powers, distance_user_ris = self.USER_RIS_DISTANCES[min_idx], distance_ris_bs = 500, distance_user_bs = self.USER_BS_DISTANCES[min_idx])
            decoding_orders_copy.pop(min_idx_copy)

        
        # SINRS = [SINR(i, decoding_orders = decoding_orders_copy, powers = powers, distance_user_ris = self.USER_RIS_DISTANCES[i], distance_ris_bs = 500, distance_user_bs = self.USER_BS_DISTANCES[i], phase_shifts = phase_shifts) for i in range(n_messages)]
        
        # print(SINRS)

        # Calculate the rate
        # This for-loop is considered when the messages
        #  of all users(except K the last user) are splitted
        #  into even parts
        RATES = []
        for k in range(0, len(SINRS) + 1, 2):
            if k != len(SINRS) - 1:
                # DONE: CHECK RATES AGAIN!!!
                RATES.append(rate(SINRS[k+1]) + rate(SINRS[k]))
            else:
                RATES.append(rate(SINRS[k]))


        # print('RATES:', RATES)

        # Update AoI for each user
        if self.step_count > 0:
            self.aoi = np.array([age_of_information(self.step_count, r_user, aoi_default=self.aoi[i]) for i,r_user in enumerate(RATES)])

        # Define immediate reward
        reward = -1 * np.sum(self.aoi)
        # Update the step_count(time step)
        self.step_count += 1
        # state = np.zeros((self.n,))
        state = self.aoi
        if self.step_count==self.epilength:
            return state, reward, True, {}
        else:
            # state[self.order[self.step_count]] += 1.
            return state, reward, False, {}

    def reset(self):
        # DONE : Finish the reset function
        self.step_count = 0
        # self.std_buf = []
        # np.random.shuffle(self.order)
        # state[self.order[self.step_count]] += 1.
        # return state

        

        # Reset AoI for all users (e.g., assume initially all are fresh at AoI = 1)
        state = np.ones((self.n_users,))
        
        return state

    def close(self):
        return None

    def render(self, mode=None):
        return np.array(self.std_buf)
    


# env = WirelessEnv()

# print(env.action_space)

# for i in range(10):
#     env.reset()
#     sample = env.action_space.sample()
#     env.step(torch.tensor(sample))
#     # print(f'phase-shifts:\n{sample[6:]}')
#     # print(f'decoding-orders:\n{sample[3:6]}')
    
#     print(f'transmit_power:\n{sample[3:6]}')





# env = ToyEnv()
# observation = env.reset()
# while True:
#     print(observation)
#     action = env.action_space.sample()
#     print(action)
#     observation, reward, done, info = env.step(action)
#     print(reward)
#     if done:
#         break











