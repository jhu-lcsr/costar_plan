import numpy as np
import tensorflow as tf
import time
import os
import math
import cv2
from keras import backend as K
from keras.models import model_from_json, Model
import json

from actor_network import ActorNetwork
from critic_network import CriticNetwork
from abstract import AbstractAgent

from memory import Memory
from grapher import Grapher
from ou_process import OUProcess

print(tf.__version__)

# references
# https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html
# https://arxiv.org/pdf/1509.02971.pdf

#===========================================================
# User Settings
#===========================================================

WHICH_GPU = "0"                 # Set "0" or "1" for either GPU, or "0,1" for both.

#display parameters
MAKE_PLOT = True                # Display and save a reward plot
MAKE_DISPLAY = True             # Display the environment

#save/load params
SAVE_WEIGHTS_PREFIX = "./latest_data_gpu2/"
LOAD_WEIGHTS = False
LOAD_WEIGHTS_PREFIX = "./latest_data_gpu2/"
LOAD_WEIGHTS_EPISODE = "3000"
STARTING_EPISODE = 0
ALREADY_TRAINED = False         # Set to true if we only want to demo the agent
EPISODE_SAVE_FREQUENCY = 1000

#training params
EPISODES = 10000000             # Total number of episodes, often manually terminated before this
PRE_LEARNING_EPS = 3            # Number of episodes spent initializing memory
EPISODE_LENGTH = 150            # Step length of each episode
TEST_EPISODE_LENGTH = 300       # Step length of a validaiton episode
BATCH_SIZE = 32                 # Batch size of experience to be learned each step
GAMMA = 0.99                    # Discount facotr in critic network
TAU = 0.005                     # Target network update rate
LRA = 0.0001                    # Learning rate for Actor
LRC = 0.001                     # Lerning rate for Critic
MEM_SIZE_FCL = 2000000          # Memory size if operating on simple float vector
MEM_SIZE_CONVOLUTIONAL = 500000 # Memory size if need to store images as states

#exploration params
EPSILON_RANGE = [1.0, 0.2]      # Epsilon initial and final values
EPSILON_STEPS = 1200000         # Frames over which to anneal epsilon
OU_MEAN = 0.0                   # OU exploration noise mean - see OUProcess.py
OU_THETA = 0.15                  # OU exploration noise theta - see OUProcess.py
OU_STD_DEV = 0.3               # OU exploration noise sigma - see OUProcess.py

#architecture & env params
ACTION_ACTIVATION = 'tanh'       # Controls range of actions. True = 'tanh' = [-1,1], False = 'sigmoid' = [0,1]
USE_ACCEL_CONTROL = False           # Control agents' acceleration instead of velocity
INCLUDE_VELOCITY_IN_STATE = False   # Include agent vx, vy in state (in addition to x, y)
PENALIZE_FOR_DISTRIBUTION = True   # Penalize agents for grouping over a single target
NUM_AGENTS = 2                      # Number of agents (all controlled by one network)
NUM_TARGETS = 2                     # Number of stationary targets
CONVOLUTIONAL = False               # Operate on images of the environment instead of float vectors
DRAW_STATE = True                 # This file should re-draw an image based on the state space
ENV_OUTPUTS_IMAGE = False           # Control if ENV outputs state-vector or state image.
NUM_NOISE_PROCS = 7
#===========================================================
# Other Constants
#===========================================================
ACTIONS_PER_AGENT = 2
IMAGE_SIDE_LENGTH = 100


class APLDDPGAgent(AbstractAgent):
    
    name = "apl_ddpg"

    def __init__(self, env, iter=200000, *args, **kwargs):
        # create the actor model
        # create the critic model
        self.env = env
        self.action_dim = sum(sum(1 for i in row if i) for row in self.env.action_space.sample())
        self.observation = env.reset()
        self.state_dim = self.observation.shape
        self.nn_action_dim = 3 # limit ddpg network output to 3 DOF
        
        
        
    def fit(self):

        MEM_SZ = MEM_SIZE_FCL
        
        
        sess = K.get_session()
        K.set_learning_phase(1)

    
        self.actor = ActorNetwork(sess, self.state_dim, self.nn_action_dim, BATCH_SIZE, TAU, LRA, convolutional=CONVOLUTIONAL, output_activation=ACTION_ACTIVATION)
        self.critic = CriticNetwork(sess, self.state_dim, self.nn_action_dim, BATCH_SIZE, TAU, LRC, convolutional=CONVOLUTIONAL)
    
        self.memory = Memory(MEM_SZ)
    
        self.actor.target_model.summary()
        self.critic.target_model.summary()
    
        if LOAD_WEIGHTS:
            self.actor.model.load_weights(LOAD_WEIGHTS_PREFIX + "actor_model_" + LOAD_WEIGHTS_EPISODE + ".h5")
            self.critic.model.load_weights(LOAD_WEIGHTS_PREFIX + "critic_model_" + LOAD_WEIGHTS_EPISODE + ".h5")
            self.actor.target_model.load_weights(LOAD_WEIGHTS_PREFIX + "actor_target_model_" + LOAD_WEIGHTS_EPISODE + ".h5")
            self.critic.target_model.load_weights(LOAD_WEIGHTS_PREFIX + "critic_target_model_" + LOAD_WEIGHTS_EPISODE + ".h5")
            print("Weights Loaded!")
    
    
        #====================================================
        #Initialize noise processes
        self.noise_procs = []
        for i in range(NUM_NOISE_PROCS):
            self.noise_procs.append(OUProcess(OU_MEAN, OU_THETA, OU_STD_DEV))
        
        #====================================================
        
        PRE_LEARNING_EPISODES = STARTING_EPISODE+PRE_LEARNING_EPS
        steps = STARTING_EPISODE*EPISODE_LENGTH
        start_time = time.time()
        last_ep_time = time.time()
        if MAKE_PLOT:
            reward_graph = Grapher()
        
        
        for ep in range(STARTING_EPISODE, EPISODES):
        
            #reset noise processes
            for ou in self.noise_procs:
                ou.reset()
        
            #start time counter
            if(ep == PRE_LEARNING_EPISODES):
                start_time = time.time()
        
            print("Episode: " + str(ep) + "  Frames: " + str(ep*EPISODE_LENGTH) + "  Uptime: " + str((time.time()-start_time)/3600.0) + " hrs    ===========")
        
            state = self.env.reset()
            agentwise_states = state
            
           
        
            play_only = (ep%10 == 0)
        
            total_reward = 0
        
            if play_only or ALREADY_TRAINED:
                for step in range(TEST_EPISODE_LENGTH):
        
                    action, control_action = self.selectAction(state, can_be_random=False, use_target=True)
                    
                    nstate, reward, done, info = self.env.step(control_action)
                    total_reward += reward
                    state = nstate
            else: 
                for step in range(EPISODE_LENGTH):
        
                        # ACT ==============================
                        epsilon = (float(steps)/float(EPSILON_STEPS))*(EPSILON_RANGE[1]-EPSILON_RANGE[0]) + EPSILON_RANGE[0]
                        
                   
                        action, control_action = self.selectAction(state, epsilon=epsilon)
                        new_state, reward, done, info = self.env.step(control_action)
                        done = (step>=EPISODE_LENGTH)
                        self.memory.addMemory(state, action, reward, new_state, done)
                        state = new_state
    
                        # LEARN ============================
                        if ep > PRE_LEARNING_EPISODES:
                            batch, idxs = self.memory.getMiniBatch(BATCH_SIZE)
                            self.learnFromBatch(batch)
        
                        # CLEANUP ==========================
                        steps += 1
        
            #we need to consider the episodes without noise to actually tell how the system is doing
            if play_only and MAKE_PLOT:
                reward_graph.addSample(total_reward)
                reward_graph.displayPlot()
        
            #calculate fph on total frames
            total_frames = (ep - PRE_LEARNING_EPISODES)*EPISODE_LENGTH
            elapsed = time.time() - start_time
            fps = total_frames/elapsed
            fph = fps*3600.0
        
            #re-calculate fps on this episode, so it updates quickly
            fps = EPISODE_LENGTH/(time.time() - last_ep_time)
            last_ep_time = time.time()
            print("fps: " + str(fps) + "  fph: " + str(fph) + "\n")
        
            #save plot and weights
            if (ep>0 and ep%EPISODE_SAVE_FREQUENCY==0) and not ALREADY_TRAINED:
        
                #plot
                if MAKE_PLOT:
                    reward_graph.savePlot(SAVE_WEIGHTS_PREFIX+"graph_"+str(ep)+".jpg")
        
                #weights
                self.actor.model.save_weights(SAVE_WEIGHTS_PREFIX+"actor_model_"+str(ep)+".h5", overwrite=True)
                self.actor.target_model.save_weights(SAVE_WEIGHTS_PREFIX+"actor_target_model_"+str(ep)+".h5", overwrite=True)
                self.critic.model.save_weights(SAVE_WEIGHTS_PREFIX+"critic_model_"+str(ep)+".h5", overwrite=True)
                self.critic.target_model.save_weights(SAVE_WEIGHTS_PREFIX+"critic_target_model_"+str(ep)+".h5", overwrite=True)
        
                #network structures (although I don't think I ever actually use these)
                with open(SAVE_WEIGHTS_PREFIX+"actor_model_"+str(ep)+".json", "w") as outfile:
                    json.dump(self.actor.model.to_json(), outfile)
                with open(SAVE_WEIGHTS_PREFIX+"actor_target_model_"+str(ep)+".json", "w") as outfile:
                    json.dump(self.actor.target_model.to_json(), outfile)
                with open(SAVE_WEIGHTS_PREFIX+"critic_model_"+str(ep)+".json", "w") as outfile:
                    json.dump(self.critic.model.to_json(), outfile)
                with open(SAVE_WEIGHTS_PREFIX+"critic_target_model_"+str(ep)+".json", "w") as outfile:
                    json.dump(self.critic.target_model.to_json(), outfile)


    def learnFromBatch(self, miniBatch):
    
        dones = np.asarray([sample['isFinal'] for sample in miniBatch])
        states = np.asarray([sample['state'] for sample in miniBatch])
        actions = np.asarray([sample['action'] for sample in miniBatch])
        new_states = np.asarray([sample['newState'] for sample in miniBatch])
        Y_batch = np.asarray([sample['reward'] for sample in miniBatch])
    
        target_q_values = self.critic.target_model.predict([new_states, self.actor.target_model.predict(new_states)])  
    
        for i in range(len(miniBatch)):
            if not dones[i]:
                Y_batch[i] = Y_batch[i] + GAMMA*target_q_values[i]
    
        self.critic.model.train_on_batch([states, actions], Y_batch)
    
        #additional operations to train actor
        temp_actions = self.actor.model.predict(states)
        grads = self.critic.gradients(states, temp_actions)
        self.actor.train(states, grads)
    
        #update target networks
        self.actor.target_train()
        self.critic.target_train()
    
    ''' This is wrong I think
    def OU(x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)
    '''
    
    def clip(self, x, minx, maxx):
        return max(minx, min(maxx, x))
    
    def selectAction(self, state, can_be_random=True, use_target=False, epsilon=1.0, permutation_num=0):
        state = np.array([state]) #add dimension to make a "batch" of 1
        
        if use_target:
            actions = self.actor.target_model.predict(state)
        else:
            actions = self.actor.model.predict(state)
    
        actions = np.squeeze(actions)
        
        #fill in zeros for all non-learned outputs
        control_actions = np.pad(actions, (0, self.action_dim-len(actions)), 'constant')
        #print control_actions
    
        #print("+++++++++++")
        #print(actions)
        i=0
        if can_be_random:
    
            #get noise
            noise = []
            #iterate over all noise procs for non-coop, or a single agent's procs for co-op
            #for n in range(permutation_num*ACTIONS_PER_AGENT, permutation_num*ACTIONS_PER_AGENT + self.action_dim):
            #    ou = self.noise_procs[n]
            #    noise.append(ou.step())
    
            for idx, a in enumerate(actions):
                ou = self.noise_procs[idx]
                noise = ou.step()                
                a = a + epsilon*noise
                #print epsilon * noise
                actions[i] = self.clip(a, -3.14, 3.14) #need to assign to actions[i], not just a.
                i += 1
        
        #print(actions)
        return actions, control_actions
    
    #Constructs an image from state vector
    def constructImageRepresentation(self, state):
        img = np.empty([IMAGE_SIDE_LENGTH, IMAGE_SIDE_LENGTH], dtype=np.uint8)
        img.fill(128)
    
        color = 255
        delta_color = int(math.floor(128/NUM_TARGETS))
        for j in range(NUM_TARGETS):
            tar = [state[2*j], state[2*j + 1]]
            cv2.circle(img, (int(tar[0]*IMAGE_SIDE_LENGTH), int(tar[1]*IMAGE_SIDE_LENGTH)), 5, 0, -1)
            cv2.circle(img, (int(tar[0]*IMAGE_SIDE_LENGTH), int(tar[1]*IMAGE_SIDE_LENGTH)), 4, color, -1)
            color -= delta_color
    
        color = 0
        for j in range(NUM_AGENTS):
            offset = 2*NUM_TARGETS
            agent = [state[offset + 2*j], state[offset + 2*j + 1]]
            #draw blank agent, no thrust display
            cv2.rectangle(img, (int(agent[0]*IMAGE_SIDE_LENGTH)-4, int(agent[1]*IMAGE_SIDE_LENGTH)-1), (int(agent[0]*IMAGE_SIDE_LENGTH)+4, int(agent[1]*IMAGE_SIDE_LENGTH)+1), color, -1)
            cv2.rectangle(img, (int(agent[0]*IMAGE_SIDE_LENGTH)-1, int(agent[1]*IMAGE_SIDE_LENGTH)-4), (int(agent[0]*IMAGE_SIDE_LENGTH)+1, int(agent[1]*IMAGE_SIDE_LENGTH)+4), color, -1)
            #first agent ia 0 since we control it, others are same color
            color = 64
    
        '''
        cv2.namedWindow('perm_image',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('perm_image', 600,600)
        cv2.imshow('perm_image', img)
        cv2.waitKey(1)
        '''
    
        img = np.array([np.subtract(img, 128)], dtype=np.float32) #zero center
        img = np.multiply(img, 1.0/128.0) #scale [-1,1]
        img = np.transpose(img, (1,2,0))
    
        return img
    
    #for co-op case, get an arrangement of the state vector for each agent.
    def getStatePermutations(self, state):
        perms = []
        for i in range(NUM_AGENTS):
    
            if CONVOLUTIONAL and not DRAW_STATE:
                perms.append(state)
            else:
                pstate = []
    
                #copy over target data
                for j in range(NUM_TARGETS*2):
                    pstate.append(state[j])
    
                #copy agent data, rotated
                for j in range(NUM_AGENTS*2):
                    rot_j = (j+(i*2))%(NUM_AGENTS*2) + (NUM_TARGETS*2)
                    pstate.append(state[rot_j])
    
                if DRAW_STATE:
                    perms.append(constructImageRepresentation(pstate))
                else:
                    perms.append(np.asarray(pstate, dtype=np.float32))
    
        return perms
    
   