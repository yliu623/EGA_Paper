"""
Deep Deterministic Policy Gradient (DDPG) class.
This class implements the DDPG algorithm for deep reinforcement learning. It inherits methods from the DRL class.
Attributes:
    env (object): The environment object for the RL task.
    dynamic (bool): Flag indicating whether the environment is dynamic or not.
    memory (list): List to store the experiences for experience replay.
    filename (str): Filename based on hyperparameters for saving models, history, and plots.
    sess (object): The current Keras session.
    TAU (float): Update rate for the target model.
    actor_lr (float): Learning rate for the actor.
    critic_lr (float): Learning rate for the critic.
    depth_nn_hidden (int): Number of hidden layers in the neural network.
    actor (object): The actor model.
    critic_Q_ex (object): The critic model for the first Q-value.
    critic_Q_ex2 (object): The critic model for the second Q-value.
    critic_Q (object): The critic model for the Q-value.
    actor_hat (object): The target actor model.
    critic_Q_ex_hat (object): The target critic model for the first Q-value.
    critic_Q_ex2_hat (object): The target critic model for the second Q-value.
    epsilon (float): Epsilon value for epsilon-greedy exploration.
    epsilon_decay (float): Decay rate for epsilon.
    epsilon_min (float): Minimum epsilon value.
    buffer_size (int): Size of the replay buffer.
    prioritized_replay_alpha (float): Alpha value for prioritized replay.
    replay_buffer (object): The replay buffer object.
    prioritized_replay_beta0 (float): Initial beta value for prioritized replay.
    prioritized_replay_beta_iters (int): Number of iterations to linearly anneal beta.
    beta_schedule (object): The schedule for beta annealing.
    prioritized_replay_eps (float): Small value for numerical stability.
    batch_size (int): Batch size for training.
    get_critic_grad (function): Function to compute the gradient of the critic.
"""

import os
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from keras.layers import Input, Dense, Lambda, concatenate, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from drl import DRL
from myenvs import TradingEnv
from replay_buffer import PrioritizedReplayBuffer
from schedules import LinearSchedule
import argparse



class DDPG(DRL): # DDPG() inherits class DRL's methods
    """
    Deep Deterministic Policy Gradient
    """

    def __init__(self, env, dynamic=False):
        # Call the parent class (DRL) constructor
        super(DDPG, self).__init__()
        
        # Initialize dynamic attribute
        self.dynamic = dynamic
        
        # Initialize memory as an empty list
        self.memory = []
        
        # Create a filename based on various hyperparameters
        self.filename = '_alr' + str(args.actor_lr) + '_clr' + str(args.critic_lr) + '_e' + str(args.epsilon_decay) + \
                        '_em' + str(args.epsilon_min) + '_tau' + str(args.TAU) + '_buf' + str(args.buffer_size) + \
                        '_b' + str(args.prioritized_replay_beta0) + '_ba' + str(args.batch_size) + '_n' + str(args.num_sim) + \
                        '_a' + str(args.prioritized_replay_alpha) + '_tr' + str(args.n_train) + '_te' + str(args.n_test) + \
                        '_nn' + str(args.depth_nn_hidden)
        
        # Create directories for saving models, history, and plots if they don't exist
        if not os.path.exists('model/'+'model'+self.filename):
            os.mkdir('model/'+'model'+self.filename)
        if not os.path.exists('history'):
            os.mkdir('history')
        if not os.path.exists('plots'):
            os.mkdir('plots')
        
        # Get the current Keras session
        self.sess = K.get_session()

        self.env = env
        self.upper_bound = self.env.action_space.high[0]
        self.lower_bound = self.env.action_space.low[0]

        # update rate for target model.
        self.TAU = args.TAU

        # learning rate for actor and critic
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.ra_c = 0 
        self.depth_nn_hidden = args.depth_nn_hidden
        self.actor = self._build_actor(learning_rate=self.actor_lr)
        self.critic_Q_ex, self.critic_Q_ex2, self.critic_Q = self._build_critic(learning_rate=self.critic_lr)
        self.critic_Q.summary()

        # target networks for actor and three critics
        self.actor_hat = self._build_actor(learning_rate=self.actor_lr)
        self.actor_hat.set_weights(self.actor.get_weights())

        self.critic_Q_ex_hat, self.critic_Q_ex2_hat, self.critic_Q_hat = self._build_critic(learning_rate=self.critic_lr)
        self.critic_Q_ex_hat.set_weights(self.critic_Q_ex.get_weights())
        self.critic_Q_ex2_hat.set_weights(self.critic_Q_ex2.get_weights())

        # epsilon of epsilon-greedy
        self.epsilon = 1.0

        # discount rate for epsilon
        self.epsilon_decay = args.epsilon_decay

        # min epsilon of epsilon-greedy.
        self.epsilon_min = args.epsilon_min

        # memory buffer for experience replay
        self.buffer_size = args.buffer_size
        self.prioritized_replay_alpha = args.prioritized_replay_alpha
        self.replay_buffer = PrioritizedReplayBuffer(self.buffer_size, alpha=self.prioritized_replay_alpha)
        self.prioritized_replay_beta0 = args.prioritized_replay_beta0
        self.prioritized_replay_beta_iters = args.prioritized_replay_beta_iters

        self.beta_schedule = LinearSchedule(self.prioritized_replay_beta_iters,
                                       initial_p=self.prioritized_replay_beta0,
                                       final_p=1.0)

        # for numerical stabiligy
        self.prioritized_replay_eps = 1e-6

        self.t = None

        # memory sample batch size
        self.batch_size = args.batch_size

        # may use for 2nd round training
        # self.policy_noise = 5000
        # self.noise_clip = 5*5000

        # gradient function
        self.get_critic_grad = self.critic_gradient()
        self.actor_optimizer()

    def load(self, tag=""):
        """
        Load the saved model weights for the actor and critic networks.
        If a tag is provided, load the model weights with the corresponding tag.
        """
        if tag == "":
            actor_file = "model/"+'model'+self.filename+"/"+"_ddpg_actor.h5"
            critic_Q_ex_file = "model/"+'model'+self.filename+"/"+"_ddpg_critic_Q_ex.h5"
            critic_Q_ex2_file = "model/"+'model'+self.filename+"/"+"_ddpg_critic_Q_ex2.h5"
        else:
            actor_file = "model/"+'model'+self.filename+"/"+"_ddpg_actor_" + tag + ".h5"
            critic_Q_ex_file = "model/"+'model'+self.filename+"/"+"_ddpg_critic_Q_ex_" + tag + ".h5"
            critic_Q_ex2_file = "model/"+'model'+self.filename+"/"+"_ddpg_critic_Q_ex2_" + tag + ".h5"

        if os.path.exists(actor_file):
            self.actor.load_weights(actor_file)
            self.actor_hat.load_weights(actor_file)
        if os.path.exists(critic_Q_ex_file):
            self.critic_Q_ex.load_weights(critic_Q_ex_file)
            self.critic_Q_ex_hat.load_weights(critic_Q_ex_file)
        if os.path.exists(critic_Q_ex2_file):
            self.critic_Q_ex2.load_weights(critic_Q_ex2_file)
            self.critic_Q_ex2_hat.load_weights(critic_Q_ex2_file)
        print('...model loaded successfully...')

    # the following pseudo code outline the actor network
    # Return the built actor model
    Function _build_actor(learning_rate):
    # Initialize input layer with shape based on the environment's state size
    inputs = InputLayer(shape=(environment_state_size))

    # Define layers based on the value of depth_nn_hidden
    If depth_nn_hidden >= 2:
        Apply batch normalization to inputs
        Add a dense layer with 64 units and ReLU activation
        Apply batch normalization
        Add a dense layer with 32 units and ReLU activation
        Add a dense output layer with 4 units and sigmoid activation

    If depth_nn_hidden >= 3:
        Apply batch normalization to inputs
        Add a dense layer with 64 units and ReLU activation
        Apply batch normalization
        Add a dense layer with 32 units and ReLU activation
        Apply batch normalization
        Add a dense layer with 16 units and ReLU activation
        Add a dense output layer with 4 units and sigmoid activation

    If depth_nn_hidden >= 4:
        Apply batch normalization to inputs
        Add a dense layer with 128 units and ReLU activation
        Apply batch normalization
        Add a dense layer with 64 units and ReLU activation
        Apply batch normalization
        Add a dense layer with 32 units and ReLU activation
        Apply batch normalization
        Add a dense layer with 16 units and ReLU activation
        Add a dense output layer with 4 units and sigmoid activation

    If depth_nn_hidden >= 5:
        Apply batch normalization to inputs
        Add a dense layer with 128 units and ReLU activation
        Apply batch normalization
        Add a dense layer with 64 units and ReLU activation
        Apply batch normalization
        Add a dense layer with 32 units and ReLU activation
        Add a dense output layer with 4 units and sigmoid activation

    If depth_nn_hidden >= 6:
        Apply batch normalization to inputs
        Add a dense layer with 256 units and ReLU activation
        Apply batch normalization
        Add a dense layer with 128 units and ReLU activation
        Apply batch normalization
        Add a dense layer with 64 units and ReLU activation
        Apply batch normalization
        Add a dense layer with 32 units and ReLU activation
        Add a dense output layer with 4 units and sigmoid activation

    If depth_nn_hidden >= 7:
        Apply batch normalization to inputs
        Add a dense layer with 512 units and ReLU activation
        Apply batch normalization
        Add a dense layer with 256 units and ReLU activation
        Apply batch normalization
        Add a dense layer with 128 units and ReLU activation
        Apply batch normalization
        Add a dense layer with 64 units and ReLU activation
        Apply batch normalization
        Add a dense layer with 32 units and ReLU activation
        Add a dense output layer with 4 units and sigmoid activation

    # Define actions with constraints
    action1 = Scale and clip the first output based on inputs
    action2 = Scale and clip the second output based on inputs and action1
    action3 = Scale and clip the third output based on inputs and action1
    action4 = Scale and clip the fourth output based on inputs and action2

    # Concatenate the action outputs into a single output layer
    output = Concatenate(action1, action2, action3, action4)

    # Create the model with inputs and the concatenated output
    model = CreateModel(inputs, output)

    # Compile the model with mean squared error loss and Adam optimizer
    model.Compile(loss="mse", optimizer=Adam(learning_rate))

    Return the compiled model

    # the following pseudo code outline the  critic network         
    Function _build_critic(learning_rate):
        # Initialize input layers
        s_inputs = InputLayer with shape equal to the environment's state size
        a_inputs = InputLayer with shape equal to 4 (for 4 actions)
    
        # Combine the state and action inputs
        x = Concatenate s_inputs and a_inputs
    
        # Apply batch normalization to the combined inputs
        x = Apply BatchNormalization to x
    
        # Define the Q_ex network
        x1 = Dense layer with 32 units and ReLU activation applied to x
        x1 = Apply BatchNormalization to x1
    
        x1 = Dense layer with 64 units and ReLU activation applied to x1
        x1 = Apply BatchNormalization to x1
    
        # Create the output layer for Q_ex without batch normalization
        output1 = Dense layer with 1 unit and linear activation applied to x1
    
        # Create the model_Q_ex using s_inputs, a_inputs as inputs and output1 as output
        model_Q_ex = CreateModel(inputs=[s_inputs, a_inputs], outputs=output1)
        Compile model_Q_ex with mean squared error loss and Adam optimizer
    
        # Define the Q_ex2 network
        x2 = Dense layer with 32 units and ReLU activation applied to x
        x2 = Apply BatchNormalization to x2
    
        x2 = Dense layer with 64 units and ReLU activation applied to x2
        x2 = Apply BatchNormalization to x2
    
        # Create the output layer for Q_ex2 without batch normalization
        output2 = Dense layer with 1 unit and linear activation applied to x2
    
        # Create the model_Q_ex2 using s_inputs, a_inputs as inputs and output2 as output
        model_Q_ex2 = CreateModel(inputs=[s_inputs, a_inputs], outputs=output2)
        Compile model_Q_ex2 with mean squared error loss and Adam optimizer
    
        # Define the Q network output using a Lambda function
        output3 = Lambda function that calculates the difference between output1 and the square root of the maximum of (output2 - output1 squared)
    
        # Create the model_Q using s_inputs, a_inputs as inputs and output3 as output
        model_Q = CreateModel(inputs=[s_inputs, a_inputs], outputs=output3)
        Compile model_Q with mean squared error loss and Adam optimizer
    
        # Return the three models: model_Q_ex, model_Q_ex2, and model_Q
        Return model_Q_ex, model_Q_ex2, model_Q


    def actor_optimizer(self):
        """Defines the actor optimizer function.
        Returns:
            function: The optimizer function for the actor.
        """
        self.ainput = self.actor.input
        aoutput = self.actor.output
        trainable_weights = self.actor.trainable_weights
        #self.action_gradient = tf.placeholder(tf.float32, shape=(None, 1))
        self.action_gradient = tf.compat.v1.placeholder(tf.float32, shape=(None, 4))
        # tf.gradients calculates dy/dx with a initial gradients for y
        #tf.gradients is not supported when eager execution is enabled. Use tf.GradientTape instead.
        # action_gradient is dq/da, so this is dq/da * da/dparams
        params_grad = tf.gradients(aoutput, trainable_weights, -self.action_gradient)
        grads = zip(params_grad, trainable_weights)
        #self.opt = tf.train.AdamOptimizer(self.actor_lr).apply_gradients(grads)
        self.opt = tf.optimizers.Adam(self.actor_lr).apply_gradients(grads)
        # print('self.opt', self.opt)
        #self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def critic_gradient(self):
        """get critic gradient function.
        Returns:
            function, gradient function for critic.
        """
        cinput = self.critic_Q.input  # Define the input layer for the critic network
        coutput = self.critic_Q.output  # Define the output layer for the critic network

        # compute the gradient of the action with respect to the Q value, dq/da.
        action_grads = K.gradients(coutput, cinput[1])  # Calculate the gradients of the output with respect to the action input

        return K.function([cinput[0], cinput[1]], action_grads)  # Return the function that computes the gradient of the action with respect to the Q value

    def egreedy_action(self, X):
        """Get actor action with OU noise.
        Arguments:
            X: state value.
        """
        # Do the epsilon greedy way
        if np.random.rand() <= self.epsilon:
            
            # Define a function to get available actions based on state X
            def get_available_actions(X):
                s1 = X[0,5]
                s2 = X[0,6]
                q = 98107
                action = [0,0,0,0]
                
                # Generate random actions within the constraints
                action[0] = np.random.uniform(0, np.minimum(s1, q)) # y1
                action[1] = np.random.uniform(0, np.minimum(s2, (q - action[0]))) # y2
                action[2] = np.random.uniform(0, s1-action[0]) # z1
                action[3] = np.random.uniform(0, s2-action[1]) # z2
                return action, s1, s2
            
            # Get the action and state values
            action, s1, s2 = get_available_actions(X)
            action_sampled = 1
            
            # The following code is commented out but can be used for a second round of training
            # action = self.actor.predict(X)[0]
            # noise = np.clip(np.random.normal(0, self.policy_noise, 4), -self.noise_clip, self.noise_clip)
            # action[0] = np.clip(action[0] + noise[0], 0, np.minimum(s1, q))
            # action[1] = np.clip(action[1] + noise[1], 0, np.minimum(s2, (q - action[0])))
            # action[2] = np.clip(action[2] + noise[2], 0, s1-action[0])
            # action[3] = np.clip(action[3] + noise[3], 0, s2-action[1])
    
        else:
            # Predict action using the actor model
            action = self.actor.predict(X)[0]
            action_sampled = 0
        
        # Return the action and other values (None in this case)
        return action, None, None, action_sampled

    def update_epsilon(self):
        """Update the epsilon value for the epsilon-greedy policy."""
        # Check if epsilon is greater than or equal to the minimum epsilon value
        if self.epsilon >= self.epsilon_min:
            # Decay epsilon by multiplying it with the epsilon decay factor
            self.epsilon *= self.epsilon_decay
    
    def remember(self, state, action, reward, next_state, done):
        """Add data to experience replay.
        Arguments:
            state: observation
            action: action
            reward: reward
            next_state: next_observation
            done: if game is done.
        """
        # Add the experience tuple to the replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)

    def process_batch(self, batch_size):
        """Process batch data.
        Arguments:
            batch_size: size of the batch.
        Returns:
            states: batch of states.
            actions: batch of actions.
            target_q_ex, target_q_ex2: batch of targets.
            weights: priority weights.
        """
        # Prioritized sample from experience replay buffer
        experience = self.replay_buffer.sample(batch_size, beta=self.beta_schedule.value(self.t))
        (states, actions, rewards, next_states, dones, weights, batch_idxes) = experience

        # Check for NaN values in states and reset them to 0
        if np.isnan(states).any():
            print('states has nan', states)
            states[np.isnan(states)] = 0
            print('states has nan reset', states)
        
        # Check for NaN values in actions and reset them to 0
        if np.isnan(actions).any():
            print('actions has nan', actions)
            actions[np.isnan(actions)] = 0
            print('actions has nan reset', actions)
        
        # Check for NaN values in rewards and reset them to 0
        if np.isnan(rewards).any():
            print('rewards', rewards)
            rewards[np.isnan(rewards)] = 0
            print('rewards reset', rewards)
        
        # Check for NaN values in next_states and reset them to 0
        if np.isnan(next_states).any():
            print('next states has nan', next_states)
            next_states[np.isnan(next_states)] = 0
            print('next_states reset', next_states)
        
        # Reshape rewards and dones for further processing
        rewards = rewards.reshape(-1, 1)
        dones = dones.reshape(-1, 1)

        # Get next actions using the target actor network
        next_actions = self.actor_hat.predict(next_states)
        if np.isnan(next_actions).any():
            print('next_actions', next_actions)
            next_actions[np.isnan(next_actions)] = 0
            print('reset next_actions', next_actions)
        
        # Prepare targets for Q_ex and Q_ex2 training
        q_ex_next = self.critic_Q_ex_hat.predict([next_states, next_actions])
        q_ex2_next = self.critic_Q_ex2_hat.predict([next_states, next_actions])
        
        # Calculate target Q values based on the Bellman equation
        target_q_ex = rewards + (1 - dones) * q_ex_next
        target_q_ex2 = rewards ** 2 + (1 - dones) * (2 * rewards * q_ex_next + q_ex2_next)

        # Use Q2 TD error as priority weight
        td_errors = self.critic_Q_ex2.predict([states, actions]) - target_q_ex2
        new_priorities = np.abs(td_errors.flatten()) + self.prioritized_replay_eps
        self.replay_buffer.update_priorities(batch_idxes, new_priorities)

        return states, actions, target_q_ex, target_q_ex2, weights

    def update_model(self, X1, X2, y1, y2, weights):
        """Update DDPG model.
        Arguments:
            X1: states.
            X2: actions.
            y1: target for Q_ex.
            y2: target for Q_ex2.
            weights: priority weights.
        Returns:
            loss_ex: critic Q_ex loss.
            loss_ex2: critic Q_ex2 loss.
        """
        # Flatten weights for training
        weights = weights.flatten()
        
        # Train the critic networks with the targets and weights
        loss_ex = self.critic_Q_ex.fit([X1, X2], y1, sample_weight=weights, verbose=0)
        loss_ex = np.mean(loss_ex.history['loss'])
        loss_ex2 = self.critic_Q_ex2.fit([X1, X2], y2, sample_weight=weights, verbose=0)
        loss_ex2 = np.mean(loss_ex2.history['loss'])

        # Predict actions using the actor network
        X3 = self.actor.predict(X1)
        
        # Get gradients of the critic with respect to the actions
        a_grads = np.array(self.get_critic_grad([X1, X3]))[0]
        
        # Update the actor network using the gradients
        self.sess.run(self.opt, feed_dict={
            self.ainput: X1,
            self.action_gradient: a_grads
        })

        return loss_ex, loss_ex2

    def update_target_model(self):
        """Soft update target model."""
        # Get current weights of the critic and actor networks
        critic_Q_ex_weights = self.critic_Q_ex.get_weights()
        critic_Q_ex2_weights = self.critic_Q_ex2.get_weights()
        actor_weights = self.actor.get_weights()

        # Get current weights of the target networks
        critic_Q_ex_hat_weights = self.critic_Q_ex_hat.get_weights()
        critic_Q_ex2_hat_weights = self.critic_Q_ex2_hat.get_weights()
        actor_hat_weights = self.actor_hat.get_weights()

        # Soft update the target networks' weights
        for i in range(len(critic_Q_ex_weights)):
            critic_Q_ex_hat_weights[i] = self.TAU * critic_Q_ex_weights[i] + (1 - self.TAU) * critic_Q_ex_hat_weights[i]

        for i in range(len(critic_Q_ex2_weights)):
            critic_Q_ex2_hat_weights[i] = self.TAU * critic_Q_ex2_weights[i] + (1 - self.TAU) * critic_Q_ex2_hat_weights[i]

        for i in range(len(actor_weights)):
            actor_hat_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU) * actor_hat_weights[i]

        # Set the updated weights to the target networks
        self.critic_Q_ex_hat.set_weights(critic_Q_ex_hat_weights)
        self.critic_Q_ex2_hat.set_weights(critic_Q_ex2_hat_weights)
        self.actor_hat.set_weights(actor_hat_weights)
        
    def train(self, episode, filename):
        """Training function.

        Args:
            episode (int): Total episodes to run.
            filename (str): Name of the file.

        Returns:
            history_details (dict): Training history details.
            history (dict): Training history.
        """
        
        # Initialize variables
        beta_discount = 0.02
        period_T = self.env.num_period - 1
        eps_history = []
        avg_reward = []
        avg_Vo = []
        avg_action = np.empty((0,4))
        w_T_store = [] # store the value at the end of each episode
        y_action_store = np.empty((0,4)) # store the average action taken for all periods during 1 episode
        V_opt_store = []
       
        history_details = {
            "episode": [], 
            "period": [],
            "period_reward": [], 
            "y1": [], 
            "y2": [], 
            "z1": [], 
            "z2": [], 
            "s1": [], 
            "s2": [],
            "pC": [], 
            "p1": [], 
            "p2": [],
            "cH": [], 
            "cP": [],
            "action_sampled": [],
            "violation_cm": [],
            "violation_s1": [],
            "violation_s2": []
        } 

        history = {
            "name": [],
            "episode": [], 
            "actor_lr": [],
            "critic_lr": [],
            "e_decay": [],
            "tau": [],
            "buffer_size": [],
            "beta0": [],
            "batch_size": [],
            "num_sim": [],
            "ttm": [],
            "n_train": [],
            "n_test": [],
            "episode_w_T": [], 
            "loss_ex": [], 
            "loss_ex2": [],
            "avg_reward": [], 
            "avg_reward_opt": [],
            "reward_gap": [], 
            "epsilon": [],
            "avg_y1": [],  
            "avg_y2": [],  
            "avg_z1": [],  
            "avg_z2": [], 
        } 
        
        # Loop through episodes
        for i in range(episode):
            observation, V_opt = self.env.reset()   
            done = False

            # for recording purpose
            y_action = np.empty((0,4)) 
            reward_store = np.empty(0) 
            self.t = i
            pt = 1
            # steps in an episode
            while not done:
                # prepare state
                x = np.array(observation).reshape(1, -1)
                if np.isnan(x).any():
                    print('In while loop: x has nan', x) 
                    x[np.isnan(x)] = 0
                    # break
                    print('In while loop reset: x has nan', x)  
                # chocie action from epsilon-greedy.
                action, _, _, action_sampled = self.egreedy_action(x)
                # states/observation transition
                observation, reward, done, info = self.env.step(x, action) # update done in the casg flow function/reward function
                # print("The new observation is {}".format(observation))
                if np.isnan(observation).any():
                    print('In while loop: observation has nan', np.array(observation))
                    observation[np.isnan(observation)] = 0
                    print('In while loop reset: observation has nan', np.array(observation))                
                y_action = np.vstack((y_action, action))
                reward_store = np.append(reward_store, reward) # append reward in each period

                self.remember(x[0], action, reward, observation, done) # x: current t state, observation: next states
                history_details["episode"].append(i)
                history_details["period"].append(pt) 
                history_details["period_reward"].append(reward)
                history_details["pC"].append(x[0][0])
                history_details["p2"].append(x[0][1])
                history_details["p1"].append(x[0][2])
                history_details["cH"].append(x[0][3])
                history_details["cP"].append(x[0][4])
                history_details["s1"].append(x[0][5])
                history_details["s2"].append(x[0][6])
                history_details["action_sampled"].append(action_sampled)
                history_details["y1"].append(action[0])  
                history_details["y2"].append(action[1]) 
                history_details["z1"].append(action[2])  
                history_details["z2"].append(action[3])  
                history_details["violation_cm"].append(action[0]+action[1]-98107-1)
                history_details["violation_s1"].append(action[0]+action[2]-x[0][5]-1)
                history_details["violation_s2"].append(action[1]+action[3]-x[0][6]-1)
                pt = pt + 1
                
                if len(self.replay_buffer) > self.batch_size:

                    # draw from memory
                    # batch of states, actions, target for Q_ex, target for Q_ex2, priority weights
                    X1, X2, y_ex, y_ex2, weights = self.process_batch(self.batch_size) 

                    # update model
                    loss_ex, loss_ex2 = self.update_model(X1, X2, y_ex, y_ex2, weights)

                    # soft update target
                    self.update_target_model()
            
            reward_store = reward_store*((1-beta_discount)**np.arange(0,period_T))
            w_T = np.sum(reward_store) # equivalent to one episode's V_opt
            w_T_store.append(w_T) # Store each episodes V_opt
            y_action_store = np.vstack((y_action_store, np.mean(y_action, axis = 0)))
        
            V_opt_store = np.append(V_opt_store, V_opt) # store each episode's V_opt
            # reduce epsilon per episode
            self.update_epsilon()
            self.gap = 1
            if i % self.gap == 0 and i > 11: # i != 0
                history["episode"].append(i)
                history["episode_w_T"].append(w_T) 
                history["loss_ex"].append(loss_ex) # need to plot loss function
                history["loss_ex2"].append(loss_ex2)                
                # save model every 1000 episode
                self.actor.save_weights("model/"+'model'+self.filename+"/"+"_ddpg_actor_" + str(int(i/1000)) + ".h5")
                self.critic_Q_ex.save_weights("model/"+'model'+self.filename+"/"+"_ddpg_critic_Q_ex_" + str(int(i/1000)) + ".h5")
                self.critic_Q_ex2.save_weights("model/"+'model'+self.filename+"/"+"_ddpg_critic_Q_ex2_" + str(int(i/1000)) + ".h5")

                episode10_avgvalue = np.mean(w_T_store[-self.gap:]) # take the average of last 10 episode
                episode10_avgaction = np.mean(y_action_store[-self.gap:], axis = 0)
                
                episode10_avg_y1 = episode10_avgaction[0]
                episode10_avg_y2 = episode10_avgaction[1]
                episode10_avg_z1 = episode10_avgaction[2]
                episode10_avg_z2 = episode10_avgaction[3]
                
                avg_reward.append(episode10_avgvalue)
                avg_action = np.vstack((avg_action, episode10_avgaction))
                
                episode10_avg_V_opt = np.mean(V_opt_store[-self.gap:])
                avg_Vo = np.append(avg_Vo, episode10_avg_V_opt)
                episode10_reward_gap = (episode10_avg_V_opt - episode10_avgvalue)/episode10_avg_V_opt

                eps_history.append(self.epsilon) 
                
                history["avg_reward"].append(episode10_avgvalue)
                history["avg_reward_opt"].append(episode10_avg_V_opt)
                history["reward_gap"].append(episode10_reward_gap)
                history["epsilon"].append(self.epsilon)
                
                history["avg_y1"].append(episode10_avg_y1)
                history["avg_y2"].append(episode10_avg_y2)
                history["avg_z1"].append(episode10_avg_z1)
                history["avg_z2"].append(episode10_avg_z2)               
                
                history["name"].append(filename)
                history["actor_lr"].append(args.actor_lr)
                history["critic_lr"].append(args.critic_lr)
                history["e_decay"].append(args.epsilon_decay)
                history["tau"].append(args.TAU)
                history["buffer_size"].append(args.buffer_size)
                history["beta0"].append(args.prioritized_replay_beta0)
                history["batch_size"].append(args.batch_size)
                history["num_sim"].append(args.num_sim)
                history["ttm"].append(args.ttm)
                history["n_train"].append(args.n_train)
                history["n_test"].append(args.n_test)
                
        self.actor.save_weights("model/"+'model'+self.filename+"/"+"_ddpg_actor.h5")
        self.critic_Q_ex.save_weights("model/"+'model'+self.filename+"/"+"_ddpg_critic_Q_ex.h5")
        self.critic_Q_ex2.save_weights("model/"+'model'+self.filename+"/"+"_ddpg_critic_Q_ex2.h5")

        return history_details, history

# This is the main file
if __name__ == "__main__":
    #################################################### PARAMS TUNING #################################
    parser = argparse.ArgumentParser(description='this is the 1st round tuning')
    parser.add_argument('-actor_lr', type = float, default=0.00001, help='actor network learning rate')
    parser.add_argument('-critic_lr', type = float, default=0.00001, help='critic network learning rate')
    parser.add_argument('-epsilon_min', type = float, default=0.1, help='minimum epsilon')
    parser.add_argument('-epsilon_decay', type = float, default=0.9994, help='epsilon decay rate')
    parser.add_argument('-TAU', type = float, default=0.00001, help='update rate for target model')
    parser.add_argument('-buffer_size', type = int, default=600000, help='buffer size of the replay buffer')
    parser.add_argument('-prioritized_replay_beta_iters', type = int, default=50001, help='prioritized replay')
    parser.add_argument('-prioritized_replay_alpha', type = float, default=0.6, help='prioritized replay alpha')  
    parser.add_argument('-prioritized_replay_beta0', type = float, default=0.4, help='prioritized replay beta')            
    parser.add_argument('-batch_size', type = int, default=128, help='prioritized replay')  
    parser.add_argument('-num_sim', type = int, default=3, help='number of simulations') #4001
    parser.add_argument('-ttm', type = int, default=10, help='time to maturity') #101
    parser.add_argument('-n_train', type = int, default=42, help='training episode') #4012
    parser.add_argument('-n_test', type = int, default=21, help='testing episode') #2001
    parser.add_argument('-depth_nn_hidden', type = int, default=2, help = 'number of hidden layers in actor nn')

    args = parser.parse_args()
    parameters = vars(args)

    ################################################### TRAINING #############################################################
    # disable GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # setup for training
    filename = '_alr' + str(args.actor_lr) + '_clr' + str(args.critic_lr) + '_e' + str(args.epsilon_decay) +'_em' + str(args.epsilon_min) + \
                '_tau' + str(args.TAU) + '_buf' + str(args.buffer_size) + '_b' + str(args.prioritized_replay_beta0) + \
                '_ba' + str(args.batch_size) + '_n' + str(args.num_sim) + '_a' + str(args.prioritized_replay_alpha) + '_tr' + str(args.n_train) + '_te' + str(args.n_test) + '_nn'+ str(args.depth_nn_hidden)

    env = TradingEnv(filename, train_flag = True, dg_random_seed=1, init_ttm=args.ttm, num_sim=args.num_sim)  
    if env.exited:
       print("Exited env class, not creating training DDPG instance") 
    else:
        ddpg = DDPG(env, dynamic=True)  
    
        # for second round training, may want to staart with a specific value of epsilon
        # train_filename = ''
        # ddpg.load(train_filename) # load last round's trained model

        train_details, training_history = ddpg.train(args.n_train, "training"+filename) # specify training episode
        ddpg.save_history(training_history, "training_history"+filename+".csv")
        ddpg.save_history(train_details, "train_details"+filename+".csv")

    #################################################### TESTING ##############################################################
    # setup for testing; use another instance for testing
    env_test = TradingEnv(filename, train_flag = False, dg_random_seed=2, init_ttm=args.ttm, num_sim=args.num_sim) 
    if env_test.exited:
        print("Exited env class, not creating testing DDPG instance") 
    else: 
        ddpg_test = DDPG(env_test)
        ddpg_test.load()
        # episode for testing:  
        test_details, testing_history = ddpg_test.test(args.n_test, "testing"+filename) # specify testing episode 
        ddpg_test.save_history(testing_history, "testing_history"+filename+".csv")
        ddpg_test.save_history(test_details, "test_details"+filename+".csv")

  
    
  