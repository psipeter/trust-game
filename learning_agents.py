import numpy as np
import random
import os
import torch
import scipy
import nengo

import matplotlib.pyplot as plt

class ActorCritic():

	class Actor(torch.nn.Module):
		def __init__(self, n_neurons, n_inputs, n_outputs):
			torch.nn.Module.__init__(self)
			self.input = torch.nn.Linear(n_inputs, n_neurons)
			self.hidden = torch.nn.Linear(n_neurons, n_neurons)
			self.output = torch.nn.Linear(n_neurons, n_outputs)
		def forward(self, x):
			x = torch.nn.functional.relu(self.input(x))
			x = torch.nn.functional.relu(self.hidden(x))
			x = torch.nn.functional.softmax(self.output(x), dim=0)
			return x

	class Critic(torch.nn.Module):
		def __init__(self, n_neurons, n_inputs, n_outputs):
			torch.nn.Module.__init__(self)
			self.input = torch.nn.Linear(n_inputs, n_neurons)
			self.hidden = torch.nn.Linear(n_neurons, n_neurons)
			self.output = torch.nn.Linear(n_neurons, n_outputs)
		def forward(self, x):
			x = torch.nn.functional.relu(self.input(x))
			x = torch.nn.functional.relu(self.hidden(x))
			x = self.output(x)
			return x

	def __init__(self, player, seed=0, n_inputs=5, n_actions=11, ID="actor-critic",
			gamma=0.99, n_neurons=100, learning_rate=3e-3):
		self.player = player
		self.ID = ID
		self.seed = seed
		self.rng = np.random.RandomState(seed=seed)
		self.gamma = gamma
		self.learning_rate = learning_rate
		self.n_inputs = n_inputs
		self.n_actions = n_actions
		self.n_neurons = n_neurons
		torch.manual_seed(seed)
		self.actor = self.Actor(n_neurons, n_inputs, self.n_actions)
		self.critic = self.Critic(n_neurons, n_inputs, 1)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.learning_rate)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.learning_rate)
		self.loss_function = torch.nn.MSELoss()
		self.action_probs_history = []
		self.critic_value_history = []
		self.state = None
		self.critic_losses = []
		self.actor_losses = []
		self.episode = 0

	def reinitialize(self, player):
		self.__init__(player, self.seed, self.n_inputs, self.n_actions, self.ID,
			self.gamma, self.n_neurons, self.learning_rate)

	def new_game(self):
		self.action_probs_history.clear()
		self.critic_value_history.clear()
		self.episode += 1

	def move(self, game):
		game_state = self.get_state(game)
		# Predict action probabilities and estimated future rewards from game_state
		action_probs = self.actor(game_state)
		critic_value = self.critic(game_state)
		# Sample action from action probability distribution
		action_dist = torch.distributions.categorical.Categorical(probs=action_probs)
		action = action_dist.sample()
		# record the actor and critic outputs for end-of-game learning
		log_prob = torch.log(action_probs.gather(index=action, dim=0))
		self.critic_value_history.append(critic_value.squeeze())
		self.action_probs_history.append(log_prob)
		# translate action into environment-appropriate signal
		self.state = action.detach().numpy() / (self.n_actions-1)
		coins = game.coins if self.player=='investor' else game.investor_give[-1]*game.match  # coins available
		move = self.state * coins
		give = int(np.clip(move, 0, coins))
		keep = int(coins - give)
		return give, keep

	def get_state(self, game):
		game_state = np.zeros((self.n_inputs))
		if self.n_inputs == 5:
			current_turn = len(game.investor_give) if self.player=='investor' else len(game.trustee_give)
			game_state[current_turn] = 1
		if self.n_inputs == 15:
			current_turn = len(game.investor_give) if self.player=='investor' else len(game.trustee_give)
			game_state[current_turn] = 1
			for t in range(current_turn):  # loop over past history and add to state
				my_gen_idx = game.turns + t
				opponent_gen_idx = 2*game.turns + t
				if self.player == 'investor':
					game_state[my_gen_idx] = game.investor_gen[t]
					game_state[opponent_gen_idx] = game.trustee_gen[t] if not np.isnan(game.trustee_gen[t]) else -1
				elif self.player == 'trustee':
					game_state[opponent_gen_idx] = game.investor_gen[t]
					game_state[my_gen_idx] = game.trustee_gen[t] if not np.isnan(game.trustee_gen[t]) else -1
		return torch.FloatTensor(game_state)

	def learn(self, game):
		# Only learn after the game has finished
		if len(game.investor_give) < game.turns: return
		# Calculate expected value from rewards
		# - At each timestep what was the total reward received after that timestep
		# - Rewards in the past are discounted by multiplying them with gamma
		# - Consider the agent's reward and the opponent's reward, weighted appropriately
		# - These are the labels for our critic
		returns = []
		discounted_sum = 0
		rewards_history = game.investor_reward if self.player=='investor' else game.trustee_reward
		for t in np.arange(game.turns-1, -1, -1):  # loop through reward history backwards
			discounted_sum = rewards_history[t] + self.gamma * discounted_sum
			ret = torch.FloatTensor([discounted_sum]).squeeze()
			returns.insert(0, ret)  # append to beginning of list
		# Calculating loss values to update our network
		history = zip(self.action_probs_history, self.critic_value_history, returns)
		actor_losses = []
		critic_losses = []
		for log_prob, val, ret in history:
			# At this point in history, the critic estimated that we would get a total reward = `value` in the future.
			# We took an action with log probability of `log_prob` and ended up recieving a total reward = `ret`.
			# The actor must be updated so that it predicts an action that leads to
			# high rewards (compared to critic's estimate) with high probability.
			diff = ret - val
			actor_losses.append(-log_prob * diff)  # actor loss
			# The critic must be updated so that it predicts a better estimate of the future rewards.
			critic_losses.append(self.loss_function(ret, val))
		actor_loss = torch.sum(torch.FloatTensor(actor_losses))
		critic_loss = torch.sum(torch.FloatTensor(critic_losses))
		loss = actor_loss + critic_loss
		loss = torch.stack(actor_losses).sum() + torch.stack(critic_losses).sum()
		self.actor_optimizer.zero_grad()
		self.critic_optimizer.zero_grad()
		loss.backward()
		self.actor_optimizer.step()
		self.critic_optimizer.step()
		self.actor_losses.append([self.episode, float(actor_loss.detach().numpy())])
		self.critic_losses.append([self.episode, float(critic_loss.detach().numpy())])



class InstanceBased():

	class Chunk():
		def __init__(self, state, action, reward, value, turn, d=0.5, epsilon=0.3):
			self.state = state
			self.action = action
			self.reward = reward
			self.value = value
			self.triggers = [turn]
			self.d = d  # decay rate for activation
			self.epsilon = epsilon  # gaussian noise added to activation
			self.activation = None  # current activation, set when populating working memory

		def set_activation(self, turn, rng):
			activation = 0
			for t in self.triggers:
				activation += (turn - t)**self.d + rng.logistic(loc=0.0, scale=self.epsilon)
			if activation <= 0:
				self.activation = 0.0
			else:
				self.activation = np.around(np.log(activation), 3)

	def __init__(self, player, seed=0, n_inputs=15, n_actions=11, ID="instance-based",
			thr_activation=0, thr_action=0.8, thr_history=0.8, epsilon=1.0, epsilon_decay=0.99,
			populate_method='state-similarity', select_method='softmax-blended-value', value_method='next-value'):
		self.player = player
		self.ID = ID
		self.seed = seed
		self.rng = np.random.RandomState(seed=seed)
		self.n_inputs = n_inputs
		self.n_actions = n_actions
		self.thr_activation = thr_activation  # activation threshold for retrieval (loading chunks from declarative into working memory)
		self.thr_action = thr_action  # action similarity threshold for retrieval (loading chunks from declarative into working memory)
		self.thr_history = thr_history  # history similarity threshold for retrieval (loading chunks from declarative into working memory)
		self.epsilon = epsilon  # probability of random action, for exploration
		self.epsilon_decay = epsilon_decay  # per-episode reduction of epsilon
		self.populate_method = populate_method  # method for determining whether a chunk in declaritive memory meets threshold
		self.select_method = select_method  # method for selecting an action based on chunks in working memory
		self.value_method = value_method  # method for assigning value to chunks during learning
		self.declarative_memory = []
		self.working_memory = []
		self.learning_memory = []
		self.state = None
		self.turn = 0  # for tracking activation within / across games

	def reinitialize(self, player):
		self.__init__(player, self.seed, self.n_inputs, self.n_actions, self.ID,
			self.thr_activation, self.thr_action, self.thr_history, 1.0, self.epsilon_decay,
			self.populate_method, self.select_method, self.value_method)

	def new_game(self):
		self.working_memory.clear()
		self.learning_memory.clear()

	def get_state(self, game):
		game_state = np.zeros((self.n_inputs))
		if self.n_inputs == 15:
			current_turn = len(game.investor_give) if self.player=='investor' else len(game.trustee_give)
			game_state[current_turn] = 1
			for t in range(current_turn):  # loop over past history and add to state
				my_gen_idx = game.turns + t
				opponent_gen_idx = 2*game.turns + t
				if self.player == 'investor':
					game_state[my_gen_idx] = game.investor_gen[t]
					game_state[opponent_gen_idx] = game.trustee_gen[t] if not np.isnan(game.trustee_gen[t]) else -1
				elif self.player == 'trustee':
					game_state[opponent_gen_idx] = game.investor_gen[t]
					game_state[my_gen_idx] = game.trustee_gen[t] if not np.isnan(game.trustee_gen[t]) else -1
		return game_state

	def populate_working_memory(self, game_state):
		self.working_memory.clear()
		for chunk in self.declarative_memory:
			chunk.set_activation(self.turn, self.rng)  # update the activation of each chunk, and store it with the chunk
			activation = chunk.activation
			if self.populate_method=="action-similarity-2":
				# identify chunk's action similarity to a fully-generous action and to a fully-greedy action
				greedy_action = 0
				generous_action = 1.0 if self.player=='investor' else 0.5  # define the generosity of a 'generous' action
				similarity_greedy = 1 - np.abs(chunk.action - greedy_action)
				similarity_generous = 1 - np.abs(chunk.action - generous_action)
				# similarity to either action is valid
				similarity = np.max([similarity_greedy, similarity_generous])
				if activation > self.thr_activation and similarity > self.thr_action:
					self.working_memory.append(chunk)
			elif self.populate_method=="state-similarity":
				# only consider the chunk if it takes place on the same turn as the current game state
				current_turn = np.where(game_state[:5]>0)[0]
				chunk_turn = np.where(chunk.state[:5]>0)[0]
				if current_turn != chunk_turn: continue
				# identify the similarity between the current move history and the chunk's move history
				current_history = game_state[5:]
				chunk_history = chunk.state[5:]
				similarity_history = 1 - np.sqrt(np.mean(np.square(current_history - chunk_history)))
				if activation > self.thr_activation and similarity_history > self.thr_history:
					self.working_memory.append(chunk)
			elif self.populate_method=="state-similarity-action-similarity-2":
				current_turn = np.where(game_state[:5]>0)[0]
				chunk_turn = np.where(chunk.state[:5]>0)[0]
				if current_turn != chunk_turn: continue
				greedy_action = 0.0  # define the generosity of a 'greedy' action
				generous_action = 1.0 if self.player=='investor' else 0.5  # define the generosity of a 'generous' action
				similarity_greedy = 1 - np.abs(chunk.action - greedy_action)
				similarity_generous = 1 - np.abs(chunk.action - generous_action)
				similarity_action = np.max([similarity_greedy, similarity_generous])
				current_history = game_state[5:]
				chunk_history = chunk.state[5:]
				similarity_history = 1 - np.sqrt(np.mean(np.square(current_history - chunk_history)))
				if activation > self.thr_activation and similarity_action > self.thr_action and similarity_history > self.thr_history:
					self.working_memory.append(chunk)

	def select_action(self, game_state):
		# if there are no chunks in working memory, select a random action
		if len(self.working_memory)==0:
			best_action = self.rng.randint(0, self.n_actions) / (self.n_actions-1)
		elif self.rng.uniform(0,1)<self.epsilon:
			best_action = self.rng.randint(0, self.n_actions) / (self.n_actions-1)
		# choose an action based on the activation, similarity, reward, and/or value of chunks in working memory
		elif self.select_method=="max-blended-value" or self.select_method=="softmax-blended-value":
			# collect chunks by actions
			actions = {}
			for chunk in self.working_memory:
				if chunk.action in actions:
					actions[chunk.action]['activations'].append(chunk.activation)
					actions[chunk.action]['rewards'].append(chunk.reward)
					actions[chunk.action]['values'].append(chunk.value)
				else:
					actions[chunk.action] = {
						'activations':[chunk.activation],
						'rewards':[chunk.reward],
						'values': [chunk.value],
						'blended': None,
					}
			# compute the blended value for each potential action as the sum of values weighted by activation
			for key in actions.keys():
				actions[key]['blended'] = np.around(np.average(actions[key]['values'], weights=actions[key]['activations']), 3)
			if self.select_method=="max-blended-value":
				# choose the action with the highest blended value
				best_action = max(actions, key=lambda v: actions[v]['blended'])
			elif self.select_method=="softmax-blended-value":
				# choose the action with probability proportional to the blended value
				arr_actions = np.array([a for a in actions])
				arr_values = np.array([actions[a]['blended'] for a in actions])
				temperature = 10 * self.epsilon
				action_probs = scipy.special.softmax(arr_values / temperature)
				best_action = self.rng.choice(arr_actions, p=action_probs)
		# create a new chunk for the chosen action, populate with more information in learn()
		new_chunk = self.Chunk(state=game_state, action=None, reward=None, value=None, turn=self.turn)
		self.learning_memory.append(new_chunk)
		return best_action

	def move(self, game):
		game_state = self.get_state(game)
		# load chunks from declarative memory into working memory
		self.populate_working_memory(game_state)
		# select an action that immitates the best chunk in working memory
		self.state = self.select_action(game_state)
		# translate action into environment-appropriate signal
		coins = game.coins if self.player=='investor' else game.investor_give[-1]*game.match  # coins available
		move = self.state * coins
		give = int(np.clip(move, 0, coins))
		keep = int(coins - give)
		self.turn += 1
		return give, keep

	def learn(self, game):
		# Only learn after the game has finished
		if len(game.investor_give) < game.turns: return
		# update value of new chunks according to some scheme
		actions = game.investor_gen if self.player=='investor' else game.trustee_gen
		actions = np.nan_to_num(actions, nan=0.0)  # convert skipped turns to action "0"
		rewards = game.investor_reward if self.player=='investor' else game.trustee_reward
		for t in np.arange(game.turns-1, -1, -1):  # loop through chunk history backwards
			chunk = self.learning_memory[t]
			chunk.action = actions[t]
			chunk.reward = rewards[t]
			if self.value_method=='reward':
				chunk.value = rewards[t]
			elif self.value_method=='game-mean':
				chunk.value = np.mean(rewards)
			elif self.value_method=='next-reward':
				chunk.value = chunk.reward if t==(game.turns-1) else chunk.reward + self.learning_memory[t+1].reward
			elif self.value_method=='next-value':
				chunk.value = chunk.reward if t==(game.turns-1) else chunk.reward + self.learning_memory[t+1].value
		# add new chunks to declarative memory
		for new_chunk in self.learning_memory:
			# Check if the chunk has identical (state, action) to a previous chunk in declarative memory.
			# If so, update that chunk's triggers, rather than adding a new chunk to declarative memory
			add_new_chunk = True
			for old_chunk in self.declarative_memory:
				if np.all(new_chunk.state == old_chunk.state) and \
						new_chunk.action == old_chunk.action and \
						new_chunk.reward == old_chunk.reward and \
						new_chunk.value == old_chunk.value:
					old_chunk.triggers.append(new_chunk.triggers[0])
					add_new_chunk = False
					break
			# Otherwise, add a new chunk to declarative memory
			if add_new_chunk:
				self.declarative_memory.append(new_chunk)
		self.epsilon *= self.epsilon_decay  # reduce exploration


class NengoActorCritic():

	class StateInput():
		def __init__(self, n_inputs):
			self.state = np.zeros((n_inputs))
		def set(self, state):
			self.state = state
		def get(self):
			return self.state

	class PastRewardInput():
		def __init__(self):
			self.history = []
		def append(self, reward):
			self.history.append(reward)
		def clear(self):
			self.history.clear()
		def get(self):
			if len(self.history)==0:
				return 0
			else:
				return self.history[-1]

	class PastProbsInput():
		def __init__(self, n_actions):
			self.n_actions = n_actions
			self.history = []
		def append(self, probs):
			self.history.append(probs)
		def clear(self):
			self.history.clear()
		def get(self):
			if len(self.history)==0:
				return np.zeros((self.n_actions))
			else:
				return self.history[-1]

	class PastActionInput():
		def __init__(self):
			self.history = []
		def append(self, action):
			self.history.append(action)
		def clear(self):
			self.history.clear()
		def get(self):
			if len(self.history)==0:
				return 0
			else:
				return self.history[-1]

	class LearningInput():
		def __init__(self):
			self.learning = 0
		def set(self, learning):
			self.learning = learning  # 0=testing, 1=training, 2=first turn, 3=last turn
		def get(self):
			return self.learning

	# an implementation of a pure Delay synapse for nengo
	class Delay(nengo.synapses.Synapse):
		def __init__(self, delay, size_in=1):
			self.delay = delay
			super().__init__(default_size_in=size_in, default_size_out=size_in)
		def make_state(self, shape_in, shape_out, dt, dtype=None, y0=None):
			return {}
		def make_step(self, shape_in, shape_out, dt, rng, state=None):
			steps = int(self.delay/dt)
			if steps == 0:
				def step_delay(t, x):
				    return x
				return step_delay
			assert steps > 0
			state = np.zeros((steps, shape_in[0]))
			state_index = np.array([0])
			def step_delay(t, x, state=state, state_index=state_index):
				result = state[state_index]
				state[state_index] = x
				state_index[:] = (state_index + 1) % state.shape[0]
				return result
			return step_delay

	def __init__(self, player, seed=0, n_inputs=6, n_actions=11, ID="nengo-actor-critic",
			critic_learning_rate=1e-4, actor_learning_rate=1e-4, gamma=0.99, n_neurons=100, dt=1e-3, tau=0.01, turn_time=2e-3,
			encoder_method='one-hot', temperature=1, temperature_decay=0.99):
		self.player = player
		self.ID = ID
		self.seed = seed
		self.rng = np.random.RandomState(seed=seed)
		self.n_inputs = n_inputs
		self.n_actions = n_actions
		self.n_neurons = n_neurons
		self.dt = dt
		self.encoder_method = encoder_method
		self.critic_learning_rate = critic_learning_rate
		self.actor_learning_rate = actor_learning_rate
		self.gamma = gamma  # discount
		self.tau = tau  # synaptic time constant
		self.delay = self.Delay(dt)  # implement a delay for learning
		self.turn_time = turn_time
		self.temperature = temperature
		self.temperature_decay = temperature_decay
		self.state = None
		self.state_input = self.StateInput(n_inputs)
		self.past_reward_input = self.PastRewardInput()
		self.past_action_input = self.PastActionInput()
		self.past_probs_input = self.PastProbsInput(n_actions)
		self.learning_input = self.LearningInput()
		self.network = None
		self.simulator = None
		self.d_critic = None
		self.d_actor = None


	def reinitialize(self, player):
		self.__init__(player, self.seed, self.n_inputs, self.n_actions, self.ID,
			self.critic_learning_rate, self.actor_learning_rate, self.gamma, self.n_neurons, self.dt, self.tau, self.turn_time,
			self.encoder_method, 1, self.temperature_decay)

	def new_game(self):
		self.network = self.build_network()
		self.simulator = nengo.Simulator(self.network, dt=self.dt, progress_bar=False)
		self.past_reward_input.clear()
		self.past_action_input.clear()
		self.past_probs_input.clear()

	def get_state(self, game):
		game_state = np.zeros((self.n_inputs))
		if self.n_inputs == 6:
			current_turn = len(game.investor_give) if self.player=='investor' else len(game.trustee_give)
			game_state[current_turn] = 1
		if self.n_inputs == 18:
			current_turn = len(game.investor_give) if self.player=='investor' else len(game.trustee_give)
			game_state[current_turn] = 1
			for t in range(current_turn):  # loop over past history and add to state
				my_gen_idx = game.turns + t
				opponent_gen_idx = 2*game.turns + t
				if self.player == 'investor':
					game_state[my_gen_idx] = game.investor_gen[t]
					game_state[opponent_gen_idx] = game.trustee_gen[t] if not np.isnan(game.trustee_gen[t]) else -1
				elif self.player == 'trustee':
					game_state[opponent_gen_idx] = game.investor_gen[t]
					game_state[my_gen_idx] = game.trustee_gen[t] if not np.isnan(game.trustee_gen[t]) else -1
		return game_state

	def build_network(self):
		network = nengo.Network(seed=self.seed)
		d_actor = self.d_actor if np.any(self.d_actor) else np.zeros((self.n_actions, self.n_neurons))
		d_critic = self.d_critic if np.any(self.d_critic) else np.zeros((1, self.n_neurons))
		if self.encoder_method=='uniform':
			intercepts = nengo.Default
			encoders = nengo.Default
		elif self.encoder_method=='one-hot':
			intercepts = nengo.dists.Uniform(0.1, 1)
			encs = []
			for dim in range(self.n_inputs):
				enc = np.zeros((self.n_inputs))
				enc[dim] = 1
				encs.append(enc)
			encoders = nengo.dists.Choice(encs)

		with network:
			def value_error_func(t, x):
				reward = x[0]  # reward for the past state/action
				past_value = x[1]  # past state
				value = x[2]  # current state
				learning = x[3]  # turns learning on/off
				if learning==0:  # no learning during testing
					error = 0  
				elif learning==1:  # normal RL update
					# error = 0
					error = reward + self.gamma*value - past_value	
				elif learning==2:  # no learning during first turn (no reward signal)
					error = 0
				elif learning==3:  # the target value in the 6th turn is simply equal to the reward (no future state)
					error = reward - past_value
				# print(f't {t} \t past {past_value:.4f} \t now {value:.4f} \t error {error:.4f}')
				return error

			def actor_error_func(t, x):
				probs = x[:self.n_actions]  # values for each action, as given by the actor (past state)
				action = int(x[-3])  # action that was previously selected by the learning agent (past state)
				value_error = x[-2]  # value error (past state)
				learning = x[-1]  # turns learning on/off
				if learning==0:  # no learning during testing
					error = np.zeros((self.n_actions))
				elif learning==1:  # normal RL update, value_error*(1-p_i) if action i is chosen, else -value_error*p_i
					error = -value_error*probs
					error[action] = value_error*(1-probs[action])
				elif learning==2:  # no learning during first turn (no reward signal)
					error = np.zeros((self.n_actions))
				elif learning==3:  # normal RL update, all signals are from the past turn
					error = -value_error*probs
					error[action] = value_error*(1-probs[action])
				return error

			state_input = nengo.Node(lambda t, x: self.state_input.get(), size_in=2, size_out=self.n_inputs)
			past_reward = nengo.Node(lambda t, x: self.past_reward_input.get(), size_in=2, size_out=1)
			past_probs = nengo.Node(lambda t, x: self.past_probs_input.get(), size_in=2, size_out=self.n_actions)
			past_action = nengo.Node(lambda t, x: self.past_action_input.get(),size_in=2, size_out=1)
			learning_input = nengo.Node(lambda t, x: self.learning_input.get(), size_in=2, size_out=1)

			state = nengo.Ensemble(
				n_neurons=self.n_neurons,
				dimensions=self.n_inputs,
				intercepts=intercepts,
				encoders=encoders,
				neuron_type=nengo.LIFRate())
			critic = nengo.Node(size_in=1)
			actor = nengo.Node(size_in=self.n_actions)
			value_error = nengo.Node(value_error_func, size_in=4)
			actor_error = nengo.Node(actor_error_func, size_in=self.n_actions+3)

			conn_state = nengo.Connection(state_input, state, synapse=None)
			conn_critic = nengo.Connection(state.neurons, critic, synapse=None, transform=d_critic,
				learning_rule_type=nengo.PES(learning_rate=self.critic_learning_rate, pre_synapse=self.delay))
			conn_actor = nengo.Connection(state.neurons, actor, synapse=None, transform=d_actor,
				learning_rule_type=nengo.PES(learning_rate=self.actor_learning_rate, pre_synapse=self.delay))

			nengo.Connection(past_reward, value_error[0], synapse=None)  # past reward
			nengo.Connection(critic, value_error[1], synapse=self.delay)  # past value
			nengo.Connection(critic, value_error[2], synapse=None)  # current value
			nengo.Connection(learning_input, value_error[3], synapse=None)  # controls whether learning is on/off
			nengo.Connection(value_error, conn_critic.learning_rule, transform=-1, synapse=None)

			nengo.Connection(past_probs, actor_error[:self.n_actions], synapse=None)  # past action probabilities
			nengo.Connection(past_action, actor_error[-3], synapse=None)  # past choice
			nengo.Connection(value_error, actor_error[-2], synapse=None)  # value error for past state
			nengo.Connection(learning_input, actor_error[-1], synapse=None)  # controls whether learning is on/off
			nengo.Connection(actor_error, conn_actor.learning_rule, transform=-1, synapse=None) 

			network.p_state_neurons = nengo.Probe(state.neurons, synapse=None)
			network.p_state = nengo.Probe(state, synapse=None)
			network.p_critic = nengo.Probe(critic, synapse=None)
			network.p_critic_delayed = nengo.Probe(critic, synapse=self.delay)
			network.p_actor = nengo.Probe(actor, synapse=None)
			network.p_reward = nengo.Probe(past_reward, synapse=None)
			network.p_value_error = nengo.Probe(value_error, synapse=None)
			network.p_actor_error = nengo.Probe(actor_error, synapse=None)
			network.p_d_actor = nengo.Probe(conn_actor, "weights", synapse=None)
			network.p_d_critic = nengo.Probe(conn_critic, "weights", synapse=None)

		return network

	def simulate_action(self):
		self.simulator.run(self.turn_time)
		# a_state = self.simulator.data[self.network.p_state_neurons]
		# x_state = self.simulator.data[self.network.p_state]
		# x_critic = self.simulator.data[self.network.p_critic]
		# x_critic_delayed = self.simulator.data[self.network.p_critic_delayed]
		x_actor = self.simulator.data[self.network.p_actor]
		# x_reward = self.simulator.data[self.network.p_reward]
		# x_value_error = self.simulator.data[self.network.p_value_error]
		# x_actor_error = self.simulator.data[self.network.p_actor_error]
		action_probs = scipy.special.softmax(x_actor[-1] / self.temperature)
		action = self.rng.choice(np.arange(self.n_actions), p=action_probs)
		# print(f"turn (now) \t {np.argmax(x_state[-1])}")
		# print(f"state (now) \t {a_state[-1]}")
		# print(f"state (now) \t {x_state[-1]}")
		# print(f"critic (now) \t {x_critic[-1][0]:.5f}")
		# print(f"critic (past) \t {x_critic_delayed[-1][0]:.5f}")
		# print(f"value error \t {x_value_error[-1][0]:.5f}")
		# print(f"actor values {x_actor[-1]}")
		# print(f"action_probs \t {action_probs}")
		# print(f"current action \t {action}")
		# print(f"actor error \t {x_actor_error[-1]}")
		return action, action_probs

	def move(self, game):
		game_state = self.get_state(game)
		# add the game state to the network's state input
		self.state_input.set(game_state)
		# collect information from the previous turn (reward, action, action probs)
		# and use them as inputs to the network on this turn (for learning)
		self.past_reward_input.append(self.get_rewards(game))
		# turn learning on/off, depending on the situation
		self.learning_input.set(self.get_learning(game))
		# simulate the network with these inputs and collect the action outputs
		action, action_probs = self.simulate_action()
		# save the action probabilities and the chosen action
		self.past_action_input.append(action)
		self.past_probs_input.append(action_probs)
		# convert action to generosity
		self.state = action / (self.n_actions-1)
		# translate action into environment-appropriate signal
		coins = game.coins if self.player=='investor' else game.investor_give[-1]*game.match  # coins available
		move = self.state * coins
		give = int(np.clip(move, 0, coins))
		keep = int(coins - give)
		return give, keep

	def get_rewards(self, game):
		rewards = game.investor_reward if self.player=='investor' else game.trustee_reward
		reward = rewards[-1] if len(rewards)>0 else 0
		return reward

	def get_learning(self, game):
		current_turn = len(game.investor_give) if self.player=='investor' else len(game.trustee_give)
		learning = 1  # on by default
		if not game.train: learning=0
		if current_turn==0: learning=2
		if current_turn>=5: learning=3
		return learning

	def learn(self, game):
		# Learning rules are applied online based on per-turn rewards, so most update happens in the move() step
		# However, we must run one final turn of simulation to permit learning on the last turn.
		# Learner and fixed agents will not make any additional moves that are added to the game history,
		# but the moves recorded on the last turn will be given an opportunity of affect weight update through PES
		give, keep = self.move(game)
		# give, keep = self.move(game)
		# save weights for the next game
		self.d_actor = self.simulator.data[self.network.p_d_actor][-1]  
		self.d_critic = self.simulator.data[self.network.p_d_critic][-1]
		# reduce exploration
		self.temperature *= self.temperature_decay