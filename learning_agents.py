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
			critic_function='V', update='discounted-sum',
			temperature=1, temperature_decay=1, gamma=0.99, n_neurons=100, learning_rate=1e-3):
		self.player = player
		self.ID = ID
		self.seed = seed
		self.rng = np.random.RandomState(seed=seed)
		self.gamma = gamma
		self.temperature = temperature
		self.temperature_decay = temperature_decay
		self.learning_rate = learning_rate
		self.n_inputs = n_inputs
		self.n_actions = n_actions
		self.n_neurons = n_neurons
		self.critic_function = critic_function
		self.update = update
		torch.manual_seed(seed)
		self.actor = self.Actor(n_neurons, n_inputs, self.n_actions)
		if critic_function=='V':  # state-value
			self.critic = self.Critic(n_neurons, n_inputs, 1)
			self.eligibility = np.zeros((self.n_inputs, 1))
		elif critic_function=='Q':  # action-value
			self.critic = self.Critic(n_neurons, n_inputs + n_actions, 1)
			self.eligibility = np.zeros((self.n_inputs, self.n_actions))
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
			self.critic_function, 1, self.temperature_decay, self.gamma, self.n_neurons, self.learning_rate)

	def new_game(self):
		self.action_probs_history.clear()
		self.critic_value_history.clear()
		self.episode += 1

	def move(self, game):
		game_state = self.get_state(game)
		# Compute action probabilities for the current state
		action_probs = self.actor(game_state)
		# Sample action from action probability distribution
		action_dist = torch.distributions.categorical.Categorical(probs=action_probs/self.temperature)
		action = action_dist.sample()
		current_turn = len(game.investor_give) if self.player=='investor' else len(game.trustee_give)
		# record the actor and critic outputs for end-of-game learning
		log_prob = torch.log(action_probs.gather(index=action, dim=0))
		# Estimate future rewards from game_state (and, for Q-function, the chosen action)
		if self.critic_function=='V':
			critic_input = game_state
		elif self.critic_function=='Q':
			action_onehot = np.zeros((self.n_actions))
			action_onehot[action.detach().numpy()] = 1
			action_onehot = torch.FloatTensor(action_onehot)
			critic_input = torch.cat((game_state, action_onehot))
		critic_value = self.critic(critic_input)
		# update the histories for learning
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
		rewards = game.investor_reward if self.player=='investor' else game.trustee_reward
		if self.update=='discounted-sum':
			# return for each timestep is the discounted sum of all future rewards
			returns = []
			discounted_sum = 0
			for t in np.arange(game.turns-1, -1, -1):  # loop through reward history backwards
				discounted_sum = rewards[t] + self.gamma * discounted_sum
				ret = torch.FloatTensor([discounted_sum]).squeeze()
				returns.insert(0, ret)  # append to beginning of list
			# return for each timestep is the immediate reward (this does not train trustees appropriately)
			# returns = [torch.FloatTensor([ret]).squeeze() for ret in rewards]
			# Calculate the loss values to update our network
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
		# elif self.update=='sarsa-lambda':
		# 	for t in np.arange(game.turns):
		# 		state = self.state_history[t]
		# 		action = self.action_history[t]
		# 		reward = rewards[t]
		# 		value = self.Q[state, action]
		# 		next_state = self.state_history[t+1] if t<game.turns-1 else None
		# 		next_action = self.action_history[t+1] if t<game.turns-1 else None
		# 		target = reward + self.Q[next_state, next_action] if t<game.turns-1 else reward
		# 		delta = self.learning_rate * (target - value)
		# 		self.z *= self.lam * self.gamma
		# 		self.z[state, action] += 1
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
		self.temperature *= self.temperature_decay



class InstanceBased():

	class Chunk():
		def __init__(self, state, action, reward, value, turn, decay=0.5, epsilon=0.3):
			self.state = state
			self.action = action
			self.reward = reward
			self.value = value
			self.triggers = [turn]
			self.decay = decay  # decay rate for activation
			self.epsilon = epsilon  # gaussian noise added to activation
			self.activation = None  # current activation, set when populating working memory

		def set_activation(self, turn, rng):
			activation = 0
			for t in self.triggers:
				activation += (turn - t)**(-self.decay)
			self.activation = np.log(activation) + rng.logistic(loc=0.0, scale=self.epsilon)

	def __init__(self, player, seed=0, n_inputs=5, n_actions=11, ID="instance-based",
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
		return game_state

	def populate_working_memory(self, game_state):
		self.working_memory.clear()
		for chunk in self.declarative_memory:
			chunk.set_activation(self.turn, self.rng)  # update the activation of each chunk, and store it with the chunk
			activation = chunk.activation
			current_turn = np.where(game_state[:5]>0)[0]
			chunk_turn = np.where(chunk.state[:5]>0)[0]
			greedy_action = 0
			generous_action = 1.0 if self.player=='investor' else 0.5  # define the generosity of a 'generous' action
			# identify chunk's action similarity to a fully-generous action and to a fully-greedy action
			similarity_greedy = 1 - np.abs(chunk.action - greedy_action)
			similarity_generous = 1 - np.abs(chunk.action - generous_action)
			similarity_action = np.max([similarity_greedy, similarity_generous])
			if self.n_inputs==15:
				# identify the similarity between the current move history and the chunk's move history
				current_history = game_state[5:]
				chunk_history = chunk.state[5:]
				similarity_history = 1 - np.sqrt(np.mean(np.square(current_history - chunk_history)))
			if self.populate_method=="action-similarity-2":
				if activation > self.thr_activation and similarity_action > self.thr_action:
					self.working_memory.append(chunk)
			elif self.populate_method=="state-similarity":
				if current_turn != chunk_turn:
					continue  # only consider the chunk if it takes place on the same turn as the current game state
				elif self.n_inputs==5:
					if activation > self.thr_activation:
						self.working_memory.append(chunk)
				elif self.n_inputs==15:
					if activation > self.thr_activation and similarity_history > self.thr_history:
						self.working_memory.append(chunk)
			elif self.populate_method=="state-similarity-action-similarity-2":
				if current_turn != chunk_turn:
					continue
				if self.n_inputs==5:
					if activation > self.thr_activation and similarity_action > self.thr_action:
						self.working_memory.append(chunk)
				elif self.n_inputs==15:
					if activation > self.thr_activation and similarity_action > self.thr_action and similarity_history > self.thr_history:
						self.working_memory.append(chunk)

	def select_action(self, game_state):
		if len(self.working_memory)==0:
			# if there are no chunks in working memory, select a random action
			best_action = self.rng.randint(0, self.n_actions) / (self.n_actions-1)
		elif self.rng.uniform(0,1)<self.epsilon:
			# epsilon random selection for exploration
			best_action = self.rng.randint(0, self.n_actions) / (self.n_actions-1)
		else:
			# choose an action based on the activation, similarity, reward, and/or value of chunks in working memory
			# collect chunks by actions
			actions = {}
			for chunk in self.working_memory:
				if chunk.action not in actions:
					actions[chunk.action] = {'activations':[], 'rewards':[], 'values': [], 'blended': None}
				actions[chunk.action]['activations'].append(chunk.activation)
				actions[chunk.action]['rewards'].append(chunk.reward)
				actions[chunk.action]['values'].append(chunk.value)
			# compute the blended value for each potential action as the sum of values weighted by activation
			for action in actions.keys():
				actions[action]['blended'] = np.average(actions[action]['values'], weights=actions[action]['activations'])
			if self.select_method=="max-blended-value":
				# choose the action with the highest blended value
				best_action = max(actions, key=lambda action: actions[action]['blended'])
			elif self.select_method=="softmax-blended-value":
				# choose the action with probability proportional to the blended value
				arr_actions = np.array([a for a in actions])
				arr_values = np.array([actions[a]['blended'] for a in actions])
				temperature = 10 * self.epsilon
				action_probs = scipy.special.softmax(arr_values / temperature)
				best_action = self.rng.choice(arr_actions, p=action_probs)
		# create a new chunk for the chosen action, populate with more information in learn()
		new_chunk = self.Chunk(state=game_state, action=None, reward=None, value=0, turn=self.turn)
		self.learning_memory.append(new_chunk)
		return best_action

	def move(self, game):
		game_state = self.get_state(game)
		# load chunks from declarative memory into working memory
		self.populate_working_memory(game_state)
		# select an action (generosity) that immitates the best chunk in working memory
		self.state = self.select_action(game_state)
		# translate action into environment-appropriate signal
		coins = game.coins if self.player=='investor' else game.investor_give[-1]*game.match  # coins available
		move = self.state * coins
		give = int(np.clip(move, 0, coins))
		keep = int(coins - give)
		self.turn += 1
		return give, keep

	def learn(self, game):
		# update value of new chunks according to some scheme
		actions = game.investor_gen if self.player=='investor' else game.trustee_gen
		rewards = game.investor_reward if self.player=='investor' else game.trustee_reward
		# for t in np.arange(game.turns):  # loop through chunk history forward
		for t in np.arange(game.turns)[::-1]:  # loop through chunk history backwards
		# for t in np.arange(game.turns-1, -1, -1)
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
		for new_chunk in self.learning_memory:
			# Check if the new chunk has identical (state, action) to a previous chunk in declarative memory.
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


	# implement a one-timestep delay
	class Delay(nengo.synapses.Synapse):
		def __init__(self, size_in=1):
			super().__init__(default_size_in=size_in, default_size_out=size_in)
		def make_state(self, shape_in, shape_out, dt, dtype=None, y0=None):
			return {}
		def make_step(self, shape_in, shape_out, dt, rng, state=None):
			x_past = np.zeros((shape_in[0]))
			def step_delay(t, x, x_past=x_past):
				result = x_past
				x_past[:] = x
				return result
			return step_delay

	def __init__(self, player, seed=0, n_inputs=6, n_actions=11, ID="nengo-actor-critic",
			critic_learning_rate=1e-6, actor_learning_rate=1e-6, gamma=0,
			n_neurons=100, dt=1e-3, tau=None, turn_time=1e-3,
			encoder_method='one-hot', temperature=1, temperature_decay=1):
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
		self.delay_activity = self.Delay(size_in=n_neurons)
		self.delay_value = self.Delay(size_in=1)
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

			class CriticNode(nengo.Node):
				def __init__(self, n_neurons, d=None, learning_rate=0, gamma=0.99):
					self.n_neurons = n_neurons
					self.size_in = 2*n_neurons + 2
					self.size_out = 2
					self.d = d if np.any(d) else np.zeros((n_neurons, 1))
					self.learning_rate = learning_rate
					self.gamma = gamma
					super().__init__(self.step, size_in=self.size_in, size_out=self.size_out)
				def step(self, t, x):
					activity = x[:self.n_neurons]  # current synaptic activities from "state" population
					past_activity = x[self.n_neurons: 2*self.n_neurons]  # delayed synaptic activities from "state" population
					past_reward = x[-2]  # reward associated with past activities
					learning = x[-1]  # gating signal for updating decoders
					value = np.dot(activity, self.d)[0]  # current state of the critic
					past_value = np.dot(past_activity, self.d)[0]  # previous state of the critic
					# calculate error signal for PES, gated by learning signal
					if learning==0: error = 0 # no learning during testing
					elif learning==1: error = past_reward + self.gamma*value - past_value  # normal RL update
					elif learning==2: error = 0  # no learning during first turn (no reward signal)
					elif learning==3: error = past_reward - past_value # the target value in the 6th turn is simply the reward
					# update based on delayed activities and the associated value error
					delta = (self.learning_rate / self.n_neurons) * past_activity.reshape(-1, 1) * error
					self.d += delta
					return [value, error]  # return the current state of the critic and the value error

			class ActorNode(nengo.Node):
				def __init__(self, n_neurons, n_actions, d=None, learning_rate=0):
					self.n_neurons = n_neurons
					self.n_actions = n_actions
					self.size_in = 2*n_neurons + n_actions + 4
					self.size_out = 2 * n_actions
					self.d = d if np.any(d) else np.zeros((n_neurons, n_actions))
					self.learning_rate = learning_rate
					super().__init__(self.step, size_in=self.size_in, size_out=self.size_out)
				def step(self, t, x):
					activity = x[:self.n_neurons]  # current synaptic activities from "state" population
					past_activity = x[self.n_neurons: 2*self.n_neurons]  # delayed synaptic activities from "state" population
					action_probs = x[2*self.n_neurons: 2*self.n_neurons+self.n_actions]  # values for each action, as given by the actor (past state)
					action = int(x[-3])  # action that was previously selected by the learning agent (past state)
					value_error = x[-2]  # value error associated with past activities fed into the critic
					learning = x[-1]  # gating signal for updating decoders
					action_values = np.dot(activity, self.d)
					# calculate error signal for PES, gated by learning signal
					error = -value_error * action_probs  # actor error is  -value_error*p_i for each non-chosen action i
					if   learning==0: error = np.zeros((self.n_actions)) # no learning during testing
					elif learning==1: error[action] = value_error*(1-action_probs[action])  # if action i was chosen, error[i] = value_error*(1-p_i)
					elif learning==2: error = np.zeros((self.n_actions))  # no learning during first turn (no reward signal)
					elif learning==3: error[action] = value_error*(1-action_probs[action])  # if action i was chosen, error[i] = value_error*(1-p_i)
					# update based on delayed activities and the associated actor error
					delta = (self.learning_rate / self.n_neurons) * past_activity.reshape(-1, 1) * error.reshape(1, -1)
					self.d[:] += delta
					result = np.concatenate((action_values, error), axis=0)  # return the current state of the actor and the actor error
					return result 

			state_input = nengo.Node(lambda t, x: self.state_input.get(), size_in=2, size_out=self.n_inputs)
			past_reward = nengo.Node(lambda t, x: self.past_reward_input.get(), size_in=2, size_out=1)
			past_probs = nengo.Node(lambda t, x: self.past_probs_input.get(), size_in=2, size_out=self.n_actions)
			past_action = nengo.Node(lambda t, x: self.past_action_input.get(),size_in=2, size_out=1)
			learning_input = nengo.Node(lambda t, x: self.learning_input.get(), size_in=2, size_out=1)

			state = nengo.Ensemble(self.n_neurons, self.n_inputs, intercepts=intercepts, encoders=encoders, neuron_type=nengo.LIFRate())
			critic = CriticNode(self.n_neurons, d=d_critic, learning_rate=self.critic_learning_rate, gamma=self.gamma)
			actor = ActorNode(self.n_neurons, self.n_actions, d=d_actor, learning_rate=self.actor_learning_rate)

			nengo.Connection(state_input, state, synapse=None)

			nengo.Connection(state.neurons, critic[:self.n_neurons], synapse=None)
			nengo.Connection(state.neurons, critic[self.n_neurons: 2*self.n_neurons], synapse=self.delay_activity)
			nengo.Connection(past_reward, critic[-2], synapse=None)
			nengo.Connection(learning_input, critic[-1], synapse=None)

			nengo.Connection(state.neurons, actor[:self.n_neurons], synapse=None)
			nengo.Connection(state.neurons, actor[self.n_neurons: 2*self.n_neurons], synapse=self.delay_activity)
			nengo.Connection(past_probs, actor[2*self.n_neurons: 2*self.n_neurons+self.n_actions], synapse=None)  # past action probabilities
			nengo.Connection(past_action, actor[-3], synapse=None)  # past choice
			nengo.Connection(critic[1], actor[-2], synapse=None)  # value error for past state
			nengo.Connection(learning_input, actor[-1], synapse=None)  # controls whether learning is on/off

			network.p_state_neurons = nengo.Probe(state.neurons, synapse=None)
			network.p_state_neurons_delayed = nengo.Probe(state.neurons, synapse=self.delay_activity)
			network.p_state = nengo.Probe(state, synapse=None)
			network.p_critic = nengo.Probe(critic[0], synapse=None)
			network.p_critic_delayed = nengo.Probe(critic[0], synapse=self.delay_value)
			network.p_actor = nengo.Probe(actor[:self.n_actions], synapse=None)
			network.p_reward = nengo.Probe(past_reward, synapse=None)
			network.p_value_error = nengo.Probe(critic[1], synapse=None)
			network.p_actor_error = nengo.Probe(actor[self.n_actions:], synapse=None)
			network.d_critic = critic.d
			network.d_actor = actor.d

		return network

	def simulate_action(self):
		self.simulator.run(self.turn_time)
		a_state = self.simulator.data[self.network.p_state_neurons]
		a_state_delayed = self.simulator.data[self.network.p_state_neurons_delayed]
		x_state = self.simulator.data[self.network.p_state]
		x_critic = self.simulator.data[self.network.p_critic]
		x_critic_delayed = self.simulator.data[self.network.p_critic_delayed]
		x_actor = self.simulator.data[self.network.p_actor]
		x_reward = self.simulator.data[self.network.p_reward]
		x_value_error = self.simulator.data[self.network.p_value_error]
		x_actor_error = self.simulator.data[self.network.p_actor_error]
		action_probs = scipy.special.softmax(x_actor[-1] / self.temperature)
		action = self.rng.choice(np.arange(self.n_actions), p=action_probs)
		# print(f"turn (now) \t {np.argmax(x_state[-1])}")
		# print(f"state (now) \t {a_state[-1]}")
		# print(f"state (past) \t {a_state_delayed[-1]}")
		# print(f"state (now) \t {x_state[-1]}")
		# print(f"critic (now) \t {x_critic[-1]}")
		# print(f"critic (past) \t {x_critic_delayed[-1]}")
		# print(f"value error \t {x_value_error[-1]}")
		# print(f"actor values \t {x_actor[-1]}")
		# print(f"action_probs \t {action_probs}")
		# print(f"current action \t {action}")
		# print(f"actor error \t {x_actor_error[-1]}")
		# print(np.argmax(x_state[-1]), x_critic[-1], action_probs)
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
		self.d_critic = self.network.d_critic
		self.d_actor = self.network.d_actor
		# reduce exploration
		self.temperature *= self.temperature_decay


class TabularQLearning():

	def __init__(self, player, seed=0, n_inputs=45, n_actions=11, ID="tabular-q-learning",
			epsilon=1.0, epsilon_decay=0.99, gamma=0.99, lam=0.9, learning_rate=1e0,
			update_type='TD-0', update_direction='forward'):
		self.player = player
		self.ID = ID
		self.seed = seed
		self.rng = np.random.RandomState(seed=seed)
		self.n_inputs = n_inputs
		self.n_actions = n_actions
		self.epsilon = epsilon  # probability of random action, for exploration
		self.epsilon_decay = epsilon_decay  # per-episode reduction of epsilon
		self.gamma = gamma  # discount factor
		self.lam = lam  #  for TD-lambda or SARSA-lambda update
		self.learning_rate = learning_rate
		self.Q = np.zeros((n_inputs, n_actions))
		self.counts = np.zeros((n_inputs, n_actions))  # visits
		self.eligibility = np.zeros((n_inputs, n_actions))  # eligibility trace
		self.state_history = []
		self.action_history = []
		self.update_type = update_type
		self.update_direction = update_direction
		self.state = None

	def reinitialize(self, player):
		self.__init__(player, self.seed, self.n_inputs, self.n_actions, self.ID,
			1, self.epsilon_decay, self.gamma, self.lam, self.learning_rate, self.update_type, self.update_direction)

	def new_game(self):
		self.state_history.clear()
		self.action_history.clear()

	def move(self, game):
		game_state = self.get_state(game)
		# Compute action probabilities for the current state
		Q_state = self.Q[game_state]
		# Sample action from q-values in the current state
		if self.rng.uniform(0, 1) < self.epsilon:
			action = self.rng.randint(self.n_actions)
		else:
			temperature = 10*self.epsilon
			action_probs = scipy.special.softmax(Q_state / temperature)
			action = self.rng.choice(np.arange(self.n_actions), p=action_probs)
			# action = np.argmax(Q_state)
		# update the histories for learning
		self.state_history.append(game_state)
		self.action_history.append(action)
		# translate action into environment-appropriate signal
		self.state = action / (self.n_actions-1)
		coins = game.coins if self.player=='investor' else game.investor_give[-1]*game.match  # coins available
		move = self.state * coins
		give = int(np.clip(move, 0, coins))
		keep = int(coins - give)
		return give, keep

	def get_state(self, game):
		game_state = 0
		if self.n_inputs == 5:
			current_turn = len(game.investor_give) if self.player=='investor' else len(game.trustee_give)
			game_state = current_turn
		if self.n_inputs == 45:
			t = len(game.investor_give) if self.player=='investor' else len(game.trustee_give)
			game_state = 9 * t
			if t==0:
				return game_state
			if self.player == 'investor':
				my_gen = game.investor_gen[-1]
				opponent_gen = game.trustee_gen[-1] if not np.isnan(game.trustee_gen[-1]) else 0
			elif self.player == 'trustee':
				opponent_gen = game.investor_gen[-1]
				my_gen = game.trustee_gen[-1] if not np.isnan(game.trustee_gen[-1]) else 0
			if 0<my_gen<=0.33: game_state +=0
			if 0.33<my_gen<=0.66: game_state +=3
			if 0.66<my_gen<=1: game_state +=6
			if 0<opponent_gen<=0.33: game_state +=0
			if 0.33<opponent_gen<=0.66: game_state +=1
			if 0.66<opponent_gen<=1: game_state +=2
			# print(game_state, t, my_gen, opponent_gen)
		return game_state

	def learn(self, game):
		self.epsilon *= self.epsilon_decay
		rewards = game.investor_reward if self.player=='investor' else game.trustee_reward
		times = np.arange(game.turns) if self.update_direction=='forward' else np.arange(game.turns)[::-1]
		if self.update_type=='TD-0':
			for t in times:
				state = self.state_history[t]
				next_state = self.state_history[t+1] if t<game.turns-1 else None
				action = self.action_history[t]
				reward = rewards[t]
				value = self.Q[state, action]
				next_value = np.max(self.Q[next_state]) if t<game.turns-1 else 0
				self.counts[state, action] += 1
				alpha = self.learning_rate / self.counts[state, action]
				# print(state, action, reward, next_state)
				self.Q[state, action] += alpha * (reward + self.gamma*next_value - value)
		elif self.update_type=='MC':
			discounted_sum = 0
			for t in times:
				discounted_sum = rewards[t] + self.gamma * discounted_sum
				state = self.state_history[t]
				next_state = self.state_history[t+1] if t<game.turns-1 else None
				action = self.action_history[t]
				reward = discounted_sum
				value = self.Q[state, action]
				next_value = np.max(self.Q[next_state]) if t<game.turns-1 else 0
				self.counts[state, action] += 1
				alpha = self.learning_rate / self.counts[state, action]
				self.Q[state, action] += alpha * (reward + self.gamma*next_value - value)
		elif self.update_type=='TD-krangle':
			# https://towardsdatascience.com/reinforcement-learning-td-%CE%BB-introduction-686a5e4f4e60
			for t in times:
				state = self.state_history[t]
				action = self.action_history[t]
				value = self.Q[state, action]
				gt_lambda = 0
				for n in range(1, game.turns-t):
					gt_tn = 0
					for r in np.arange(t, t+n):
						gt_tn += self.gamma**r * rewards[r]
					state_n = self.state_history[n]
					action_n = self.action_history[n]
					gt_tn += self.gamma**n * self.Q[state_n, action_n]
					gt_lambda += self.lam**(n-1) * gt_tn
				gt_lambda *= (1 - self.lam)
				gt_lambda += self.lam**(n-1) * rewards[t]
				self.counts[state, action] += 1
				alpha = self.learning_rate / self.counts[state, action]
				self.Q[state, action] += alpha * (gt_lambda - value)
				# print(self.Q)
		elif self.update_type=='TD-krangle2':
			for t in times:
				# normal TD-0 update
				state = self.state_history[t]
				next_state = self.state_history[t+1] if t<game.turns-1 else None
				action = self.action_history[t]
				reward = rewards[t]
				value = self.Q[state, action]
				next_value = np.max(self.Q[next_state]) if t<game.turns-1 else 0
				self.counts[state, action] += 1
				alpha = self.learning_rate / self.counts[state, action]
				self.Q[state, action] += alpha * (reward + self.gamma*next_value - value)
				# run through history up to this point and do many TD-0 updates
				for n in np.arange(t)[::-1]:
					print(n)
					state = self.state_history[n]
					next_state = self.state_history[n+1]
					action = self.action_history[n]
					reward = rewards[n]
					value = self.Q[state, action]
					next_value = np.max(self.Q[next_state])
					alpha = self.gamma**t * self.learning_rate / self.counts[state, action]
					self.Q[state, action] += alpha * (reward + self.gamma*next_value - value)
		elif self.update_type=='TD-krangle3':
			# for t in times:
			# 	# normal TD-0 update
			# 	state = self.state_history[t]
			# 	next_state = self.state_history[t+1] if t<game.turns-1 else None
			# 	action = self.action_history[t]
			# 	reward = rewards[t]
			# 	value = self.Q[state, action]
			# 	next_value = np.max(self.Q[next_state]) if t<game.turns-1 else 0
			# 	self.counts[state, action] += 1
			# 	alpha = self.learning_rate / self.counts[state, action]
			# 	self.Q[state, action] += alpha * (reward + self.gamma*next_value - value)
			# # one final update based on the total score
			reward = np.sum(rewards)
			for t in times:
				# normal TD-0 update
				state = self.state_history[t]
				next_state = self.state_history[t+1] if t<game.turns-1 else None
				action = self.action_history[t]
				value = self.Q[state, action]
				next_value = np.max(self.Q[next_state]) if t<game.turns-1 else 0
				self.counts[state, action] += 1
				alpha = self.learning_rate / self.counts[state, action]
				self.Q[state, action] += alpha * (reward + self.gamma*next_value - value)
		elif self.update_type=='TD-lambda':
			# http://incompleteideas.net/book/ebook/node78.html
			for t in times:
				state = self.state_history[t]
				action = self.action_history[t]
				reward = rewards[t]
				value = self.Q[state, action]
				next_state = self.state_history[t+1] if t<game.turns-1 else None
				# next_action = self.action_history[t+1] if t<game.turns-1 else None
				# next_value = self.Q[next_state, next_action] if t<game.turns-1 else 0
				next_value = np.max(self.Q[next_state]) if t<game.turns-1 else 0
				self.counts[state, action] += 1
				delta = self.learning_rate / self.counts[state, action] * (reward + self.gamma*next_value - value)
				self.eligibility[state, action] = 1
				self.Q += delta*self.eligibility
				self.eligibility *= self.lam * self.gamma