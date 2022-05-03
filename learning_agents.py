import numpy as np
import random
import os
import torch
import scipy
import nengo
# import nengolib
import nengo_spa
from collections import namedtuple, deque
from itertools import count
from utils import *


class TQ():
	# Tabular Q-learning agent
	def __init__(self, player, ID="TQ", seed=0, n_actions=11, n_states=155,
			learning_rate=1, gamma=0.6, epsilon_decay=0.005,
			svo_min=0.1, svo_max=0.4, orientation="proself",
			normalize=True, randomize=False):
		self.player = player
		self.ID = ID
		self.seed = seed
		self.randomize = randomize
		self.normalize = normalize
		self.rng = np.random.RandomState(seed=seed)
		self.n_states = n_states
		self.n_actions = n_actions
		self.svo_min = svo_min
		self.svo_max = svo_max
		if self.randomize:
			self.gamma = self.rng.uniform(0.6, 0.9)
			self.learning_rate = self.rng.uniform(0.8, 1.2)
			self.orientation = "proself" if self.rng.uniform(0,1) < 0.5 else "prosocial"
			self.svo = 0 if self.orientation=="proself" else self.rng.uniform(self.svo_min, self.svo_max)
		else:
			self.gamma = gamma
			self.learning_rate = learning_rate
			self.orientation = orientation
			self.svo = 0 if self.orientation=="proself" else self.svo_max
		self.epsilon_decay = epsilon_decay
		self.Q = np.zeros((self.n_states, self.n_actions))
		self.state_history = []
		self.action_history = []
		self.state = None
		self.episode = 0

	def reinitialize(self, player):
		self.player = player
		self.Q = np.zeros((self.n_states, self.n_actions))
		self.state_history = []
		self.action_history = []
		self.state = None
		self.episode = 0

	def new_game(self, game):
		self.state_history.clear()
		self.action_history.clear()

	def move(self, game):
		game_state = get_state(self.player, game=game, agent='TQ')
		# Compute action probabilities for the current state
		Q_state = self.Q[game_state]
		# Sample action from q-values in the current state
		epsilon = 1 - self.episode * self.epsilon_decay
		if self.rng.uniform(0, 1) < epsilon:
			action = self.rng.randint(self.n_actions)
		else:
			action = np.argmax(Q_state)
		# convert action to number of coins given/kept
		self.state = action / (self.n_actions-1)
		give, keep, action_idx = action_to_coins(self.player, self.state, self.n_actions, game)
		# save state and actions for learning
		self.state_history.append(game_state)
		self.action_history.append(action_idx)
		return give, keep

	def learn(self, game):
		self.episode += 1
		rewards = get_rewards(self.player, self.svo, game, self.normalize)
		for t in np.arange(game.turns):
			state = self.state_history[t]
			action = self.action_history[t]
			value = self.Q[state, action]
			if t==(game.turns-1):
				delta = rewards[t] - value
			else:
				next_state = self.state_history[t+1]
				next_action = self.action_history[t+1]
				next_value = np.max(self.Q[next_state])  # Q-learning
				# next_value = self.Q[next_state, next_action]  # SARSA
				dR = (1-self.gamma)*rewards[t] if self.normalize else rewards[t]
				dT = self.gamma*next_value
				delta = dR + dT - value
			self.Q[state, action] += self.learning_rate * delta



class DQN():

	class Critic(torch.nn.Module):
		def __init__(self, n_neurons, n_states, n_actions):
			torch.nn.Module.__init__(self)
			self.input = torch.nn.Linear(n_states, n_neurons)
			self.hidden = torch.nn.Linear(n_neurons, n_neurons)
			self.output = torch.nn.Linear(n_neurons, n_actions)
		def forward(self, x):
			x = torch.nn.functional.relu(self.input(x))
			x = torch.nn.functional.relu(self.hidden(x))
			x = self.output(x)
			return x

	class ReplayMemory:
		def __init__(self, capacity):
			self.capacity = capacity
			self.memory = []
		def push(self, transition):
			self.memory.append(transition)
			if len(self.memory) > self.capacity:
				del self.memory[0]
		def sample(self, batch_size):
			return random.sample(self.memory, batch_size)
		def __len__(self):
			return len(self.memory)

	def __init__(self, player, seed=0, n_states=155, n_actions=11, n_neurons=30, ID="DQN",
			learning_rate=2e-2, gamma=0.99, epsilon_decay=0.0025, batch_size=30, target_update=100,
			svo_min=0.1, svo_max=0.4, orientation="proself",
			normalize=False, randomize=False):
		self.player = player
		self.ID = ID
		self.seed = seed
		self.rng = np.random.RandomState(seed=seed)
		self.randomize = randomize
		self.normalize = normalize
		self.n_states = n_states
		self.n_actions = n_actions
		self.n_neurons = n_neurons
		self.batch_size = batch_size
		self.target_update = target_update
		self.svo_min = svo_min
		self.svo_max = svo_max
		if self.randomize:
			self.gamma = self.rng.uniform(0.6, 0.9)
			self.learning_rate = self.rng.uniform(1e-2, 3e-2)
			self.orientation = "proself" if self.rng.uniform(0,1) < 0.5 else "prosocial"
			self.svo = 0 if self.orientation=="proself" else self.rng.uniform(self.svo_min, self.svo_max)
		else:
			self.gamma = gamma
			self.learning_rate = learning_rate
			self.orientation = orientation
			self.svo = 0 if self.orientation=="proself" else self.svo_max
		torch.manual_seed(seed)
		self.epsilon_decay = epsilon_decay
		self.critic = self.Critic(self.n_neurons, self.n_states, self.n_actions)
		self.target = self.Critic(self.n_neurons, self.n_states, self.n_actions)
		self.optimizer = torch.optim.Adam(self.critic.parameters(), self.learning_rate)
		# self.value_history = []
		self.replay_memory = self.ReplayMemory(10000)
		self.state_history = []
		self.action_history = []
		self.state = None
		self.episode = 0

	def reinitialize(self, player):
		self.player = player
		self.critic = self.Critic(self.n_neurons, self.n_states, self.n_actions)
		self.target = self.Critic(self.n_neurons, self.n_states, self.n_actions)
		self.optimizer = torch.optim.Adam(self.critic.parameters(), self.learning_rate)
		self.replay_memory = self.ReplayMemory(10000)
		# self.value_history = []
		self.state_history = []
		self.action_history = []
		self.state = None
		self.episode = 0

	def new_game(self, game):
		# self.value_history.clear()
		self.state_history.clear()
		self.action_history.clear()

	def move(self, game):
		game_state = get_state(self.player, game, agent="DQN", dim=self.n_states)
		# Estimate the value of the current game_state
		with torch.no_grad():
			values = self.critic(game_state)
		# Choose and action based on thees values and some exploration strategy
		epsilon = 1 - self.episode * self.epsilon_decay
		if self.rng.uniform(0, 1) < epsilon:
			action = torch.LongTensor([self.rng.randint(self.n_actions)])
		else:
			action = torch.argmax(values)
		# translate action into environment-appropriate signal
		self.state = action.detach().numpy() / (self.n_actions-1)
		give, keep, action_idx = action_to_coins(self.player, self.state, self.n_actions, game)
		# self.value_history.append(values)
		# self.value_history.append(values[action_idx])
		self.state_history.append(game_state)
		self.action_history.append(torch.LongTensor([action_idx]))
		return give, keep

	def learn(self, game):
		self.episode += 1
		rewards = get_rewards(self.player, self.svo, game, self.normalize)
		for t in range(game.turns):
			state = self.state_history[t]
			action = self.action_history[t]
			reward = torch.FloatTensor([rewards[t]])
			next_state = self.state_history[t+1] if t<game.turns-1 else torch.FloatTensor(np.zeros((self.n_states)))
			self.replay_memory.push((state, action, next_state, reward))
		if len(self.replay_memory) < self.batch_size: return
		transitions = self.replay_memory.sample(self.batch_size)
		batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)
		batch_state = torch.reshape(torch.cat(batch_state), shape=(self.batch_size, self.n_states))
		batch_action = torch.reshape(torch.cat(batch_action), shape=(self.batch_size, 1))
		batch_next_state = torch.reshape(torch.cat(batch_next_state), shape=(self.batch_size, self.n_states))
		batch_reward = torch.cat(batch_reward)
		current_values = self.critic(batch_state).gather(1, batch_action).squeeze()
		expected_rewards = self.target(batch_next_state).detach().max(1)[0]
		# print(self.critic(batch_state))
		# print(batch_action)
		# print(current_values)
		# print(expected_rewards)
		# print(batch_reward)
		# raise
		if self.normalize:
			pass
		else:
			expected_values = batch_reward + (self.gamma * expected_rewards)
		# print(expected_values)
		# loss = torch.nn.functional.mse_loss(current_values, expected_values)
		loss = torch.nn.functional.huber_loss(current_values, expected_values)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		if self.episode % self.target_update == 0:
			self.target.load_state_dict(self.critic.state_dict())

	# def learn(self, game):
	# 	self.episode += 1
	# 	rewards = get_rewards(self.player, self.svo, game, self.normalize)
	# 	losses = []
	# 	for t in np.arange(game.turns):
	# 		action = self.action_history[t]
	# 		value = self.value_history[t][action]
	# 		next_action = self.action_history[t+1] if t<game.turns-1 else 0
	# 		next_value = torch.max(self.value_history[t+1]) if t<game.turns-1 else 0
	# 		# next_value = self.value_history[t+1][next_action] if t<game.turns-1 else 0
	# 		dR = (1-self.gamma)*torch.FloatTensor([rewards[t]]) if self.normalize else torch.FloatTensor([rewards[t]])
	# 		dT = self.gamma*next_value
	# 		delta = dR + dT - value
	# 		losses.append(delta**2)
	# 	loss = torch.stack(losses).sum()
	# 	self.optimizer.zero_grad()
	# 	loss.backward()
	# 	self.optimizer.step()


class IBL():

	class Chunk():
		def __init__(self, state, action, reward, value, episode, decay, epsilon):
			self.state = state
			self.action = action
			self.reward = reward
			self.value = value
			self.triggers = [episode]
			self.decay = decay  # decay rate for activation
			self.epsilon = epsilon  # gaussian noise added to activation

		def get_activation(self, episode, rng):
			activation = 0
			for t in self.triggers:
				activation += (episode - t)**(-self.decay)
			return np.log(activation) + rng.logistic(loc=0.0, scale=self.epsilon)

	def __init__(self, player, seed=0, n_actions=11, ID="IBL", representation='turn-coin',
			populate_method='state-similarity', value_method='next-value',
			thr_activation=0, thr_action=0.8, thr_state=0.9, friendliness=0, randomize=False,
			learning_method='TD0', gamma=0.99, decay=0.5, epsilon=0.3, biased_exploration=False, bias=0.75,
			explore_method='epsilon', explore_start=1, explore_decay=0.007, explore_decay_method='linear'):
		self.player = player
		self.ID = ID
		self.seed = seed
		self.rng = np.random.RandomState(seed=seed)
		self.representation = representation
		# self.n_actions = 11 if self.player=='investor' else 31
		self.n_actions = n_actions
		self.n_inputs = get_n_inputs(representation, player, self.n_actions)
		self.learning_method = learning_method
		self.randomize = randomize
		if self.randomize:
			self.gamma = self.rng.uniform(0.5, 1)
			self.decay = self.rng.uniform(0.4, 0.6)
			self.epsilon = self.rng.uniform(0.2, 0.4)
			# self.friendliness = self.rng.uniform(0, 0.4)
			# self.friendliness = 0
			if self.player=='investor':
				if self.rng.uniform(0,1)<0.5: self.friendliness = 0
				else: self.friendliness = 0.2
			elif self.player=='trustee':
				if self.rng.uniform(0,1)<0.5: self.friendliness = 0
				else: self.friendliness = 0.4 #0.3
		else:
			self.gamma = gamma
			self.friendliness = friendliness
			self.decay = decay  # decay rate for memory chunks
			self.epsilon = epsilon  # logistic noise in memory retrieval
		self.thr_activation = thr_activation  # activation threshold for retrieval (loading chunks from declarative into working memory)
		self.thr_action = thr_action  # action similarity threshold for retrieval (loading chunks from declarative into working memory)
		# self.thr_state = thr_state  # state similarity threshold for retrieval (loading chunks from declarative into working memory)
		self.thr_state = 1 - 1/self.n_actions
		self.explore_start = explore_start  # probability of random action, for exploration
		self.explore_decay = explore_decay
		self.explore_method = explore_method
		self.explore_decay_method = explore_decay_method
		self.biased_exploration = biased_exploration
		self.bias = bias
		self.populate_method = populate_method  # method for determining whether a chunk in declaritive memory meets threshold
		self.value_method = value_method  # method for assigning value to chunks during learning
		self.declarative_memory = []
		self.working_memory = []
		self.learning_memory = []
		self.state = None
		self.episode = 0  # for tracking activation within / across games

	def reinitialize(self, player, ID, seed):
		self.__init__(player=player, ID=ID, seed=seed)

	def new_game(self, game):
		self.working_memory.clear()
		self.learning_memory.clear()
		self.rng.shuffle(self.declarative_memory)

	def move(self, game):
		game_state = get_state(self.player, self.representation, game=game, return_type='index', n_actions=self.n_actions)
		# load chunks from declarative memory into working memory
		self.populate_working_memory(game_state, game)
		# select an action (generosity) that immitates the best chunk in working memory
		self.state = self.select_action()
		# create a new chunk for the chosen action, populate with more information in learn()
		new_chunk = self.Chunk(game_state, None, None, None, self.episode, self.decay, self.epsilon)
		self.learning_memory.append(new_chunk)
		# translate action into environment-appropriate signal
		give, keep, action_idx = action_to_coins(self.player, self.state, self.n_actions, game)
		return give, keep

	def populate_working_memory(self, game_state, game):
		self.working_memory.clear()
		current_turn = len(game.investor_give) if self.player=='investor' else len(game.trustee_give)
		greedy_action = 0
		generous_action = 1.0 if self.player=='investor' else 0.5
		for chunk in self.declarative_memory:
			activation = chunk.get_activation(self.episode, self.rng)
			# identify the similarity between the chunk's action and each of the candidate actions
			similarity_greedy = 1 - np.abs(chunk.action - greedy_action)
			similarity_generous = 1 - np.abs(chunk.action - generous_action)
			similarity_action = np.max([similarity_greedy, similarity_generous])
			# identify the similarity between the chunk's state and the current game state
			if self.representation=='turn':
				chunk_turn = chunk.state
				similarity_state = 1 if current_turn==chunk_turn else 0
			elif self.representation=='turn-coin':
				if self.player=='investor':
					chunk_turn = chunk.state
					similarity_state = 1 if current_turn==chunk_turn else 0
				else:
					chunk_turn = int(chunk.state / (game.coins*game.match+1))
					if current_turn==chunk_turn:
						chunk_coins = chunk.state - chunk_turn*(game.coins*game.match+1)
						current_coins = game.investor_give[-1]*game.match
						diff_coin = np.abs(chunk_coins - current_coins)
						similarity_state = 1.0 - diff_coin / (game.coins*game.match+1)
					else:
						similarity_state = 0
			# load the chunk into working memory if various checks on activation, action similarity, and/or state similarity are passed
			pass_activation = activation > self.thr_activation
			pass_action = similarity_action > self.thr_action
			pass_state = similarity_state > self.thr_state
			if self.populate_method=="action-similarity" and pass_activation and pass_action:
				self.working_memory.append(chunk)
			elif self.populate_method=="state-similarity" and pass_activation and pass_state:
				self.working_memory.append(chunk)
			elif self.populate_method=="state-action-similarity" and pass_activation and pass_action and pass_state:
				self.working_memory.append(chunk)

	def select_action(self):
		if len(self.working_memory)==0:
			# if there are no chunks in working memory, select a random action
			selected_action = self.rng.randint(0, self.n_actions) / (self.n_actions-1)
		else:
			# choose an action based on the activation, similarity, reward, and/or value of chunks in working memory
			# collect chunks by actions
			actions = {}
			for action in np.arange(self.n_actions)/(self.n_actions-1):
				actions[action] = {'activations':[], 'rewards':[], 'values': [], 'blended': None}
			for chunk in self.working_memory:
				if chunk.action not in actions:
					actions[chunk.action] = {'activations':[], 'rewards':[], 'values': [], 'blended': None}
				actions[chunk.action]['activations'].append(chunk.get_activation(self.episode, self.rng))
				actions[chunk.action]['rewards'].append(chunk.reward)
				actions[chunk.action]['values'].append(chunk.value)
			# compute the blended value for each potential action as the sum of values weighted by activation
			# if there are no chunks corresponding to this action, set the default blended value to zero
			for action in actions.keys():
				if len(actions[action]['activations']) > 0:
					actions[action]['blended'] = np.average(actions[action]['values'], weights=actions[action]['activations'])
				else:
					actions[action]['blended'] = 0
			# select an action based on the exploration scheme
			if self.explore_decay_method == 'linear':
				explore = np.max([0, self.explore_start - self.explore_decay*self.episode])
			elif self.explore_decay_method =='exponential':
				explore = self.explore_start * np.exp(-self.explore_decay*self.episode)
			elif self.explore_decay_method == 'power':
				explore = self.explore_start * np.power(self.explore_decay, self.episode)
			if self.explore_method=='epsilon':
				if self.rng.uniform(0, 1) < explore:
					if self.biased_exploration:
						biased_actions = [0, int(self.n_actions/2), self.n_actions-1]
						unbiased_actions = np.delete(np.arange(self.n_actions), biased_actions)
						if self.rng.uniform(0,1)<self.bias:
							selected_action = biased_actions[self.rng.randint(len(biased_actions))] / (self.n_actions-1)
						else:
							selected_action = unbiased_actions[self.rng.randint(len(unbiased_actions))] / (self.n_actions-1)
					else:
						selected_action = self.rng.randint(0, self.n_actions) / (self.n_actions-1)
				else:
					selected_action = max(actions, key=lambda action: actions[action]['blended'])
			elif self.explore_method=='boltzmann':
				action_gens = np.array([a for a in actions])
				action_values = np.array([actions[a]['blended'] for a in actions])
				action_probs = scipy.special.softmax(action_values / explore)
				selected_action = self.rng.choice(action_gens, p=action_probs)
		return selected_action

	def learn(self, game):
		# update value of new chunks according to some scheme
		actions = game.investor_gen if self.player=='investor' else game.trustee_gen
		rewards = game.investor_reward if self.player=='investor' else game.trustee_reward
		rewards_other = game.trustee_reward if self.player=='investor' else game.investor_reward
		if self.learning_method=='MC':
			returns = []
			return_sum = 0
			for t in np.arange(game.turns)[::-1]:
				return_sum += rewards[t]
				returns.insert(0, return_sum)
		else:
			returns = (1.0-self.friendliness)*np.array(rewards) + self.friendliness*np.array(rewards_other)
		for t in np.arange(game.turns):
			chunk = self.learning_memory[t]
			chunk.action = actions[t]
			chunk.reward = returns[t]
			# estimate the value of the next chunk by retrieving all similar chunks and computing their blended value
			next_reward_blended = 0
			next_value_blended = 0
			if t<(game.turns-1):
				next_turn = t+1
				next_chunk_rewards = []
				next_chunk_values = []
				next_chunk_activations = []
				for recalled_chunk in self.declarative_memory:
					recalled_activation = recalled_chunk.get_activation(self.episode, self.rng)
					if self.representation=='turn':
						chunk_turn = recalled_chunk.state
						similarity_state = 1 if next_turn==chunk_turn else 0
					elif self.representation=='turn-coin':
						if self.player=='investor':
							chunk_turn = recalled_chunk.state
							similarity_state = 1 if next_turn==chunk_turn else 0
						else:
							chunk_turn = int(recalled_chunk.state / (game.coins*game.match+1))
							if next_turn==chunk_turn:
								chunk_coins = recalled_chunk.state - chunk_turn*(game.coins*game.match+1)
								current_coins = game.investor_give[next_turn]*game.match
								diff_coin = np.abs(chunk_coins - current_coins)
								similarity_state = 1.0 - diff_coin / (game.coins*game.match+1)
							else:
								similarity_state = 0
					pass_activation = recalled_activation > self.thr_activation
					pass_state = similarity_state > self.thr_state
					if pass_activation and pass_state:
						next_chunk_rewards.append(recalled_chunk.reward)
						next_chunk_values.append(recalled_chunk.value)
						next_chunk_activations.append(recalled_activation)
				if len(next_chunk_values)>0:
					next_reward_blended = np.average(next_chunk_rewards, weights=next_chunk_activations)
					next_value_blended = np.average(next_chunk_values, weights=next_chunk_activations)
			if self.value_method=='reward':
				chunk.value = returns[t]
			elif self.value_method=='next-reward':
				chunk.value = chunk.reward + self.gamma*next_reward_blended
			elif self.value_method=='next-value':
				chunk.value = chunk.reward + self.gamma*next_value_blended

		for new_chunk in self.learning_memory:
			# Check if the new chunk has identical (state, action) to a previous chunk in declarative memory.
			# If so, update that chunk's triggers, rather than adding a new chunk to declarative memory
			add_new_chunk = True
			for old_chunk in self.declarative_memory:
				if np.all(new_chunk.state == old_chunk.state) and new_chunk.action == old_chunk.action:
					old_chunk.triggers.append(new_chunk.triggers[0])
					old_chunk.reward = new_chunk.reward
					old_chunk.value = new_chunk.value
					add_new_chunk = False
					break
			# Otherwise, add a new chunk to declarative memory
			if add_new_chunk:
				self.declarative_memory.append(new_chunk)
		self.episode += 1





class SPA():

	class Environment():
		def __init__(self, ssp_dim, n_actions, t1, t2, t3, tR, rng, gamma, dt=1e-3):
			self.state = np.zeros((ssp_dim))
			self.n_actions = n_actions
			self.rng = rng
			self.reward = 0
			self.t1 = t1
			self.t2 = t2
			self.t3 = t3
			self.tR = tR
			self.dt = dt
			self.random_choice = np.zeros((self.n_actions))
			self.gamma = gamma
		def set_state(self, state):
			self.state = state
		def set_reward(self, player, game, friendliness):
			rewards = game.investor_reward if player=='investor' else game.trustee_reward
			rewards_other = game.trustee_reward if player=='investor' else game.investor_reward
			reward = (1-friendliness)*rewards[-1]+friendliness*rewards_other[-1] if len(rewards)>0 else 0
			self.reward = reward / (game.coins * game.match) # if player=='investor' else reward / game.coins
			if player=='investor' and len(game.investor_give)<5:
				self.reward *= (1-self.gamma)
			if player=='trustee' and len(game.trustee_give)<5:
				self.reward *= (1-self.gamma)
		def set_random_choice(self, epsilon):
			if self.rng.uniform(0, 1) < epsilon:
				random_action = self.rng.randint(self.n_actions)
				self.random_choice = -np.ones((self.n_actions))
				self.random_choice[random_action] = 1
			else:
				self.random_choice = np.zeros((self.n_actions))
		def get_state(self):
			return self.state
		def get_reward(self):
			return self.reward
		def get_random_choice(self, t):
			T = t % (self.t1 + self.t2 + self.t3)
			if 0 <= T <= self.t1 + self.t2:
				return np.zeros_like(self.random_choice)   # don't explore in stage 1 or 2
			else:
				return self.random_choice  # explore in stage 3
		def get_buffer(self, t):
			T = t % (self.t1 + self.t2 + self.t3)
			if 0 <= T <= self.t1 + self.t2:
				return 0  # don't buffer in stage 1 or 2
			else:
				return 1  # buffer in stage 3
		def get_replay(self, t):
			T = t % (self.t1 + self.t2 + self.t3)
			if self.t1 < T <= self.t1 + self.t2:
				return 1  # replay in stage 2
			else:
				return 0  # don't replay in stage 1 or 3
		def get_reset(self, t):
			T = t % (self.t1 + self.t2 + self.t3)
			# reset only for the first tR seconds of eahc phase
			if 0 <= T < self.tR:
				return 1
			elif self.t1 < T < self.t1 + self.tR:
				return 1
			elif self.t1 + self.t2 < T < self.t1 + self.t2 + self.tR:
				return 1
			else:
				return 0

	def __init__(self, player, seed=0, n_actions=11, ID="SPA",
			learning_rate=1e-8, n_neurons=5000, n_array=500, ssp_dim=100, sparsity=0.1,
			gate_mode="direct", memory_mode="array", 
			dt=1e-3, t1=1e-1, t2=1e-1, t3=1e-1, tR=1e-2, 
			explore_method='epsilon', explore=1, explore_decay=0.01, gamma=0.6, friendliness=0):
		self.player = player
		self.ID = ID
		self.seed = seed
		self.rng = np.random.RandomState(seed=seed)
		self.ssp_dim = ssp_dim
		self.n_actions = n_actions
		self.n_neurons = n_neurons
		self.n_array = n_array
		self.dt = dt
		self.gamma = gamma
		self.friendliness = friendliness
		self.learning_rate = learning_rate
		self.t1 = t1
		self.t2 = t2
		self.t3 = t3
		self.tR = tR
		self.memory_mode = memory_mode
		self.gate_mode = gate_mode
		self.explore_method = explore_method
		self.explore = explore
		self.explore_decay = explore_decay
		self.env = self.Environment(self.ssp_dim, self.n_actions, t1, t2, t3, tR, self.rng, self.gamma)
		self.decoders = np.zeros((self.n_neurons, self.n_actions))
		self.network = None
		self.simulator = None
		self.state = None
		self.episode = 0
		self.sparsity = sparsity
		self.turn_basis = make_unitary(np.fft.fft(nengo.dists.UniformHypersphere().sample(1, ssp_dim)))
		self.coin_basis = make_unitary(np.fft.fft(nengo.dists.UniformHypersphere().sample(1, ssp_dim)))
		self.state_intercept = nengo.dists.Choice([self.sparsity_to_x_intercept(ssp_dim, self.sparsity)])
		self.encoders = self.make_encoders()
		self.sM = np.zeros((ssp_dim))

	def sparsity_to_x_intercept(self, d, p):
		sign = 1
		if p > 0.5:
			p = 1.0 - p
			sign = -1
		return sign * np.sqrt(1-scipy.special.betaincinv((d-1)/2.0, 0.5, 2*p))

	def make_encoders(self, mode="inputs", n_samples=100):
		encoders = []
		for t in range(6):
			if self.player == "investor":
				encoders.append(encode_state(t, 10, self.turn_basis, self.coin_basis))
			if self.player == "trustee":
				for c in [3,30]:
					encoders.append(encode_state(t, c, self.turn_basis, self.coin_basis))
		encoders = np.array(encoders)
		return encoders

	def reinitialize(self, player, ID, seed):
		self.__init__(player=player, ID=ID, seed=seed)
		self.network = self.build_network()
		self.simulator = nengo.Simulator(self.network, dt=self.dt, seed=self.seed, progress_bar=True)

	def new_game(self, game):
		self.env.__init__(self.ssp_dim, self.n_actions, self.t1, self.t2, self.t3, self.tR, self.rng, self.gamma)
		self.episode += 1
		self.simulator.reset(self.seed)

	def build_network(self):
		n_actions = self.n_actions
		ssp_dim = self.ssp_dim
		n_neurons = self.n_neurons
		n_array = self.n_array
		seed = self.seed
		network = nengo.Network(seed=seed)
		network.config[nengo.Ensemble].neuron_type = nengo.LIFRate()
		network.config[nengo.Ensemble].max_rates = nengo.dists.Uniform(400, 400)
		network.config[nengo.Probe].synapse = None
		with network:

			class LearningNode(nengo.Node):
				def __init__(self, n_neurons, n_actions, decoders, learning_rate):
					self.n_neurons = n_neurons
					self.n_actions = n_actions
					self.size_in = n_neurons + n_actions
					self.size_out = n_actions
					self.decoders = decoders
					self.learning_rate = learning_rate
					super().__init__(self.step, size_in=self.size_in, size_out=self.size_out)
				def step(self, t, x):
					n_neurons = self.n_neurons
					n_actions = self.n_actions
					state_activities = x[:n_neurons]
					error = x[n_neurons: n_neurons+n_actions]
					delta = self.learning_rate * state_activities.reshape(-1, 1) * error.reshape(1, -1)
					self.decoders[:] += delta
					Q = np.dot(state_activities, self.decoders)
					return Q

			def StateGate(n_neurons, n_array, dim, seed, mode="array"):
				net = nengo.Network(seed=seed)
				with net:
					net.a = nengo.Node(size_in=dim)
					net.b = nengo.Node(size_in=dim)
					net.output = nengo.Node(size_in=dim)
					if mode=="neuron":
						wInh = -1e1*np.ones((n_neurons, 1))
						net.gate_a = nengo.Ensemble(n_array, 1)
						net.gate_b = nengo.Ensemble(n_array, 1)
						net.ens_a = nengo.Ensemble(n_neurons, dim)
						net.ens_b = nengo.Ensemble(n_neurons, dim)
						nengo.Connection(net.a, net.ens_a, synapse=None)
						nengo.Connection(net.b, net.ens_b, synapse=None)
						nengo.Connection(net.ens_a, net.output, synapse=None)
						nengo.Connection(net.ens_b, net.output, synapse=None)
						nengo.Connection(net.gate_a, net.ens_a.neurons, transform=wInh, synapse=None)
						nengo.Connection(net.gate_b, net.ens_b.neurons, transform=wInh, synapse=None)
					elif mode=="array":
						wInh = -1e1*np.ones((n_array*dim, 1))
						net.gate_a = nengo.Ensemble(n_array, 1)
						net.gate_b = nengo.Ensemble(n_array, 1)
						net.ens_a = nengo.networks.EnsembleArray(n_array, dim)
						net.ens_b = nengo.networks.EnsembleArray(n_array, dim)
						net.ens_a.add_neuron_input()
						net.ens_b.add_neuron_input()
						nengo.Connection(net.a, net.ens_a.input, synapse=None)
						nengo.Connection(net.b, net.ens_b.input, synapse=None)
						nengo.Connection(net.ens_a.output, net.output, synapse=None)
						nengo.Connection(net.ens_b.output, net.output, synapse=None)
						nengo.Connection(net.gate_a, net.ens_a.neuron_input, transform=wInh, synapse=None)
						nengo.Connection(net.gate_b, net.ens_b.neuron_input, transform=wInh, synapse=None)
					elif mode=="direct":
						net.gate_a = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
						net.gate_b = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
						net.ens_a = nengo.Ensemble(1, dim+1, neuron_type=nengo.Direct())
						net.ens_b = nengo.Ensemble(1, dim+1, neuron_type=nengo.Direct())
						nengo.Connection(net.a, net.ens_a[:-1], synapse=None)
						nengo.Connection(net.b, net.ens_b[:-1], synapse=None)
						nengo.Connection(net.gate_a, net.ens_a[-1], function=lambda x: 1-x, synapse=None)
						nengo.Connection(net.gate_b, net.ens_b[-1], function=lambda x: 1-x, synapse=None)
						nengo.Connection(net.ens_a, net.output, function=lambda x: x[:-1]*x[-1], synapse=None)
						nengo.Connection(net.ens_b, net.output, function=lambda x: x[:-1]*x[-1], synapse=None)
				return net

			class GatedNode(nengo.Node):
				def __init__(self, ssp_dim):
					self.ssp_dim = ssp_dim
					self.size_in = ssp_dim + 1
					self.size_out = ssp_dim
					self.memory = np.zeros((ssp_dim))
					super().__init__(self.step, size_in=self.size_in, size_out=self.size_out)
				def step(self, t, x):
					ssp = x[:-1]
					gate = int(x[-1])
					if gate==1:
						self.memory = ssp
					return self.memory

			def StateGateMemory(n_neurons, n_array, dim, seed, n_gates=1, gain=0.1, synapse=0, mode="array"):
				net = nengo.Network(seed=seed)
				with net:
					net.state = nengo.Node(size_in=dim)
					net.output = nengo.Node(size_in=dim)
					if mode=="neuron":
						wInh = -1e1*np.ones((n_neurons, 1))
						net.gate = nengo.Ensemble(n_array, 1)
						net.mem = nengo.Ensemble(n_neurons, dim)
						net.diff = nengo.Ensemble(n_neurons, dim)
						nengo.Connection(net.state, net.diff, synapse=None)
						nengo.Connection(net.diff, net.mem, transform=gain, synapse=synapse)
						nengo.Connection(net.mem, net.mem, synapse=synapse)
						nengo.Connection(net.mem, net.diff, transform=-1, synapse=synapse)
						nengo.Connection(net.gate, net.diff.neurons, transform=wInh, synapse=None)
						nengo.Connection(net.mem, net.output, synapse=None)
					elif mode=="array":
						wInh = -1e1*np.ones((n_array*dim, 1))
						net.gate = nengo.Ensemble(n_array, 1)
						net.mem = nengo.networks.EnsembleArray(n_array, dim, radius=0.3)
						net.diff = nengo.networks.EnsembleArray(n_array, dim, radius=0.3)
						net.diff.add_neuron_input()
						nengo.Connection(net.state, net.diff.input, synapse=None)
						nengo.Connection(net.diff.output, net.mem.input, transform=gain, synapse=synapse)
						nengo.Connection(net.mem.output, net.mem.input, synapse=synapse)
						nengo.Connection(net.mem.output, net.diff.input, transform=-1, synapse=synapse)
						nengo.Connection(net.gate, net.diff.neuron_input, transform=wInh, synapse=None)
						nengo.Connection(net.mem.output, net.output, synapse=None)
					elif mode=="direct":
						net.gate = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
						net.mem = GatedNode(dim)
						nengo.Connection(net.state, net.mem[:-1], synapse=None)
						nengo.Connection(net.gate, net.mem[-1], function=lambda x: 1-x, synapse=None)
						nengo.Connection(net.mem, net.output, synapse=None)
				return net

			def GatedMemory(n_neurons, dim, seed, n_gates=1, gain=1, synapse=0, onehot=True):
				net = nengo.Network(seed=seed)
				wInh = -1e1*np.ones((n_neurons*dim, 1))
				with net:
					net.state = nengo.Node(size_in=dim)
					net.gates = [nengo.Ensemble(n_neurons, 1) for g in range(n_gates)]
					net.mem = nengo.networks.EnsembleArray(n_neurons, dim, intercepts=nengo.dists.Uniform(0,1), encoders=nengo.dists.Choice([[1]]))
					net.diff = nengo.networks.EnsembleArray(n_neurons, dim)  # calculate difference between stored value and input
					net.diff.add_neuron_input()
					net.output = nengo.Node(size_in=dim)
					nengo.Connection(net.state, net.diff.input, synapse=None)
					nengo.Connection(net.diff.output, net.mem.input, transform=gain, synapse=synapse)  # feed difference into integrator
					nengo.Connection(net.mem.output, net.mem.input, synapse=synapse)  # memory feedback
					nengo.Connection(net.mem.output, net.diff.input, transform=-1, synapse=synapse)  # calculate difference between stored value and input
					for g in range(n_gates):
						nengo.Connection(net.gates[g], net.diff.neuron_input, function=lambda x: 1-x, transform=wInh, synapse=None)  # gate the inputs
					nengo.Connection(net.mem.output, net.output, synapse=None)
				return net

			def OneHotIndexer(n_neurons, dim, n_actions, seed):
				net = nengo.Network(seed=seed)
				T = 1.0 / np.sqrt(2)
				wInh = -1e0 * np.ones((n_neurons, n_neurons))
				assert (dim==1 or dim==n_actions)
				with net:
					net.values = nengo.Node(size_in=dim, label="values")
					net.onehot = nengo.Node(size_in=n_actions, label="onehot")
					net.bias = nengo.Node(-1)
					net.output = nengo.Node(size_in=n_actions, label='output')
					net.ens_values = nengo.networks.EnsembleArray(n_neurons, n_actions)
					net.ens_onehot = nengo.networks.EnsembleArray(n_neurons, n_actions,
						intercepts=nengo.dists.Uniform(0.1,1), encoders=nengo.dists.Choice([[1]]))
					net.ens_neg_onehot = nengo.networks.EnsembleArray(n_neurons, n_actions,
						intercepts=nengo.dists.Uniform(0.1,1), encoders=nengo.dists.Choice([[-1]]))
					if dim==1:
						for a in range(n_actions):
							nengo.Connection(net.values, net.ens_values.input[a], synapse=None)
					elif dim==n_actions:
						nengo.Connection(net.values, net.ens_values.input, synapse=None)
					nengo.Connection(net.onehot, net.ens_onehot.input, synapse=None)
					for a in range(n_actions):
						nengo.Connection(net.bias, net.ens_neg_onehot.ea_ensembles[a], synapse=None)
						nengo.Connection(net.ens_onehot.ea_ensembles[a], net.ens_neg_onehot.ea_ensembles[a], synapse=None)
						nengo.Connection(net.ens_neg_onehot.ea_ensembles[a].neurons, net.ens_values.ea_ensembles[a].neurons, transform=wInh, synapse=None)
					nengo.Connection(net.ens_values.output, net.output, synapse=None)
				return net

			def IndependentAccumulator(n_neurons, n_actions, seed, thr=0.9, Tff=5e-2, Tfb=-5e-2):
				net = nengo.Network(seed=seed)
				wReset = -1e1 * np.ones((n_neurons, 1))
				with net:
					net.input = nengo.Node(size_in=n_actions)
					net.reset = nengo.Node(size_in=1)
					net.acc = nengo.networks.EnsembleArray(n_neurons, n_actions, intercepts=nengo.dists.Uniform(0, 1), encoders=nengo.dists.Choice([[1]]))
					net.inh = nengo.networks.EnsembleArray(n_neurons, n_actions, intercepts=nengo.dists.Uniform(thr, 1), encoders=nengo.dists.Choice([[1]]))
					net.output = nengo.Node(size_in=n_actions)
					nengo.Connection(net.input, net.acc.input, synapse=None, transform=Tff)
					nengo.Connection(net.acc.output, net.acc.input, synapse=0)
					nengo.Connection(net.acc.output, net.inh.input, synapse=0)
					for a in range(n_actions):
						nengo.Connection(net.reset, net.acc.ea_ensembles[a].neurons, synapse=None, transform=wReset)
						for a2 in range(n_actions):
							if a!=a2:
								nengo.Connection(net.inh.ea_ensembles[a], net.acc.ea_ensembles[a2], synapse=0, transform=Tfb)
					nengo.Connection(net.acc.output, net.output, synapse=None)
				return net

			# inputs from environment
			state_input = nengo.Node(lambda t, x: self.env.get_state(), size_in=2, size_out=ssp_dim)
			reward_input = nengo.Node(lambda t, x: self.env.get_reward(), size_in=2, size_out=1)
			random_choice_input = nengo.Node(lambda t, x: self.env.get_random_choice(t), size_in=2, size_out=n_actions)
			replay_input = nengo.Node(lambda t, x: self.env.get_replay(t), size_in=2, size_out=1)
			buffer_input = nengo.Node(lambda t, x: self.env.get_buffer(t), size_in=2, size_out=1)
			reset_input = nengo.Node(lambda t, x: self.env.get_reset(t), size_in=2, size_out=1)

			# ensembles and nodes
			state = nengo.Ensemble(n_neurons, ssp_dim, intercepts=self.state_intercept)
			state_memory = StateGateMemory(n_neurons, n_array, ssp_dim, seed, mode=self.memory_mode)
			state_gate = StateGate(n_neurons, n_array, ssp_dim, seed, mode=self.gate_mode)
			# state_cleanup = nengo.networks.AssociativeMemory(self.encoders, self.encoders, n_array)

			critic = nengo.networks.EnsembleArray(n_array, n_actions)
			error = nengo.networks.EnsembleArray(n_array, n_actions, radius=0.2)
			learning = LearningNode(n_neurons, n_actions, self.decoders, self.learning_rate)

			choice_memory = GatedMemory(n_array, n_actions, gain=0.3, seed=seed)
			value_memory = GatedMemory(n_array, 1, gain=0.3, n_gates=2, seed=seed)

			buffered_value_product = OneHotIndexer(n_array, 1, n_actions, seed=seed)
			reward_product = OneHotIndexer(n_array, 1, n_actions, seed=seed)
			replayed_value_product = OneHotIndexer(n_array, n_actions, n_actions, seed=seed)
			compressed_value_product = OneHotIndexer(n_array, n_actions, n_actions, seed=seed)

			choice = IndependentAccumulator(n_array, n_actions, seed=seed)

			# send the current state to state_memory, update only in (stage 3)
			nengo.Connection(state_input, state_memory.state, synapse=None)
			nengo.Connection(buffer_input, state_memory.gate, function=lambda x: 1-x, synapse=None)

			# send either the current state (stage 1 or 3) OR previous state (stage 2) to the state population, gated by replay
			nengo.Connection(state_input, state_gate.a, synapse=None)
			nengo.Connection(state_memory.output, state_gate.b, synapse=None)
			nengo.Connection(replay_input, state_gate.gate_a, synapse=None)
			nengo.Connection(replay_input, state_gate.gate_b, function=lambda x: 1-x, synapse=None)
			# nengo.Connection(state_gate.output, state_cleanup.input, synapse=None)
			# nengo.Connection(state_cleanup.output, state, synapse=None)
			nengo.Connection(state_gate.output, state, synapse=None)

			# state to critic connection, computes Q function, updates with DeltaQ from error population
			nengo.Connection(state.neurons, learning[:n_neurons], synapse=None)
			nengo.Connection(error.output, learning[n_neurons:], synapse=None)
			nengo.Connection(learning, critic.input, synapse=0)

			# Q values sent to WTA competition in choice
			nengo.Connection(critic.output, choice.input, synapse=None)
			nengo.Connection(random_choice_input, choice.input, synapse=None)
			nengo.Connection(reset_input, choice.reset, synapse=None)

			# before learning (stage 1), store the Q value of the current state, indexed by the best action in the new state
			nengo.Connection(critic.output, compressed_value_product.values, synapse=None)
			nengo.Connection(choice.output, compressed_value_product.onehot, synapse=None)
			for ens in compressed_value_product.ens_values.ea_ensembles:
				# sum all dimensions, reducing the one-hot vector to a 1D estimate of Q(s',a')
				nengo.Connection(ens, value_memory.state, synapse=None)
			nengo.Connection(replay_input, value_memory.gates[0], synapse=None, function=lambda x: 1-x)
			nengo.Connection(buffer_input, value_memory.gates[1], synapse=None, function=lambda x: 1-x)

			# after learning (stage 3), store the action selected by the choice ensemble in choice memory
			nengo.Connection(choice.output, choice_memory.state, synapse=None)
			nengo.Connection(buffer_input, choice_memory.gates[0], synapse=None)

			# during replay (stage 2), index all components of the error signal by the action stored in choice memory
			# so that updates to the decoders only affect the dimensions corresponding to a0
			nengo.Connection(value_memory.output, buffered_value_product.values, synapse=None)
			nengo.Connection(choice_memory.output, buffered_value_product.onehot, synapse=None)
			nengo.Connection(buffered_value_product.output, error.input, synapse=None, transform=self.gamma)
			nengo.Connection(reward_input, reward_product.values, synapse=None)
			nengo.Connection(choice_memory.output, reward_product.onehot, synapse=None)
			nengo.Connection(reward_product.output, error.input, synapse=None)
			nengo.Connection(critic.output, replayed_value_product.values, synapse=None)
			nengo.Connection(choice_memory.output, replayed_value_product.onehot, synapse=None)
			nengo.Connection(replayed_value_product.output, error.input, synapse=None, transform=-1)

			# turn learning off until replay (stage 3)
			for a in range(n_actions):
				nengo.Connection(replay_input, error.ea_ensembles[a].neurons,
					function=lambda x: 1-x, synapse=None, transform=-1e5*np.ones((n_array, 1)))

			network.p_state_input = nengo.Probe(state_input)
			network.p_reward_input = nengo.Probe(reward_input)
			network.p_random_choice_input = nengo.Probe(random_choice_input)
			network.p_replay_input = nengo.Probe(replay_input)
			network.p_buffer_input = nengo.Probe(buffer_input)
			network.p_reset_input = nengo.Probe(reset_input)

			network.p_state_memory = nengo.Probe(state_memory.output)
			network.p_state = nengo.Probe(state)
			network.p_spikes = nengo.Probe(state.neurons)

			network.p_critic = nengo.Probe(critic.output)
			network.p_learning = nengo.Probe(learning)
			network.p_error = nengo.Probe(error.output)
			network.p_choice = nengo.Probe(choice.output)

			network.p_state_memory = nengo.Probe(state_memory.output)
			network.p_value_memory = nengo.Probe(value_memory.output)
			network.p_choice_memory = nengo.Probe(choice_memory.output)
			network.p_reward_product = nengo.Probe(reward_product.output)
			network.p_replayed_value_product = nengo.Probe(replayed_value_product.output)
			network.p_buffered_value_product = nengo.Probe(buffered_value_product.output)			

		return network

	def move(self, game):
		
		game_state = get_state(self.player, "ssp", game=game, return_type='ssp', dim=self.ssp_dim, n_actions=self.n_actions, rng=self.rng,
			turn_basis=self.turn_basis, coin_basis=self.coin_basis)

		self.env.set_reward(self.player, game, self.friendliness)  # reward for the this turn depends on actions taken last turn
		self.env.set_state(game_state)
		self.env.set_random_choice(self.explore - self.explore_decay*self.episode)

		# print("Stage 1")
		self.simulator.run(self.t1, progress_bar=False)  # store Q(s',a*)
		print('critic', np.around(self.simulator.data[self.network.p_critic][-1], 2))
		# state0 = self.simulator.data[self.network.p_state][-1]
		# spikes0 = self.simulator.data[self.network.p_spikes][-1]
		# mag0 = np.sqrt(np.sum(np.square(state0)))
		# print(f"magnitude (current): {np.around(mag0, 2)}")

		# print("Stage 2")
		self.simulator.run(self.t2, progress_bar=False)  # replay Q(s,a), recall Q(s',a') from value memory, and learn
		# state1 = self.simulator.data[self.network.p_state][-1]
		# spikes1 = self.simulator.data[self.network.p_spikes][-1]
		# simMR = measure_similarity(self.sM, state1)
		# sim01 = measure_similarity(state0, state1)
		# mag1 = np.sqrt(np.sum(np.square(state1)))
		# print(f'similarity (past, current): {sim01:.3}')
		# print(f'similarity (past , recall): {simMR:.3}')
		# print(f"magnitude (recall): {np.around(mag1, 2)}")

		# print("Stage 3")
		self.simulator.run(self.t3, progress_bar=False)  # choose a'
		choice = self.simulator.data[self.network.p_choice][-1]
		action = np.argmax(choice)
		self.state = action / (self.n_actions-1)  # translate action into environment-appropriate signal
		give, keep, action_idx = action_to_coins(self.player, self.state, self.n_actions, game)
		self.sM = self.simulator.data[self.network.p_state][-1]

		return give, keep

	def learn(self, game):
		pass