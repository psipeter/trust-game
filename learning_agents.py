import numpy as np
import random
import os
import torch
import scipy
import nengo
import nengolib
from utils import *

class TabularQLearning():

	def __init__(self, player, seed=0, n_actions=11, ID="tabular-q-learning", representation='turn-coin',
			explore_method='epsilon', explore_start=1, explore_decay=0.005, explore_decay_method='linear',
			learning_method='TD0', learning_rate=1e0, gamma=0.9, lambd=0, randomize=False, friendliness=0):
		self.player = player
		self.ID = ID
		self.seed = seed
		self.rng = np.random.RandomState(seed=seed)
		self.representation = representation
		self.n_inputs = get_n_inputs(representation, player, n_actions)
		self.n_actions = n_actions
		self.explore_method = explore_method
		self.explore_start = explore_start
		self.explore_decay_method = explore_decay_method
		self.randomize = randomize
		if self.randomize:
			self.gamma = self.rng.uniform(0, 1)
			self.explore_decay = self.rng.uniform(0.3, 0.5)
			if self.player=='investor':
				if self.rng.uniform(0,1)<0.5: self.friendliness = 0
				else: self.friendliness = 0.1
			elif self.player=='trustee':
				if self.rng.uniform(0,1)<0.5: self.friendliness = 0
				else: self.friendliness = 0.4
		else:
			self.gamma = gamma
			self.friendliness = friendliness
			self.explore_decay = explore_decay
		self.lambd = lambd
		self.learning_rate = learning_rate
		self.learning_method = learning_method
		self.Q = np.zeros((self.n_inputs, self.n_actions))
		self.E = np.zeros((self.n_inputs, self.n_actions))  # eligibility trace
		self.state_history = []
		self.action_history = []
		self.state = None
		self.episode = 0

	def reinitialize(self, player, ID, seed):
		self.__init__(player=player, ID=ID, seed=seed)

	def new_game(self, game):
		self.state_history.clear()
		self.action_history.clear()
		self.E = np.zeros((self.n_inputs, self.n_actions))

	def move(self, game):
		game_state = get_state(self.player, self.representation, game=game, return_type='index', n_actions=self.n_actions)
		# Compute action probabilities for the current state
		Q_state = self.Q[game_state]
		# Sample action from q-values in the current state
		if self.explore_decay_method == 'linear':
			explore = self.explore_start - self.explore_decay*self.episode
		elif self.explore_decay_method =='exponential':
			explore = self.explore_start * np.exp(-self.explore_decay*self.episode)
		elif self.explore_decay_method == 'power':
			explore = self.explore_start * np.power(self.explore_decay, self.episode)
		if self.explore_method=='epsilon':
			if self.rng.uniform(0, 1) < explore:
				action = self.rng.randint(self.n_actions)
			else:
				action = np.argmax(Q_state)
		elif self.explore_method=='boltzmann':
			action_probs = scipy.special.softmax(Q_state / explore)
			action = self.rng.choice(np.arange(self.n_actions), p=action_probs)
		elif self.explore_method=='upper-confidence-bound':
			ucb = np.sqrt(2*np.log(self.episode)/(self.C[game_state]+np.ones((self.n_actions))))
			values = Q_state + ucb
			idx_max = np.argwhere(values==np.max(values)).flatten()  # get indices where the values are at a maximum
			action = self.rng.choice(idx_max)
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
			# returns = (np.array(rewards) + self.friendliness*np.array(rewards_other)) / (1+self.friendliness)
		for t in np.arange(game.turns):
			state = self.state_history[t]
			action = self.action_history[t]
			next_state = self.state_history[t+1] if t<game.turns-1 else None
			next_action = self.action_history[t+1] if t<game.turns-1 else None
			value = self.Q[state, action]
			next_value = np.max(self.Q[next_state]) if t<game.turns-1 else 0
			# next_value = self.Q[next_state, next_action] if t<game.turns-1 else 0
			delta = returns[t] + self.gamma*next_value - value
			if self.learning_method=='ET':
				self.E[state, action] += 1
				self.Q += self.learning_rate * delta * self.E
				self.E *= self.gamma * self.lambd
			else:
				self.Q[state, action] += self.learning_rate * delta



class NormalizedQLearning():

	def __init__(self, player, seed=0, n_actions=11, ID="normalized-q-learning", representation='turn-coin',
			explore_method='epsilon', explore_start=1, explore_decay=0.005, explore_decay_method='linear',
			learning_rate=1e0, gamma=0.6, randomize=False):
		self.player = player
		self.ID = ID
		self.seed = seed
		self.rng = np.random.RandomState(seed=seed)
		self.representation = representation
		self.n_states = get_n_inputs(representation, player, n_actions)
		self.n_actions = n_actions
		self.explore_method = explore_method
		self.explore_start = explore_start
		self.explore_decay_method = explore_decay_method
		self.randomize = randomize
		self.gamma = gamma
		self.explore_decay = explore_decay
		self.learning_rate = learning_rate
		self.friendliness = 0
		self.Q = np.zeros((self.n_states, self.n_actions))
		self.state_history = []
		self.action_history = []
		self.state = None
		self.episode = 0

	def reinitialize(self, player, ID, seed):
		self.__init__(player=player, ID=ID, seed=seed)

	def new_game(self, game):
		self.state_history.clear()
		self.action_history.clear()

	def move(self, game):
		game_state = get_state(self.player, self.representation, game=game, return_type='index', n_actions=self.n_actions)
		# Compute action probabilities for the current state
		Q_state = self.Q[game_state]
		# Sample action from q-values in the current state
		explore = self.explore_start - self.explore_decay*self.episode
		if self.explore_method=='epsilon':
			if self.rng.uniform(0, 1) < explore:
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
		rewards = game.investor_reward if self.player=='investor' else game.trustee_reward
		rewards = np.array(rewards)/(game.coins * game.match)
		for t in np.arange(game.turns):
			state = self.state_history[t]
			action = self.action_history[t]
			value = self.Q[state, action]
			if t<game.turns-1:
				next_state = self.state_history[t+1]
				next_action = self.action_history[t+1]
				next_value = np.max(self.Q[next_state])
				# next_value = self.Q[next_state, next_action]
				dR = (1-self.gamma)*rewards[t]
				dT = self.gamma*next_value
				delta = dR + dT - value
			else:
				next_state = None
				next_action = None
				next_value = None
				dR = rewards[t]
				dT = 0
				delta = dR + dT - value
			self.Q[state, action] += self.learning_rate * delta
		for s in self.state_history:
			print(self.Q[state])


class TabularActorCritic():

	def __init__(self, player, seed=0, n_actions=5, ID="tabular-actor-critic", representation='turn-gen-opponent',
			explore_method='boltzmann', explore=100, explore_decay=0.995,
			learning_method='TD0', critic_rate=1e-1, actor_rate=1e-1, gamma=0.99, lambd=0.8):
		self.player = player
		self.ID = ID
		self.seed = seed
		self.rng = np.random.RandomState(seed=seed)
		self.representation = representation
		self.n_inputs = get_n_inputs(representation, player, n_actions)
		self.n_actions = n_actions
		self.explore_method = explore_method
		self.explore = explore
		self.explore_decay = explore_decay
		self.gamma = gamma
		self.lambd = lambd
		self.learning_method = learning_method
		self.critic_rate = critic_rate
		self.actor_rate = actor_rate
		self.critic = np.zeros((self.n_inputs, 1))
		self.actor = np.zeros((self.n_inputs, self.n_actions))
		self.E = np.zeros((self.n_inputs, self.n_actions))
		self.state_history = []
		self.action_history = []
		self.action_probs_history = []
		self.state = None
		self.episode = 0

	def reinitialize(self, player):
		self.__init__(player)

	def new_game(self, game):
		self.state_history.clear()
		self.action_history.clear()
		self.action_probs_history.clear()
		self.episode += 1
		self.E = np.zeros((self.n_inputs, self.n_actions))

	def move(self, game):
		game_state = get_state(self.player, self.representation, game=game, return_type='index', n_actions=self.n_actions)
		# Compute action probabilities for the current state
		action_values = self.actor[game_state]
		critic_values = self.critic[game_state]
		# Sample action from actor probability distribution
		if self.explore_method=='boltzmann':
			temperature = self.explore*np.power(self.explore_decay, self.episode)
			action_probs = scipy.special.softmax(action_values / temperature)
			action = self.rng.choice(np.arange(self.n_actions), p=action_probs)			
		# translate action into environment-appropriate signal
		self.state = action / (self.n_actions-1)
		give, keep, action_idx = action_to_coins(self.player, self.state, self.n_actions, game)
		# update the histories for learning
		self.state_history.append(game_state)
		self.action_history.append(action_idx)
		self.action_probs_history.append(action_probs)
		return give, keep

	def learn(self, game):
		rewards = game.investor_reward if self.player=='investor' else game.trustee_reward
		if self.learning_method=='MC':
			returns = []
			return_sum = 0
			for t in np.arange(game.turns)[::-1]:
				return_sum += rewards[t]
				returns.insert(0, return_sum)
		else:
			returns = rewards
		for t in np.arange(game.turns):
			state = self.state_history[t]
			next_state = self.state_history[t+1] if t<game.turns-1 else None
			action = self.action_history[t]
			value = self.critic[state]
			next_value = self.critic[next_state] if t<game.turns-1 else 0
			action_probs = self.action_probs_history[t]
			delta = returns[t] + self.gamma*next_value - value
			if self.learning_method=='ET':
				for a in range(self.n_actions):
					if a==action:
						self.actor[state, a] += self.actor_rate * delta * (1-action_probs[a])
						self.E[state, a] += (1-action_probs[a])
				self.E *= self.gamma * self.lambd
				self.critic[state] += self.critic_rate * delta
			else:
				for a in range(self.n_actions):
					if a==action: self.actor[state, a] += self.actor_rate * delta * (1-action_probs[a])
					# else: self.actor[state, a] -= self.actor_rate * delta * action_probs[a]
				self.critic[state] += self.critic_rate * delta


class DeepQLearning():

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

	def __init__(self, player, seed=0, n_actions=11, n_neurons=200, ID="deep-q-learning", representation='turn-coin', 
			explore_method='epsilon', explore_start=1, explore_decay=0.007, explore_decay_method='linear',
			learning_method='TD0', randomize=True, friendliness=0, critic_rate=3e-2, gamma=0.9, biased_exploration=True, bias=0.75):
		self.player = player
		self.ID = ID
		self.seed = seed
		self.rng = np.random.RandomState(seed=seed)
		self.randomize = randomize
		if self.randomize:
			self.gamma = self.rng.uniform(0.5, 1)
			self.critic_rate = self.rng.uniform(3e-3, 3e-2)
			# self.friendliness = self.rng.uniform(0, 0.3)
			if self.player=='investor':
				if self.rng.uniform(0,1)<0.5: self.friendliness = 0
				else: self.friendliness = 0.2 # 0.15
			elif self.player=='trustee':
				if self.rng.uniform(0,1)<0.5: self.friendliness = 0
				else: self.friendliness = 0.3
		else:
			self.gamma = gamma
			self.friendliness = friendliness
			self.critic_rate = critic_rate
		self.explore_decay = explore_decay
		self.representation = representation
		self.explore_start = explore_start
		self.explore_method = explore_method
		self.explore_decay_method = explore_decay_method
		self.biased_exploration = biased_exploration
		self.bias = bias
		self.n_actions = n_actions
		self.n_inputs = get_n_inputs(representation, player, self.n_actions)
		self.n_neurons = n_neurons
		self.learning_method = learning_method
		torch.manual_seed(seed)
		self.critic = self.Critic(self.n_neurons, self.n_inputs, self.n_actions)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.critic_rate)
		self.critic_value_history = []
		self.action_history = []
		self.state = None
		self.episode = 0

	def reinitialize(self, player, ID, seed):
		self.__init__(player=player, ID=ID, seed=seed)

	def new_game(self, game):
		self.critic_value_history.clear()
		self.action_history.clear()

	def move(self, game):
		game_state = get_state(self.player, self.representation, game=game, return_type='tensor',
			dim=self.n_inputs, n_actions=self.n_actions)
		# Estimate the value of the current game_state
		critic_values = self.critic(game_state)
		# Choose and action based on thees values and some exploration strategy
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
						random_action = biased_actions[self.rng.randint(len(biased_actions))]
					else:
						random_action = unbiased_actions[self.rng.randint(len(unbiased_actions))]
				else:
					random_action = self.rng.randint(self.n_actions)
				action = torch.LongTensor([random_action])
			else:
				action = torch.argmax(critic_values)
		elif self.explore_method=='boltzmann':
			action_probs = torch.nn.functional.softmax(critic_values / explore, dim=0)
			action_dist = torch.distributions.categorical.Categorical(probs=action_probs)
			action = action_dist.sample()	
		else:
			action = torch.argmax(critic_values)
		# translate action into environment-appropriate signal
		self.state = action.detach().numpy() / (self.n_actions-1)
		give, keep, action_idx = action_to_coins(self.player, self.state, self.n_actions, game)
		self.critic_value_history.append(critic_values)
		# self.critic_value_history.append(critic_values[action_idx])
		self.action_history.append(action_idx)
		return give, keep

	def learn(self, game):
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
			# returns = rewards - self.fairness*np.abs(np.array(rewards) - np.array(rewards_other))
		critic_losses = []
		for t in np.arange(game.turns):
			action = self.action_history[t]
			value = self.critic_value_history[t][action]
			next_action = self.action_history[t+1] if t<game.turns-1 else 0
			next_value = torch.max(self.critic_value_history[t+1]) if t<game.turns-1 else 0
			# next_value = self.critic_value_history[t+1][next_action] if t<game.turns-1 else 0
			reward = torch.FloatTensor([returns[t]])
			delta = reward + self.gamma*next_value - value
			critic_loss = delta**2
			critic_losses.append(critic_loss)
		critic_loss = torch.stack(critic_losses).sum()
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()
		self.episode += 1


class InstanceBased():

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

	def __init__(self, player, seed=0, n_actions=11, ID="instance-based", representation='turn-coin',
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


class NengoQLearning():

	class StateInput():
		def __init__(self, n_inputs):
			self.state = np.zeros((n_inputs))
		def set(self, state):
			self.state = state
		def get(self):
			return self.state

	class PastRewardInput():
		def __init__(self, friendliness, n_actions):
			self.history = []
			self.friendliness = friendliness
			self.n_actions = n_actions
			self.past_action = None
		def set(self, player, game):
			rewards = game.investor_reward if player=='investor' else game.trustee_reward
			rewards_other = game.trustee_reward if player=='investor' else game.investor_reward
			reward = (1-self.friendliness)*rewards[-1]+self.friendliness*rewards_other[-1] if len(rewards)>0 else 0
			max_reward = game.coins * game.match
			one_hot_reward = np.zeros((self.n_actions))
			one_hot_reward[self.past_action] = reward / max_reward
			self.history.append(one_hot_reward)
		def clear(self):
			self.history.clear()
		def get(self):
			return self.history[-1] if len(self.history)>0 else 0

	class PastActionInput():
		def __init__(self, n_actions):
			self.history = []
			self.n_actions = n_actions
		def set(self, action):
			one_hot_action = np.zeros((self.n_actions))
			one_hot_action[action] = 1
			self.history.append(one_hot_action)
		def clear(self):
			self.history.clear()
		def get(self):
			return self.history[-1] if len(self.history)>0 else 0

	def __init__(self, player, seed=0, n_actions=11, ID="nengo-q-learning", representation='turn-coin',
			encoder_method='one-hot', learning_rate=3e-7, n_neurons=300, dt=1e-3, turn_time=3e-2, q=30,
			explore_method='epsilon', explore=1, explore_decay=0.007, gamma=0.99, friendliness=0,
			biased_exploration=True, bias=0.75, thalalmus_network=True, randomize=True):
		self.player = player
		self.ID = ID
		self.seed = seed
		self.rng = np.random.RandomState(seed=seed)
		self.representation = representation
		self.n_inputs = get_n_inputs(representation, player, n_actions, extra_turn=1)
		self.n_actions = n_actions
		self.n_neurons = n_neurons
		self.dt = dt
		self.encoder_method = encoder_method
		self.randomize = randomize
		if self.randomize:
			self.gamma = self.rng.uniform(0.5, 1)
			self.learning_rate = self.rng.uniform(1e-7, 1e-6)
			# self.friendliness = self.rng.uniform(0, 0.3)
			if self.player=='investor':
				if self.rng.uniform(0,1)<0.5: self.friendliness = 0
				else: self.friendliness = 0.2
			elif self.player=='trustee':
				if self.rng.uniform(0,1)<0.5: self.friendliness = 0
				else: self.friendliness = 0.4
		else:
			self.gamma = gamma
			self.friendliness = friendliness
			self.learning_rate = learning_rate
		self.q = q 
		self.turn_time = turn_time
		self.explore_method = explore_method
		self.explore = explore
		self.explore_decay = explore_decay
		self.thalalmus_network = thalalmus_network
		self.biased_exploration = biased_exploration
		self.bias = bias
		self.state_input = self.StateInput(self.n_inputs)
		self.reward_input = self.PastRewardInput(self.friendliness, self.n_actions)
		self.action_input = self.PastActionInput(self.n_actions)
		self.delay = nengolib.synapses.PadeDelay(turn_time, q)
		self.d_critic = np.zeros((self.n_inputs, self.n_actions))
		self.network = None
		self.simulator = None
		self.state = None
		self.episode = 0

	def reinitialize(self, player, ID, seed):
		self.__init__(player=player, ID=ID, seed=seed)
		self.network = self.build_network()
		self.simulator = nengo.Simulator(self.network, dt=self.dt, seed=self.seed, progress_bar=False)

	def new_game(self, game):
		self.reward_input.clear()
		self.action_input.clear()
		self.simulator.reset(self.seed)
		self.episode += 1

	def onehot_encoders(self, dims, n_neurons):
		encoders = np.zeros((n_neurons, dims))
		for n in range(n_neurons):
			d = n % dims
			encoders[n, d] = [1,-1][self.rng.randint(2)]
		return encoders

	def build_network(self):
		n_actions = self.n_actions
		n_inputs = self.n_inputs
		n_neurons = self.n_neurons
		seed = self.seed
		network = nengo.Network(seed=seed)
		network.config[nengo.Ensemble].seed = seed
		network.config[nengo.Ensemble].neuron_type = nengo.LIFRate()
		network.config[nengo.Connection].seed = seed
		network.config[nengo.Probe].synapse = None
		intercepts = nengo.dists.Uniform(0.1, 0.1)
		encoders = self.onehot_encoders(n_actions, n_neurons*n_actions)
		radius = 3
		with network:

			class PESNode(nengo.Node):
				def __init__(self, n_neurons, d, learning_rate):
					self.n_neurons = n_neurons
					self.dimensions = d.shape[1]
					self.size_in = 2*n_neurons + self.dimensions
					self.size_out = self.dimensions
					self.d = d
					self.learning_rate = learning_rate
					super().__init__(self.step, size_in=self.size_in, size_out=self.size_out)
				def step(self, t, x):
					activity = x[:self.n_neurons]
					past_activity = x[self.n_neurons: 2*self.n_neurons]
					error = x[2*self.n_neurons:]
					delta = self.learning_rate * past_activity.reshape(-1, 1) * error.reshape(1, -1)
					self.d[:] += delta
					value = np.dot(activity, self.d)
					return value

			class CleanupNode(nengo.Node):
				def __init__(self, n_actions):
					self.n_actions = n_actions
					self.size_in = n_actions
					self.size_out = n_actions
					super().__init__(self.step, size_in=self.size_in, size_out=self.size_out)
				def step(self, t, x):
					one_hot = np.zeros((self.n_actions))
					one_hot[np.argmax(x)] = 1
					return one_hot

			def ScalarProduct(n_neurons, n_actions, seed, neuron_type=nengo.LIFRate(), mag=1):
				net = nengo.Network(seed=seed)
				T = 1.0 / np.sqrt(2)
				with net:
					net.input_a = net.A = nengo.Node(size_in=1, label="input_a")
					net.input_b = net.B = nengo.Node(size_in=n_actions, label="input_b")
					net.output = nengo.Node(size_in=n_actions, label="output")
					net.sq1 = nengo.networks.EnsembleArray(n_neurons, n_actions, neuron_type=neuron_type, ens_dimensions=1, radius=mag*np.sqrt(2), seed=seed)
					net.sq2 = nengo.networks.EnsembleArray(n_neurons, n_actions, neuron_type=neuron_type, ens_dimensions=1, radius=mag*np.sqrt(2), seed=seed)
					nengo.Connection(net.input_a, net.sq1.input, seed=seed, synapse=None, transform=T*np.ones((n_actions, 1)))
					nengo.Connection(net.input_b, net.sq1.input, seed=seed, synapse=None, transform=T)
					nengo.Connection(net.input_a, net.sq2.input, seed=seed, synapse=None, transform=T*np.ones((n_actions, 1)))
					nengo.Connection(net.input_b, net.sq2.input, seed=seed, synapse=None, transform=-T)
					sq1_out = net.sq1.add_output('square', np.square)
					nengo.Connection(sq1_out, net.output, transform=.5, synapse=None)
					sq2_out = net.sq2.add_output('square', np.square)
					nengo.Connection(sq2_out, net.output, transform=-.5, synapse=None)
					net.p_a = nengo.Probe(net.input_a, synapse=None)
					net.p_b = nengo.Probe(net.input_b, synapse=None)
					net.p_out = nengo.Probe(net.output, synapse=None)
				return net

			state_input = nengo.Node(lambda t, x: self.state_input.get(), size_in=2, size_out=self.n_inputs)
			reward_input = nengo.Node(lambda t, x: self.reward_input.get(), size_in=2, size_out=n_actions)
			past_action = nengo.Node(lambda t, x: self.action_input.get(), size_in=2, size_out=n_actions)

			state = nengo.Ensemble(n_inputs, 1, intercepts=intercepts, encoders=nengo.dists.Choice([[1]]))
			learning = PESNode(n_inputs, self.d_critic, self.learning_rate)
			critic = nengo.Ensemble(n_neurons*n_actions, n_actions, radius=radius, encoders=encoders)
			error = nengo.Ensemble(n_neurons*n_actions, n_actions, radius=radius, encoders=encoders)
			current_onehot = nengo.networks.Product(n_neurons*n_actions, n_actions)
			current_max = nengo.Ensemble(n_actions*n_neurons, n_actions, radius=radius, encoders=encoders)
			current_value = ScalarProduct(n_neurons, n_actions, seed, mag=radius)
			past_value = nengo.networks.Product(n_neurons*n_actions, n_actions)
			cleanup = CleanupNode(n_actions)
			for ens in current_onehot.all_ensembles: ens.radius = radius
			for ens in past_value.all_ensembles: ens.radius = radius

			if self.thalalmus_network:
				softmax = nengo.Ensemble(1, n_actions, neuron_type=nengo.Direct())
				bg = nengo.networks.BasalGanglia(n_actions, n_neurons, input_bias=0)
				thalamus = nengo.networks.Thalamus(n_actions, n_neurons)
				for ens in bg.all_ensembles: ens.neuron_type = nengo.LIFRate()
				for ens in thalamus.all_ensembles: ens.neuron_type = nengo.LIFRate()

			nengo.Connection(state_input, state.neurons, synapse=None)
			nengo.Connection(state.neurons, learning[:n_inputs], synapse=None)
			nengo.Connection(state.neurons, learning[n_inputs: 2*n_inputs], synapse=self.delay)
			nengo.Connection(error, learning[2*n_inputs:], synapse=0)
			nengo.Connection(learning, critic, synapse=None)
			nengo.Connection(critic, current_onehot.input_a, synapse=None)
			nengo.Connection(cleanup, current_onehot.input_b, synapse=None)
			nengo.Connection(current_onehot.output, current_max, synapse=None)
			nengo.Connection(current_max, current_value.input_a, function=lambda x: np.sum(x), synapse=None)
			nengo.Connection(past_action, current_value.input_b, synapse=None)
			nengo.Connection(critic, past_value.input_a, synapse=self.delay)
			nengo.Connection(past_action, past_value.input_b, synapse=None)
			nengo.Connection(current_value.output, error, transform=self.gamma, synapse=None)
			nengo.Connection(past_value.output, error, transform=-1, synapse=None)
			nengo.Connection(reward_input, error, synapse=None)

			if self.thalalmus_network:
				nengo.Connection(critic, bg.input, synapse=None, function=lambda x: scipy.special.softmax(10*x))
				nengo.Connection(bg.output, thalamus.input, synapse=None)
				nengo.Connection(thalamus.output, cleanup, synapse=None)
			else:
				nengo.Connection(critic, cleanup, synapse=None)

			network.p_input = nengo.Probe(state_input)
			network.p_state = nengo.Probe(state.neurons)
			network.p_critic = nengo.Probe(critic)
			network.p_learning = nengo.Probe(learning)
			network.p_error = nengo.Probe(error)
			network.p_cleanup = nengo.Probe(cleanup)
			if self.thalalmus_network:
				network.p_bg = nengo.Probe(bg.output)
				network.p_thalamus = nengo.Probe(thalamus.output)
			network.critic = critic
			network.error = error

		return network

	def simulate_action(self):
		self.simulator.run(self.turn_time, progress_bar=False)
		critic = self.simulator.data[self.network.p_critic][-1]
		learning = self.simulator.data[self.network.p_learning][-1]
		cleanup = self.simulator.data[self.network.p_cleanup][-1]
		if self.thalalmus_network:
			bg = self.simulator.data[self.network.p_bg][-1]
			thalamus = self.simulator.data[self.network.p_thalamus][-1]
		if self.explore_method=='epsilon':
			epsilon = self.explore - self.explore_decay*self.episode
			if self.rng.uniform(0, 1) < epsilon:
				if self.biased_exploration:
					biased_actions = [0, int(self.n_actions/2), self.n_actions-1]
					unbiased_actions = np.delete(np.arange(self.n_actions), biased_actions)
					if self.rng.uniform(0,1)<self.bias:
						action = biased_actions[self.rng.randint(len(biased_actions))]
					else:
						action = unbiased_actions[self.rng.randint(len(unbiased_actions))]						
				else:
					action = self.rng.randint(self.n_actions)
			else:
				# action = np.argmax(thalamus)
				action = np.argmax(cleanup)
		# print('thalamus', self.simulator.data[self.network.p_thalamus][-1])
		# print('cleanup', self.simulator.data[self.network.p_cleanup][-1])
		# print('learn', np.around(np.max(np.abs(learning)), 2))
		# print('criti', np.around(np.max(np.abs(critic)), 2))
		# print('perfect', np.argmax(thalamus))
		# print('thalam', np.argmax(thal))
		# thal_val = np.around(np.max(thalamus), 2)
		# print(f'thalamus value \t {np.around(thalamus, 2)}')
		# print(f'same \t {thal_val}') if np.argmax(cleanup) == np.argmax(thalamus) else print(f'diff \t {thal_val}')
		return action

	def move(self, game):
		game_state = get_state(self.player, self.representation, game=game, return_type='one-hot',
			dim=self.n_inputs, n_actions=self.n_actions)
		# add the game state to the network's state input
		self.state_input.set(game_state)
		# use reward from the previous turn for online learning
		self.reward_input.set(self.player, game)
		# turn learning off during testing
		if not game.train: self.network.error.learning_rate = 0
		# simulate the network with these inputs and collect the action outputs
		action = self.simulate_action()
		# translate action into environment-appropriate signal
		self.state = action / (self.n_actions-1)
		give, keep, action_idx = action_to_coins(self.player, self.state, self.n_actions, game)
		# save the chosen action for online learning in the next turn
		self.action_input.set(action_idx)
		self.reward_input.past_action = action_idx
		return give, keep

	def learn(self, game):
		# Learning rules are applied online based on per-turn rewards, so most update happens in the move() step
		# However, we must run one final turn of simulation to permit learning on the last turn.
		# Learner and fixed agents will not make any additional moves that are added to the game history,
		# but the moves recorded on the last turn will be given an opportunity of affect weight update through PES
		give, keep = self.move(game)




class NQ2():

	class Environment():
		def __init__(self, n_states, n_actions, t1, t2, t3, tR, rng, gamma, pos=1, neg=-1, dt=1e-3):
			self.state = np.zeros((n_states))
			self.n_actions = n_actions
			self.rng = rng
			self.reward = 0
			self.t1 = t1
			self.t2 = t2
			self.t3 = t3
			self.t_all = t1+t2+t3
			self.tR = tR
			self.dt = dt
			self.replay = 0
			self.buffer = 0
			self.reset = np.zeros((int(self.t_all/self.dt)+1))
			self.explore = np.zeros((self.n_actions))
			self.pos = pos
			self.neg = neg
			self.gamma = gamma
			for t in np.arange(0, self.t_all, self.dt):
				idx = int((t%self.t_all)/self.dt)
				if 0<t<self.tR: self.reset[idx] = 1
				if self.t1<t<self.t1+self.tR: self.reset[idx] = 1
				if self.t2<t<self.t2+self.tR: self.reset[idx] = 1
		def set_state(self, state):
			self.state = state
		def set_reward(self, player, game, friendliness):
			rewards = game.investor_reward if player=='investor' else game.trustee_reward
			rewards_other = game.trustee_reward if player=='investor' else game.investor_reward
			reward = (1-friendliness)*rewards[-1]+friendliness*rewards_other[-1] if len(rewards)>0 else 0
			self.reward = np.array(reward) / (game.coins * game.match)
			if player=='investor' and len(game.investor_give)<5:
				self.reward *= (1-self.gamma)
			if player=='trustee' and len(game.trustee_give)<5:
				self.reward *= (1-self.gamma)

		def set_explore(self, epsilon):
			if self.rng.uniform(0, 1) < epsilon:
				random_action = self.rng.randint(self.n_actions)
				# one_hot = np.zeros((self.n_actions))
				one_hot = self.neg * np.ones((self.n_actions))
				one_hot[random_action] = self.pos
				# one_hot[~random_action] = self.neg
				self.explore = one_hot
			else:
				self.explore = np.zeros((self.n_actions))
		def get_state(self):
			return self.state
		def get_reward(self):
			return self.reward
		def get_replay(self):
			return self.replay
		def get_buffer(self):
			return self.buffer
		def get_explore(self):
			return self.explore
		def get_reset(self, t):
			idx = int((t%self.t_all)/self.dt)
			return self.reset[idx]

	def __init__(self, player, seed=0, n_actions=11, ID="NQ2", representation='turn-coin',
			encoder_method='one-hot', learning_rate=3e-8, n_neurons=500, dt=1e-3, t1=2e-2, t2=2e-2, t3=2e-2, tR=3e-3, radius=1,
			explore_method='epsilon', explore=1, explore_decay=0.005, gamma=0.6, friendliness=0):
		self.player = player
		self.ID = ID
		self.seed = seed
		self.rng = np.random.RandomState(seed=seed)
		self.representation = representation
		self.n_states = get_n_inputs(representation, player, n_actions, extra_turn=1)
		self.n_actions = n_actions
		self.n_neurons = n_neurons
		self.dt = dt
		self.encoder_method = encoder_method
		self.gamma = gamma
		self.friendliness = friendliness
		self.learning_rate = learning_rate
		self.radius = radius
		self.t1 = t1
		self.t2 = t2
		self.t3 = t3
		self.tR = tR
		self.explore_method = explore_method
		self.explore = explore
		self.explore_decay = explore_decay
		self.env = self.Environment(self.n_states, self.n_actions, t1, t2, t3, tR, self.rng, self.gamma)
		self.decoders = np.zeros((self.n_states, self.n_actions))
		self.network = None
		self.simulator = None
		self.state = None
		self.episode = 0

	def reinitialize(self, player, ID, seed):
		self.__init__(player=player, ID=ID, seed=seed)
		self.network = self.build_network()
		self.simulator = nengo.Simulator(self.network, dt=self.dt, seed=self.seed, progress_bar=False)

	def new_game(self, game):
		self.env.__init__(self.n_states, self.n_actions, self.t1, self.t2, self.t3, self.tR, self.rng, self.gamma)
		self.simulator.reset(self.seed)
		self.episode += 1

	def build_network(self):
		n_actions = self.n_actions
		n_states = self.n_states
		n_neurons = self.n_neurons
		seed = self.seed
		radius = self.radius
		network = nengo.Network(seed=seed)
		network.config[nengo.Ensemble].neuron_type = nengo.LIFRate()
		network.config[nengo.Probe].synapse = None
		intercepts = nengo.dists.Uniform(0.1, 0.1)
		wInh = -1e5*np.ones((n_neurons*n_actions, 1))
		if self.encoder_method=='one-hot':
			encoders = np.zeros((n_neurons*n_actions, n_actions))
			for n in range(encoders.shape[0]):
				enc = np.zeros((n_actions))
				enc[self.rng.randint(n_actions)] = [1,-1][self.rng.randint(2)]
				encoders[n] = np.array(enc)
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

			def GatedSwitch(n_neurons, dim, seed):
				net = nengo.Network(seed=seed)
				wInh = -2e0*np.ones((n_neurons*dim, 1))
				with net:
					net.a = nengo.Node(size_in=dim)
					net.b = nengo.Node(size_in=dim)
					# net.gate = nengo.Node(size_in=1)
					net.gate = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
					net.output = nengo.Node(size_in=dim)
					net.ens_a = nengo.networks.EnsembleArray(n_neurons, dim)
					net.ens_b = nengo.networks.EnsembleArray(n_neurons, dim)
					net.ens_a.add_neuron_input()
					net.ens_b.add_neuron_input()
					nengo.Connection(net.a, net.ens_a.input, synapse=None)
					nengo.Connection(net.b, net.ens_b.input, synapse=None)
					nengo.Connection(net.ens_a.output, net.output, synapse=None)
					nengo.Connection(net.ens_b.output, net.output, synapse=None)
					nengo.Connection(net.gate, net.ens_a.neuron_input, transform=wInh, synapse=None)
					nengo.Connection(net.gate, net.ens_b.neuron_input, transform=wInh, function=lambda x: 1-x, synapse=None)
				return net

			class ChoiceNode(nengo.Node):
				def __init__(self, n_actions):
					self.n_actions = n_actions
					self.size_in = n_actions
					self.size_out = n_actions
					super().__init__(self.step, size_in=self.size_in, size_out=self.size_out)
				def step(self, t, x):
					one_hot = np.zeros((self.n_actions))
					action = np.argmax(x)
					one_hot[action] = 1
					# print(one_hot)
					return one_hot

			def GatedMemory(n_neurons, dim, seed, radius=1, onehot=False, n_gates=1, gain=1, synapse=0):
				net = nengo.Network(seed=seed)
				wInh = -2e0*np.ones((n_neurons*dim, 1))
				with net:
					net.state = nengo.Node(size_in=dim)
					net.gates = [nengo.Ensemble(1, 1, neuron_type=nengo.Direct()) for _ in range(n_gates)]
					if onehot:  # integrator to store value
						net.mem = nengo.networks.EnsembleArray(n_neurons, dim, intercepts=nengo.dists.Uniform(0,1), encoders=nengo.dists.Choice([[1]]), radius=1)
					else:
						net.mem = nengo.networks.EnsembleArray(n_neurons, dim, radius=radius)
					net.diff = nengo.networks.EnsembleArray(n_neurons, dim, radius=radius)  # calculate difference between stored value and input
					net.diff.add_neuron_input()
					net.output = nengo.Ensemble(1, dim, neuron_type=nengo.Direct())
					nengo.Connection(net.state, net.diff.input, synapse=None)
					nengo.Connection(net.diff.output, net.mem.input, transform=gain, synapse=synapse)  # feed difference into integrator
					nengo.Connection(net.mem.output, net.mem.input, synapse=synapse)  # memory feedback
					nengo.Connection(net.mem.output, net.diff.input, transform=-1, synapse=synapse)  # calculate difference between stored value and input
					for g in range(n_gates):
						nengo.Connection(net.gates[g], net.diff.neuron_input, function=lambda x: 1-x, transform=wInh, synapse=None)  # gate the inputs
					nengo.Connection(net.mem.output, net.output, synapse=None)
				return net

			def VectorProduct(n_neurons, n_actions, seed, radius=1):
				net = nengo.Network(seed=seed)
				wInh = -2e0 * np.ones((n_neurons, 1))
				with net:
					net.vector = nengo.Node(size_in=n_actions, label="vector")
					net.onehot = nengo.Node(size_in=n_actions, label="onehot")
					net.output = nengo.Ensemble(1, n_actions, neuron_type=nengo.Direct())
					net.a = nengo.networks.EnsembleArray(n_neurons, n_actions, radius=radius)
					net.b = nengo.networks.EnsembleArray(n_neurons, n_actions, radius=1)
					nengo.Connection(net.vector, net.a.input, synapse=None)
					nengo.Connection(net.onehot, net.b.input, synapse=None)
					for action in range(n_actions):
						# for each non-chosen action in the one-hot vector, inhibit the corresponding dimension in the input vector
						nengo.Connection(net.b.output[action], net.a.ea_ensembles[action].neurons, transform=wInh)
					nengo.Connection(net.a.output, net.output, synapse=None)
				return net

			def ScalarProduct(n_neurons, n_actions, seed, radius=1, transform=1):
				net = nengo.Network(seed=seed)
				wInh = -2e0 * np.ones((n_neurons, 1))
				with net:
					net.scalar = nengo.Node(size_in=1, label="scalar")
					net.onehot = nengo.Node(size_in=n_actions, label="onehot")
					net.output = nengo.Ensemble(1, n_actions, neuron_type=nengo.Direct())
					net.a = nengo.networks.EnsembleArray(n_neurons, n_actions, radius=radius)
					net.b = nengo.networks.EnsembleArray(n_neurons, n_actions, radius=1)
					nengo.Connection(net.onehot, net.b.input, synapse=None)
					for action in range(n_actions):
						# project the scalar to each dimension of an ensemble array
						nengo.Connection(net.scalar, net.a.input[action], synapse=None)
						# for each non-chosen action in the one-hot vector, inhibit the corresponding dimension in the input vector
						nengo.Connection(net.b.output[action], net.a.ea_ensembles[action].neurons, transform=wInh)
					nengo.Connection(net.a.output, net.output, synapse=None, transform=transform)
				return net

			def GatedAccumulator(n_neurons, n_actions, seed, thr=0.97, Tff=1e-2, Tfb=-3e-2, radius=5):
				net = nengo.Network(seed=seed)
				wInhGate = -1e-5 * np.ones((n_neurons*n_actions, n_neurons*n_actions))
				wReset = -1e1 * np.ones((n_neurons*n_actions, 1))
				with net:
					net.input = nengo.Node(size_in=n_actions)
					net.reset = nengo.Node(size_in=1)
					net.gate = nengo.networks.EnsembleArray(n_neurons, n_actions, neuron_type=nengo.LIFRate(), radius=radius)
					net.acc = nengo.networks.EnsembleArray(n_neurons, n_actions, neuron_type=nengo.LIFRate())
					net.inh = nengo.networks.EnsembleArray(n_neurons, n_actions, neuron_type=nengo.LIFRate(),
						intercepts=nengo.dists.Uniform(thr, 1), encoders=nengo.dists.Choice([[1]]))
					net.output = nengo.Node(size_in=n_actions)
					net.gate.add_neuron_input()
					net.acc.add_neuron_input()
					net.inh.add_neuron_output()
					nengo.Connection(net.input, net.gate.input)
					nengo.Connection(net.gate.output, net.acc.input, synapse=None, transform=Tff)
					nengo.Connection(net.acc.output, net.acc.input, synapse=0)
					nengo.Connection(net.acc.output, net.inh.input, synapse=0)
					nengo.Connection(net.inh.neuron_output, net.gate.neuron_input, synapse=0, transform=wInhGate)
					for a in range(n_actions):
						for a2 in range(n_actions):
							if a!=a2:
								nengo.Connection(net.inh.ea_ensembles[a], net.acc.ea_ensembles[a2], synapse=0, transform=Tfb)
					nengo.Connection(net.acc.output, net.output, synapse=None)
					nengo.Connection(net.reset, net.acc.neuron_input, synapse=None, transform=wReset)
					nengo.Connection(net.reset, net.gate.neuron_input, synapse=None, transform=wReset)
				return net

			# inputs from environment
			state_input = nengo.Node(lambda t, x: self.env.get_state(), size_in=2, size_out=n_states)
			reward = nengo.Node(lambda t, x: self.env.get_reward(), size_in=2, size_out=1)
			replay = nengo.Node(lambda t, x: self.env.get_replay(), size_in=2, size_out=1)
			buffer = nengo.Node(lambda t, x: self.env.get_buffer(), size_in=2, size_out=1)
			explore = nengo.Node(lambda t, x: self.env.get_explore(), size_in=2, size_out=n_actions)
			reset = nengo.Node(lambda t, x: self.env.get_reset(t), size_in=2, size_out=1)

			# ensembles and nodes
			state = nengo.networks.EnsembleArray(1, n_states, intercepts=intercepts, encoders=nengo.dists.Choice([[1]]))
			critic = nengo.networks.EnsembleArray(n_neurons, n_actions, radius=radius)
			error = nengo.networks.EnsembleArray(n_neurons, n_actions, radius=1)
			learning = LearningNode(n_states, n_actions, self.decoders, self.learning_rate)
			# normalize = GatedAccumulator(n_neurons, n_actions, radius=radius, seed=seed)
			# normalize = nengo.Ensemble(1, n_actions, neuron_type=nengo.Direct())
			choice = ChoiceNode(n_actions)
			state_memory = GatedMemory(n_neurons, n_states, gain=0.2, seed=seed, onehot=True)
			choice_memory = GatedMemory(n_neurons, n_actions, gain=0.2, seed=seed, onehot=True)
			value_memory = GatedMemory(n_neurons, 1, seed=seed, gain=0.2, n_gates=2, radius=radius)
			state_gate = GatedSwitch(n_neurons, n_states, seed=seed)
			compressed_value_product = VectorProduct(n_neurons, n_actions, seed=seed, radius=radius)
			replayed_value_product = VectorProduct(n_neurons, n_actions, seed=seed, radius=radius)
			buffered_value_product = ScalarProduct(n_neurons, n_actions, seed=seed, radius=radius)
			reward_product = ScalarProduct(n_neurons, n_actions, seed=seed, radius=1)  # , transform=1-self.gamma
			# basal_ganglia = nengo.networks.BasalGanglia(n_actions, n_neurons, input_bias=0.0)
			# thalamus = nengo.networks.Thalamus(n_actions, n_neurons)
			# cleanup = nengo.networks.AssociativeMemory(np.eye(n_actions), n_neurons=n_neurons, seed=seed)
			# cleanup.add_wta_network()
			onehot = nengo.Ensemble(1, n_actions, neuron_type=nengo.Direct())

			# inputs: current state to state memory
			nengo.Connection(state_input, state_memory.state, synapse=None)
			nengo.Connection(buffer, state_memory.gates[0], synapse=None)

			# inputs: current state (stage 1 or 3) OR previous state (stage 2) to state population, gated by replay
			nengo.Connection(state_input, state_gate.a, synapse=None)
			nengo.Connection(state_memory.output, state_gate.b, synapse=None)
			nengo.Connection(replay, state_gate.gate, synapse=None)
			nengo.Connection(state_gate.output, state.input, synapse=None)

			# state to critic connection, computes Q function, updates with DeltaQ from error population
			state.add_neuron_output()
			nengo.Connection(state.neuron_output, learning[:n_states], synapse=None)
			nengo.Connection(error.output, learning[n_states:], synapse=None)
			nengo.Connection(learning, critic.input, synapse=0)

			# Q values sent to WTA competition in choice
			# nengo.Connection(critic.output, basal_ganglia.input, synapse=None)
			# nengo.Connection(explore, basal_ganglia.input, synapse=None)
			# nengo.Connection(basal_ganglia.output, choice, synapse=None)
			nengo.Connection(critic.output, choice, synapse=None)
			nengo.Connection(explore, choice, synapse=None)
			nengo.Connection(choice, onehot, synapse=None)

			# before learning (stage 1), store the Q value of the current state, indexed by the best action in the new state
			nengo.Connection(critic.output, compressed_value_product.vector, synapse=None)
			nengo.Connection(onehot, compressed_value_product.onehot, synapse=None, function=lambda x: 1-x)
			for ens in compressed_value_product.a.ea_ensembles:  # sum all dimensions, reducing the one-hot vector to a 1D estimate of Q(s',a')
				nengo.Connection(ens, value_memory.state, synapse=None)
			nengo.Connection(replay, value_memory.gates[0], synapse=None, function=lambda x: 1-x)
			nengo.Connection(buffer, value_memory.gates[1], synapse=None, function=lambda x: 1-x)

			# after learning (stage 3), store the action selected by the choice ensemble in choice memory
			nengo.Connection(onehot, choice_memory.state, synapse=None)
			nengo.Connection(buffer, choice_memory.gates[0], synapse=None)

			# during replay (stage 2), index all components of the error signal by the action stored in choice memory
			# so that updates to the decoders only affect the dimensions corresponding to a0
			nengo.Connection(value_memory.output, buffered_value_product.scalar, synapse=None)
			nengo.Connection(choice_memory.output, buffered_value_product.onehot, synapse=None, function=lambda x: 1-x)
			nengo.Connection(buffered_value_product.output, error.input, synapse=None, transform=self.gamma)
			nengo.Connection(reward, reward_product.scalar, synapse=None)
			nengo.Connection(choice_memory.output, reward_product.onehot, synapse=None, function=lambda x: 1-x)
			nengo.Connection(reward_product.output, error.input, synapse=None)
			nengo.Connection(critic.output, replayed_value_product.vector, synapse=None)
			nengo.Connection(choice_memory.output, replayed_value_product.onehot, synapse=None, function=lambda x: 1-x)
			nengo.Connection(replayed_value_product.output, error.input, synapse=None, transform=-1)

			# turn learning off until replay (stage 3)
			error.add_neuron_input()
			nengo.Connection(replay, error.neuron_input, synapse=None, function=lambda x: 1-x, transform=wInh)

			network.p_state = nengo.Probe(state.neuron_output)
			network.p_state_memory = nengo.Probe(state_memory.output)
			network.p_critic = nengo.Probe(critic.output)
			network.p_learning = nengo.Probe(learning)
			network.p_error = nengo.Probe(error.output)
			# network.p_normalize = nengo.Probe(normalize.output)
			# network.p_basal_ganglia_in = nengo.Probe(basal_ganglia.input)
			# network.p_basal_ganglia_out = nengo.Probe(basal_ganglia.output)
			# network.p_thalamus = nengo.Probe(thalamus.output)
			network.p_choice = nengo.Probe(choice)
			network.p_onehot = nengo.Probe(onehot)
			network.p_buffer = nengo.Probe(buffer)
			network.p_replay = nengo.Probe(replay)
			network.p_value_memory = nengo.Probe(value_memory.output)
			network.p_value_diff = nengo.Probe(value_memory.diff.output)
			network.p_choice_memory = nengo.Probe(choice_memory.output)
			network.p_compressed_value_product = nengo.Probe(compressed_value_product.output)
			network.p_replayed_value_product = nengo.Probe(replayed_value_product.output)
			network.p_buffered_value_product = nengo.Probe(buffered_value_product.output)
			network.p_reward_product = nengo.Probe(reward_product.output)
			network.p_reset = nengo.Probe(reset)

		return network

	def move(self, game):
		
		# print("Stage 1") # assess the Q value of s', compute a=argmax(Q(s')), and store it in value memory
		self.env.set_reward(self.player, game, self.friendliness)  # reward for the this turn depends on actions taken last turn
		game_state = get_state(self.player, self.representation, game=game, return_type='one-hot', dim=self.n_states, n_actions=self.n_actions)
		self.env.set_state(game_state)
		self.env.set_explore(0)
		self.env.buffer = 0  # do not save the current state to a state memory buffer
		self.env.replay = 0  # do not replay items from memory buffers
		self.simulator.run(self.t1, progress_bar=False)  # store Q(s',a*)
		critic = self.simulator.data[self.network.p_critic]
		value_memory = self.simulator.data[self.network.p_value_memory]
		# print('state', np.around(self.simulator.data[self.network.p_state][-1], 2))
		# print('state memory', np.around(self.simulator.data[self.network.p_state_memory][-1], 2))
		print('critic', np.around(self.simulator.data[self.network.p_critic][-1], 2))
		# print('normalize', np.around(self.simulator.data[self.network.p_normalize][-1], 2))
		# print('bg_in', np.around(self.simulator.data[self.network.p_basal_ganglia_in][-1], 2))
		# print('bg_out', np.around(self.simulator.data[self.network.p_basal_ganglia_out][-1], 2))
		# print('thalamus', np.around(self.simulator.data[self.network.p_thalamus][-1], 2))
		# print('choice', np.around(self.simulator.data[self.network.p_choice][-1], 2))
		# print('onehot', np.around(self.simulator.data[self.network.p_onehot][-1], 2))
		# print('compressed value product', np.around(self.simulator.data[self.network.p_compressed_value_product][-1], 2))
		# print('value memory', np.around(self.simulator.data[self.network.p_value_memory][-1], 2))
		# print(f"critic range: \t {np.around(np.min(critic[-1]), 2)} to {np.around(np.max(critic[-1]), 2)}")
		# print(f'value memory error: \t {100*np.around(np.abs(value_memory[-1]-np.max(critic[-1]))/(np.max(critic[-1])), 2)[0]}%')
		# print(f"argmax critic \t {np.argmax(self.simulator.data[self.network.p_critic][-1])}")
		# print(f"argmax bg  \t {np.argmax(self.simulator.data[self.network.p_basal_ganglia_out][-1])}")
		# print(f"argmax thal \t {np.argmax(self.simulator.data[self.network.p_thalamus][-1])}")
		# print('choice memory', np.around(self.simulator.data[self.network.p_choice_memory][-1], 2))

		# print("Stage 2")  # recall s, a, retrieve Q(s',a*), and R(s,a); compute Q(s,a), compute dQ, and to TD(0) with PES
		self.env.set_explore(0)
		self.env.buffer = 0  # do not save the current state to a state memory buffer
		self.env.replay = 1  # replay items from memory buffers
		self.simulator.run(self.t2, progress_bar=False)  # replay Q(s,a), recall Q(s',a') from value memory, and learn
		# print('state', np.around(self.simulator.data[self.network.p_state][-1], 2))
		# print('critic', np.around(self.simulator.data[self.network.p_critic][-1], 2))
		# print('choice memory', np.around(self.simulator.data[self.network.p_choice_memory][-1], 2))
		# print('value diff', np.around(self.simulator.data[self.network.p_value_diff][-1], 2))
		# print('value memory', np.around(self.simulator.data[self.network.p_value_memory][-1], 2))
		# print('reward', self.env.reward)
		# print('reward product', np.around(self.simulator.data[self.network.p_reward_product][-1], 2))
		# print('error', np.around(self.simulator.data[self.network.p_error][-1], 2))
		# print('replayed value product', np.around(self.simulator.data[self.network.p_replayed_value_product][-1], 2))
		# print('buffered value product', np.around(self.simulator.data[self.network.p_buffered_value_product][-1], 2))
		# print('thalamus', np.around(self.simulator.data[self.network.p_thalamus][-1], 2))
		# print('choice', np.around(self.simulator.data[self.network.p_choice][-1], 2))
		# print('bg_in', np.around(self.simulator.data[self.network.p_basal_ganglia_in][-1], 2))
		# print('bg_out', np.around(self.simulator.data[self.network.p_basal_ganglia_out][-30:], 2))

		# print("Stage 3")  # choose a' with exploration and store s' and a' for next turn
		epsilon = self.explore - self.explore_decay*self.episode
		self.env.set_explore(epsilon)
		self.env.buffer = 1  # save the current state to a state memory buffer
		self.env.replay = 0  # do not replay items from memory buffers
		self.simulator.run(self.t3, progress_bar=False)  # choose a'
		choice = self.simulator.data[self.network.p_onehot][-1]
		action = np.argmax(choice)
		self.state = action / (self.n_actions-1)  # translate action into environment-appropriate signal
		give, keep, action_idx = action_to_coins(self.player, self.state, self.n_actions, game)
		# print('value diff', np.around(self.simulator.data[self.network.p_value_diff][-1], 2))
		# print('state', np.around(self.simulator.data[self.network.p_state][-1], 2))
		# print('critic', np.around(self.simulator.data[self.network.p_critic][-1], 2))
		# print('thalamus', np.around(self.simulator.data[self.network.p_thalamus][-1], 2))
		# print('choice', np.around(self.simulator.data[self.network.p_choice][-1], 2))
		# print('onehot', np.around(self.simulator.data[self.network.p_onehot][-1], 2))
		# print('value memory', np.around(self.simulator.data[self.network.p_value_memory][-1], 2))
		# print('critic', np.around(self.simulator.data[self.network.p_critic][-1], 2))
		# print('normalize_in', np.around(self.simulator.data[self.network.p_normalize_in][-1], 2))
		# print('bg_in', np.around(self.simulator.data[self.network.p_basal_ganglia_in][-1], 2))
		# print('bg_out', np.around(self.simulator.data[self.network.p_basal_ganglia_out][-30:], 2))
		# print('state memory', np.around(self.simulator.data[self.network.p_state_memory][-20:], 2))
		# print('choice memory', np.around(self.simulator.data[self.network.p_choice_memory][-20:], 2))

		return give, keep

	def learn(self, game):
		pass




class NQ3():

	class Environment():
		def __init__(self, n_states, n_actions, t1, t2, t3, rng, gamma):
			self.state = np.zeros((n_states))
			self.n_actions = n_actions
			self.rng = rng
			self.reward = 0
			self.t1 = t1
			self.t2 = t2
			self.t3 = t3
			self.t_all = t1+t2+t3
			self.replay = 0
			self.buffer = 0
			self.explore = np.zeros((self.n_actions))
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
		def set_explore(self, epsilon):
			if self.rng.uniform(0, 1) < epsilon:
				random_action = self.rng.randint(self.n_actions)
				one_hot = np.zeros((self.n_actions))
				one_hot[random_action] = 1e3
				self.explore = one_hot
			else:
				self.explore = np.zeros((self.n_actions))
		def get_state(self):
			return self.state
		def get_reward(self):
			return self.reward
		def get_replay(self):
			return self.replay
		def get_buffer(self):
			return self.buffer
		def get_explore(self):
			return self.explore

	def __init__(self, player, seed=0, n_actions=11, ID="NQ2", representation='turn-coin',
			encoder_method='one-hot', learning_rate=1e-6, n_neurons=300, dt=1e-3, t1=1e-2, t2=1e-2, t3=1e-2,
			explore_method='epsilon', explore=1, explore_decay=0.005, gamma=0.8, friendliness=0):
		self.player = player
		self.ID = ID
		self.seed = seed
		self.rng = np.random.RandomState(seed=seed)
		self.representation = representation
		self.n_states = get_n_inputs(representation, player, n_actions, extra_turn=1)
		self.n_actions = n_actions
		self.n_neurons = n_neurons
		self.dt = dt
		self.encoder_method = encoder_method
		self.gamma = gamma
		self.friendliness = friendliness
		self.learning_rate = learning_rate
		self.t1 = t1
		self.t2 = t2
		self.t3 = t3
		self.explore_method = explore_method
		self.explore = explore
		self.explore_decay = explore_decay
		self.env = self.Environment(self.n_states, self.n_actions, t1, t2, t3, self.rng, self.gamma)
		self.decoders = np.zeros((self.n_states, self.n_actions))
		self.network = None
		self.simulator = None
		self.state = None
		self.episode = 0

	def reinitialize(self, player, ID, seed):
		self.__init__(player=player, ID=ID, seed=seed)
		self.network = self.build_network()
		self.simulator = nengo.Simulator(self.network, dt=self.dt, seed=self.seed, progress_bar=False)

	def new_game(self, game):
		self.env.__init__(self.n_states, self.n_actions, self.t1, self.t2, self.t3, self.rng, self.gamma)
		self.simulator.reset(self.seed)
		self.network.value_memory.memory = 0
		self.episode += 1

	def build_network(self):
		n_actions = self.n_actions
		n_states = self.n_states
		n_neurons = self.n_neurons
		seed = self.seed
		network = nengo.Network(seed=seed)
		network.config[nengo.Ensemble].neuron_type = nengo.LIFRate()
		network.config[nengo.Probe].synapse = None
		wInh = -1e5*np.ones((n_neurons*n_actions, 1))
		if self.encoder_method=='one-hot':
			encoders = np.zeros((n_neurons*n_actions, n_actions))
			for n in range(encoders.shape[0]):
				enc = np.zeros((n_actions))
				enc[self.rng.randint(n_actions)] = [1,-1][self.rng.randint(2)]
				encoders[n] = np.array(enc)
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

			class StateGate(nengo.Node):
				def __init__(self, n_states):
					self.n_states = n_states
					self.size_in = 2*n_states + 1
					self.size_out = n_states
					super().__init__(self.step, size_in=self.size_in, size_out=self.size_out)
				def step(self, t, x):
					n_states = self.n_states
					state_now = x[:n_states]
					state_past = x[n_states: 2*n_states]
					replay = int(x[-1])
					if replay==1:  # pass current state
						passed_state = state_past
					else:  # pass past state
						passed_state = state_now
					# print(t, passed_state)
					return passed_state

			def GatedMemory(n_neurons, dim, seed, n_gates=1, gain=1, synapse=0):
				net = nengo.Network(seed=seed)
				wInh = -2e0*np.ones((n_neurons*dim, 1))
				with net:
					net.state = nengo.Node(size_in=dim)
					net.gates = [nengo.Ensemble(1, 1, neuron_type=nengo.Direct()) for _ in range(n_gates)]
					net.mem = nengo.networks.EnsembleArray(n_neurons, dim,
						intercepts=nengo.dists.Uniform(0,1), encoders=nengo.dists.Choice([[1]]))
					net.diff = nengo.networks.EnsembleArray(n_neurons, dim)  # calculate difference between stored value and input
					net.diff.add_neuron_input()
					net.output = nengo.Ensemble(1, dim, neuron_type=nengo.Direct())
					nengo.Connection(net.state, net.diff.input, synapse=None)
					nengo.Connection(net.diff.output, net.mem.input, transform=gain, synapse=synapse)  # feed difference into integrator
					nengo.Connection(net.mem.output, net.mem.input, synapse=synapse)  # memory feedback
					nengo.Connection(net.mem.output, net.diff.input, transform=-1, synapse=synapse)  # calculate difference between stored value and input
					for g in range(n_gates):
						nengo.Connection(net.gates[g], net.diff.neuron_input, function=lambda x: 1-x, transform=wInh, synapse=None)  # gate the inputs
					nengo.Connection(net.mem.output, net.output, synapse=None)
				return net

			class ValueMemoryNode(nengo.Node):
				def __init__(self, n_actions):
					self.n_actions = n_actions
					self.size_in = 2*n_actions + 2
					self.size_out = 1
					self.memory = 0
					super().__init__(self.step, size_in=self.size_in, size_out=self.size_out)
				def step(self, t, x):
					n_actions = self.n_actions
					critic = x[:n_actions]
					choice = x[n_actions: 2*n_actions]
					replay = int(x[-2])
					buffer = int(x[-1])
					if buffer==0 and replay==0:  # store the (current) Q value indexed by the (current) chosen action
						self.memory = np.dot(critic, choice)
					# print(t, np.around(critic, 2), np.around(choice, 2), np.around(self.memory, 2))
					return self.memory

			class ChoiceNode(nengo.Node):
				def __init__(self, n_actions):
					self.n_actions = n_actions
					self.size_in = n_actions
					self.size_out = n_actions
					super().__init__(self.step, size_in=self.size_in, size_out=self.size_out)
				def step(self, t, x):
					one_hot = np.zeros((self.n_actions))
					action = np.argmax(x)
					one_hot[action] = 1
					# print(one_hot)
					return one_hot

			def ScalarProduct(n_neurons, n_actions, seed):
				net = nengo.Network(seed=seed)
				T = 1.0 / np.sqrt(2)
				with net:
					net.input_a = net.A = nengo.Node(size_in=1, label="input_a")
					net.input_b = net.B = nengo.Node(size_in=n_actions, label="input_b")
					net.output = nengo.Node(size_in=n_actions, label="output")
					net.sq1 = nengo.networks.EnsembleArray(n_neurons, n_actions, radius=np.sqrt(2), neuron_type=nengo.Direct(), seed=seed)
					net.sq2 = nengo.networks.EnsembleArray(n_neurons, n_actions, radius=np.sqrt(2), neuron_type=nengo.Direct(), seed=seed)
					nengo.Connection(net.input_a, net.sq1.input, synapse=None, transform=T*np.ones((n_actions, 1)))
					nengo.Connection(net.input_b, net.sq1.input, synapse=None, transform=T)
					nengo.Connection(net.input_a, net.sq2.input, synapse=None, transform=T*np.ones((n_actions, 1)))
					nengo.Connection(net.input_b, net.sq2.input, synapse=None, transform=-T)
					sq1_out = net.sq1.add_output('square', np.square)
					nengo.Connection(sq1_out, net.output, transform=.5, synapse=None)
					sq2_out = net.sq2.add_output('square', np.square)
					nengo.Connection(sq2_out, net.output, transform=-.5, synapse=None)
					net.p_a = nengo.Probe(net.input_a, synapse=None)
					net.p_b = nengo.Probe(net.input_b, synapse=None)
					net.p_out = nengo.Probe(net.output, synapse=None)
				return net

			def VectorProduct(n_neurons, n_actions, seed):
				net = nengo.Network(seed=seed)
				T = 1.0 / np.sqrt(2)
				with net:
					net.input_a = net.A = nengo.Node(size_in=n_actions, label="input_a")
					net.input_b = net.B = nengo.Node(size_in=n_actions, label="input_b")
					net.output = nengo.Node(size_in=n_actions, label="output")
					net.sq1 = nengo.networks.EnsembleArray(n_neurons, n_actions, radius=np.sqrt(2), neuron_type=nengo.Direct(), seed=seed)
					net.sq2 = nengo.networks.EnsembleArray(n_neurons, n_actions, radius=np.sqrt(2), neuron_type=nengo.Direct(), seed=seed)
					nengo.Connection(net.input_a, net.sq1.input, synapse=None, transform=T)
					nengo.Connection(net.input_b, net.sq1.input, synapse=None, transform=T)
					nengo.Connection(net.input_a, net.sq2.input, synapse=None, transform=T)
					nengo.Connection(net.input_b, net.sq2.input, synapse=None, transform=-T)
					sq1_out = net.sq1.add_output('square', np.square)
					nengo.Connection(sq1_out, net.output, transform=.5, synapse=None)
					sq2_out = net.sq2.add_output('square', np.square)
					nengo.Connection(sq2_out, net.output, transform=-.5, synapse=None)
					net.p_a = nengo.Probe(net.input_a, synapse=None)
					net.p_b = nengo.Probe(net.input_b, synapse=None)
					net.p_out = nengo.Probe(net.output, synapse=None)
				return net

			class ErrorGate(nengo.Node):
				def __init__(self, n_actions):
					self.n_actions = n_actions
					self.size_in = n_actions + 1
					self.size_out = n_actions
					super().__init__(self.step, size_in=self.size_in, size_out=self.size_out)
				def step(self, t, x):
					n_actions = self.n_actions
					error = x[:n_actions]
					gate = int(x[-1])
					if gate==1:  # pass current state
						return error
					else:  # pass past state
						return 0*error

			# inputs from environment
			state_input = nengo.Node(lambda t, x: self.env.get_state(), size_in=2, size_out=n_states)
			reward = nengo.Node(lambda t, x: self.env.get_reward(), size_in=2, size_out=1)
			replay = nengo.Node(lambda t, x: self.env.get_replay(), size_in=2, size_out=1)
			buffer = nengo.Node(lambda t, x: self.env.get_buffer(), size_in=2, size_out=1)
			explore = nengo.Node(lambda t, x: self.env.get_explore(), size_in=2, size_out=n_actions)

			# ensembles and nodes
			state = nengo.networks.EnsembleArray(1, n_states, seed=seed,
				intercepts=nengo.dists.Uniform(0.1, 0.1), encoders=nengo.dists.Choice([[1]]))
			critic = nengo.networks.EnsembleArray(n_neurons, n_actions, seed=seed)
			# error = ErrorGate(n_actions)
			error = nengo.networks.EnsembleArray(n_neurons, n_actions, seed=seed, radius=0.2)
			learning = LearningNode(n_states, n_actions, self.decoders, self.learning_rate)
			choice = ChoiceNode(n_actions)
			# state_memory = StateMemoryNode(n_states)
			state_memory = GatedMemory(n_neurons, n_states, gain=0.3, seed=seed)
			choice_memory = GatedMemory(n_neurons, n_actions, gain=0.3, seed=seed)
			value_memory = ValueMemoryNode(n_actions)
			state_gate = StateGate(n_states)
			replayed_value_product = VectorProduct(n_neurons, n_actions, seed=seed)
			buffered_value_product = ScalarProduct(n_neurons, n_actions, seed=seed)
			reward_product = ScalarProduct(n_neurons, n_actions, seed=seed)

			# inputs: current state to state memory
			nengo.Connection(state_input, state_memory.state, synapse=None)
			nengo.Connection(buffer, state_memory.gates[0], synapse=None)

			# inputs: current state (stage 1 or 3) OR previous state (stage 2) to state population, gated by replay
			nengo.Connection(state_input, state_gate[:n_states], synapse=None)
			nengo.Connection(state_memory.output, state_gate[n_states: 2*n_states], synapse=None)
			nengo.Connection(replay, state_gate[-1], synapse=None)
			nengo.Connection(state_gate, state.input, synapse=None)

			# state to critic connection, computes Q function, updates with DeltaQ from error population
			state.add_neuron_output()
			nengo.Connection(state.neuron_output, learning[:n_states], synapse=None)
			nengo.Connection(error.output, learning[n_states:], synapse=None)
			nengo.Connection(learning, critic.input, synapse=0)

			# Q values sent to WTA competition in choice
			nengo.Connection(critic.output, choice, synapse=None)
			nengo.Connection(explore, choice, synapse=None)

			# before learning (stage 1), store the Q value of the current state, indexed by the best action in the new state
			nengo.Connection(critic.output, value_memory[:n_actions], synapse=None)
			nengo.Connection(choice, value_memory[n_actions: 2*n_actions], synapse=None)
			nengo.Connection(replay, value_memory[-2], synapse=None)
			nengo.Connection(buffer, value_memory[-1], synapse=None)

			# after learning (stage 3), store the action selected by the choice ensemble in choice memory
			nengo.Connection(choice, choice_memory.state, synapse=None)
			nengo.Connection(buffer, choice_memory.gates[0], synapse=None)

			# during replay (stage 2), index all components of the error signal by the action stored in choice memory
			# so that updates to the decoders only affect the dimensions corresponding to a0
			nengo.Connection(value_memory, buffered_value_product.input_a, synapse=None)
			nengo.Connection(choice_memory.output, buffered_value_product.input_b, synapse=None)
			nengo.Connection(buffered_value_product.output, error.input, synapse=None, transform=self.gamma)
			nengo.Connection(reward, reward_product.input_a, synapse=None)
			nengo.Connection(choice_memory.output, reward_product.input_b, synapse=None)
			nengo.Connection(reward_product.output, error.input, synapse=None)
			nengo.Connection(critic.output, replayed_value_product.input_a, synapse=None)
			nengo.Connection(choice_memory.output, replayed_value_product.input_b, synapse=None)
			nengo.Connection(replayed_value_product.output, error.input, synapse=None, transform=-1)

			# turn learning off until replay (stage 3)
			error.add_neuron_input()
			nengo.Connection(replay, error.neuron_input, function=lambda x: 1-x, synapse=None, transform=wInh)

			network.value_memory = value_memory
			network.p_state = nengo.Probe(state.neuron_output)
			network.p_critic = nengo.Probe(critic.output)
			network.p_learning = nengo.Probe(learning)
			network.p_reward = nengo.Probe(reward)
			network.p_error = nengo.Probe(error.output)
			network.p_choice = nengo.Probe(choice)
			network.p_buffer = nengo.Probe(buffer)
			network.p_replay = nengo.Probe(replay)
			network.p_value_memory = nengo.Probe(value_memory)
			network.p_state_memory = nengo.Probe(state_memory.output)
			network.p_choice_memory = nengo.Probe(choice_memory.output)

		return network

	def move(self, game):
		
		# print("Stage 1")
		self.env.set_reward(self.player, game, self.friendliness)  # reward for the this turn depends on actions taken last turn
		game_state = get_state(self.player, self.representation, game=game, return_type='one-hot', dim=self.n_states, n_actions=self.n_actions)
		self.env.set_state(game_state)
		self.env.set_explore(0)
		self.env.buffer = 0  # do not save the current state to a state memory buffer
		self.env.replay = 0  # do not replay items from memory buffers
		self.simulator.run(self.t1, progress_bar=False)  # store Q(s',a*)
		# print('state', np.around(self.simulator.data[self.network.p_state][-1], 2))
		# print('state memory', np.around(self.simulator.data[self.network.p_state_memory][-10:], 2))
		# print('critic', np.around(self.simulator.data[self.network.p_critic][-1], 2))
		# print('choice', np.around(self.simulator.data[self.network.p_choice][-1], 2))
		# print('value memory', np.around(self.simulator.data[self.network.p_value_memory][-1], 2))

		# print("Stage 2")
		self.env.set_explore(0)
		self.env.buffer = 0  # do not save the current state to a state memory buffer
		self.env.replay = 1  # replay items from memory buffers
		self.simulator.run(self.t2, progress_bar=False)  # replay Q(s,a), recall Q(s',a') from value memory, and learn
		# print('state', np.around(self.simulator.data[self.network.p_state][-1], 2))
		# print('state memory', np.around(self.simulator.data[self.network.p_state_memory][-10:], 2))
		# print('critic', np.around(self.simulator.data[self.network.p_critic][-1], 2))
		# print('choice memory', np.around(self.simulator.data[self.network.p_choice_memory][-1], 2))
		# print('value memory', np.around(self.simulator.data[self.network.p_value_memory][-1], 2))
		# print('reward', np.around(self.simulator.data[self.network.p_reward][-1], 2))
		# print('error', np.around(self.simulator.data[self.network.p_error][-1], 2))
		# print('error', np.around(np.min(self.simulator.data[self.network.p_error][-1]), 2),
		# 	np.around(np.max(self.simulator.data[self.network.p_error][-1]), 2))

		# print("Stage 3:")
		epsilon = self.explore - self.explore_decay*self.episode
		self.env.set_explore(epsilon)
		self.env.buffer = 1  # save the current state to a state memory buffer
		self.env.replay = 0  # do not replay items from memory buffers
		self.simulator.run(self.t3, progress_bar=False)  # choose a'
		choice = self.simulator.data[self.network.p_choice][-1]
		action = np.argmax(choice)
		self.state = action / (self.n_actions-1)  # translate action into environment-appropriate signal
		give, keep, action_idx = action_to_coins(self.player, self.state, self.n_actions, game)
		# print('state memory', np.around(self.simulator.data[self.network.p_state_memory][-10:], 2))
		# print('state', np.where(self.simulator.data[self.network.p_state][-1]>0)[0])
		# print('critic', np.around(self.simulator.data[self.network.p_critic][-1], 2))
		# print('choice', np.around(self.simulator.data[self.network.p_choice][-1], 2))
		# print('value memory', np.around(self.simulator.data[self.network.p_value_memory][-1], 2))
		return give, keep

	def learn(self, game):
		pass