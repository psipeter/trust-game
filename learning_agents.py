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
			explore_method='boltzmann', explore_start=30, explore_decay=0.1, explore_decay_method='exponential',
			learning_method='TD0', learning_rate=1e0, gamma=0.99, lambd=0, randomize=True):
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
			explore = np.max([0, self.explore_start - self.explore_decay*self.episode])
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

	def __init__(self, player, seed=0, n_actions=11, n_neurons=100, ID="deep-q-learning", representation='turn-coin', 
			# explore_method='epsilon', explore_start=1, explore_decay=0.001, explore_decay_method='linear',
			explore_method='epsilon', explore_start=1, explore_decay=0.007, explore_decay_method='linear',
			learning_method='TD0', randomize=True, friendliness=0, critic_rate=1e-2, gamma=0.9, biased_exploration=False, bias=0.5):
		self.player = player
		self.ID = ID
		self.seed = seed
		self.rng = np.random.RandomState(seed=seed)
		self.randomize = randomize
		if self.randomize:
			self.gamma = self.rng.uniform(0, 1)
			# self.critic_rate = self.rng.uniform(3e-3, 3e-2)
			self.critic_rate = self.rng.uniform(1e-3, 1e-2)
			# self.friendliness = self.rng.uniform(0, 0.3)
			if self.player=='investor':
				# self.friendliness = self.rng.uniform(0, 0.2)
				if self.rng.uniform(0,1)<0.5: self.friendliness = 0
				else: self.friendliness = 0.2
			elif self.player=='trustee':
				# self.friendliness = self.rng.uniform(0, 0.4)
				if self.rng.uniform(0,1)<0.5: self.friendliness = 0
				else: self.friendliness = 0.4
			# 	else: self.friendliness = 0.2
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
					assert self.n_actions>3
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



class DeepActorCritic():

	class Actor(torch.nn.Module):
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

	def __init__(self, player, seed=0, n_actions=5, n_neurons=200, ID="deep-actor-critic", representation='turn-gen-opponent',
			explore_method='boltzmann', explore=100, explore_decay=0.995,
			learning_method='TD0', critic_rate=1e-3, actor_rate=1e-3, gamma=0.99):
		self.player = player
		self.ID = ID
		self.seed = seed
		self.rng = np.random.RandomState(seed=seed)
		self.gamma = gamma
		self.representation = representation
		self.learning_method = learning_method
		self.actor_rate = actor_rate
		self.critic_rate = critic_rate
		self.explore = explore
		self.explore_decay = explore_decay
		self.explore_method = explore_method
		self.n_inputs = get_n_inputs(representation, player, n_actions)
		self.n_actions = n_actions
		self.n_neurons = n_neurons
		torch.manual_seed(seed)
		self.actor = self.Actor(self.n_neurons, self.n_inputs, self.n_actions)
		self.critic = self.Critic(self.n_neurons, self.n_inputs, 1)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.critic_rate)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.actor_rate)
		self.action_probs_history = []
		self.critic_value_history = []
		self.state = None
		self.episode = 0

	def reinitialize(self, player):
		self.__init__(player)

	def new_game(self, game):
		self.action_probs_history.clear()
		self.critic_value_history.clear()
		self.episode += 1

	def move(self, game):
		game_state = get_state(self.player, self.representation, game=game, return_type='tensor',
			dim=self.n_inputs, n_actions=self.n_actions)
		# Compute action probabilities for the current state and sample action from that distribution
		action_values = self.actor(game_state)
		if self.explore_method=='boltzmann':
			temperature = self.explore * np.power(self.explore_decay, self.episode)
			action_probs = torch.nn.functional.softmax(action_values / temperature, dim=0)
			action_dist = torch.distributions.categorical.Categorical(probs=action_probs)
			action = action_dist.sample()
		else:
			action = torch.argmax(action_values)
		# Estimate value of the current game state
		critic_values = self.critic(game_state)
		# translate action into environment-appropriate signal
		self.state = action.detach().numpy() / (self.n_actions-1)
		give, keep, action_idx = action_to_coins(self.player, self.state, self.n_actions, game)
		# record the actor and critic outputs for end-of-game learning
		# log_prob = torch.log(action_probs.gather(index=torch.LongTensor([action_idx]), dim=0))
		log_prob = torch.log(action_probs.gather(index=action, dim=0))
		self.critic_value_history.append(critic_values)
		self.action_probs_history.append(log_prob)
		return give, keep

	def learn(self, game):
		rewards = game.investor_reward if self.player=='investor' else game.trustee_reward
		actor_losses = []
		critic_losses = []
		if self.learning_method=='MC':
			returns = []
			return_sum = 0
			for t in np.arange(game.turns)[::-1]:
				return_sum += rewards[t]
				returns.insert(0, return_sum)
		else:
			returns = rewards
		for t in np.arange(game.turns):
			value = self.critic_value_history[t]
			next_value = self.critic_value_history[t+1] if t<game.turns-1 else 0
			reward = torch.FloatTensor([returns[t]])
			log_prob = self.action_probs_history[t]
			delta = reward + self.gamma*next_value - value
			actor_loss = -log_prob * delta
			critic_loss = delta**2
			actor_losses.append(actor_loss)
			critic_losses.append(critic_loss)
		actor_loss = torch.stack(actor_losses).sum()
		critic_loss = torch.stack(critic_losses).sum()
		combined_loss = actor_loss + critic_loss
		# print(combined_loss, actor_loss, critic_loss)
		self.actor_optimizer.zero_grad()
		self.critic_optimizer.zero_grad()
		combined_loss.backward()
		# actor_loss.backward(retain_graph=True)
		# critic_loss.backward(retain_graph=True)
		self.actor_optimizer.step()
		self.critic_optimizer.step()



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
			thr_activation=0, thr_action=0.8, thr_state=0.9, friendliness=0, randomize=True,
			learning_method='TD0', gamma=0.99, decay=0.5, epsilon=0.3, biased_exploration=True, bias=0.5,
			# explore_method='epsilon', explore_start=1, explore_decay=0.001, explore_decay_method='linear'):
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
			self.gamma = self.rng.uniform(0, 1)
			self.decay = self.rng.uniform(0.4, 0.6)
			self.epsilon = self.rng.uniform(0.2, 0.4)
			# self.friendliness = self.rng.uniform(0, 0.3)
			# self.friendliness = 0
			if self.player=='investor':
				# self.friendliness = self.rng.uniform(0, 0.2)
				if self.rng.uniform(0,1)<0.5: self.friendliness = 0
				else: self.friendliness = 0.2
			elif self.player=='trustee':
				# self.friendliness = self.rng.uniform(0, 0.4)
				if self.rng.uniform(0,1)<0.5: self.friendliness = 0
				else: self.friendliness = 0.4
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
						assert self.n_actions>3
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

	class ExploreInput():
		def __init__(self, n_actions, rng, value_pos=2, value_neg=-1):
			self.n_actions = n_actions
			self.rng = rng
			self.value_pos = value_pos
			self.value_neg = value_neg
			self.vector = None
		def set(self, explore_start, explore_decay, explore_decay_method, episode):
			if explore_decay_method == 'linear':
				explore = explore_start - explore_decay*episode
			self.vector = np.zeros((self.n_actions))
			if self.rng.uniform(0,1) < explore:
				random_action = self.rng.randint(self.n_actions)
				self.vector[~random_action] = self.value_neg
				self.vector[random_action] = self.value_pos
		def get(self):
			return self.vector

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
			encoder_method='one-hot', learning_rate=3e-7, n_neurons=300, dt=1e-3, turn_time=3e-2, q=10,
			explore_method='epsilon', explore_start=1, explore_decay=0.007, explore_decay_method='linear',
			gamma=0.99, friendliness=0):
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
		self.learning_rate = learning_rate
		self.gamma = gamma
		self.q = q
		self.turn_time = turn_time
		self.friendliness = friendliness
		self.explore_method = explore_method
		self.explore_start = explore_start
		self.explore_decay = explore_decay
		self.explore_decay_method = explore_decay_method
		self.state_input = self.StateInput(self.n_inputs)
		self.reward_input = self.PastRewardInput(self.friendliness, self.n_actions)
		self.action_input = self.PastActionInput(self.n_actions)
		self.explore_input = self.ExploreInput(self.n_actions, self.rng)
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

			class ErrorNode(nengo.Node):
				def __init__(self, n_actions, turn_time, rng):
					self.n_actions = n_actions
					self.size_in = 5*self.n_actions
					self.size_out = n_actions
					self.turn_time = turn_time
					self.rng = rng
					super().__init__(self.step, size_in=self.size_in, size_out=self.size_out)
				def step(self, t, x):
					value = x[:self.n_actions]  # current state of the critic
					past_value = x[self.n_actions: 2*self.n_actions]  # previous state of the critic
					past_reward = x[2*self.n_actions: 3*self.n_actions]  # reward associated with past activities
					past_action = x[3*self.n_actions: 4*self.n_actions]  # action chosen on the previous turn
					current_thalamus = x[4*self.n_actions: 5*self.n_actions]  # action chosen on the previous turn
					# current_value = np.multiply(value, current_thalamus)
					current_action = np.argmax(current_thalamus)
					current_value = value[current_action]
					# print('max', np.around(np.max(value), 2))
					# print('val', np.around(value_of_current_action, 2))
					error = past_reward
					if 0<t<=self.turn_time:
						error *= 0
					elif self.turn_time<t<=5*self.turn_time:
						# error += np.multiply(np.max(value), past_action)
						error += np.multiply(current_value, past_action)
						error -= np.multiply(past_value, past_action)
					elif 5*self.turn_time<t:
						error -= np.multiply(past_value, past_action)
					return error

			state_input = nengo.Node(lambda t, x: self.state_input.get(), size_in=2, size_out=self.n_inputs)
			past_reward = nengo.Node(lambda t, x: self.reward_input.get(), size_in=2, size_out=n_actions)
			past_action = nengo.Node(lambda t, x: self.action_input.get(), size_in=2, size_out=n_actions)
			explore_input = nengo.Node(lambda t, x: self.explore_input.get(), size_in=2, size_out=n_actions)

			state = nengo.Ensemble(n_inputs, 1, intercepts=nengo.dists.Uniform(0.1, 0.1), encoders=nengo.dists.Choice([[1]]))
			learning = PESNode(n_inputs, self.d_critic, self.learning_rate)
			# critic = nengo.Ensemble(n_actions*n_neurons, n_actions, radius=2)
			critic = nengo.Ensemble(1, n_actions, neuron_type=nengo.Direct())
			error = ErrorNode(n_actions, turn_time=self.turn_time, rng=self.rng)
			basal_ganglia = nengo.networks.BasalGanglia(n_actions, n_neurons)
			thalamus = nengo.networks.Thalamus(n_actions, n_neurons)
			probs = nengo.Ensemble(1, n_actions, neuron_type=nengo.Direct())
			# current_choice = nengo.networks.Product(n_neurons, n_actions)
			# past_choice = nengo.networks.Product(n_neurons, n_actions)

			nengo.Connection(state_input, state.neurons, synapse=None)
			nengo.Connection(state.neurons, learning[:n_inputs], synapse=None)
			nengo.Connection(state.neurons, learning[n_inputs: 2*self.n_inputs], synapse=self.delay)
			nengo.Connection(error, learning[2*n_inputs:], synapse=0)
			nengo.Connection(learning, critic, synapse=None)

			nengo.Connection(critic, error[:n_actions], transform=self.gamma, synapse=None)
			nengo.Connection(critic, error[n_actions: 2*n_actions], synapse=self.delay)
			nengo.Connection(past_reward, error[2*n_actions: 3*n_actions], synapse=None)
			nengo.Connection(past_action, error[3*n_actions: 4*n_actions], synapse=None)
			nengo.Connection(thalamus.output, error[4*n_actions: 5*n_actions], synapse=None)

			nengo.Connection(critic, probs, function=lambda x: scipy.special.softmax(30*x), synapse=None)
			nengo.Connection(probs, basal_ganglia.input, synapse=None)
			nengo.Connection(explore_input, basal_ganglia.input, synapse=None)  # epsilon random action
			nengo.Connection(basal_ganglia.output, thalamus.input, synapse=None)

			# nengo.Connection(thalamus.output, current_choice.input_a, synapse=None)
			# nengo.Connection(critic, current_choice.input_b, synapse=None)
			# nengo.Connection(current_choice.output, error[:n_actions], transform=self.gamma, synapse=None)

			# nengo.Connection(thalamus.output, past_choice.input_a, synapse=self.delay)
			# nengo.Connection(critic, past_choice.input_b, synapse=self.delay)
			# nengo.Connection(past_choice.output, error[n_actions: 2*n_actions], transform=self.gamma, synapse=None)


			network.p_input = nengo.Probe(state_input)
			network.p_state = nengo.Probe(state.neurons)
			network.p_learning = nengo.Probe(learning)
			network.p_critic = nengo.Probe(critic)
			network.p_probs = nengo.Probe(probs)
			network.p_error = nengo.Probe(error)
			network.p_bg = nengo.Probe(basal_ganglia.output)
			network.p_thalamus = nengo.Probe(thalamus.output)

		return network

	def simulate_action(self):
		self.simulator.run(self.turn_time, progress_bar=False)
		critic = self.simulator.data[self.network.p_critic][-1]
		probs = self.simulator.data[self.network.p_probs][-1]
		bg = self.simulator.data[self.network.p_bg][-1]
		thalamus = self.simulator.data[self.network.p_thalamus][-1]
		# best_actions = np.where(thalamus==np.amax(thalamus))[0]
		# print(thalamus, best_actions)
		# action = best_actions[self.rng.randint(len(best_actions))]
		action = np.argmax(thalamus)
		# print('critic \t', np.around(critic, 2))
		# print('probs \t', np.around(probs, 2), '\t', np.around(np.sum(probs),2))
		# print('basal \t', np.around(bg, 2))
		print('thalam \t', np.around(thalamus, 2))
		return action

	def move(self, game):
		game_state = get_state(self.player, self.representation, game=game, return_type='one-hot',
			dim=self.n_inputs, n_actions=self.n_actions)
		# add the game state to the network's state input
		self.state_input.set(game_state)
		self.reward_input.set(self.player, game)
		self.explore_input.set(self.explore_start, self.explore_decay, self.explore_decay_method, self.episode)
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



# class NengoActorCritic():

# 	class StateInput():
# 		def __init__(self, n_inputs):
# 			self.state = np.zeros((n_inputs))
# 		def set(self, state):
# 			self.state = state
# 		def get(self):
# 			return self.state

# 	class RewardInput():
# 		def __init__(self, friendliness):
# 			self.history = []
# 			self.friendliness = friendliness
# 		def set(self, player, game):
# 			rewards = game.investor_reward if player=='investor' else game.trustee_reward
# 			rewards_other = game.trustee_reward if player=='investor' else game.investor_reward
# 			reward = (1.0-self.friendliness)*rewards[-1] + self.friendliness*rewards_other[-1] if len(rewards)>0 else 0
# 			max_reward = game.coins * game.match
# 			self.history.append(reward / max_reward)
# 		def clear(self):
# 			self.history.clear()
# 		def get(self):
# 			return self.history[-1] if len(self.history)>0 else 0

# 	class ExploreInput():
# 		def __init__(self, n_actions, rng, value_pos=1, value_neg=0, biased_exploration=False, bias=0.5):
# 			self.n_actions = n_actions
# 			self.rng = rng
# 			self.value_pos = value_pos
# 			self.value_neg = value_neg
# 			self.vector = None
# 			self.epsilon = None
# 			self.temperature = None
# 			self.biased_exploration = biased_exploration
# 			self.bias = bias
# 		def set(self, explore_start, explore_decay, episode):
# 			self.explore = explore_start - explore_decay*episode
# 			# self.explore = explore_start*np.power(explore_decay, episode)
# 			self.vector = np.zeros((self.n_actions))
# 			if self.rng.uniform(0,1) < self.explore:
# 				if self.biased_exploration:
# 					assert self.n_actions>3
# 					biased_actions = [0, int(self.n_actions/2), self.n_actions-1]
# 					unbiased_actions = np.delete(np.arange(self.n_actions), biased_actions)
# 					if self.rng.uniform(0,1)<self.bias:
# 						random_action = biased_actions[self.rng.randint(len(biased_actions))]
# 					else:
# 						random_action = unbiased_actions[self.rng.randint(len(unbiased_actions))]
# 					self.vector[~random_action] = self.value_neg
# 					self.vector[random_action] = self.value_pos
# 				else:
# 					random_action = self.rng.randint(self.n_actions)
# 					self.vector[~random_action] = self.value_neg
# 					self.vector[random_action] = self.value_pos
# 			else:
# 				print('\n choice')
# 		def get(self):
# 			return self.vector

# 	def __init__(self, player, seed=0, n_actions=5, ID="nengo-actor-critic", representation='turn-gen-opponent',
# 			encoder_method='one-hot', actor_rate=1e-7, critic_rate=1e-7, softmax_rate=3e-4,
# 			n_neurons=100, dt=1e-3, turn_time=1e-1, q=6,
# 			explore_method='epsilon', explore_start=1, explore_decay=0.007, explore_decay_method='linear',
# 			# explore_method='epsilon', explore_start=100, explore_decay=0.95, explore_decay_method='power',
# 			biased_exploration=False, bias=0.5,
# 			gamma=0.99, friendliness=0, randomize=False):
# 		self.player = player
# 		self.ID = ID
# 		self.seed = seed
# 		self.rng = np.random.RandomState(seed=seed)
# 		self.representation = representation
# 		self.n_inputs = get_n_inputs(representation, player, n_actions, extra_turn=1)
# 		self.n_actions = n_actions
# 		self.n_neurons = n_neurons
# 		self.dt = dt
# 		self.encoder_method = encoder_method
# 		self.randomize = randomize
# 		if self.randomize:
# 			self.gamma = self.rng.uniform(0, 1)
# 			self.critic_rate = self.rng.uniform(1e-6, 1e-6)
# 			self.actor_rate = self.rng.uniform(1e-6, 1e-6)
# 			self.softmax_rate = self.rng.uniform(1e-4, 1e-4)
# 			if self.player=='investor':
# 				if self.rng.uniform(0,1)<0.5: self.friendliness = 0
# 				else: self.friendliness = 0.2
# 			elif self.player=='trustee':
# 				if self.rng.uniform(0,1)<0.5: self.friendliness = 0
# 				else: self.friendliness = 0.4
# 		else:
# 			self.gamma = gamma
# 			self.friendliness = friendliness
# 			self.critic_rate = critic_rate
# 			self.actor_rate = actor_rate
# 			self.softmax_rate = softmax_rate
# 		# self.delay = nengolib.synapses.DiscreteDelay(int(turn_time/dt))
# 		self.delay = nengolib.synapses.PadeDelay(turn_time, q)
# 		self.turn_time = turn_time
# 		self.explore_start = explore_start
# 		self.explore_decay = explore_decay
# 		self.state_input = self.StateInput(self.n_inputs)
# 		self.reward_input = self.RewardInput(friendliness)
# 		self.explore_input = self.ExploreInput(self.n_actions, self.rng, biased_exploration=biased_exploration, bias=bias)
# 		self.encoders, self.intercepts = self.build_encoders()
# 		self.d_critic = np.zeros((self.n_neurons, 1))
# 		self.d_actor = np.zeros((self.n_neurons, self.n_actions))
# 		self.d_softmax = np.zeros((self.n_actions*self.n_neurons, self.n_actions))
# 		self.state = None
# 		self.network_train = None
# 		self.network_test = None
# 		self.sim_train = None
# 		self.sim_test = None
# 		self.episode = 0


# 	def reinitialize(self, player, ID, seed):
# 		self.__init__(player=player, ID=ID, seed=seed)
# 		self.network_train = self.build_network_train()
# 		self.sim_train = nengo.Simulator(self.network_train, dt=self.dt, progress_bar=True)

# 	def new_game(self, game):
# 		if game.train:
# 			self.reward_input.clear()
# 			self.sim_train.reset(self.seed)
# 			self.episode += 1
# 		else:
# 			self.network_test = self.build_network_test()
# 			self.sim_test = nengo.Simulator(self.network_test, dt=self.dt, progress_bar=False)

# 	def build_encoders(self):
# 		if self.encoder_method=='uniform':
# 			intercepts = nengo.Default
# 			encoders = nengo.Default
# 		elif self.encoder_method=='one-hot':
# 			intercepts = nengo.dists.Uniform(0.1, 1)
# 			encs = []
# 			for dim in range(self.n_inputs):
# 				enc = np.zeros((self.n_inputs))
# 				enc[dim] = 1
# 				encs.append(enc)
# 			encoders = []
# 			for i in range(self.n_neurons):
# 				encoders.append(encs[i%len(encs)])
# 			# encoders = nengo.dists.Choice(encs)
# 		return encoders, intercepts

# 	def build_network_train(self):
# 		seed = self.seed
# 		n_neurons = self.n_neurons
# 		n_actions = self.n_actions
# 		n_inputs = self.n_inputs
# 		w_inh = -1e5*np.ones((self.n_neurons, 1))
# 		network = nengo.Network(seed=seed)
# 		network.config[nengo.Ensemble].seed = seed
# 		network.config[nengo.Ensemble].neuron_type = nengo.LIFRate()
# 		network.config[nengo.Connection].seed = seed
# 		network.config[nengo.Probe].synapse = None
# 		with network:

# 			class PESNode(nengo.Node):
# 				def __init__(self, n_neurons, d, learning_rate):
# 					self.n_neurons = n_neurons
# 					self.dimensions = d.shape[1]
# 					self.size_in = 2*n_neurons + self.dimensions
# 					self.size_out = self.dimensions
# 					self.d = d
# 					self.learning_rate = learning_rate
# 					super().__init__(self.step, size_in=self.size_in, size_out=self.size_out)
# 				def step(self, t, x):
# 					activity = x[:self.n_neurons]
# 					past_activity = x[self.n_neurons: 2*self.n_neurons]
# 					error = x[2*self.n_neurons:]
# 					if self.learning_rate > 0:
# 						delta = (self.learning_rate / self.n_neurons) * past_activity.reshape(-1, 1) * error.reshape(1, -1)
# 						self.d[:] += delta
# 					value = np.dot(activity, self.d)
# 					return value

# 			def learning_control(t, x):
# 				if t<=self.turn_time:
# 					return [1,1,1]  # inhibit all learning on first turn
# 				if self.turn_time < t <= 5*self.turn_time:
# 					return [0,0,0]  # all learning is active
# 				if 5*self.turn_time < t:
# 					return [1,0,0]  # inhibit the value of critic to ignore the max(next_value) term on turn 6

# 			state_input = nengo.Node(lambda t, x: self.state_input.get(), size_in=2, size_out=n_inputs)
# 			past_reward = nengo.Node(lambda t, x: self.reward_input.get(), size_in=2, size_out=1)
# 			explore = nengo.Node(lambda t, x: self.explore_input.get(), size_in=2, size_out=n_actions)
# 			inhibit_learning = nengo.Node(learning_control, size_in=2, size_out=3)
# 			softmax_pes = nengo.PES(learning_rate=self.softmax_rate)
# 			softmax_node = nengo.Ensemble(1, n_actions, neuron_type=nengo.Direct())
# 			actor_node = nengo.Ensemble(1, n_actions, neuron_type=nengo.Direct())

# 			state = nengo.Ensemble(n_neurons, n_inputs, intercepts=self.intercepts, encoders=self.encoders)
# 			reward = nengo.Ensemble(n_neurons, 1)
# 			critic = nengo.Ensemble(n_neurons, 1)
# 			actor = nengo.Ensemble(n_actions*n_neurons, n_actions)
# 			critic_gate = nengo.networks.EnsembleArray(n_neurons, 3)
# 			critic_error = nengo.Ensemble(n_neurons, 1)
# 			critic_pes = PESNode(n_neurons, d=self.d_critic, learning_rate=self.critic_rate)
# 			actor_pes = PESNode(n_neurons, d=self.d_actor, learning_rate=self.actor_rate)
# 			actor_error = nengo.Ensemble(n_actions*n_neurons, n_actions+1)
# 			probs = nengo.Ensemble(n_actions*n_neurons, n_actions)
# 			# softmax_error = nengo.Ensemble(1, n_actions, neuron_type=nengo.Direct())
# 			product_chosen = nengo.networks.Product(n_neurons, n_actions)
# 			product_unchosen = nengo.networks.Product(n_neurons, n_actions)
# 			basal_ganglia = nengo.networks.BasalGanglia(n_actions, n_neurons)
# 			thalamus = nengo.networks.Thalamus(n_actions, n_neurons)

# 			# inputs from environment
# 			nengo.Connection(state_input, state, synapse=None, seed=seed)
# 			nengo.Connection(past_reward, reward, synapse=None, seed=seed)
# 			# update decoders with PES learning
# 			nengo.Connection(state.neurons, actor_pes[:n_neurons], synapse=None)
# 			nengo.Connection(state.neurons, actor_pes[n_neurons: 2*n_neurons], synapse=self.delay)
# 			nengo.Connection(actor_error, actor_pes[2*n_neurons:], function=lambda x: x[-1]*x[:-1], synapse=0)
# 			nengo.Connection(state.neurons, critic_pes[:n_neurons], synapse=None)
# 			nengo.Connection(state.neurons, critic_pes[n_neurons: 2*self.n_neurons], synapse=self.delay)
# 			nengo.Connection(critic_error, critic_pes[2*n_neurons:], synapse=0)
# 			# compute actor and critic values
# 			nengo.Connection(actor_pes, actor, synapse=None)
# 			nengo.Connection(critic_pes, critic, synapse=None)
# 			# compute actor and critic errors
# 			nengo.Connection(critic, critic_gate.input[0], transform=self.gamma, synapse=None)
# 			nengo.Connection(critic, critic_gate.input[1], transform=-1, synapse=self.delay)
# 			nengo.Connection(reward, critic_gate.input[2], synapse=None)
# 			nengo.Connection(critic_gate.output[0], critic_error, synapse=None)
# 			nengo.Connection(critic_gate.output[1], critic_error, synapse=None)
# 			nengo.Connection(critic_gate.output[2], critic_error, synapse=None)
# 			# control learning on turns 0 and 6
# 			nengo.Connection(inhibit_learning[0], critic_gate.ea_ensembles[0].neurons, transform=w_inh, synapse=None)
# 			nengo.Connection(inhibit_learning[1], critic_gate.ea_ensembles[1].neurons, transform=w_inh, synapse=None)
# 			nengo.Connection(inhibit_learning[2], critic_gate.ea_ensembles[2].neurons, transform=w_inh, synapse=None)
# 			# compute action probabilities and select an action
# 			# nengo.Connection(actor, probs, function=lambda x: scipy.special.softmax(x), synapse=None)
# 			# conn_softmax = nengo.Connection(actor.neurons, probs, transform=self.d_softmax.T, synapse=None)
# 			# conn_softmax.learning_rule_type = softmax_pes
# 			nengo.Connection(actor, actor_node, synapse=None)
# 			nengo.Connection(actor_node, softmax_node, function=lambda x: scipy.special.softmax(x), synapse=None)
# 			nengo.Connection(softmax_node, probs, synapse=None)
# 			# nengo.Connection(probs, softmax_error, synapse=None)  # actual
# 			# nengo.Connection(softmax_node, softmax_error, transform=-1, synapse=None)  # target
# 			# nengo.Connection(softmax_error, conn_softmax.learning_rule, synapse=0)

# 			nengo.Connection(probs, basal_ganglia.input, synapse=None)
# 			nengo.Connection(explore, basal_ganglia.input, synapse=None)  # epsilon random action
# 			nengo.Connection(basal_ganglia.output, thalamus.input, synapse=None)
# 			# create one-hot vector of the chosen action probability for computing actor error
# 			# (1-p_a) for chosen action
# 			nengo.Connection(probs, product_chosen.input_a, function=lambda x: 1-x, synapse=self.delay)
# 			nengo.Connection(thalamus.output, product_chosen.input_b, synapse=self.delay)
# 			# -p_a for unchosen action
# 			# nengo.Connection(probs, product_unchosen.input_a, synapse=self.delay)
# 			# nengo.Connection(basal_ganglia.output, product_unchosen.input_b, transform=0.5, synapse=self.delay)
# 			# actor_error to actor_pes computes value_error * transformed probabilities
# 			nengo.Connection(product_chosen.output, actor_error[:n_actions], synapse=None)
# 			# nengo.Connection(product_unchosen.output, actor_error[:n_actions], synapse=None)
# 			nengo.Connection(critic_error, actor_error[-1], synapse=None)

# 			network.p_actor = nengo.Probe(actor)
# 			network.p_critic = nengo.Probe(critic)
# 			network.p_reward = nengo.Probe(reward)
# 			network.p_critic_error = nengo.Probe(critic_error)
# 			network.p_actor_error = nengo.Probe(actor_error)
# 			network.p_probs = nengo.Probe(probs)
# 			network.p_softmax = nengo.Probe(softmax_node)
# 			network.p_basal_ganglia = nengo.Probe(basal_ganglia.output)
# 			network.p_thalamus = nengo.Probe(thalamus.output)
# 			network.p_product_chosen = nengo.Probe(product_chosen.output)
# 			network.p_product_unchosen = nengo.Probe(product_unchosen.output)
# 			# network.p_d_softmax = nengo.Probe(conn_softmax, "weights")
# 			network.actor_pes = actor_pes
# 			network.critic_pes = critic_pes

# 		return network


# 	def build_network_test(self):
# 		seed = self.seed
# 		n_neurons = self.n_neurons
# 		n_actions = self.n_actions
# 		n_inputs = self.n_inputs
# 		network = nengo.Network(seed=seed)
# 		network.config[nengo.Ensemble].seed = seed
# 		network.config[nengo.Ensemble].neuron_type = nengo.LIFRate()
# 		network.config[nengo.Connection].seed = seed
# 		network.config[nengo.Probe].synapse = None
# 		with network:
# 			state_input = nengo.Node(lambda t, x: self.state_input.get(), size_in=2, size_out=n_inputs)
# 			state = nengo.Ensemble(n_neurons, n_inputs, intercepts=self.intercepts, encoders=self.encoders)
# 			actor = nengo.Ensemble(n_actions*n_neurons, n_actions)
# 			probs = nengo.Ensemble(n_actions*n_neurons, n_actions)
# 			basal_ganglia = nengo.networks.BasalGanglia(n_actions, n_neurons)
# 			thalamus = nengo.networks.Thalamus(n_actions, n_neurons)
# 			softmax_node = nengo.Ensemble(1, n_actions, neuron_type=nengo.Direct())
# 			actor_node = nengo.Ensemble(1, n_actions, neuron_type=nengo.Direct())
# 			nengo.Connection(state_input, state, synapse=None, seed=seed)
# 			nengo.Connection(state.neurons, actor, synapse=None, transform=self.d_actor.T)
# 			# nengo.Connection(actor, probs, function=lambda x: scipy.special.softmax(x), synapse=None)
# 			# nengo.Connection(actor.neurons, probs, transform=self.d_softmax.T, synapse=None)
# 			nengo.Connection(actor, actor_node, synapse=None)
# 			nengo.Connection(actor_node, softmax_node, function=lambda x: scipy.special.softmax(x), synapse=None)
# 			nengo.Connection(softmax_node, probs, synapse=None)
# 			nengo.Connection(probs, basal_ganglia.input, synapse=None)
# 			nengo.Connection(basal_ganglia.output, thalamus.input, synapse=None)
# 			network.p_actor = nengo.Probe(actor)
# 			network.p_probs = nengo.Probe(probs)
# 			network.p_basal_ganglia = nengo.Probe(basal_ganglia.output)
# 			network.p_thalamus = nengo.Probe(thalamus.output)
# 		return network


# 	def act_train(self):
# 		self.sim_train.run(self.turn_time, progress_bar=False)
# 		actor = self.sim_train.data[self.network_train.p_actor][-1]
# 		actor_error = self.sim_train.data[self.network_train.p_actor_error][-1]
# 		critic = self.sim_train.data[self.network_train.p_critic][-1]
# 		critic_error = self.sim_train.data[self.network_train.p_critic_error][-1]
# 		probs = self.sim_train.data[self.network_train.p_probs][-1]
# 		product_chosen = self.sim_train.data[self.network_train.p_product_chosen][-1]
# 		product_unchosen = self.sim_train.data[self.network_train.p_product_unchosen][-1]
# 		softmax = self.sim_train.data[self.network_train.p_softmax][-1]
# 		# self.d_softmax = self.sim_train.data[self.network_train.p_d_softmax][-1]
# 		bg = self.sim_train.data[self.network_train.p_basal_ganglia][-1]
# 		thalamus = self.sim_train.data[self.network_train.p_thalamus][-1]
# 		action = np.argmax(thalamus) if np.std(thalamus)>0.1 else self.rng.randint(self.n_actions)
# 		# print('critic \t', np.around(critic, 2))
# 		# print('actor \t', np.around(actor, 2))
# 		# print('probs \t', np.around(np.sum(probs), 2),   '\t', np.around(probs, 2))
# 		print('probs \t', np.around(probs, 2))
# 		# print('softmax \t', np.around(np.sum(softmax), 2), '\t', np.around(softmax, 2))
# 		print('bg \t', np.around(bg, 2))
# 		print('thal \t', np.around(thalamus, 2))
# 		# print('prod chosen \t', np.around(product_chosen, 2))
# 		# print('prod unchosen \t', np.around(product_unchosen, 2))
# 		# print('critic error \t', np.around(critic_error, 2))
# 		# print('actor_error \t', np.around(actor_error, 2))
# 		# print('action', action)
# 		return action, probs

# 	def act_test(self):
# 		self.sim_test.run(self.turn_time, progress_bar=False)
# 		actor = self.sim_test.data[self.network_test.p_actor][-1]
# 		probs = self.sim_test.data[self.network_test.p_probs][-1]
# 		bg = self.sim_test.data[self.network_test.p_basal_ganglia][-1]
# 		thalamus = self.sim_test.data[self.network_test.p_thalamus][-1]
# 		action = np.argmax(thalamus)
# 		return action

# 	def move(self, game):
# 		game_state = get_state(self.player, self.representation, game=game, return_type='one-hot',
# 			dim=self.n_inputs, n_actions=self.n_actions)
# 		self.state_input.set(game_state)  # add the game state to the network's state input
# 		if game.train:
# 			self.reward_input.set(self.player, game)  # use reward from the previous turn for online learning
# 			self.explore_input.set(self.explore_start, self.explore_decay, self.episode)  # epsilon-exploration
# 			action, probs = self.act_train()  # simulate the network with these inputs and collect the action outputs
# 		else:
# 			action = self.act_test()
# 		# translate action into environment-appropriate signal
# 		self.state = action / (self.n_actions-1)
# 		give, keep, action_idx = action_to_coins(self.player, self.state, self.n_actions, game)
# 		return give, keep

# 	def learn(self, game):
# 		# Learning rules are applied online based on per-turn rewards, so decoder update happens in the move() step
# 		# However, we must run one final turn of simulation to permit learning on the last turn.
# 		# Learner and fixed agents will not make any additional moves that are added to the game history,
# 		# but the moves recorded on the last turn will be given an opportunity of affect weight update through PES
# 		if game.train: give, keep = self.move(game)

class NengoActorCritic():

	class StateInput():
		def __init__(self, n_inputs):
			self.state = np.zeros((n_inputs))
		def set(self, state):
			self.state = state
		def get(self):
			return self.state

	class RewardInput():
		def __init__(self):
			self.history = []
		def set(self, player, game):
			rewards = game.investor_reward if player=='investor' else game.trustee_reward
			reward = rewards[-1] if len(rewards)>0 else 0
			max_reward = game.coins * game.match
			self.history.append(reward / max_reward)
		def clear(self):
			self.history.clear()
		def get(self):
			return self.history[-1] if len(self.history)>0 else 0

	class ExploreInput():
		def __init__(self, n_actions, rng, value_pos=1, value_neg=0):
			self.n_actions = n_actions
			self.rng = rng
			self.value_pos = value_pos
			self.value_neg = value_neg
			self.vector = None
			self.epsilon = None
			self.temperature = None
		def set(self, explore, explore_decay, episode):
			self.epsilon = np.power(explore_decay, episode)
			self.temperature = explore*np.power(explore_decay, episode)
			action = self.rng.randint(self.n_actions)
			self.vector = np.zeros((self.n_actions+1))
			if self.rng.uniform(0,1) < self.epsilon:
				self.vector[~action] = self.value_neg
				self.vector[action] = self.value_pos
			self.vector[-1] = self.temperature
		def get(self):
			return self.vector

	def __init__(self, player, seed=0, n_actions=5, ID="nengo-actor-critic", representation='turn-gen-opponent',
			encoder_method='one-hot', actor_rate=3e-6, critic_rate=3e-6, n_neurons=200, dt=1e-3, turn_time=1e-1, q=6,
			explore_method='boltzmann', explore=100, explore_decay=0.995, gamma=0.99):
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
		self.actor_rate = actor_rate
		self.critic_rate = critic_rate
		self.gamma = gamma 
		# self.delay = nengolib.synapses.DiscreteDelay(int(turn_time/dt))
		self.delay = nengolib.synapses.PadeDelay(turn_time, q)
		self.turn_time = turn_time
		self.explore_method = explore_method
		self.explore = explore
		self.explore_decay = explore_decay
		self.state_input = self.StateInput(self.n_inputs)
		self.reward_input = self.RewardInput()
		self.explore_input = self.ExploreInput(self.n_actions, self.rng)
		self.encoders, self.intercepts = self.build_encoders()
		self.d_critic = np.zeros((self.n_neurons, 1))
		self.d_actor = np.zeros((self.n_neurons, self.n_actions))
		self.state = None
		self.network = None
		self.simulator = None
		self.episode = 0


	def reinitialize(self, player, ID, seed):
		self.__init__(player=player, ID=ID, seed=seed)
		self.network = self.build_network()
		self.simulator = nengo.Simulator(self.network, dt=self.dt, progress_bar=True)

	def new_game(self, game):
		self.reward_input.clear()
		self.simulator.reset(self.seed)
		self.episode += 1

	def build_encoders(self):
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
			encoders = []
			for i in range(self.n_neurons):
				encoders.append(encs[i%len(encs)])
			# encoders = nengo.dists.Choice(encs)
		return encoders, intercepts

	def build_network(self):
		seed = self.seed
		n_neurons = self.n_neurons
		n_actions = self.n_actions
		n_inputs = self.n_inputs
		w_inh = -1e5*np.ones((self.n_neurons, 1))
		network = nengo.Network(seed=seed)
		network.config[nengo.Ensemble].seed = seed
		network.config[nengo.Ensemble].neuron_type = nengo.LIFRate()
		network.config[nengo.Connection].seed = seed
		network.config[nengo.Probe].synapse = None
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
					if self.learning_rate > 0:
						delta = (self.learning_rate / self.n_neurons) * past_activity.reshape(-1, 1) * error.reshape(1, -1)
						self.d[:] += delta
					value = np.dot(activity, self.d)
					return value

			def learning_control(t, x):
				if t<=self.turn_time:
					return [1,1,1]  # inhibit all learning on first turn
				if self.turn_time < t <= 5*self.turn_time:
					return [0,0,0]  # all learning is active
				if 5*self.turn_time < t:
					return [1,0,0]  # inhibit the value of critic to ignore the max(next_value) term on turn 6

			state_input = nengo.Node(lambda t, x: self.state_input.get(), size_in=2, size_out=n_inputs)
			past_reward = nengo.Node(lambda t, x: self.reward_input.get(), size_in=2, size_out=1)
			explore = nengo.Node(lambda t, x: self.explore_input.get(), size_in=2, size_out=n_actions+1)
			inhibit_learning = nengo.Node(learning_control, size_in=2, size_out=3)

			state = nengo.Ensemble(n_neurons, n_inputs, intercepts=self.intercepts, encoders=self.encoders)
			reward = nengo.Ensemble(n_neurons, 1)
			critic = nengo.Ensemble(n_neurons, 1)
			actor = nengo.Ensemble(n_actions*n_neurons, n_actions)
			critic_gate = nengo.networks.EnsembleArray(n_neurons, 3)
			critic_error = nengo.Ensemble(n_neurons, 1)
			critic_pes = PESNode(n_neurons, d=self.d_critic, learning_rate=self.critic_rate)
			actor_pes = PESNode(n_neurons, d=self.d_actor, learning_rate=self.actor_rate)
			actor_error = nengo.Ensemble(n_actions*n_neurons, n_actions+1)
			probs = nengo.Ensemble(n_actions*n_neurons, n_actions)
			product = nengo.networks.Product(n_neurons, n_actions)
			basal_ganglia = nengo.networks.BasalGanglia(n_actions, n_neurons)
			thalamus = nengo.networks.Thalamus(n_actions, n_neurons)

			# inputs from environment
			nengo.Connection(state_input, state, synapse=None, seed=seed)
			nengo.Connection(past_reward, reward, synapse=None, seed=seed)

			# update decoders with PES learning
			nengo.Connection(state.neurons, actor_pes[:n_neurons], synapse=None)
			nengo.Connection(state.neurons, actor_pes[n_neurons: 2*n_neurons], synapse=self.delay)
			nengo.Connection(actor_error, actor_pes[2*n_neurons:], function=lambda x: x[-1]*x[:-1], synapse=0)
			nengo.Connection(state.neurons, critic_pes[:n_neurons], synapse=None)
			nengo.Connection(state.neurons, critic_pes[n_neurons: 2*self.n_neurons], synapse=self.delay)
			nengo.Connection(critic_error, critic_pes[2*n_neurons:], synapse=0)

			# compute actor and critic values
			nengo.Connection(actor_pes, actor, synapse=None)
			nengo.Connection(critic_pes, critic, synapse=None)

			# compute actor and critic errors
			nengo.Connection(critic, critic_gate.input[0], transform=self.gamma, synapse=None)
			nengo.Connection(critic, critic_gate.input[1], transform=-1, synapse=self.delay)
			nengo.Connection(reward, critic_gate.input[2], synapse=None)
			nengo.Connection(critic_gate.output[0], critic_error, synapse=None)
			nengo.Connection(critic_gate.output[1], critic_error, synapse=None)
			nengo.Connection(critic_gate.output[2], critic_error, synapse=None)

			# control learning on turns 0 and 6
			nengo.Connection(inhibit_learning[0], critic_gate.ea_ensembles[0].neurons, transform=w_inh, synapse=None)
			nengo.Connection(inhibit_learning[1], critic_gate.ea_ensembles[1].neurons, transform=w_inh, synapse=None)
			nengo.Connection(inhibit_learning[2], critic_gate.ea_ensembles[2].neurons, transform=w_inh, synapse=None)

			# compute action probabilities and select an action
			nengo.Connection(actor, probs, function=lambda x: scipy.special.softmax(x), synapse=None)
			nengo.Connection(probs, basal_ganglia.input, synapse=None)
			nengo.Connection(explore[:-1], basal_ganglia.input, synapse=None)  # epsilon random action
			nengo.Connection(basal_ganglia.output, thalamus.input, synapse=None)

			# create one-hot vector of the chosen action probability for computing actor error
			nengo.Connection(probs, product.input_a, function=lambda x: 1-x, synapse=self.delay)
			nengo.Connection(thalamus.output, product.input_b, synapse=self.delay)
			nengo.Connection(product.output, actor_error[:n_actions], synapse=None)
			nengo.Connection(critic_error, actor_error[-1], synapse=None)

			network.p_actor = nengo.Probe(actor)
			network.p_critic = nengo.Probe(critic)
			network.p_reward = nengo.Probe(reward)
			network.p_critic_error = nengo.Probe(critic_error)
			network.p_actor_error = nengo.Probe(actor_error)
			network.p_probs = nengo.Probe(probs)
			network.p_basal_ganglia = nengo.Probe(basal_ganglia.output)
			network.p_thalamus = nengo.Probe(thalamus.output)
			network.p_product = nengo.Probe(product.output)
			network.actor_pes = actor_pes
			network.critic_pes = critic_pes

		return network

	def simulate_action(self):
		self.simulator.run(self.turn_time, progress_bar=False)
		actor = self.simulator.data[self.network.p_actor][-1]
		actor_error = self.simulator.data[self.network.p_actor_error][-1]
		critic = self.simulator.data[self.network.p_critic][-1]
		critic_error = self.simulator.data[self.network.p_critic_error][-1]
		probs = self.simulator.data[self.network.p_probs][-1]
		# action = self.simulator.data[self.network.p_choice][-1]
		bg = self.simulator.data[self.network.p_basal_ganglia][-1]
		thalamus = self.simulator.data[self.network.p_thalamus][-1]
		action = np.argmax(thalamus) if np.std(thalamus)>0.1 else self.rng.randint(self.n_actions)
		# print('critic \t', critic)
		# print('actor \t', actor)
		# print('probs \t', probs)
		# print('thal \t', thalamus)
		# print('product \t', product)
		# print('bg \t', bg)
		# print('critic error', critic_error)
		# print('critic error', critic_error)
		# print('actor_error', actor_error)
		# print('action', action)
		return action, probs

	def move(self, game):
		game_state = get_state(self.player, self.representation, game=game, return_type='one-hot',
			dim=self.n_inputs, n_actions=self.n_actions)
		# add the game state to the network's state input
		self.state_input.set(game_state)
		# use reward from the previous turn for online learning
		self.reward_input.set(self.player, game)
		# turn learning on/off, depending on the situation
		if not game.train:
			self.network.critic_pes.learning_rate = 0
			self.network.actor_pes.learning_rate = 0
		# epsilon-exploration
		self.explore_input.set(self.explore, self.explore_decay, self.episode)
		# simulate the network with these inputs and collect the action outputs
		action, probs = self.simulate_action()
		# translate action into environment-appropriate signal
		self.state = action / (self.n_actions-1)
		give, keep, action_idx = action_to_coins(self.player, self.state, self.n_actions, game)
		return give, keep

	def learn(self, game):
		# Learning rules are applied online based on per-turn rewards, so decoder update happens in the move() step
		# However, we must run one final turn of simulation to permit learning on the last turn.
		# Learner and fixed agents will not make any additional moves that are added to the game history,
		# but the moves recorded on the last turn will be given an opportunity of affect weight update through PES
		give, keep = self.move(game)