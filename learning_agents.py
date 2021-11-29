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
			explore_method='boltzmann', explore=100, explore_decay=0.92,
			learning_method='TD0', learning_rate=1e0, gamma=0.99, lambd=0.8):
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
		self.learning_rate = learning_rate
		self.learning_method = learning_method
		self.Q = np.zeros((self.n_inputs, self.n_actions))
		self.E = np.zeros((self.n_inputs, self.n_actions))  # eligibility trace
		self.state_history = []
		self.action_history = []
		self.state = None
		self.episode = 0

	def reinitialize(self, player):
		self.__init__(player)

	def new_game(self):
		self.state_history.clear()
		self.action_history.clear()
		self.E = np.zeros((self.n_inputs, self.n_actions))
		self.episode += 1

	def move(self, game):
		game_state = get_state(self.player, self.representation, game=game, return_type='index', n_actions=self.n_actions)
		# Compute action probabilities for the current state
		Q_state = self.Q[game_state]
		# Sample action from q-values in the current state
		if self.explore_method=='epsilon':
			epsilon = self.explore*np.power(self.explore_decay, self.episode)
			if self.rng.uniform(0, 1) < epsilon:
				action = self.rng.randint(self.n_actions)
			else:
				action = np.argmax(Q_state)
		elif self.explore_method=='boltzmann':
			temperature = self.explore*np.power(self.explore_decay, self.episode)
			action_probs = scipy.special.softmax(Q_state / temperature)
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
			action = self.action_history[t]
			next_state = self.state_history[t+1] if t<game.turns-1 else None
			next_action = self.action_history[t+1] if t<game.turns-1 else None
			reward = rewards[t]
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

	def __init__(self, player, seed=0, n_actions=11, ID="tabular-actor-critic", representation='turn-gen-opponent',
			explore_method='boltzmann', explore=100, explore_decay=0.998,
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

	def new_game(self):
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

	def __init__(self, player, seed=0, n_actions=11, n_neurons=200, ID="deep-q-learning", representation='turn-gen-opponent',
			explore_method='boltzmann', explore=100, explore_decay=0.998,
			learning_method='TD0', critic_rate=1e-2, gamma=0.99, lambd=0.8):
		self.player = player
		self.ID = ID
		self.seed = seed
		self.rng = np.random.RandomState(seed=seed)
		self.gamma = gamma
		self.lambd = lambd
		self.representation = representation
		self.critic_rate = critic_rate
		self.explore = explore
		self.explore_decay = explore_decay
		self.explore_method = explore_method
		self.n_inputs = get_n_inputs(representation, player, n_actions)
		self.n_actions = n_actions
		self.n_neurons = n_neurons
		self.learning_method = learning_method
		torch.manual_seed(seed)
		self.critic = self.Critic(self.n_neurons, self.n_inputs, self.n_actions)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.critic_rate)
		self.critic_value_history = []
		self.action_history = []
		self.state = None
		self.episode = 0

	def reinitialize(self, player):
		self.__init__(player)

	def new_game(self):
		self.critic_value_history.clear()
		self.action_history.clear()
		self.episode += 1

	def move(self, game):
		game_state = get_state(self.player, self.representation, game=game, return_type='tensor',
			dim=self.n_inputs, n_actions=self.n_actions)
		# Estimate the value of the current game_state
		critic_values = self.critic(game_state)
		# Choose and action based on thees values and some exploration strategy
		if self.explore_method=='epsilon':
			epsilon = self.explore*np.power(self.explore_decay, self.episode)
			if self.rng.uniform(0, 1) < epsilon:
				action = torch.LongTensor([self.rng.randint(self.n_actions)])
			else:
				action = torch.argmax(critic_values)
		elif self.explore_method=='boltzmann':
			temperature = self.explore*np.power(self.explore_decay, self.episode)
			action_probs = torch.nn.functional.softmax(critic_values / temperature, dim=0)
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
		if self.learning_method=='MC':
			returns = []
			return_sum = 0
			for t in np.arange(game.turns)[::-1]:
				return_sum += rewards[t]
				returns.insert(0, return_sum)
		else:
			returns = rewards
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

	def __init__(self, player, seed=0, n_actions=11, n_neurons=200, ID="deep-actor-critic", representation='turn-gen-opponent',
			explore_method='boltzmann', explore=100, explore_decay=0.998,
			learning_method='TD0', critic_rate=2e-3, actor_rate=2e-3, gamma=0.99):
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

	def new_game(self):
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
		def __init__(self, state, action, reward, value, episode, decay=0.5, epsilon=0.3):
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

	def __init__(self, player, seed=0, n_actions=11, ID="instance-based", representation='turn-gen-opponent',
			populate_method='state-similarity', value_method='next-value',
			thr_activation=0, thr_action=0.8, thr_state=0.8,
			learning_method='TD0', gamma=0.99,
			explore_method='boltzmann', explore=100, explore_decay=0.93):
		self.player = player
		self.ID = ID
		self.seed = seed
		self.rng = np.random.RandomState(seed=seed)
		self.representation = representation
		self.n_inputs = get_n_inputs(representation, player, n_actions)
		self.n_actions = n_actions
		self.learning_method = learning_method
		self.gamma = gamma
		self.thr_activation = thr_activation  # activation threshold for retrieval (loading chunks from declarative into working memory)
		self.thr_action = thr_action  # action similarity threshold for retrieval (loading chunks from declarative into working memory)
		self.thr_state = thr_state  # state similarity threshold for retrieval (loading chunks from declarative into working memory)
		self.explore = explore  # probability of random action, for exploration
		self.explore_decay = explore_decay  # per-episode reduction of exploration
		self.explore_method = explore_method
		self.populate_method = populate_method  # method for determining whether a chunk in declaritive memory meets threshold
		self.value_method = value_method  # method for assigning value to chunks during learning
		self.declarative_memory = []
		self.working_memory = []
		self.learning_memory = []
		self.state = None
		self.episode = 0  # for tracking activation within / across games

	def reinitialize(self, player):
		self.__init__(player)

	def new_game(self):
		self.working_memory.clear()
		self.learning_memory.clear()
		self.rng.shuffle(self.declarative_memory)
		self.episode += 1

	def move(self, game):
		game_state = get_state(self.player, self.representation, game=game, return_type='index', n_actions=self.n_actions)
		# load chunks from declarative memory into working memory
		self.populate_working_memory(game_state, game)
		# select an action (generosity) that immitates the best chunk in working memory
		self.state = self.select_action()
		# create a new chunk for the chosen action, populate with more information in learn()
		new_chunk = self.Chunk(state=game_state, action=None, reward=None, value=None, episode=self.episode)
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
				if current_turn==chunk_turn: similarity_state = 1
				else: similarity_state = 0
			elif self.representation=='turn-gen-opponent':
				chunk_turn = int(chunk.state / (self.n_actions+1))
				if current_turn==chunk_turn:
					gens_current = game.trustee_gen if self.player=='investor' else game.investor_gen
					index_chunk = chunk.state % (self.n_actions+1)
					if self.player=='investor':
						# get generosity of opponent for the current game state
						if current_turn==0: gen_current = 0
						elif np.isnan(gens_current[-1]): gen_current = -1
						else: gen_current = gens_current[-1]
						# get generosity of opponent for the remembered chunk's game state					
						if index_chunk==0: gen_chunk = 0
						elif index_chunk == self.n_actions: gen_chunk = -1
						else: gen_chunk = index_chunk / (self.n_actions-1)
					else:
						gen_current = gens_current[-1]
						gen_chunk = index_chunk / (self.n_actions-1)
					if gen_current==-1 and gen_chunk==-1:
						similarity_state = 1
					elif gen_current==-1 or gen_chunk==-1:
						similarity_state = 0
					else:
						similarity_state = 1 - np.abs(gen_current - gen_chunk)
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
			if self.explore_method=='epsilon':
				epsilon = self.explore * np.power(self.explore_decay, self.episode)
				if self.rng.uniform(0, 1) < epsilon:
					selected_action = self.rng.randint(0, self.n_actions) / (self.n_actions-1)
				else:
					selected_action = max(actions, key=lambda action: actions[action]['blended'])
			elif self.explore_method=='boltzmann':
				temperature = self.explore * np.power(self.explore_decay, self.episode)
				action_gens = np.array([a for a in actions])
				action_values = np.array([actions[a]['blended'] for a in actions])
				action_probs = scipy.special.softmax(action_values / temperature)
				selected_action = self.rng.choice(action_gens, p=action_probs)
		return selected_action

	def learn(self, game):
		# update value of new chunks according to some scheme
		actions = game.investor_gen if self.player=='investor' else game.trustee_gen
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
			chunk = self.learning_memory[t]
			next_chunk = self.learning_memory[t+1] if t<(game.turns-1) else None
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
				for next_chunk in self.declarative_memory:
					activation = next_chunk.get_activation(self.episode, self.rng)
					if self.representation=='turn':
						chunk_turn = next_chunk.state
						similarity_state = 1 if next_turn==chunk_turn else 0
					elif self.representation=='turn-gen-opponent':
						chunk_turn = int(next_chunk.state / (self.n_actions+1))
						if next_turn==chunk_turn:
							gens_current = game.trustee_gen if self.player=='investor' else game.investor_gen
							index_chunk = next_chunk.state % (self.n_actions+1)
							if self.player=='investor':
								# get generosity of opponent for the next turn's game state
								if np.isnan(gens_current[next_turn]): gen_current = -1
								else: gen_current = gens_current[next_turn]
								# get generosity of opponent for the remembered chunk's game state					
								if index_chunk == self.n_actions: gen_chunk = -1
								else: gen_chunk = index_chunk / (self.n_actions-1)
							else:
								gen_current = gens_current[next_turn]
								gen_chunk = index_chunk / (self.n_actions-1)
							if gen_current==-1 and gen_chunk==-1:
								similarity_state = 1
							elif gen_current==-1 or gen_chunk==-1:
								similarity_state = 0
							else:
								similarity_state = 1 - np.abs(gen_current - gen_chunk)
						else:
							similarity_state = 0
					pass_activation = activation > self.thr_activation
					pass_state = similarity_state > self.thr_state
					if pass_activation and pass_state:
						next_chunk_rewards.append(next_chunk.reward)
						next_chunk_values.append(next_chunk.value)
						next_chunk_activations.append(activation)
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


class NengoQLearning():

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
		def set(self, player, game):
			rewards = game.investor_reward if player=='investor' else game.trustee_reward
			reward = rewards[-1] if len(rewards)>0 else 0
			self.history.append(reward)
		def clear(self):
			self.history.clear()
		def get(self):
			return self.history[-1] if len(self.history)>0 else 0

	class PastActionInput():
		def __init__(self):
			self.history = []
		def set(self, action):
			self.history.append(action)
		def clear(self):
			self.history.clear()
		def get(self):
			return self.history[-1] if len(self.history)>0 else 0

	class LearningInput():
		def __init__(self):
			self.learning = 0
		def set(self, train):
			self.learning = 1 if train else 0
		def get(self):
			return self.learning

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

	# Terry's code for LMUs
	class LMU(nengo.Process):
		def __init__(self, theta, q, size_in=1):
			self.q = q              # number of internal state dimensions per input
			self.theta = theta      # size of time window (in seconds)
			self.size_in = size_in  # number of inputs
			# Do Aaron's math to generate the matrices
			#  https://github.com/arvoelke/nengolib/blob/master/nengolib/synapses/analog.py#L536
			Q = np.arange(q, dtype=np.float64)
			R = (2*Q + 1)[:, None] / theta
			j, i = np.meshgrid(Q, Q)
			self.A = np.where(i < j, -1, (-1.)**(i-j+1)) * R
			self.B = (-1.)**Q[:, None] * R
			super().__init__(default_size_in=size_in, default_size_out=q*size_in)

		def make_step(self, shape_in, shape_out, dt, rng, state=None):
			state = np.zeros((self.q, self.size_in))
			# Handle the fact that we're discretizing the time step
			#  https://en.wikipedia.org/wiki/Discretization#Discretization_of_linear_state_space_models
			Ad = scipy.linalg.expm(self.A*dt)
			Bd = np.dot(np.dot(np.linalg.inv(self.A), (Ad-np.eye(self.q))), self.B)
			# this code will be called every timestep
			def step_legendre(t, x, state=state):
				state[:] = np.dot(Ad, state) + np.dot(Bd, x[None, :])
				return state.T.flatten()
			return step_legendre

		def get_weights_for_delays(self, r):
			# compute the weights needed to extract the value at time r
			# from the network (r=0 is right now, r=1 is theta seconds ago)
			r = np.asarray(r)
			m = np.asarray([scipy.special.legendre(i)(2*r - 1) for i in range(self.q)])
			return m.reshape(self.q, -1).T

	class Delay:
		def __init__(self, dimensions, timesteps=1):
			self.history = np.zeros((timesteps+1, dimensions))
		def step(self, t, x):
			self.history = np.roll(self.history, -1, axis=0)
			self.history[-1] = x
			return self.history[0]


	def __init__(self, player, seed=0, n_actions=11, ID="nengo-q-learning", representation='turn-gen-opponent',
			encoder_method='one-hot', learning_rate=1e-4, n_neurons=100, dt=1e-3,
			turn_time=1e-2, q=10,
			explore_method='boltzmann', explore=100, explore_decay=0.998, gamma=0.99):
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
		self.q = q  # order of LMU
		self.turn_time = turn_time
		self.explore_method = explore_method
		self.explore = explore
		self.explore_decay = explore_decay
		self.state_input = self.StateInput(self.n_inputs)
		self.reward_input = self.PastRewardInput()
		self.action_input = self.PastActionInput()
		self.learning_input = self.LearningInput()
		self.encoders, self.intercepts = self.build_encoders()
		self.d_critic = np.zeros((self.n_neurons, self.n_actions))
		self.network = self.build_network()
		self.simulator = nengo.Simulator(self.network, dt=self.dt, seed=self.seed)
		self.state = None
		self.episode = 0

	def reinitialize(self, player):
		self.__init__(player)

	def new_game(self):
		self.reward_input.clear()
		self.action_input.clear()
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
		network = nengo.Network(seed=self.seed)
		with network:

			class CriticNode(nengo.Node):
				def __init__(self, n_neurons, n_actions, turn_time, d, learning_rate, gamma):
					self.n_neurons = n_neurons
					self.n_actions = n_actions
					self.size_in = 2*n_neurons + 3
					self.size_out = n_actions
					self.turn_time = turn_time
					self.d = d
					self.learning_rate = learning_rate
					self.gamma = gamma
					super().__init__(self.step, size_in=self.size_in, size_out=self.size_out)
				def step(self, t, x):
					activity = x[:self.n_neurons]  # current synaptic activities from "state" population
					past_activity = x[self.n_neurons: 2*self.n_neurons]  # delayed synaptic activities from "state" population
					past_action = int(x[-3])  # action chosen on the previous turn
					past_reward = x[-2]  # reward associated with past activities
					learning = x[-1]  # gating signal for updating decoders
					value = np.dot(activity, self.d)  # current state of the critic
					past_value = np.dot(past_activity, self.d)  # previous state of the critic
					# print(t, activity[:5], past_activity[:5], past_reward, past_action, learning)
					if 0<t<=self.turn_time:
						error = 0
					elif self.turn_time<t<=5*self.turn_time:
						error = learning * (past_reward + self.gamma*np.max(value) - past_value[past_action]) # TD0 update
					elif 5*self.turn_time<t:
						error = learning * (past_reward - past_value[past_action]) # TD0 update on last turn
					delta = (self.learning_rate / self.n_neurons) * past_activity * error  # PES update
					self.d[:,past_action] += delta
					return value  # return the current state of the critic (Q-value)

			state_input = nengo.Node(lambda t, x: self.state_input.get(), size_in=2, size_out=self.n_inputs)
			past_reward = nengo.Node(lambda t, x: self.reward_input.get(), size_in=2, size_out=1)
			past_action = nengo.Node(lambda t, x: self.action_input.get(), size_in=2, size_out=1)
			learning_input = nengo.Node(lambda t, x: self.learning_input.get(), size_in=2, size_out=1)

			state = nengo.Ensemble(self.n_neurons, self.n_inputs, seed=self.seed,
				intercepts=self.intercepts, encoders=self.encoders, neuron_type=nengo.LIFRate())
			state_delayed = nengo.Ensemble(self.n_neurons, self.n_inputs, seed=self.seed,
				intercepts=self.intercepts, encoders=self.encoders, neuron_type=nengo.LIFRate())
			# delay = self.Delay(dimensions=self.n_inputs, timesteps=int(self.turn_time/self.dt))
			# delay = self.LMU(theta=self.turn_time, q=self.q, size_in=self.n_inputs)
			# memory = nengo.Node(delay)
			# memory = nengo.Node(delay.step, size_in=self.n_inputs, size_out=self.n_inputs)
			memory = []
			for dim in range(self.n_inputs):
				memory.append(nengolib.networks.RollingWindow(
					theta=self.turn_time, n_neurons=100, neuron_type=nengo.LIFRate(), legendre=True, dimensions=self.q, process=None))
			critic = CriticNode(self.n_neurons, self.n_actions,
				turn_time=self.turn_time, d=self.d_critic, learning_rate=self.learning_rate, gamma=self.gamma)

			# compute decoders from LMU to decode the activity of the state population turn_time seconds ago
			# d_delay = np.kron(np.eye(self.n_inputs), delay.get_weights_for_delays(r=1)) # r=1 corresponds to delay=theta
			# nengo.Connection(memory, state_delayed, transform=d_delay, synapse=None)

			nengo.Connection(state_input, state, synapse=None)
			# nengo.Connection(state_input, memory, synapse=None)
			for dim in range(self.n_inputs):
				nengo.Connection(state_input[dim], memory[dim].input, synapse=None)
				nengo.Connection(memory[dim].output, state_delayed[dim], synapse=None)
			# nengo.Connection(memory, state_delayed, synapse=None)

			nengo.Connection(state.neurons, critic[:self.n_neurons], synapse=None)
			nengo.Connection(state_delayed.neurons, critic[self.n_neurons: 2*self.n_neurons], synapse=None)
			nengo.Connection(past_action, critic[-3], synapse=None)
			nengo.Connection(past_reward, critic[-2], synapse=None)
			nengo.Connection(learning_input, critic[-1], synapse=None)

			network.p_input = nengo.Probe(state_input, synapse=None)
			network.p_state = nengo.Probe(state, synapse=None)
			network.p_critic = nengo.Probe(critic, synapse=None)
			network.d_critic = critic.d

		return network

	def simulate_action(self):
		self.simulator.run(self.turn_time)
		x_critic = self.simulator.data[self.network.p_critic][-1]
		if self.explore_method=='epsilon':
			epsilon = self.explore*np.power(self.explore_decay, self.episode)
			if self.rng.uniform(0, 1) < epsilon:
				action = self.rng.randint(self.n_actions)
			else:
				action = np.argmax(x_critic)
		elif self.explore_method=='boltzmann':
			temperature = self.explore*np.power(self.explore_decay, self.episode)
			action_probs = scipy.special.softmax(x_critic / temperature)
			action = self.rng.choice(np.arange(self.n_actions), p=action_probs)
		else:
			action = torch.argmax(x_critic)
		return action

	def move(self, game):
		game_state = get_state(self.player, self.representation, game=game, return_type='one-hot',
			dim=self.n_inputs, n_actions=self.n_actions)
		# add the game state to the network's state input
		self.state_input.set(game_state)
		# use reward from the previous turn for online learning
		self.reward_input.set(self.player, game)
		# turn learning off during testing
		self.learning_input.set(game.train)
		# simulate the network with these inputs and collect the action outputs
		action = self.simulate_action()
		# translate action into environment-appropriate signal
		self.state = action / (self.n_actions-1)
		give, keep, action_idx = action_to_coins(self.player, self.state, self.n_actions, game)
		# save the chosen action for online learning in the next turn
		self.action_input.set(action_idx)
		return give, keep

	def learn(self, game):
		# Learning rules are applied online based on per-turn rewards, so most update happens in the move() step
		# However, we must run one final turn of simulation to permit learning on the last turn.
		# Learner and fixed agents will not make any additional moves that are added to the game history,
		# but the moves recorded on the last turn will be given an opportunity of affect weight update through PES
		give, keep = self.move(game)
		# save weights for the next game
		# self.d_critic = self.network.d_critic


# class NengoQLearning():

# 	class StateInput():
# 		def __init__(self, n_inputs):
# 			self.state = np.zeros((n_inputs))
# 		def set(self, state):
# 			self.state = state
# 		def get(self):
# 			return self.state

# 	class PastRewardInput():
# 		def __init__(self):
# 			self.history = []
# 		def set(self, player, game):
# 			rewards = game.investor_reward if player=='investor' else game.trustee_reward
# 			reward = rewards[-1] if len(rewards)>0 else 0
# 			self.history.append(reward)
# 		def clear(self):
# 			self.history.clear()
# 		def get(self):
# 			return self.history[-1] if len(self.history)>0 else 0

# 	class PastActionInput():
# 		def __init__(self):
# 			self.history = []
# 		def set(self, action):
# 			self.history.append(action)
# 		def clear(self):
# 			self.history.clear()
# 		def get(self):
# 			return self.history[-1] if len(self.history)>0 else 0

# 	class LearningInput():
# 		def __init__(self):
# 			self.learning = 0
# 		def set(self, train):
# 			self.learning = 1 if train else 0
# 		def get(self):
# 			return self.learning

# 	def __init__(self, player, seed=0, n_actions=11, ID="nengo-q-learning", representation='turn-gen-opponent',
# 			encoder_method='one-hot', learning_rate=1e-4, n_neurons=100, dt=1e-3, turn_time=1e-2, q=10,
# 			explore_method='boltzmann', explore=100, explore_decay=0.93, gamma=0.99):
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
# 		self.learning_rate = learning_rate
# 		self.gamma = gamma
# 		self.q = q  # order of LMU
# 		self.turn_time = turn_time
# 		self.explore_method = explore_method
# 		self.explore = explore
# 		self.explore_decay = explore_decay
# 		self.state_input = self.StateInput(self.n_inputs)
# 		self.reward_input = self.PastRewardInput()
# 		self.action_input = self.PastActionInput()
# 		self.learning_input = self.LearningInput()
# 		self.encoders, self.intercepts = self.build_encoders()
# 		self.d_critic = np.zeros((self.n_neurons, self.n_actions))
# 		self.state = None
# 		self.network = None
# 		self.simulator = None
# 		self.episode = 0

# 	def reinitialize(self, player):
# 		self.__init__(player)

# 	def new_game(self):
# 		self.network = self.build_network()
# 		self.simulator = nengo.Simulator(self.network, dt=self.dt, progress_bar=False)
# 		self.reward_input.clear()
# 		self.action_input.clear()
# 		self.episode += 1

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

# 	def build_network(self):
# 		network = nengo.Network(seed=self.seed)
# 		with network:

# 			class CriticNode(nengo.Node):
# 				def __init__(self, n_neurons, n_actions, turn_time, d, learning_rate, gamma):
# 					self.n_neurons = n_neurons
# 					self.n_actions = n_actions
# 					self.size_in = 2*n_neurons + 3
# 					self.size_out = n_actions
# 					self.turn_time = turn_time
# 					self.d = d
# 					self.learning_rate = learning_rate
# 					self.gamma = gamma
# 					super().__init__(self.step, size_in=self.size_in, size_out=self.size_out)
# 				def step(self, t, x):
# 					activity = x[:self.n_neurons]  # current synaptic activities from "state" population
# 					past_activity = x[self.n_neurons: 2*self.n_neurons]  # delayed synaptic activities from "state" population
# 					past_action = int(x[-3])  # action chosen on the previous turn
# 					past_reward = x[-2]  # reward associated with past activities
# 					learning = x[-1]  # gating signal for updating decoders
# 					value = np.dot(activity, self.d)  # current state of the critic
# 					past_value = np.dot(past_activity, self.d)  # previous state of the critic
# 					# print(t, activity[:20], past_activity[:20])
# 					if 0<t<=self.turn_time:
# 						error = 0
# 					elif self.turn_time<t<=5*self.turn_time:
# 						error = learning * (past_reward + self.gamma*np.max(value) - past_value[past_action]) # TD0 update
# 					elif 5*self.turn_time<t:
# 						error = learning * (past_reward - past_value[past_action]) # TD0 update on last turn
# 					delta = (self.learning_rate / self.n_neurons) * past_activity * error  # PES update
# 					self.d[:,past_action] += delta
# 					return value  # return the current state of the critic (Q-value)

# 			state_input = nengo.Node(lambda t, x: self.state_input.get(), size_in=2, size_out=self.n_inputs)
# 			past_reward = nengo.Node(lambda t, x: self.reward_input.get(), size_in=2, size_out=1)
# 			past_action = nengo.Node(lambda t, x: self.action_input.get(), size_in=2, size_out=1)
# 			learning_input = nengo.Node(lambda t, x: self.learning_input.get(), size_in=2, size_out=1)

# 			state = nengo.Ensemble(self.n_neurons, self.n_inputs,
# 				intercepts=self.intercepts, encoders=self.encoders, neuron_type=nengo.LIFRate())
# 			critic = CriticNode(self.n_neurons, self.n_actions,
# 				turn_time=self.turn_time, d=self.d_critic, learning_rate=self.learning_rate, gamma=self.gamma)
# 			# one LMU per neuron in state
# 			state_delayed = []
# 			for n in range(self.n_neurons):
# 				state_delayed.append(
# 					nengolib.networks.RollingWindow(
# 						theta=self.turn_time, n_neurons=100, neuron_type=nengo.LIFRate(), legendre=True,
# 						process=nengo.processes.WhiteSignal(period=1, high=100, rms=600, y0=0, seed=self.seed+n)))
# 				nengo.Connection(state.neurons[n], state_delayed[n].input, synapse=None)
# 				nengo.Connection(state_delayed[n].output, critic[self.n_neurons+n: self.n_neurons+n+1], synapse=None)

# 			nengo.Connection(state_input, state, synapse=None)
# 			nengo.Connection(state.neurons, critic[:self.n_neurons], synapse=None)
# 			nengo.Connection(past_action, critic[-3], synapse=None)
# 			nengo.Connection(past_reward, critic[-2], synapse=None)
# 			nengo.Connection(learning_input, critic[-1], synapse=None)

# 			network.p_critic = nengo.Probe(critic, synapse=None)
# 			network.d_critic = critic.d

# 		return network

# 	def simulate_action(self):
# 		self.simulator.run(self.turn_time)
# 		x_critic = self.simulator.data[self.network.p_critic][-1]
# 		if self.explore_method=='epsilon':
# 			epsilon = self.explore*np.power(self.explore_decay, self.episode)
# 			if self.rng.uniform(0, 1) < epsilon:
# 				action = self.rng.randint(self.n_actions)
# 			else:
# 				action = np.argmax(x_critic)
# 		elif self.explore_method=='boltzmann':
# 			temperature = self.explore*np.power(self.explore_decay, self.episode)
# 			action_probs = scipy.special.softmax(x_critic / temperature)
# 			action = self.rng.choice(np.arange(self.n_actions), p=action_probs)
# 		else:
# 			action = torch.argmax(x_critic)
# 		return action

# 	def move(self, game):
# 		game_state = get_state(self.player, self.representation, game=game, return_type='one-hot',
# 			dim=self.n_inputs, n_actions=self.n_actions)
# 		# add the game state to the network's state input
# 		self.state_input.set(game_state)
# 		# use reward from the previous turn for online learning
# 		self.reward_input.set(self.player, game)
# 		# turn learning off during testing
# 		self.learning_input.set(game.train)
# 		# simulate the network with these inputs and collect the action outputs
# 		action = self.simulate_action()
# 		# translate action into environment-appropriate signal
# 		self.state = action / (self.n_actions-1)
# 		give, keep, action_idx = action_to_coins(self.player, self.state, self.n_actions, game)
# 		# save the chosen action for online learning in the next turn
# 		self.action_input.set(action_idx)
# 		return give, keep

# 	def learn(self, game):
# 		# Learning rules are applied online based on per-turn rewards, so most update happens in the move() step
# 		# However, we must run one final turn of simulation to permit learning on the last turn.
# 		# Learner and fixed agents will not make any additional moves that are added to the game history,
# 		# but the moves recorded on the last turn will be given an opportunity of affect weight update through PES
# 		give, keep = self.move(game)
# 		# save weights for the next game
# 		self.d_critic = self.network.d_critic



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
		def set(self, player, game):
			rewards = game.investor_reward if player=='investor' else game.trustee_reward
			reward = rewards[-1] if len(rewards)>0 else 0
			self.history.append(reward)
		def clear(self):
			self.history.clear()
		def get(self):
			return self.history[-1] if len(self.history)>0 else 0

	class PastActionInput():
		def __init__(self):
			self.history = []
		def set(self, action):
			self.history.append(action)
		def clear(self):
			self.history.clear()
		def get(self):
			return self.history[-1] if len(self.history)>0 else 0

	class PastProbsInput():
		def __init__(self, n_actions):
			self.n_actions = n_actions
			self.history = []
		def set(self, probs):
			self.history.append(probs)
		def clear(self):
			self.history.clear()
		def get(self):
			if len(self.history)==0:
				return np.zeros((self.n_actions))
			else:
				return self.history[-1]

	class LearningInput():
		def __init__(self):
			self.learning = 0
		def set(self, player, game):
			current_turn = len(game.investor_give) if player=='investor' else len(game.trustee_give)
			self.learning = 1  # on by default
			if not game.train: self.learning = 0  # testing
			if current_turn==0: self.learning = 2  # first turn
			if current_turn>=5: self.learning = 3  # last turn
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

	def __init__(self, player, seed=0, n_actions=11, ID="nengo-q-learning", representation='turn-gen-opponent',
			encoder_method='one-hot', actor_rate=3e-4, critic_rate=3e-4, n_neurons=200, dt=1e-3, turn_time=1e-3,
			tau_fast=0.01, tau_slow=0.1,
			explore_method='boltzmann', explore=100, explore_decay=0.98, gamma=0.99):
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
		self.tau_fast = tau_fast
		self.tau_slow = tau_slow
		self.delay = self.Delay(size_in=n_neurons)
		self.turn_time = turn_time
		self.explore_method = explore_method
		self.explore = explore
		self.explore_decay = explore_decay
		self.state_input = self.StateInput(self.n_inputs)
		self.reward_input = self.PastRewardInput()
		self.action_input = self.PastActionInput()
		self.probs_input = self.PastProbsInput(self.n_actions)
		self.learning_input = self.LearningInput()
		self.encoders, self.intercepts = self.build_encoders()
		self.d_critic = np.zeros((self.n_neurons, 1))
		self.d_actor = np.zeros((self.n_neurons, self.n_actions))
		self.state = None
		self.network = None
		self.simulator = None
		self.episode = 0


	def reinitialize(self, player):
		self.__init__(player)

	def new_game(self):
		self.network = self.build_network()
		self.simulator = nengo.Simulator(self.network, dt=self.dt, progress_bar=False)
		self.reward_input.clear()
		self.action_input.clear()
		self.probs_input.clear()

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
		network = nengo.Network(seed=self.seed)
		with network:

			class CriticNode(nengo.Node):
				def __init__(self, n_neurons, d=None, learning_rate=0, gamma=0.99):
					self.n_neurons = n_neurons
					self.size_in = 2*n_neurons + 2
					self.size_out = 1
					self.d = d
					self.learning_rate = learning_rate
					self.gamma = gamma
					super().__init__(self.step, size_in=self.size_in, size_out=self.size_out)
				def step(self, t, x):
					activity = x[:self.n_neurons]  # current synaptic activities from "state" population
					past_activity = x[self.n_neurons: 2*self.n_neurons]  # delayed synaptic activities from "state" population
					past_reward = x[-2]  # reward associated with past activities
					learning = x[-1]  # gating signal for updating decoders
					# calculate value error signal for PES, gated by learning signal
					value = np.dot(activity, self.d)  # current state of the critic
					past_value = np.dot(past_activity, self.d)  # previous state of the critic
					if learning==0: error = 0 # no learning during testing
					elif learning==1: error = past_reward + self.gamma*np.max(value) - past_value  # normal RL update
					elif learning==2: error = 0  # no learning during first turn (no reward signal)
					elif learning==3: error = past_reward - past_value # the target value in the 6th turn is simply the reward
					# update decoders based on delayed activities and the associated value error
					delta = (self.learning_rate / self.n_neurons) * past_activity.reshape(-1, 1) * error
					self.d[:] += delta
					return error  # return the value error, which is fed to the ActorNode for online learning

			class ActorNode(nengo.Node):
				def __init__(self, n_neurons, n_actions, d=None, learning_rate=0):
					self.n_neurons = n_neurons
					self.n_actions = n_actions
					self.size_in = 2*n_neurons + n_actions + 2
					self.size_out = n_actions
					self.d = d
					self.learning_rate = learning_rate
					super().__init__(self.step, size_in=self.size_in, size_out=self.size_out)
				def step(self, t, x):
					activity = x[:self.n_neurons]  # current synaptic activities from "state" population
					past_activity = x[self.n_neurons: 2*self.n_neurons]  # delayed synaptic activities from "state" population
					action_probs = x[2*self.n_neurons: 2*self.n_neurons+self.n_actions]  # values for each action, as given by the actor (past state)
					past_action = int(x[-2])  # action that was previously selected by the learning agent (past state)
					value_error = x[-1]  # value error associated with past activities, from CriticNode
					# calculate actor error signal for PES, gated by learning signal
					action_values = np.dot(activity, self.d)
					actor_error = -value_error * action_probs  # non-chosen actions
					actor_error[past_action] = value_error*(1-action_probs[past_action])  # chosen action
					# update decoders based on delayed activities and the associated actor error
					delta = (self.learning_rate / self.n_neurons) * past_activity.reshape(-1, 1) * actor_error.reshape(1, -1)
					self.d[:] += delta
					return action_values  # return the current state of the actor (value of each potential action in current state)

			state_input = nengo.Node(lambda t, x: self.state_input.get(), size_in=2, size_out=self.n_inputs)
			past_reward = nengo.Node(lambda t, x: self.reward_input.get(), size_in=2, size_out=1)
			past_action = nengo.Node(lambda t, x: self.action_input.get(),size_in=2, size_out=1)
			past_probs = nengo.Node(lambda t, x: self.probs_input.get(), size_in=2, size_out=self.n_actions)
			learning_input = nengo.Node(lambda t, x: self.learning_input.get(), size_in=2, size_out=1)

			state = nengo.Ensemble(self.n_neurons, self.n_inputs, intercepts=self.intercepts, encoders=self.encoders, neuron_type=nengo.LIFRate())
			critic = CriticNode(self.n_neurons, d=self.d_critic, learning_rate=self.critic_rate, gamma=self.gamma)
			actor = ActorNode(self.n_neurons, self.n_actions, d=self.d_actor, learning_rate=self.actor_rate)

			nengo.Connection(state_input, state, synapse=None)
			nengo.Connection(state.neurons, critic[:self.n_neurons], synapse=None)
			nengo.Connection(state.neurons, critic[self.n_neurons: 2*self.n_neurons], synapse=self.delay)
			nengo.Connection(past_reward, critic[-2], synapse=None)
			nengo.Connection(learning_input, critic[-1], synapse=None)

			nengo.Connection(state.neurons, actor[:self.n_neurons], synapse=None)
			nengo.Connection(state.neurons, actor[self.n_neurons: 2*self.n_neurons], synapse=self.delay)
			nengo.Connection(past_probs, actor[2*self.n_neurons: 2*self.n_neurons+self.n_actions], synapse=None)  # past action probabilities
			nengo.Connection(past_action, actor[-2], synapse=None)  # past choice
			nengo.Connection(critic, actor[-1], synapse=None)  # value error

			network.p_actor = nengo.Probe(actor, synapse=None)
			network.d_critic = critic.d
			network.d_actor = actor.d

		return network

	def simulate_action(self):
		self.simulator.run(self.turn_time)
		x_actor = self.simulator.data[self.network.p_actor][-1]
		if self.explore_method=='boltzmann':
			temperature = self.explore*np.power(self.explore_decay, self.episode)
			action_probs = scipy.special.softmax(x_actor / temperature)
			action = self.rng.choice(np.arange(self.n_actions), p=action_probs)
		return action, action_probs

	def move(self, game):
		game_state = get_state(self.player, self.representation, game=game, return_type='one-hot',
			dim=self.n_inputs, n_actions=self.n_actions)
		# add the game state to the network's state input
		self.state_input.set(game_state)
		# use reward from the previous turn for online learning
		self.reward_input.set(self.player, game)
		# turn learning on/off, depending on the situation
		self.learning_input.set(self.player, game)
		# simulate the network with these inputs and collect the action outputs
		action, action_probs = self.simulate_action()
		# translate action into environment-appropriate signal
		self.state = action / (self.n_actions-1)
		give, keep, action_idx = action_to_coins(self.player, self.state, self.n_actions, game)
		# save the chosen action for online learning in the next turn
		self.action_input.set(action_idx)
		self.probs_input.set(action_probs)
		return give, keep

	def learn(self, game):
		# Learning rules are applied online based on per-turn rewards, so most update happens in the move() step
		# However, we must run one final turn of simulation to permit learning on the last turn.
		# Learner and fixed agents will not make any additional moves that are added to the game history,
		# but the moves recorded on the last turn will be given an opportunity of affect weight update through PES
		give, keep = self.move(game)
		# save weights for the next game
		self.d_critic = self.network.d_critic
		self.d_actor = self.network.d_actor


