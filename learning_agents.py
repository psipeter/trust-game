import numpy as np
import random
import os
import torch
import scipy
import nengo
from utils import *

class TabularQLearning():

	def __init__(self, player, seed=0, n_actions=11, ID="tabular-q-learning", representation='turn',
			explore_method='boltzmann', explore=100, explore_decay=0.995, gamma=0.99, lam=1, learning_rate=1):
		self.player = player
		self.ID = ID
		self.seed = seed
		self.rng = np.random.RandomState(seed=seed)
		self.representation = representation
		self.n_inputs = get_n_inputs(representation, player, n_actions)
		self.n_actions = n_actions
		self.explore_method = explore_method
		self.explore = explore  # probability of random action, for exploration
		self.explore_decay = explore_decay  # per-episode reduction of epsilon
		self.gamma = gamma  # discount factor
		self.lam = lam  #  for TD-lambda or SARSA-lambda update
		self.learning_rate = learning_rate
		self.Q = np.zeros((self.n_inputs, self.n_actions))
		self.C = np.zeros((self.n_inputs, self.n_actions))  # visits
		self.eligibility = np.zeros((self.n_inputs, self.n_actions))  # eligibility trace
		self.state_history = []
		self.action_history = []
		self.state = None
		self.episode = 0

	def reinitialize(self, player):
		self.__init__(player, self.seed, self.n_actions, self.ID, self.representation, 
			self.explore_method,self.explore, self.explore_decay, self.gamma, self.lam, self.learning_rate)

	def new_game(self):
		self.state_history.clear()
		self.action_history.clear()
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
		# update the histories for learning
		self.state_history.append(game_state)
		# translate action into environment-appropriate signal
		self.state = action / (self.n_actions-1)
		give, keep, action_idx = action_to_coins(self.player, self.state, self.n_actions, game)
		self.action_history.append(action_idx)
		return give, keep

	def learn(self, game):
		rewards = game.investor_reward if self.player=='investor' else game.trustee_reward
		for t in np.arange(game.turns):
			state = self.state_history[t]
			next_state = self.state_history[t+1] if t<game.turns-1 else None
			action = self.action_history[t]
			next_action = self.action_history[t+1] if t<game.turns-1 else None
			reward = rewards[t]
			value = self.Q[state, action]
			# next_value = np.max(self.Q[next_state]) if t<game.turns-1 else 0
			next_value = self.Q[next_state, next_action] if t<game.turns-1 else 0
			self.C[state, action] += 1
			alpha = self.learning_rate / self.C[state, action]
			# alpha = self.learning_rate
			self.Q[state, action] += alpha * (reward + self.gamma*next_value - value)



class TabularActorCritic():

	def __init__(self, player, seed=0, n_actions=11, ID="tabular-actor-critic", representation='turn',
			explore_method='boltzmann', explore=100, explore_decay=0.995, gamma=0.99, critic_rate=1e-3, actor_rate=1e-3):
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
		self.gamma = gamma  # discount factor
		self.critic_rate = critic_rate
		self.actor_rate = actor_rate
		self.critic = np.zeros((self.n_inputs, self.n_actions))
		# self.critic = np.zeros((self.n_inputs, 1))
		self.actor = np.zeros((self.n_inputs, self.n_actions))
		self.C = np.zeros((self.n_inputs, self.n_actions))
		self.state_history = []
		self.action_history = []
		self.action_probs_history = []
		self.state = None
		self.episode = 0

	def reinitialize(self, player):
		self.__init__(player, self.seed, self.n_actions, self.ID, self.representation, 
			self.explore_method,self.explore, self.explore_decay, self.gamma, self.critic_rate, self.actor_rate)

	def new_game(self):
		self.state_history.clear()
		self.action_history.clear()
		self.action_probs_history.clear()
		self.episode += 1

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
		self.action_probs_history.append(np.log(action_probs[action_idx]))
		# self.action_probs_history.append(action_probs)
		return give, keep

	def learn(self, game):
		rewards = game.investor_reward if self.player=='investor' else game.trustee_reward
		for t in np.arange(game.turns):
			state = self.state_history[t]
			next_state = self.state_history[t+1] if t<game.turns-1 else None
			action = self.action_history[t]
			next_action = self.action_history[t+1] if t<game.turns-1 else None
			log_prob = self.action_probs_history[t]
			reward = rewards[t]
			value = self.critic[state, action]
			next_value = self.critic[next_state, next_action] if t<game.turns-1 else 0
			# value = self.critic[state]
			# next_value = self.critic[next_state] if t<game.turns-1 else 0
			self.C[state, action] += 1
			alpha_actor = self.actor_rate / self.C[state, action]
			alpha_critic = self.critic_rate / self.C[state, action]
			# delta = np.clip(reward + self.gamma*next_value - value, -10, 10)
			delta = reward + self.gamma*next_value - value
			actor_loss = alpha_actor * -log_prob * delta
			critic_loss = alpha_critic * delta**2
			self.actor[state, action] += actor_loss
			self.critic[state, action] += critic_loss
			# self.critic[state] += critic_loss
			# for a in range(self.n_actions):
			# 	if a==action: self.actor[state, a] += alpha_actor*delta*(1-self.action_probs_history[t][a])
			# 	else: self.actor[state, a] += alpha_actor*-delta*self.action_probs_history[t][a]


class TabularModelBased():

	def __init__(self, player, seed=0, n_actions=11, ID="tabular-model-based", representation='turn-coin',
			explore_method='boltzmann', explore=100, explore_decay=0.99, gamma=0.99, learning_rate=1e-3):
		self.player = player
		self.ID = ID
		self.seed = seed
		self.rng = np.random.RandomState(seed=seed)
		self.representation = representation
		self.n_inputs = get_n_inputs(representation, player, n_actions)
		self.n_actions = n_actions
		self.explore = explore  # probability of random action, for exploration
		self.explore_method = explore_method
		self.explore_decay = explore_decay  # per-episode reduction of epsilon
		self.gamma = gamma  # discount factor
		self.learning_rate = learning_rate
		self.Q = np.zeros((self.n_inputs, self.n_actions))  # values
		self.T = np.zeros((self.n_inputs, self.n_actions, self.n_inputs))  # transition probabilities
		self.C = np.zeros((self.n_inputs, self.n_actions, self.n_inputs))  # visits
		self.state_history = []
		self.action_history = []
		self.state = None
		self.episode = 0

	def reinitialize(self, player):
		self.__init__(player, self.seed, self.n_actions, self.ID, self.representation, 
			self.explore_method, self.explore, self.explore_decay, self.gamma, self.learning_rate)

	def new_game(self):
		self.state_history.clear()
		self.action_history.clear()
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
			ucb = np.sqrt(2*np.log(self.episode)/(np.sum(self.C[game_state], axis=1)+np.ones((self.n_actions))))
			values = Q_state + ucb
			idx_max = np.argwhere(values==np.max(values)).flatten()  # get indices where the values are at a maximum
			action = self.rng.choice(idx_max)
		else:
			action = np.argmax(Q_state)
		# update the histories for learning
		self.state_history.append(game_state)
		# translate action into environment-appropriate signal
		self.state = action / (self.n_actions-1)
		give, keep, action_idx = action_to_coins(self.player, self.state, self.n_actions, game)
		self.action_history.append(action_idx)
		return give, keep

	def learn(self, game):
		rewards = game.investor_reward if self.player=='investor' else game.trustee_reward
		for t in np.arange(game.turns):
			state = self.state_history[t]
			next_state = self.state_history[t+1] if t<game.turns-1 else 0
			action = self.action_history[t]
			next_action = self.action_history[t+1] if t<game.turns-1 else None
			reward = rewards[t]
			value = self.Q[state, action]
			# next_value = self.Q[next_state, next_action] if t<game.turns-1 else 0
			self.C[state, action, next_state] += 1
			self.T[state, action, :] = self.C[state, action, :] / np.sum(self.C[state, action])
			alpha = self.learning_rate # / np.sum(self.C[state, action])
			weighted_next_value = 0
			for s in range(self.n_inputs):
				weighted_next_value += self.T[state, action, s] * np.max(self.Q[s])
			self.Q[state, action] += alpha * (reward + self.gamma*weighted_next_value - value)














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

	def __init__(self, player, seed=0, n_actions=11, ID="deep-q-learning", representation='turn-coin',
			update_method='TD-0', update_direction='forward', friendliness=0,
			explore_method='boltzmann', explore=100, explore_decay=0.99, gamma=0.99, n_neurons=200, critic_rate=3e-3):
		self.player = player
		self.ID = ID
		self.seed = seed
		self.rng = np.random.RandomState(seed=seed)
		self.gamma = gamma
		self.representation = representation
		self.critic_rate = critic_rate
		self.explore = explore
		self.explore_decay = explore_decay
		self.explore_method = explore_method
		self.n_inputs = get_n_inputs(representation, player, n_actions)
		self.n_actions = n_actions
		self.n_neurons = n_neurons
		self.friendliness = friendliness
		self.update_method = update_method
		self.update_direction = update_direction
		torch.manual_seed(seed)
		self.critic = self.Critic(self.n_neurons, self.n_inputs, self.n_actions)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.critic_rate)
		self.loss_function = torch.nn.MSELoss()
		self.critic_value_history = []
		self.state = None
		self.episode = 0

	def reinitialize(self, player):
		self.__init__(player, self.seed, self.n_actions, self.ID, self.representation,
			self.update_method, self.update_direction, self.friendliness,
			self.explore_method, self.explore, self.explore_decay, self.gamma, self.n_neurons, self.critic_rate)

	def new_game(self):
		self.critic_value_history.clear()
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
		self.critic_value_history.append(critic_values[action_idx])
		return give, keep

	def learn(self, game):
		rewards = game.investor_reward if self.player=='investor' else game.trustee_reward
		rewards_other = game.trustee_reward if self.player=='investor' else game.investor_reward
		critic_losses = []

		if self.update_method=='TD-0':
			times = np.arange(game.turns) if self.update_direction=='forward' else np.arange(game.turns)[::-1]
			for t in times:
				value = self.critic_value_history[t]
				next_value = self.critic_value_history[t+1] if t<game.turns-1 else 0
				reward = torch.FloatTensor([rewards[t]])
				reward_other = torch.FloatTensor([rewards_other[t]])
				R = (1-self.friendliness)*reward + self.friendliness*reward_other
				delta = R + self.gamma*next_value - value
				critic_loss = self.loss_function(value, R+self.gamma*next_value)
				critic_losses.append(critic_loss)

		if self.update_method=='MC':
			discounted_sum = 0
			times = np.arange(game.turns)[::-1]  # backward
			for t in times:
				value = self.critic_value_history[t]
				discounted_sum = rewards[t] + self.gamma * discounted_sum
				reward = torch.FloatTensor([discounted_sum]).squeeze()
				delta = reward - value
				critic_loss = self.loss_function(value, reward)
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

	def __init__(self, player, seed=0, n_actions=11, ID="deep-actor-critic", representation='turn-coin',
			update_method='TD-0', update_direction='forward', friendliness=0,
			explore_method='boltzmann', explore=100, explore_decay=0.99, gamma=0.99, n_neurons=200, critic_rate=1e-3, actor_rate=1e-3):
		self.player = player
		self.ID = ID
		self.seed = seed
		self.rng = np.random.RandomState(seed=seed)
		self.gamma = gamma
		self.representation = representation
		self.actor_rate = actor_rate
		self.critic_rate = critic_rate
		self.explore = explore
		self.explore_decay = explore_decay
		self.explore_method = explore_method
		self.n_inputs = get_n_inputs(representation, player, n_actions)
		self.n_actions = n_actions
		self.n_neurons = n_neurons
		self.friendliness = friendliness
		self.update_method = update_method
		self.update_direction = update_direction
		torch.manual_seed(seed)
		self.actor = self.Actor(self.n_neurons, self.n_inputs, self.n_actions)
		# self.critic = self.Critic(self.n_neurons, self.n_inputs, self.n_actions)
		self.critic = self.Critic(self.n_neurons, self.n_inputs, 1)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.critic_rate)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.actor_rate)
		self.loss_function = torch.nn.MSELoss()
		self.action_probs_history = []
		self.critic_value_history = []
		self.state = None
		self.episode = 0

	def reinitialize(self, player):
		self.__init__(player, self.seed, self.n_actions, self.ID, self.representation,
			self.update_method, self.update_direction, self.friendliness,
			self.explore_method, self.explore, self.explore_decay, self.gamma, self.n_neurons, self.critic_rate, self.actor_rate)

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
		# critic_value = critic_values[action_idx]
		log_prob = torch.log(action_probs.gather(index=action, dim=0))
		self.critic_value_history.append(critic_values)
		self.action_probs_history.append(log_prob)
		return give, keep

	def learn(self, game):
		rewards = game.investor_reward if self.player=='investor' else game.trustee_reward
		rewards_other = game.trustee_reward if self.player=='investor' else game.investor_reward
		actor_losses = []
		critic_losses = []

		if self.update_method=='TD-0':
			times = np.arange(game.turns) if self.update_direction=='forward' else np.arange(game.turns)[::-1]
			for t in times:
				value = self.critic_value_history[t]
				next_value = self.critic_value_history[t+1] if t<game.turns-1 else 0
				reward = torch.FloatTensor([rewards[t]])
				reward_other = torch.FloatTensor([rewards_other[t]])
				R = (1-self.friendliness)*reward + self.friendliness*reward_other
				log_prob = self.action_probs_history[t]
				delta = R + self.gamma*next_value - value
				actor_loss = -log_prob * delta
				critic_loss = self.loss_function(value, R+self.gamma*next_value)
				actor_losses.append(actor_loss)
				critic_losses.append(critic_loss)

		if self.update_method=='MC':
			discounted_sum = 0
			times = np.arange(game.turns)[::-1]  # backward
			for t in times:
				value = self.critic_value_history[t]
				discounted_sum = rewards[t] + self.gamma * discounted_sum
				reward = torch.FloatTensor([discounted_sum]).squeeze()
				log_prob = self.action_probs_history[t]
				delta = reward - value
				actor_loss = -log_prob * delta
				critic_loss = self.loss_function(value, reward)
				actor_losses.append(actor_loss)
				critic_losses.append(critic_loss)

		actor_loss = torch.stack(actor_losses).sum()
		critic_loss = torch.stack(critic_losses).sum()
		# combined_loss = actor_loss + critic_loss
		# print(combined_loss, actor_loss, critic_loss)
		self.actor_optimizer.zero_grad()
		self.critic_optimizer.zero_grad()
		# combined_loss.backward()
		actor_loss.backward(retain_graph=True)
		critic_loss.backward(retain_graph=True)
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

	def __init__(self, player, seed=0, n_actions=11, ID="instance-based", representation='turn-coin', 
			update_method='TD-0', gamma=0.99, thr_activation=0, thr_action=0.8, thr_state=0.8,
			explore_method='epsilon', explore=10, explore_decay=0.99, update_direction='forward',
			populate_method='state-similarity', select_method='max-blended-value', value_method='next-value'):
		self.player = player
		self.ID = ID
		self.seed = seed
		self.rng = np.random.RandomState(seed=seed)
		self.representation = representation
		self.n_inputs = get_n_inputs(representation, player, n_actions)
		self.n_actions = n_actions
		self.update_method = update_method
		self.update_direction = update_direction
		self.gamma = gamma
		self.thr_activation = thr_activation  # activation threshold for retrieval (loading chunks from declarative into working memory)
		self.thr_action = thr_action  # action similarity threshold for retrieval (loading chunks from declarative into working memory)
		self.thr_state = thr_state  # state similarity threshold for retrieval (loading chunks from declarative into working memory)
		self.explore = explore  # probability of random action, for exploration
		self.explore_decay = explore_decay  # per-episode reduction of exploration
		self.explore_method = explore_method
		self.populate_method = populate_method  # method for determining whether a chunk in declaritive memory meets threshold
		self.select_method = select_method  # method for selecting an action based on chunks in working memory
		self.value_method = value_method  # method for assigning value to chunks during learning
		self.declarative_memory = []
		self.working_memory = []
		self.learning_memory = []
		self.state = None
		self.episode = 0  # for tracking activation within / across games

	def reinitialize(self, player):
		self.__init__(player, self.seed, self.n_actions, self.ID, self.representation,
			self.update_method, self.gamma, self.thr_activation, self.thr_action, self.thr_state,
			self.explore_method, self.explore, self.explore_decay, self.update_direction,
			self.populate_method, self.select_method, self.value_method)

	def new_game(self):
		self.working_memory.clear()
		self.learning_memory.clear()
		self.rng.shuffle(self.declarative_memory)
		self.episode += 1

	def move(self, game):
		game_state = get_state(self.player, self.representation, game=game, return_type='index')
		# load chunks from declarative memory into working memory
		self.populate_working_memory(game_state, game)
		# select an action (generosity) that immitates the best chunk in working memory
		self.state = self.select_action()
		# create a new chunk for the chosen action, populate with more information in learn()
		new_chunk = self.Chunk(state=game_state, action=None, reward=None, value=0, episode=self.episode)
		self.learning_memory.append(new_chunk)
		# translate action into environment-appropriate signal
		coins = game.coins if self.player=='investor' else game.investor_give[-1]*game.match  # coins available
		move = self.state * coins
		give = int(np.clip(move, 0, coins))
		keep = int(coins - give)
		return give, keep

	def populate_working_memory(self, game_state, game):
		self.working_memory.clear()
		current_turn = len(game.investor_give) if self.player=='investor' else len(game.trustee_give)
		greedy_action = 0
		generous_action = 1.0 if self.player=='investor' else 0.5
		for chunk in self.declarative_memory:
			# update the activation of each chunk, and store it with the chunk
			activation = chunk.get_activation(self.episode, self.rng)
			# identify chunk's action similarity to a fully-generous action and to a fully-greedy action
			similarity_greedy = 1 - np.abs(chunk.action - greedy_action)
			similarity_generous = 1 - np.abs(chunk.action - generous_action)
			similarity_action = np.max([similarity_greedy, similarity_generous])

			if self.representation=='turn':
				chunk_turn = chunk.state
				if current_turn==chunk_turn: similarity_state = 1
				else: similarity_state = 0
			elif self.representation=='turn-coin':
				if self.player=='investor':
					chunk_turn = chunk.state
					if current_turn!=chunk_turn: similarity_state = 0
					else: similarity_state = 1
				elif self.player=='trustee':
					chunk_turn = int(chunk.state / (game.coins*game.match+1))
					if current_turn!=chunk_turn: similarity_state = 0
					else:
						# identify the difference in available coins between the current state and the remembered chunk
						available_current = game_state % (game.coins*game.match+1)
						available_chunk = chunk.state % (game.coins*game.match+1)
						diff = np.abs(available_chunk-available_current)
						similarity_state = 1 - diff/(game.coins*game.match+1)

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
		explore = self.explore*np.power(self.explore_decay, self.episode)
		if len(self.working_memory)==0:
			# if there are no chunks in working memory, select a random action
			selected_action = self.rng.randint(0, self.n_actions) / (self.n_actions-1)
		elif self.explore_method=='epsilon' and self.rng.uniform(0, 1) < explore:
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
			if self.select_method=="max-blended-value":
				# choose the action with the highest blended value
				selected_action = max(actions, key=lambda action: actions[action]['blended'])
			elif self.select_method=="softmax-blended-value":
				# choose the action with probability proportional to the blended value
				action_gens = np.array([a for a in actions])
				action_values = np.array([actions[a]['blended'] for a in actions])
				if self.explore_method=='boltzmann':
					action_probs = scipy.special.softmax(action_values / explore)
				else:
					action_probs = scipy.special.softmax(action_values)
				selected_action = self.rng.choice(action_gens, p=action_probs)
		return selected_action

	def learn(self, game):
		# update value of new chunks according to some scheme
		actions = game.investor_gen if self.player=='investor' else game.trustee_gen
		rewards = game.investor_reward if self.player=='investor' else game.trustee_reward
		times = np.arange(game.turns) if self.update_direction=='forward' else np.arange(game.turns)[::-1]
		for t in times:
			chunk = self.learning_memory[t]
			next_chunk = self.learning_memory[t+1] if t<(game.turns-1) else None
			chunk.action = actions[t]
			chunk.reward = rewards[t]
			if self.update_method=='TD-0':
				# estimate the value of the next chunk by retrieving all similar chunks and computing their blended value
				next_reward = 0
				next_value = 0
				if t<(game.turns-1):
					next_turn = t+1
					similar_chunks = []
					for next_chunk in self.declarative_memory:
						# update the activation of each chunk, and store it with the chunk
						activation = next_chunk.get_activation(self.episode, self.rng)
						if self.representation=='turn':
							chunk_turn = next_chunk.state
							similarity_state = 1 if next_turn==chunk_turn else 0
						elif self.representation=='turn-coin':
							if self.player=='investor':
								chunk_turn = next_chunk.state
								similarity_state = 1 if next_turn==chunk_turn else 0
							elif self.player=='trustee':
								chunk_turn = int(next_chunk.state / (game.coins*game.match+1))
								if next_turn!=chunk_turn: similarity_state = 0
								else:
									# identify the difference in available coins between the current state and the remembered chunk
									available_current = game.investor_give[next_turn]*game.match
									available_chunk = next_chunk.state % (game.coins*game.match+1)
									diff = np.abs(available_chunk-available_current)
									similarity_state = 1 - diff/(game.coins*game.match+1)
						pass_activation = activation > self.thr_activation
						pass_state = similarity_state > self.thr_state
						if pass_activation and pass_state:
							similar_chunks.append(next_chunk)
					if len(similar_chunks)>0:
						next_reward =  np.mean([next_chunk.reward for next_chunk in similar_chunks])
						next_value =  np.mean([next_chunk.value for next_chunk in similar_chunks])
				if self.value_method=='reward':
					chunk.value = rewards[t]
				elif self.value_method=='game-mean':
					chunk.value = np.mean(rewards)
				elif self.value_method=='next-reward':
					chunk.value = chunk.reward + self.gamma*next_reward
				elif self.value_method=='next-value':
					chunk.value = chunk.reward + self.gamma*next_value
			elif self.update_method=='MC':
				if self.value_method=='reward':
					chunk.value = chunk.reward
				elif self.value_method=='game-mean':
					chunk.value = np.mean(rewards)
				elif self.value_method=='next-reward':
					chunk.value = chunk.reward + self.gamma*next_chunk.reward if t<(game.turns-1) else chunk.reward
				elif self.value_method=='next-value':
					chunk.value = chunk.reward + self.gamma*next_chunk.value if t<(game.turns-1) else chunk.reward

		for new_chunk in self.learning_memory:
			# Check if the new chunk has identical (state, action) to a previous chunk in declarative memory.
			# If so, update that chunk's triggers, rather than adding a new chunk to declarative memory
			add_new_chunk = True
			for old_chunk in self.declarative_memory:
				# if np.all(new_chunk.state == old_chunk.state) and \
				# 		  new_chunk.action == old_chunk.action and \
				# 		  new_chunk.reward == old_chunk.reward and \
				# 		  new_chunk.value == old_chunk.value:
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

	def __init__(self, player, seed=0, n_actions=5, ID="nengo-q-learning", representation='turn-coin',
			learning_rate=1e-6, gamma=0.99, n_neurons=200, dt=1e-3, tau=None, turn_time=1e-3,
			encoder_method='one-hot', explore_method='boltzmann', explore=100, explore_decay=0.99):
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
		self.gamma = gamma  # discount
		self.tau = tau  # synaptic time constant
		self.delay_activity = self.Delay(size_in=n_neurons)
		self.delay_value = self.Delay(size_in=1)
		self.turn_time = turn_time
		self.explore_method = explore_method
		self.explore = explore
		self.explore_decay = explore_decay
		self.state_input = self.StateInput(self.n_inputs)
		self.reward_input = self.PastRewardInput()
		self.action_input = self.PastActionInput()
		self.learning_input = self.LearningInput()
		self.encoders, self.intercepts = self.build_encoders()
		self.d_critic = np.zeros((self.n_actions, self.n_neurons))
		self.state = None
		self.network = None
		self.simulator = None
		self.episode = 0

	def reinitialize(self, player):
		self.__init__(player, self.seed, self.n_actions, self.ID, self.representation,
			self.learning_rate, self.gamma, self.n_neurons, self.dt, self.tau, self.turn_time,
			self.encoder_method, self.explore_method, self.explore, self.explore_decay)

	def new_game(self):
		self.network = self.build_network()
		self.simulator = nengo.Simulator(self.network, dt=self.dt, progress_bar=False)
		self.reward_input.clear()
		self.action_input.clear()
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
				def __init__(self, n_neurons, n_actions, d=None, learning_rate=0, gamma=0.99):
					self.n_neurons = n_neurons
					self.n_actions = n_actions
					self.size_in = 2*n_neurons + 3
					self.size_out = n_actions
					self.d = d if np.any(d) else np.zeros((n_neurons, n_actions))
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
					# calculate error signal for PES, gated by learning signal
					if learning==0: error = 0 # no learning during testing
					# elif learning==1: error = past_reward + self.gamma*value[past_action] - past_value[past_action]  # TODO: SARSA
					elif learning==1: error = past_reward + self.gamma*np.max(value) - past_value[past_action]  # normal RL update
					elif learning==2: error = 0  # no learning during first turn (no reward signal)
					elif learning==3: error = past_reward - past_value[past_action] # the target value in the 6th turn is simply the reward
					# update based on delayed activities and the associated value error
					# print(t, int(np.sum(past_activity)), past_value)
					delta = (self.learning_rate / self.n_neurons) * past_activity * error
					self.d[:,past_action] += delta
					return value  # return the current state of the critic (Q-value)

			state_input = nengo.Node(lambda t, x: self.state_input.get(), size_in=2, size_out=self.n_inputs)
			past_reward = nengo.Node(lambda t, x: self.reward_input.get(), size_in=2, size_out=1)
			past_action = nengo.Node(lambda t, x: self.action_input.get(), size_in=2, size_out=1)
			learning_input = nengo.Node(lambda t, x: self.learning_input.get(), size_in=2, size_out=1)

			state = nengo.Ensemble(self.n_neurons, self.n_inputs, intercepts=self.intercepts, encoders=self.encoders, neuron_type=nengo.LIFRate())
			critic = CriticNode(self.n_neurons, self.n_actions, d=self.d_critic, learning_rate=self.learning_rate, gamma=self.gamma)

			nengo.Connection(state_input, state, synapse=None)
			nengo.Connection(state.neurons, critic[:self.n_neurons], synapse=None)
			nengo.Connection(state.neurons, critic[self.n_neurons: 2*self.n_neurons], synapse=self.delay_activity)
			nengo.Connection(past_action, critic[-3], synapse=None)
			nengo.Connection(past_reward, critic[-2], synapse=None)
			nengo.Connection(learning_input, critic[-1], synapse=None)

			network.p_state = nengo.Probe(state.neurons, synapse=None)
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
		# turn learning on/off, depending on the situation
		self.learning_input.set(self.player, game)
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
		self.d_critic = self.network.d_critic


























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


