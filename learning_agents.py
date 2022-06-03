import numpy as np
import random
import os
import torch
import scipy
import nengo
# import nengolib
import nengo_spa
import itertools
from utils import *


class TQ():
	# Tabular Q-learning agent
	def __init__(self, player, ID="TQ", seed=0, n_actions=11, n_states=155,
			learning_rate=1, gamma=0.8, epsilon_decay=0.02,
			orientation="proself",
			normalize=False, randomize=False):
		self.player = player
		self.ID = ID
		self.seed = seed
		self.randomize = randomize
		self.normalize = normalize
		self.rng = np.random.RandomState(seed=seed)
		self.n_states = n_states
		self.n_actions = n_actions
		self.w_s = 1
		if self.randomize:
			self.gamma = self.rng.uniform(0.6, 1.0)
			self.learning_rate = self.rng.uniform(0.9, 1.0)
			self.orientation = "proself" if self.rng.uniform(0,1) < 0.5 else "prosocial"
			self.w_o = 0 if self.orientation=="proself" else self.rng.uniform(0, 0.3)
			self.w_i = 0 if self.orientation=="proself" else self.rng.uniform(0, 0.5)
			self.epsilon_decay = self.rng.uniform(0.015, 0.025)
		else:
			self.gamma = gamma
			self.learning_rate = learning_rate
			self.orientation = orientation
			self.w_o = 0 if self.orientation=="proself" else 0.2
			self.w_i = 0 if self.orientation=="proself" else 0.2
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
		# epsilon = 1 - self.episode * self.epsilon_decay
		epsilon = np.exp(-self.episode*self.epsilon_decay)
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
		rewards = get_rewards(self.player, game, self.w_s, self.w_o, self.w_i, self.normalize, self.gamma)
		for t in np.arange(game.turns):
			state = self.state_history[t]
			action = self.action_history[t]
			value = self.Q[state, action]
			if t==(game.turns-1):
				next_value = 0
			else:
				next_state = self.state_history[t+1]
				next_action = self.action_history[t+1]
				next_value = np.max(self.Q[next_state])  # Q-learning
				# next_value = self.Q[next_state, next_action]  # SARSA
			delta = rewards[t] + self.gamma*next_value - value
			self.Q[state, action] += self.learning_rate * delta



class DQN():

	class Critic(torch.nn.Module):
		def __init__(self, n_neurons, n_states, n_actions):
			torch.nn.Module.__init__(self)
			self.input = torch.nn.Linear(n_states, n_neurons)
			self.hidden = torch.nn.Linear(n_neurons, n_neurons)
			self.output = torch.nn.Linear(n_neurons, n_actions)
			self.apply(self.init_params)
		def forward(self, x):
			x = torch.nn.functional.relu(self.input(x))
			x = torch.nn.functional.relu(self.hidden(x))
			x = self.output(x)
			return x
		def init_params(self, m):
			classname = m.__class__.__name__
			if classname.find("Linear") != -1:
				m.weight.data.normal_(0, 1)
				m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))

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

	def __init__(self, player, seed=0, n_states=156, n_actions=11, n_neurons=30, ID="DQN",
			learning_rate=3e-2, gamma=0.9, epsilon_decay=0.0025, batch_size=20, target_update=50,
			orientation="proself", representation="one-hot", batch_learn=False,
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
		self.representation = representation
		self.batch_learn = batch_learn
		self.turn_basis = make_unitary(np.fft.fft(nengo.dists.UniformHypersphere().sample(1, n_states, rng=self.rng)))
		self.coin_basis = make_unitary(np.fft.fft(nengo.dists.UniformHypersphere().sample(1, n_states, rng=self.rng)))
		self.w_s = 1
		if self.randomize:
			self.gamma = self.rng.uniform(0.8, 1)
			self.learning_rate = self.rng.uniform(1e-2, 3e-2)
			self.orientation = "proself" if self.rng.uniform(0,1) < 0.5 else "prosocial"
			self.w_o = 0 if self.orientation=="proself" else self.rng.uniform(0.2, 0.4)
			self.w_i = 0 if self.orientation=="proself" else self.rng.uniform(0, 0)
			self.epsilon_decay = epsilon_decay
		else:
			self.gamma = gamma
			self.learning_rate = learning_rate
			self.orientation = orientation
			self.w_o = 0 if self.orientation=="proself" else 0.3
			self.w_i = 0 if self.orientation=="proself" else 0.5
			self.epsilon_decay = epsilon_decay
		torch.manual_seed(seed)
		self.critic = self.Critic(self.n_neurons, self.n_states, self.n_actions)
		self.target = self.Critic(self.n_neurons, self.n_states, self.n_actions) if self.batch_learn else None
		self.optimizer = torch.optim.Adam(self.critic.parameters(), self.learning_rate)
		self.value_history = []
		self.replay_memory = self.ReplayMemory(10000) if self.batch_learn else None
		self.state_history = []
		self.action_history = []
		self.state = None
		self.episode = 0

	def reinitialize(self, player):
		self.player = player
		torch.manual_seed(self.seed)
		self.critic = self.Critic(self.n_neurons, self.n_states, self.n_actions)
		self.target = self.Critic(self.n_neurons, self.n_states, self.n_actions) if self.batch_learn else None
		self.optimizer = torch.optim.Adam(self.critic.parameters(), self.learning_rate)
		self.replay_memory = self.ReplayMemory(10000) if self.batch_learn else None
		self.value_history = []
		self.state_history = []
		self.action_history = []
		self.state = None
		self.episode = 0

	def new_game(self, game):
		self.value_history.clear()
		self.state_history.clear()
		self.action_history.clear()

	def move(self, game):
		game_state = get_state(self.player, game, agent="DQN", dim=self.n_states,
			turn_basis=self.turn_basis, coin_basis=self.coin_basis, representation=self.representation)
		# Estimate the value of the current game_state
		if self.batch_learn:
			with torch.no_grad():
				values = self.critic(game_state)
		else:
			values = self.critic(game_state)			
		# Choose and action based on thees values and some exploration strategy
		epsilon = 1 - self.episode * self.epsilon_decay
		# epsilon = np.exp(-self.episode*self.epsilon_decay)
		if self.rng.uniform(0, 1) < epsilon:
			action = torch.LongTensor([self.rng.randint(self.n_actions)])
		else:
			action = torch.argmax(values)
		# translate action into environment-appropriate signal
		self.state = action.detach().numpy() / (self.n_actions-1)
		give, keep, action_idx = action_to_coins(self.player, self.state, self.n_actions, game)
		if self.batch_learn:
			self.value_history.append(values)
		else:
			self.value_history.append(values[action_idx])
		self.state_history.append(game_state)
		self.action_history.append(torch.LongTensor([action_idx]))
		# self.action_history.append(action_idx)
		return give, keep

	def learn(self, game):
		self.episode += 1
		rewards = get_rewards(self.player, game, self.w_s, self.w_o, self.w_i, self.normalize, self.gamma)
		if self.batch_learn:
			for t in range(game.turns):
				state = self.state_history[t]
				action = self.action_history[t]
				reward = torch.FloatTensor([rewards[t]])
				if t<game.turns-1:
					next_state = self.state_history[t+1]
				else:
					next_state = torch.FloatTensor(np.zeros((self.n_states)))
				self.replay_memory.push((state, action, next_state, reward))
			if len(self.replay_memory) < self.batch_size: return
			transitions = self.replay_memory.sample(self.batch_size)
			batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)
			batch_state = torch.reshape(torch.cat(batch_state), shape=(self.batch_size, self.n_states))
			batch_action = torch.reshape(torch.cat(batch_action), shape=(self.batch_size, 1))
			batch_next_state = torch.reshape(torch.cat(batch_next_state), shape=(self.batch_size, self.n_states))
			batch_reward = torch.cat(batch_reward)
			current_values = self.critic(batch_state).gather(1, batch_action).squeeze()
			# mask the "final" states, which are tensors with all zeroses
			non_final_mask = torch.tensor(tuple(map(lambda s: torch.count_nonzero(s)>0, batch_next_state)), dtype=torch.bool)
			non_final_next_states = torch.cat([s for s in batch_next_state if torch.count_nonzero(s)>0])
			non_final_next_states = torch.reshape(non_final_next_states, shape=(torch.sum(non_final_mask).item(), self.n_states))
			expected_rewards = torch.zeros(self.batch_size)
			expected_rewards[non_final_mask] = self.target(non_final_next_states).detach().max(1)[0]
			expected_values = batch_reward + self.gamma*expected_rewards
			masked_rewards = torch.zeros(self.batch_size)
			masked_rewards[non_final_mask] = batch_reward[non_final_mask]
			loss = torch.nn.functional.huber_loss(current_values, expected_values)
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
			if self.episode % self.target_update == 0:
				self.target.load_state_dict(self.critic.state_dict())
		else:
			losses = []
			for t in np.arange(game.turns):
				value = self.value_history[t]
				reward = torch.FloatTensor([rewards[t]])
				if t==(game.turns-1):
					next_value = 0
				else:
					next_value = torch.max(self.value_history[t+1])
				delta = reward + self.gamma*next_value - value
				losses.append(delta**2)
			loss = torch.stack(losses).sum()
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

class IBL():

	class Chunk():
		def __init__(self, turn, coins, action, reward, value, episode, decay, epsilon):
			self.turn = turn
			self.coins = coins
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

	def __init__(self, player, ID="IBL", seed=0, n_actions=11,
			gamma=0.8, epsilon_decay=0.01,
			orientation="proself",
			activation_decay=0.5, activation_noise=0.3, thr_activation=0,
			normalize=False, randomize=False):
		self.player = player
		self.ID = ID
		self.seed = seed
		self.randomize = randomize
		self.normalize = normalize
		self.rng = np.random.RandomState(seed=seed)
		self.n_actions = n_actions
		self.w_s = 1
		if self.randomize:
			self.gamma = self.rng.uniform(0.8, 1.0)
			self.activation_decay = self.rng.uniform(0.4, 0.5)
			self.activation_noise = self.rng.uniform(0.2, 0.3)
			self.orientation = "proself" if self.rng.uniform(0,1) < 0.5 else "prosocial"
			self.w_o = 0 if self.orientation=="proself" else self.rng.uniform(0, 1)
			self.w_i = 0 if self.orientation=="proself" else self.rng.uniform(0, 1)
			self.epsilon_decay = epsilon_decay
		else:
			self.gamma = gamma
			self.activation_decay = activation_decay
			self.activation_noise = activation_noise
			self.orientation = orientation
			self.epsilon_decay = epsilon_decay
			self.w_o = 0 if self.orientation=="proself" else 0.3
			self.w_i = 0 if self.orientation=="proself" else 0.5
		self.thr_activation = thr_activation  # activation threshold for retrieval (loading chunks from declarative into working memory)
		self.declarative_memory = []
		self.working_memory = []
		self.learning_memory = []
		self.state = None
		self.episode = 0

	def reinitialize(self, player):
		self.player = player
		self.declarative_memory = []
		self.working_memory = []
		self.learning_memory = []
		self.state = None
		self.episode = 0

	def new_game(self, game):
		self.working_memory.clear()
		self.learning_memory.clear()
		self.rng.shuffle(self.declarative_memory)

	def move(self, game):
		turn, coins = get_state(self.player, game, "IBL")
		# load chunks from declarative memory into working memory
		self.populate_working_memory(turn, coins, game)
		# select an action (generosity) that immitates the best chunk in working memory
		self.state = self.select_action()
		# create a new chunk for the chosen action, populate with more information in learn()
		new_chunk = self.Chunk(turn, coins, None, None, None, self.episode, self.activation_decay, self.activation_noise)
		self.learning_memory.append(new_chunk)
		# translate action into environment-appropriate signal
		give, keep, action_idx = action_to_coins(self.player, self.state, self.n_actions, game)
		return give, keep

	def populate_working_memory(self, current_turn, current_coins, game):
		self.working_memory.clear()
		for chunk in self.declarative_memory:
			activation = chunk.get_activation(self.episode, self.rng)
			similarity_state = 1 if current_turn==chunk.turn and current_coins==chunk.coins else 0
			pass_activation = activation > self.thr_activation
			pass_state = similarity_state > 0
			if pass_activation and pass_state:
				self.working_memory.append(chunk)

	def select_action(self):
		if len(self.working_memory)==0:
			# if there are no chunks in working memory, select a random action
			selected_action = self.rng.randint(0, self.n_actions) / (self.n_actions-1)
		else:
			# choose an action based on the activation, similarity, reward, and/or value of chunks in working memory
			actions = {}
			# collect chunks by actionsrValues
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
			epsilon = 1 - self.episode * self.epsilon_decay
			# epsilon = np.exp(-self.episode*self.epsilon_decay)
			if self.rng.uniform(0, 1) < epsilon:
				selected_action = self.rng.randint(0, self.n_actions) / (self.n_actions-1)
			else:
				selected_action = max(actions, key=lambda action: actions[action]['blended'])
		return selected_action

	def learn(self, game):
		rewards = get_rewards(self.player, game, self.w_s, self.w_o, self.w_i, self.normalize, self.gamma)
		actions = game.investor_gen if self.player=='investor' else game.trustee_gen
		for t in np.arange(game.turns):
			# update value of new chunks
			chunk = self.learning_memory[t]
			chunk.action = actions[t]
			chunk.reward = rewards[t]
			# estimate the value of the next chunk by retrieving all similar chunks and computing their blended value
			if t==game.turns-1:
				chunk.value = chunk.reward
			else:
				next_turn = t+1
				next_coins = game.coins if self.player=="investor" else game.investor_give[next_turn]*game.match
				expected_value = 0
				rValues = []
				rActivations = []
				# recall all chunks in working memory and compare to current chunk
				for rChunk in self.declarative_memory:
					rActivation = rChunk.get_activation(self.episode, self.rng)
					similarity_state = 1 if next_coins==rChunk.coins else 0	
					pass_activation = rActivation > self.thr_activation
					pass_state = similarity_state > 0
					if pass_activation and pass_state:
						rValues.append(rChunk.value)
						rActivations.append(rActivation)
				if len(rValues)>0:
					expected_value = np.average(rValues, weights=rActivations)
				chunk.value = chunk.reward + self.gamma*expected_value

		for nChunk in self.learning_memory:
			# Check if the new chunk has identical (state, action) to a previous chunk in declarative memory.
			# If so, update that chunk's triggers, rather than adding a new chunk to declarative memory
			add_nChunk = True
			for rChunk in self.declarative_memory:
				if nChunk.turn==rChunk.turn and nChunk.coins==rChunk.coins and nChunk.action == rChunk.action:
					rChunk.triggers.append(nChunk.triggers[0])
					rChunk.reward = nChunk.reward
					rChunk.value = nChunk.value
					add_nChunk = False
					break
			# Otherwise, add a new chunk to declarative memory
			if add_nChunk:
				self.declarative_memory.append(nChunk)
		self.episode += 1





class SPA():

	class Environment():
		def __init__(self, player, n_states, n_actions, t1, t2, t3, tR, rng, gamma, w_s, w_o, w_i, normalize=True, dt=1e-3):
			self.player = player
			self.state = np.zeros((n_states))
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
			self.w_s = w_s
			self.w_o = w_o
			self.w_i = w_i
			self.normalize = normalize
		def set_state(self, state):
			self.state = state
		def set_reward(self, game):
			rewards_self = np.array(game.investor_reward) if self.player=='investor' else np.array(game.trustee_reward)
			rewards_other = np.array(game.trustee_reward) if self.player=='investor' else np.array(game.investor_reward)
			rewards = self.w_s*rewards_self + self.w_o*rewards_other - self.w_i*np.abs(rewards_self-rewards_other)
			if self.normalize:
				rewards = rewards / (game.coins * game.match)
				if len(rewards)==0:
					self.reward = 0
				elif len(rewards)<5:
					self.reward = (1-self.gamma)*rewards[-1]
				elif len(rewards)==5:
					self.reward = rewards[-1]
			else:
				self.reward = rewards[-1]
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
		def save_value(self, t):
			T = t % (self.t1 + self.t2 + self.t3)
			if 0 <= T <= self.t1:
				return 1  # save value Q(s',a*) in state s' in stage 1
			else:
				return 0		
		def save_state(self, t):
			T = t % (self.t1 + self.t2 + self.t3)
			if self.t1 + self.t2 < T <= self.t1 + self.t2 + self.t3:
				return 1  # save state s' in stage 3
			else:
				return 0
		def save_choice(self, t):
			T = t % (self.t1 + self.t2 + self.t3)
			if self.t1 + self.t2 < T <= self.t1 + self.t2 + self.t3:
				return 1  # save choice a (with exploration) in stage 3
			else:
				return 0
		def do_replay(self, t):
			T = t % (self.t1 + self.t2 + self.t3)
			if self.t1 < T <= self.t1 + self.t2:
				return 1  # recall state s in stage 2
			else:
				return 0
		def do_reset(self, t):
			T = t % (self.t1 + self.t2 + self.t3)
			# reset only for the first tR seconds of each phase
			if 0 <= T < self.tR:
				return 1
			elif self.t1 < T < self.t1 + self.tR:
				return 1
			elif self.t1 + self.t2 < T < self.t1 + self.t2 + self.tR:
				return 1
			else:
				return 0


	def __init__(self, player, seed=0, n_actions=11, ID="SPA",
			learning_rate=3e-8, n_neurons=1000, n_array=500,
			n_states=100, sparsity=0.05, turn_exp=3, coin_exp=0.3,
			gate_mode="array", memory_mode="array", randomize=False, normalize=True,
			dt=1e-3, t1=1e-1, t2=1e-1, t3=1e-1, tR=1e-2, orientation="proself",
			epsilon_decay=0.005, gamma=0.6):
		self.player = player
		self.ID = ID
		self.seed = seed
		self.rng = np.random.RandomState(seed=seed)
		self.n_states = n_states
		self.n_actions = n_actions
		self.n_neurons = n_neurons
		self.n_array = n_array
		self.dt = dt
		self.normalize = normalize
		self.randomize = randomize
		self.w_s = 1
		if self.randomize:
			self.gamma = self.rng.uniform(0.6, 0.6)
			self.learning_rate = self.rng.uniform(1e-8, 2e-8)
			self.orientation = "proself"# if self.rng.uniform(0,1) < 0.5 else "prosocial"
			self.w_o = 0 if self.orientation=="proself" else self.rng.uniform(0.2, 0.3)
			self.w_i = 0 if self.orientation=="proself" else self.rng.uniform(0.2, 0.3)
			self.epsilon_decay = epsilon_decay
		else:
			self.gamma = gamma
			self.learning_rate = learning_rate
			self.orientation = orientation
			self.w_o = 0 if self.orientation=="proself" else 0.3
			self.w_i = 0 if self.orientation=="proself" else 0.3
			self.epsilon_decay = epsilon_decay
		self.t1 = t1
		self.t2 = t2
		self.t3 = t3
		self.tR = tR
		self.memory_mode = memory_mode
		self.gate_mode = gate_mode
		self.epsilon_decay = epsilon_decay
		self.env = self.Environment(self.player, self.n_states, self.n_actions, t1, t2, t3, tR,
			self.rng, self.gamma, self.w_s, self.w_o, self.w_i, self.normalize)
		self.decoders = np.zeros((self.n_neurons, self.n_actions))
		self.network = None
		self.simulator = None
		self.state = None
		self.episode = 0
		self.sparsity = sparsity
		self.turn_exp = turn_exp
		self.coin_exp = coin_exp
		self.sampler = nengo.dists.UniformHypersphere()
		self.turn_basis = make_unitary(np.fft.fft(self.sampler.sample(1, n_states, rng=self.rng)))
		self.coin_basis = make_unitary(np.fft.fft(self.sampler.sample(1, n_states, rng=self.rng)))
		self.intercept = nengo.dists.Choice([self.sparsity_to_x_intercept(n_states, self.sparsity)])
		self.encoders = self.find_good_encoders()
		self.sM = np.zeros((n_states))


	def encode_state(self, t, c):
		return np.fft.ifft(self.turn_basis**(t*self.turn_exp) * self.coin_basis**(c*self.coin_exp)).real.squeeze()

	def sparsity_to_x_intercept(self, d, p):
		sign = 1
		if p > 0.5:
			p = 1.0 - p
			sign = -1
		return sign * np.sqrt(1-scipy.special.betaincinv((d-1)/2.0, 0.5, 2*p))

	def find_good_encoders(self, iterations=10):

		class NodeInput():
			def __init__(self, dim):
				self.state = np.zeros((dim))
			def set_state(self, state):
				self.state = state
			def get_state(self):
				return self.state

		ssp_input = NodeInput(self.n_states)
		encoders = self.sampler.sample(self.n_neurons, self.n_states, rng=self.rng)
		for i in range(iterations):
			network = nengo.Network(seed=self.seed)
			network.config[nengo.Ensemble].neuron_type = nengo.LIFRate()
			network.config[nengo.Ensemble].max_rates = nengo.dists.Uniform(300, 400)
			network.config[nengo.Probe].synapse = None
			with network:
				ssp_node = nengo.Node(lambda t, x: ssp_input.get_state(), size_in=2, size_out=self.n_states)
				ens = nengo.Ensemble(self.n_neurons, self.n_states, encoders=encoders, intercepts=self.intercept)
				nengo.Connection(ssp_node, ens, synapse=None, seed=self.seed)
				p_spikes = nengo.Probe(ens.neurons, synapse=None)
			sim = nengo.Simulator(network, progress_bar=False)

			spikes = []
			trials = []
			for turn in range(5):
				for coin in range(31):
					trials.append([turn, coin])
					sim.reset(self.seed)
					ssp = self.encode_state(turn, coin)
					ssp_input.set_state(ssp)
					sim.run(0.001, progress_bar=False)
					spk = sim.data[p_spikes][-1]
					spikes.append(spk)
			spikes = np.array(spikes)
			inactives = list(np.where(np.sum(spikes, axis=0)==0)[0])

			non_uniques = []
			for pair in itertools.combinations(range(5*31), 2):
				spikes_a = spikes[pair[0]]
				spikes_b = spikes[pair[1]]
				for n in range(self.n_neurons):
					s_a = spikes_a[n]
					s_b = spikes_b[n]
					if s_a>0 and s_b>0 and -1 < s_a-s_b < 1:
						non_uniques.append(n)

			bad_neurons = np.sort(np.unique(inactives+non_uniques))
			print(f"iteration {i}")
			print(f"non unique neurons: {len(np.sort(np.unique(non_uniques)))}")
			print(f"quiet neurons: {len(inactives)}")
			# print(f"non unique neurons: {np.sort(np.unique(non_uniques))}")
			# print(f"quiet neurons: {inactives}")
			if len(bad_neurons)==0: break

			new_encoders = self.sampler.sample(self.n_neurons, self.n_states, rng=self.rng)
			for n in range(self.n_neurons):
				if n not in bad_neurons:
					new_encoders[n] = encoders[n]
			encoders = np.array(new_encoders)
			
		return encoders

	def reinitialize(self, player):
		self.player = player
		self.decoders = np.zeros((self.n_neurons, self.n_actions))
		self.network = self.build_network()
		self.simulator = nengo.Simulator(self.network, dt=self.dt, seed=self.seed, progress_bar=True)
		self.episode = 0

	def new_game(self, game):
		self.env.__init__(self.player, self.n_states, self.n_actions, self.t1, self.t2, self.t3, self.tR, self.rng, self.gamma,
			self.w_s, self.w_o, self.w_i, self.normalize)
		self.episode += 1
		self.simulator.reset(self.seed)

	def build_network(self):
		n_actions = self.n_actions
		n_states = self.n_states
		n_neurons = self.n_neurons
		n_array = self.n_array
		seed = self.seed
		network = nengo.Network(seed=seed)
		network.config[nengo.Ensemble].neuron_type = nengo.LIFRate()
		network.config[nengo.Ensemble].max_rates = nengo.dists.Uniform(300, 400)
		network.config[nengo.Probe].synapse = None
		with network:

			# Network Definitions
			class LearningNode(nengo.Node):
				# implements PES learning rule
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

			def Gate(n_neurons, dim, seed):
				# receives two inputs (e.g. states) and two gating signals (which must be opposite, e.g. [0,1] or [1,0])
				# returns input A if gateA is open, returns input B if gateB is open
				net = nengo.Network(seed=seed)
				wInh = -1e1*np.ones((n_neurons*dim, 1))
				with net:
					net.a = nengo.Node(size_in=dim)
					net.b = nengo.Node(size_in=dim)
					net.output = nengo.Node(size_in=dim)
					net.gate_a = nengo.Ensemble(n_neurons, 1)
					net.gate_b = nengo.Ensemble(n_neurons, 1)
					net.ens_a = nengo.networks.EnsembleArray(n_neurons, dim, radius=0.3)
					net.ens_b = nengo.networks.EnsembleArray(n_neurons, dim, radius=0.3)
					net.ens_a.add_neuron_input()
					net.ens_b.add_neuron_input()
					nengo.Connection(net.a, net.ens_a.input, synapse=None)
					nengo.Connection(net.b, net.ens_b.input, synapse=None)
					nengo.Connection(net.ens_a.output, net.output, synapse=None)
					nengo.Connection(net.ens_b.output, net.output, synapse=None)
					nengo.Connection(net.gate_a, net.ens_a.neuron_input, transform=wInh, synapse=None)
					nengo.Connection(net.gate_b, net.ens_b.neuron_input, transform=wInh, synapse=None)
				return net

			def Memory(n_neurons, dim, seed, gain=0.1, radius=1, synapse=0, onehot_cleanup=False):
				# gated difference memory, saves "state" to the memory if "gate" is open, otherwise maintains "state" in the memory
				wInh = -1e1*np.ones((n_neurons*dim, 1))
				net = nengo.Network(seed=seed)
				with net:
					net.state = nengo.Node(size_in=dim)
					net.output = nengo.Node(size_in=dim)
					net.gate = nengo.Node(size_in=1)
					net.mem = nengo.networks.EnsembleArray(n_neurons, dim, radius=radius)
					net.diff = nengo.networks.EnsembleArray(n_neurons, dim, radius=radius)
					net.diff.add_neuron_input()
					nengo.Connection(net.state, net.diff.input, synapse=None)
					nengo.Connection(net.diff.output, net.mem.input, transform=gain, synapse=synapse)
					nengo.Connection(net.mem.output, net.mem.input, synapse=synapse)
					nengo.Connection(net.mem.output, net.diff.input, transform=-1, synapse=synapse)
					nengo.Connection(net.gate, net.diff.neuron_input, transform=wInh, synapse=None)
					if onehot_cleanup:  # for choice memory
						net.onehot = nengo.networks.EnsembleArray(n_neurons, dim, intercepts=nengo.dists.Uniform(0.5,1), encoders=nengo.dists.Choice([[1]]))
						for a in range(dim):
							nengo.Connection(net.mem.ea_ensembles[a], net.onehot.ea_ensembles[a], function=lambda x: np.around(x), synapse=None)
						nengo.Connection(net.onehot.output, net.output, synapse=None)
					else:  # for value and state memories
						nengo.Connection(net.mem.output, net.output, synapse=None)
				return net

			def IndependentAccumulator(n_neurons, dim, seed, thr=0.9, Tff=1e-1, Tfb=-1e-1):
				# WTA selection, each dimension of "input" accumulates in a seperate integrator (one dim of an ensemble array)
				# at a rate "Tff" until one reaches a value "thr". That dimension then 'de-accumulates' each other dimension
				# at a rate "Tfb" until they reach a value of zero
				net = nengo.Network(seed=seed)
				wReset = -1e1 * np.ones((n_neurons, 1))
				with net:
					net.input = nengo.Node(size_in=dim)
					net.reset = nengo.Node(size_in=1)
					net.acc = nengo.networks.EnsembleArray(n_neurons, dim, intercepts=nengo.dists.Uniform(0, 1), encoders=nengo.dists.Choice([[1]]))
					net.inh = nengo.networks.EnsembleArray(n_neurons, dim, intercepts=nengo.dists.Uniform(thr, 1), encoders=nengo.dists.Choice([[1]]))
					net.output = nengo.Node(size_in=dim)
					nengo.Connection(net.input, net.acc.input, synapse=None, transform=Tff)
					nengo.Connection(net.acc.output, net.acc.input, synapse=0)
					nengo.Connection(net.acc.output, net.inh.input, synapse=0)
					for a in range(dim):
						nengo.Connection(net.reset, net.acc.ea_ensembles[a].neurons, synapse=None, transform=wReset)
						for a2 in range(dim):
							if a!=a2:
								nengo.Connection(net.inh.ea_ensembles[a], net.acc.ea_ensembles[a2], synapse=0, transform=Tfb)
					nengo.Connection(net.acc.output, net.output, synapse=None)
				return net

			def Compressor(n_neurons, dim, seed):
				# receives a full vector of values and a one-hot choice vector, and takes the dot product
				# this requires inhibiting all non-chosen dimensions of "value", then summing all dimensions
				net = nengo.Network(seed=seed)
				wInh = -1e1 * np.ones((n_neurons, 1))
				with net:
					net.values = nengo.Node(size_in=dim)
					net.choice = nengo.Node(size_in=dim)
					net.output = nengo.Node(size_in=1)
					net.bias = nengo.Node(np.ones((dim)))
					net.ens = nengo.networks.EnsembleArray(n_neurons, dim)
					net.inh = nengo.networks.EnsembleArray(n_neurons, dim, intercepts=nengo.dists.Uniform(0.1,1), encoders=nengo.dists.Choice([[1]]))
					nengo.Connection(net.values, net.ens.input, synapse=None)
					nengo.Connection(net.choice, net.inh.input, transform=-1, synapse=None)
					nengo.Connection(net.bias, net.inh.input, synapse=None)
					for a in range(dim):
						nengo.Connection(net.inh.output[a], net.ens.ea_ensembles[a].neurons, transform=wInh, synapse=None)
						nengo.Connection(net.ens.ea_ensembles[a], net.output, synapse=None)
				return net

			def Expander(n_neurons, dim, seed):
				# receives a single vector and a one-hot choice vector, and scales the one-hot vector by "value"
				# to avoid multiplication, this requires creating a new vector with each entry equal to "value", then inhibiting all but one dim
				net = nengo.Network(seed=seed)
				wInh = -1e1 * np.ones((n_neurons, 1))
				with net:
					net.value = nengo.Node(size_in=1)
					net.choice = nengo.Node(size_in=dim)
					net.output = nengo.Node(size_in=dim)
					net.bias = nengo.Node(np.ones((dim)))
					net.ens = nengo.networks.EnsembleArray(n_neurons, dim)
					net.inh = nengo.networks.EnsembleArray(n_neurons, dim, intercepts=nengo.dists.Uniform(0.1,1), encoders=nengo.dists.Choice([[1]]))
					nengo.Connection(net.value, net.ens.input, transform=np.ones((dim, 1)), synapse=None)
					nengo.Connection(net.choice, net.inh.input, transform=-1, synapse=None)
					nengo.Connection(net.bias, net.inh.input, synapse=None)
					nengo.Connection(net.ens.output, net.output, synapse=None)
					for a in range(dim):
						nengo.Connection(net.inh.output[a], net.ens.ea_ensembles[a].neurons, transform=wInh, synapse=None)
				return net

			def Selector(n_neurons, dim, seed):
				# receives a full vector of values and a one-hot choice vector, and takes element-wise product
				# same as "Compressor", but without the final summation
				net = nengo.Network(seed=seed)
				wInh = -1e1 * np.ones((n_neurons, 1))
				with net:
					net.values = nengo.Node(size_in=dim)
					net.choice = nengo.Node(size_in=dim)
					net.output = nengo.Node(size_in=dim)
					net.bias = nengo.Node(np.ones((dim)))
					net.ens = nengo.networks.EnsembleArray(n_neurons, dim)
					net.inh = nengo.networks.EnsembleArray(n_neurons, dim, intercepts=nengo.dists.Uniform(0.1,1), encoders=nengo.dists.Choice([[1]]))
					nengo.Connection(net.values, net.ens.input, synapse=None)
					nengo.Connection(net.choice, net.inh.input, transform=-1, synapse=None)
					nengo.Connection(net.bias, net.inh.input, synapse=None)
					nengo.Connection(net.ens.output, net.output, synapse=None)
					for a in range(dim):
						nengo.Connection(net.inh.output[a], net.ens.ea_ensembles[a].neurons, transform=wInh, synapse=None)
				return net

			# Inputs from environment and from control systems
			state_input = nengo.Node(lambda t, x: self.env.get_state(), size_in=2, size_out=n_states)
			reward_input = nengo.Node(lambda t, x: self.env.get_reward(), size_in=2, size_out=1)
			random_choice_input = nengo.Node(lambda t, x: self.env.get_random_choice(t), size_in=2, size_out=n_actions)
			replay_switch = nengo.Node(lambda t, x: self.env.do_replay(t), size_in=2, size_out=1)
			save_state_switch = nengo.Node(lambda t, x: self.env.save_state(t), size_in=2, size_out=1)
			save_value_switch = nengo.Node(lambda t, x: self.env.save_value(t), size_in=2, size_out=1)
			save_choice_switch = nengo.Node(lambda t, x: self.env.save_choice(t), size_in=2, size_out=1)
			reset_switch = nengo.Node(lambda t, x: self.env.do_reset(t), size_in=2, size_out=1)

			# Nodes, Ensembles, and Networks
			state = nengo.Ensemble(n_neurons, n_states, encoders=self.encoders, intercepts=self.intercept)
			state_memory = Memory(n_array, n_states, seed, radius=0.3)
			state_gate = Gate(n_array, n_states, seed)
			critic = nengo.networks.EnsembleArray(n_array, n_actions)
			error = nengo.networks.EnsembleArray(n_array, n_actions, radius=0.2)
			learning = LearningNode(n_neurons, n_actions, self.decoders, self.learning_rate)
			choice = IndependentAccumulator(n_array, n_actions, seed)
			choice_memory = Memory(n_array, n_actions, seed, onehot_cleanup=True)
			value_memory = Memory(n_array, 1, seed)
			value_compressor = Compressor(n_array, n_actions, seed)
			value_expander = Expander(n_array, n_actions, seed)
			reward_expander = Expander(n_array, n_actions, seed)
			replay_selector = Selector(n_array, n_actions, seed)

			# Connections
			# stage 3: load the current state to state_memory
			nengo.Connection(state_input, state_memory.state, synapse=None)
			nengo.Connection(save_state_switch, state_memory.gate, function=lambda x: 1-x, synapse=None)

			# stage 1-3: send the current state (stage 1 or 3) OR the recalled previous state (stage 2) to the state population
			nengo.Connection(state_input, state_gate.a, synapse=None)
			nengo.Connection(state_memory.output, state_gate.b, synapse=None)
			nengo.Connection(replay_switch, state_gate.gate_a, synapse=None)
			nengo.Connection(replay_switch, state_gate.gate_b, function=lambda x: 1-x, synapse=None)
			nengo.Connection(state_gate.output, state, synapse=None)

			# stage 1-3: state to critic connection, computes Q function, updates with DeltaQ from error population
			nengo.Connection(state.neurons, learning[:n_neurons], synapse=None)
			nengo.Connection(error.output, learning[n_neurons:], synapse=None)
			nengo.Connection(learning, critic.input, synapse=0)

			# stage 1-3: Q values sent to WTA competition in choice
			nengo.Connection(critic.output, choice.input, synapse=None)
			nengo.Connection(random_choice_input, choice.input, synapse=None)
			nengo.Connection(reset_switch, choice.reset, synapse=None)

			# stage 1: save Q(s',a*)
			nengo.Connection(critic.output, value_compressor.values, synapse=None)
			nengo.Connection(choice.output, value_compressor.choice, synapse=None)
			nengo.Connection(value_compressor.output, value_memory.state, synapse=None)
			nengo.Connection(save_value_switch, value_memory.gate, function=lambda x: 1-x, synapse=None)

			# stage 3: choose action a in state s' (with exploration) and save it in choice_memory
			nengo.Connection(choice.output, choice_memory.state, synapse=None)
			nengo.Connection(save_choice_switch, choice_memory.gate, function=lambda x: 1-x, synapse=None)

			# stage 2: create vectors for value, reward, and replayed value, each with a nonzero entry at the previous choice
			# this ensures updates to the decoders only affect the dimensions corresponding to the chosen action a
			nengo.Connection(value_memory.output, value_expander.value, synapse=None)
			nengo.Connection(choice_memory.output, value_expander.choice, synapse=None)
			nengo.Connection(reward_input, reward_expander.value, synapse=None)
			nengo.Connection(choice_memory.output, reward_expander.choice, synapse=None)
			nengo.Connection(critic.output, replay_selector.values, synapse=None)
			nengo.Connection(choice_memory.output, replay_selector.choice, synapse=None)

			# stage 2: sum all inputs to compute the overall error: dQ = R(s,a) + gamma*Q(s',a*) - Q(s,a)
			nengo.Connection(value_expander.output, error.input, synapse=None, transform=self.gamma)  # gamma*Q(s',a*)
			nengo.Connection(reward_expander.output, error.input, synapse=None)  # R(s,a)
			nengo.Connection(replay_selector.output, error.input, synapse=None, transform=-1)  # -Q(s,a)

			# stage 1,3: turn learning off
			error.add_neuron_input()
			wInh = -1e1*np.ones((n_array*n_actions, 1))
			nengo.Connection(replay_switch, error.neuron_input, function=lambda x: 1-x, transform=wInh, synapse=None)			

			# Probes
			network.p_replay_switch = nengo.Probe(replay_switch)
			network.p_save_state_switch = nengo.Probe(save_state_switch)
			network.p_save_value_switch = nengo.Probe(save_value_switch)
			network.p_save_choice_switch = nengo.Probe(save_choice_switch)
			network.p_state = nengo.Probe(state)
			network.p_critic = nengo.Probe(critic.output)
			network.p_error = nengo.Probe(error.output)
			network.p_choice = nengo.Probe(choice.output)

			network.p_choice_memory = nengo.Probe(choice_memory.output)
			network.p_reward = nengo.Probe(reward_input)
			network.p_reward_expander = nengo.Probe(reward_expander.output)


		return network

	def move(self, game):
		epsilon = 1 - self.episode * self.epsilon_decay
		game_state = get_state(self.player, game, "SPA", dim=self.n_states, representation="ssp",
			turn_basis=self.turn_basis, coin_basis=self.coin_basis, turn_exp=self.turn_exp, coin_exp=self.coin_exp)
		self.env.set_reward(game)
		self.env.set_state(game_state)
		self.env.set_random_choice(epsilon)

		# print("Stage 1")
		self.simulator.run(self.t1, progress_bar=False)  # store Q(s',a*)
		print('critic', np.around(self.simulator.data[self.network.p_critic][-1], 2))
		
		# print("Stage 2")
		self.simulator.run(self.t2, progress_bar=False)  # replay Q(s,a), recall Q(s',a') from value memory, and learn
		# print('error', np.around(self.simulator.data[self.network.p_error][-1], 2))
		
		# print("Stage 3")
		self.simulator.run(self.t3, progress_bar=False)  # choose a'
		
		choice = self.simulator.data[self.network.p_choice][-1]
		action = np.argmax(choice)
		self.state = action / (self.n_actions-1)  # translate action into environment-appropriate signal
		give, keep, action_idx = action_to_coins(self.player, self.state, self.n_actions, game)

		return give, keep

	def learn(self, game):
		pass