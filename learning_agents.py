import numpy as np
import random
import os
import torch
import scipy

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

	def __init__(self, player, seed=0, n_inputs=15, ID="actor-critic",
			gamma=0.99, n_neurons=100, learning_rate=3e-3):
		self.player = player
		self.ID = ID
		self.seed = seed
		self.rng = np.random.RandomState(seed=seed)
		self.gamma = gamma
		self.learning_rate = learning_rate
		self.n_inputs = n_inputs
		self.n_actions = 11 if self.player=='investor' else 31
		self.n_neurons = n_neurons
		torch.manual_seed(seed)
		self.actor = self.Actor(n_neurons, n_inputs, self.n_actions)
		self.critic = self.Critic(n_neurons, n_inputs, 1)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.learning_rate)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.learning_rate)
		self.loss_function = torch.nn.MSELoss()
		self.action_probs_history = []
		self.critic_value_history = []
		self.rewards_history = []
		self.state = None
		self.critic_losses = []
		self.actor_losses = []
		self.episode = 0

	def reinitialize(self, player):
		self.__init__(player, self.seed, self.n_inputs, self.ID,
			self.gamma, self.n_neurons, self.learning_rate)

	def new_game(self):
		self.action_probs_history.clear()
		self.critic_value_history.clear()
		self.rewards_history.clear()
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
		move = action.detach().numpy()
		coins = game.coins if self.player=='investor' else game.investor_give[-1]*game.match  # coins available
		give = int(np.clip(move, 0, coins))
		keep = int(coins - give)
		self.update_state(give, keep)  # for reference
		return give, keep

	def update_state(self, give, keep):
		self.state = give/(give+keep) if give+keep>0 else np.NaN

	def get_state(self, game):
		game_state = np.zeros((self.n_inputs))
		if self.n_inputs == 15:
			current_turn = len(game.investor_give) if self.player=='investor' else len(game.investor_give)-1
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

	def __init__(self, player, seed=0, n_inputs=15, ID="instance-based",
			thr_activation=0, thr_action=0.8, thr_history=0.8, epsilon=1.0, epsilon_decay=0.99,
			populate_method='state-similarity', select_method='softmax-blended-value', value_method='next-value'):
		self.player = player
		self.ID = ID
		self.seed = seed
		self.rng = np.random.RandomState(seed=seed)
		self.n_inputs = n_inputs
		self.n_actions = 11 if self.player=='investor' else 31
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
		self.__init__(player, self.seed, self.n_inputs, self.ID,
			self.thr_activation, self.thr_action, self.thr_history, 1.0, self.epsilon_decay,
			self.populate_method, self.select_method, self.value_method)

	def new_game(self):
		self.working_memory.clear()
		self.learning_memory.clear()

	def get_state(self, game):
		game_state = np.zeros((self.n_inputs))
		if self.n_inputs == 15:
			current_turn = len(game.investor_give) if self.player=='investor' else len(game.investor_give)-1
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
				similarity_greedy = 1 - np.abs(chunk.action/self.n_actions - greedy_action)
				similarity_generous = 1 - np.abs(chunk.action/self.n_actions - generous_action)
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
				similarity_greedy = 1 - np.abs(chunk.action/self.n_actions - greedy_action)
				similarity_generous = 1 - np.abs(chunk.action/self.n_actions - generous_action)
				similarity_action = np.max([similarity_greedy, similarity_generous])
				current_history = game_state[5:]
				chunk_history = chunk.state[5:]
				similarity_history = 1 - np.sqrt(np.mean(np.square(current_history - chunk_history)))
				if activation > self.thr_activation and similarity_action > self.thr_action and similarity_history > self.thr_history:
					self.working_memory.append(chunk)

	def select_action(self, game_state):
		# if there are no chunks in working memory, select a random action
		if len(self.working_memory)==0:
			best_action = self.rng.randint(0, self.n_actions)	
		elif self.rng.uniform(0,1)<self.epsilon:
			best_action = self.rng.randint(0, self.n_actions)	
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
		move = self.select_action(game_state)
		# translate action into environment-appropriate signal
		coins = game.coins if self.player=='investor' else game.investor_give[-1]*game.match  # coins available
		give = int(np.clip(move, 0, coins))
		keep = int(coins - give)
		self.update_state(give, keep)  # for reference
		self.turn += 1
		return give, keep

	def update_state(self, give, keep):
		self.state = give/(give+keep) if give+keep>0 else np.NaN

	def learn(self, game):
		# update value of new chunks according to some scheme
		actions = game.investor_give if self.player=='investor' else game.trustee_give
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