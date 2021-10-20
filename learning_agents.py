import numpy as np
import random
import os
import torch

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

	def __init__(self, player, seed=0, n_inputs=15, n_actions=11, ID="actor-critic",
			w_self=1, w_other=0, gamma=0.99, n_neurons=100, learning_rate=3e-3):
		self.player = player
		self.ID = ID
		self.seed = seed
		self.rng = np.random.RandomState(seed=seed)
		self.gamma = gamma
		self.learning_rate = learning_rate
		self.w_self = w_self
		self.w_other = w_other
		self.n_inputs = n_inputs
		self.n_actions = n_actions
		self.n_neurons = n_neurons
		torch.manual_seed(seed)
		self.actor = self.Actor(n_neurons, n_inputs, n_actions)
		self.critic = self.Critic(n_neurons, n_inputs, 1)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.learning_rate)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.learning_rate)
		self.loss_function = torch.nn.MSELoss()
		self.action_probs_history = []
		self.critic_value_history = []
		self.rewards_history_self = []
		self.rewards_history_other = []
		self.state = None
		self.critic_losses = []
		self.actor_losses = []
		self.episode = 0

	def new_game(self):
		# Clear the loss and reward history
		self.action_probs_history.clear()
		self.critic_value_history.clear()
		self.rewards_history_self.clear()
		self.rewards_history_other.clear()
		self.episode += 1

	def move(self, game):
		game_state = self.get_state(game)
		# Predict action probabilities and estimated future rewards from game_state
		action_probs = self.actor(game_state)
		critic_value = self.critic(game_state)
		# Sample action from action probability distribution
		action_dist = torch.distributions.categorical.Categorical(probs=action_probs)
		action = action_dist.sample()
		move = action.detach().numpy()
		log_prob = torch.log(action_probs.gather(index=action, dim=0))
		self.critic_value_history.append(critic_value.squeeze())
		self.action_probs_history.append(log_prob)
		coins = game.coins if self.player=='investor' else game.investor_give[-1]*game.match  # coins available
		give = int(np.clip(move, 0, coins))
		keep = int(coins - give)
		self.update_state(give, keep)  # for reference
		return give, keep

	def update_state(self, give, keep):
		self.state = give/(give+keep) if give+keep>0 else np.NaN

	def get_state(self, game):
		game_state = np.zeros((self.n_inputs))
		if self.n_inputs == 1:
			current_turn = len(game.investor_give)
			game_state[0] = current_turn/game.turns  # normalize 0-1
		elif self.n_inputs == 2:
			if self.player == 'investor':
				current_turn = len(game.investor_give)
				game_state[0] = current_turn/game.turns  # normalize 0-1
				if current_turn > 0: # second turn and beyond
					game_state[1] = game.trustee_gen[-1]  if not np.isnan(game.trustee_gen[-1]) else -1
				else:  # first turn
					game_state[1] = 0
			elif self.player == 'trustee':
				current_turn = len(game.investor_give)
				game_state[0] = current_turn/game.turns  # normalize 0-1
				game_state[1] = game.investor_gen[-1]
		elif self.n_inputs == 3:  # [my_gen, opponent_gen, turn]
			if self.player == 'investor':
				current_turn = len(game.investor_give)
				game_state[2] = current_turn/game.turns  # normalize 0-1
				if current_turn > 0: # second turn and beyond
					game_state[0] = game.investor_gen[-1]
					game_state[1] = game.trustee_gen[-1]  if not np.isnan(game.trustee_gen[-1]) else -1
				else:  # first turn
					game_state[0] = -1
					game_state[1] = -1
			elif self.player == 'trustee':
				current_turn = len(game.investor_give)
				game_state[2] = current_turn/game.turns  # normalize 0-1
				game_state[0] = game.investor_gen[-1]
				if current_turn > 1:
					game_state[1] = game.trustee_gen[-1]  if not np.isnan(game.trustee_gen[-1]) else -1
				else:
					game_state[1] = -1
		if self.n_inputs == 5:
			if self.player == 'investor': game_state[len(game.investor_give)] = 1
			if self.player == 'trustee': game_state[len(game.investor_give)-1] = 1
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
		discounted_sum_self = 0
		discounted_sum_other = 0
		rewards_history_self = game.investor_reward if self.player=='investor' else game.trustee_reward
		rewards_history_other = game.trustee_reward if self.player=='investor' else game.investor_reward
		for t in np.arange(game.turns-1, -1, -1):  # loop through reward history backwards
			discounted_sum_self = rewards_history_self[t] + self.gamma * discounted_sum_self
			discounted_sum_other = rewards_history_other[t] + self.gamma * discounted_sum_other
			ret = torch.FloatTensor([self.w_self*discounted_sum_self + self.w_other*discounted_sum_other]).squeeze()
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



class SoftActorCritic():

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

	class ReplayMemory():
		def __init__(self, rng, batch_size):
			self.memory = []
			self.rng = rng
			self.batch_size = batch_size
		def push(self, transition):
			self.memory.append(transition)
		def sample(self):
			idx = self.rng.randint(0, len(self.memory), size=self.batch_size)
			states, actions, next_states, rewards = [], [], [], []
			for i in idx:
				states.append(self.memory[i][0].unsqueeze(0))
				actions.append(self.memory[i][1].unsqueeze(0))
				next_states.append(self.memory[i][2].unsqueeze(0))
				rewards.append(self.memory[i][3].unsqueeze(0))
			# reshape into (batchsize, dim) tensors, then remove gr
			states = torch.cat(states, dim=0)
			actions = torch.cat(actions, dim=0)
			next_states = torch.cat(next_states, dim=0)
			rewards = torch.cat(rewards, dim=0)
			return states, actions, next_states, rewards

	def __init__(self, player, seed=0, n_inputs=15, n_actions=11, ID="soft-actor-critic",
			w_self=1, w_other=0, gamma=0.99, n_neurons=100, learning_rate=1e-3, batch_size=300,
			alpha=1):
		self.player = player
		self.ID = ID
		self.seed = seed
		self.rng = np.random.RandomState(seed=seed)
		self.gamma = gamma
		self.alpha = alpha
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.w_self = w_self
		self.w_other = w_other
		self.n_inputs = n_inputs
		self.n_actions = n_actions
		self.n_neurons = n_neurons
		torch.manual_seed(seed)
		self.actor = self.Actor(n_neurons, n_inputs, n_actions)
		self.critic = self.Critic(n_neurons, n_inputs, n_actions)
		self.target = self.Critic(n_neurons, n_inputs, n_actions)
		self.memory = self.ReplayMemory(self.rng, self.batch_size)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.learning_rate)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.learning_rate)
		self.actor_loss_function = torch.nn.KLDivLoss(reduction='batchmean')
		self.critic_loss_function = torch.nn.MSELoss(reduction='mean')
		self.state = None
		self.critic_losses = []
		self.actor_losses = []
		self.episode = 0

	def new_game(self):
		self.state = None
		self.episode += 1

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

	def update_state(self, give, keep):
		self.state = give/(give+keep) if give+keep>0 else np.NaN

	def update_memory(self, game):
		game_states = []
		for current_turn in range(game.turns):
			game_state = np.zeros((self.n_inputs))
			if self.n_inputs == 15:
				game_state[current_turn] = 1
				for t in range(current_turn):
					my_gen_idx = game.turns + t
					opponent_gen_idx = 2*game.turns + t
					if self.player == 'investor':
						game_state[my_gen_idx] = game.investor_gen[t]
						game_state[opponent_gen_idx] = game.trustee_gen[t] if not np.isnan(game.trustee_gen[t]) else -1
					elif self.player == 'trustee':
						game_state[opponent_gen_idx] = game.investor_gen[t]
						game_state[my_gen_idx] = game.trustee_gen[t] if not np.isnan(game.trustee_gen[t]) else -1
			game_states.append(torch.FloatTensor(game_state))
		# add a final "finished game" state to allow learning on the last turn
		# game_states.append(-1.0*torch.ones((1, self.n_inputs)))
		# add an (state, action, next_state, reward) transition entry to memory
		for t in range(game.turns-1):
			s = game_states[t]
			sp = game_states[t+1]
			if self.player=='investor':
				a = torch.LongTensor([game.investor_give[t]])  # dtype=int
				r = torch.FloatTensor([1.0*game.investor_reward[t]])
			else:
				a = torch.LongTensor([game.trustee_give[t]])  # dtype=int
				r = torch.FloatTensor([1.0*game.trustee_reward[t]])
			self.memory.push([s,a,sp,r])

	def move(self, game):
		game_state = self.get_state(game)
		action_dist = torch.distributions.categorical.Categorical(probs=self.actor(game_state))
		action = action_dist.sample() # Sample action from action probability distribution
		move = action.detach().numpy()
		coins = game.coins if self.player=='investor' else game.investor_give[-1]*game.match  # coins available
		give = int(np.clip(move, 0, coins))
		keep = int(coins - give)
		self.update_state(give, keep)  # for reference
		return give, keep


	def learn(self, game):
		# push all the transitions from the game that just ended into memory
		# then go through the entire memory (across games) and do batch learning
		# batch_size random transitions (s,as',r) are taken from experience replay memory
		self.update_memory(game)
		if len(self.memory.memory) < self.batch_size:
			return
		states, actions, next_states, rewards = self.memory.sample()

		# loss for actor network is the KL divergence between the action distribution and the predicted Q distribution
		action_probs = self.actor(next_states)
		next_Qs = self.target(next_states)
		soft_Qs = torch.nn.functional.softmax(next_Qs, dim=1)
		actor_loss = self.actor_loss_function(action_probs, soft_Qs)
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		# torch.nn.utils.clip_grad_value_(self.actor.parameters(), clip_value=1)
		self.actor_optimizer.step()
	
		# loss for critic network is measured from error between current and expected values
		action_probs = self.actor(next_states)
		# build a distribution for the action probabilities, then sample from it
		action_dist = torch.distributions.categorical.Categorical(probs=action_probs)
		next_actions = action_dist.sample().unsqueeze(0)
		current_Qs = self.critic(states)
		next_Qs = self.target(next_states)
		# sample the next Q value by choosing the sampled actions and adding the entropy of the policy network
		current_values = current_Qs.gather(index=actions, dim=1)
		next_values = next_Qs.gather(index=next_actions, dim=1)
		entropy = action_dist.entropy()
		expected_future_rewards = torch.transpose(next_values + self.alpha*entropy, 0, 1)
		expected_values = rewards + self.gamma * expected_future_rewards
		critic_loss = self.critic_loss_function(current_values, expected_values)
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		# torch.nn.utils.clip_grad_value_(self.critic.parameters(), clip_value=1)
		self.critic_optimizer.step()

		self.actor_losses.append([self.episode, float(actor_loss.detach().numpy())])
		self.critic_losses.append([self.episode, float(critic_loss.detach().numpy())])

		if self.episode % 100 == 0:
			self.target.load_state_dict(self.critic.state_dict())

		# OR update the actor network by reducing the log probability of choosing an action in a state minus the Q value of the state-action pair
		# log_prob_actions = log(self.actor(next_states))
		# log_probs = gather(log_prob_actions, next_actions, batch_dims=1)
		# actor_loss = keras.losses.MeanSquaredError()(log_probs, next_values)