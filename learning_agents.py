import numpy as np
import random
import os
from tensorflow import keras, convert_to_tensor, expand_dims
from tensorflow.math import log


class ActorCritic():
	# Adapted from https://keras.io/examples/rl/actor_critic_cartpole/
	def __init__(self, player, seed=0, n_inputs=3, n_actions=11, ID="actor-critic",
			w_self=1, w_other=0, gamma=0.99, n_neurons=300, learning_rate=1e-3):
		self.player = player
		self.ID = ID
		self.seed = seed
		self.gamma = gamma
		self.learning_rate = learning_rate
		self.w_self = w_self
		self.w_other = w_other
		self.n_inputs = n_inputs
		self.n_actions = n_actions
		self.n_neurons = n_neurons
		self.inputs = keras.layers.Input(shape=(n_inputs,))
		self.common = keras.layers.Dense(n_neurons, activation="relu")(self.inputs)
		self.action = keras.layers.Dense(n_actions, activation="softmax")(self.common)
		self.critic = keras.layers.Dense(1)(self.common)
		self.model = keras.Model(inputs=self.inputs, outputs=[self.action, self.critic])
		self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
		# self.loss_function = keras.losses.MeanAbsoluteError()
		self.loss_function = keras.losses.Huber()
		self.action_probs_history = []
		self.critic_value_history = []
		self.rewards_history_self = []
		self.rewards_history_other = []
		self.state = None

	def new_game(self):
		# Clear the loss and reward history
		self.action_probs_history.clear()
		self.critic_value_history.clear()
		self.rewards_history_self.clear()
		self.rewards_history_other.clear()

	def move(self, game):
		game_state = self.get_state(game)
		game_state = convert_to_tensor(game_state)
		game_state = expand_dims(game_state, 0)
		# Predict action probabilities and estimated future rewards from game_state
		action_probs, critic_value = self.model(game_state)
		self.critic_value_history.append(critic_value[0, 0])
		# Sample action from action probability distribution
		action = np.random.choice(self.n_actions, p=np.squeeze(action_probs))
		self.action_probs_history.append(log(action_probs[0, action]))
		coins = game.coins if self.player=='investor' else game.investor_give[-1]*game.match  # coins available
		give = int(np.clip(action, 0, coins))
		keep = int(coins - give)
		self.update_state(give, keep)  # for reference
		return give, keep

	def update_state(self, give, keep):
		self.state = give/(give+keep) if give+keep>0 else np.NaN

	def get_state(self, game):
		game_state = np.zeros((self.n_inputs))
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
		return game_state

	# def get_state(self, game):
	# 	game_state = np.zeros((self.n_inputs))
	# 	if self.player == 'investor':
	# 		current_turn = len(game.investor_give)
	# 		game_state[0] = current_turn
	# 		if current_turn > 0: # second turn and beyond
	# 			game_state[1] = game.trustee_gen[-1]  if not np.isnan(game.trustee_gen[-1]) else -1
	# 		else:  # first turn
	# 			game_state[1] = 0
	# 	elif self.player == 'trustee':
	# 		current_turn = len(game.investor_give)
	# 		game_state[0] = current_turn
	# 		game_state[1] = game.investor_gen[-1]
	# 	return game_state

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
			returns.insert(0, self.w_self*discounted_sum_self + self.w_other*discounted_sum_other)  # append to beginning of list
		# Calculating loss values to update our network
		history = zip(self.action_probs_history, self.critic_value_history, returns)
		actor_losses = []
		critic_losses = []
		for log_prob, value, ret in history:
			# At this point in history, the critic estimated that we would get a
			# total reward = `value` in the future. We took an action with log probability
			# of `log_prob` and ended up recieving a total reward = `ret`.
			# The actor must be updated so that it predicts an action that leads to
			# high rewards (compared to critic's estimate) with high probability.
			diff = ret - value
			actor_losses.append(-log_prob * diff)  # actor loss
			# The critic must be updated so that it predicts a better estimate of the future rewards.
			critic_losses.append(self.loss_function(expand_dims(value, 0), expand_dims(ret, 0)))
		# print('actor loss', sum(actor_losses))
		# print('critic loss', sum(critic_losses))
		loss_value = sum(actor_losses) + sum(critic_losses)
		grads = game.tape.gradient(loss_value, self.model.trainable_variables)
		self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))  # Backpropagation