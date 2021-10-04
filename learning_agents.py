import numpy as np
import random
import os
import tensorflow as tf
from tensorflow import keras


class ActorCritic():
	# Adapted from https://keras.io/examples/rl/actor_critic_cartpole/
	def __init__(self, player, seed=0, n_inputs=3, n_actions=11, ID="actor-critic",
			gamma=0.99, n_neurons=100, learning_rate=3e-3):
		self.player = player
		self.ID = ID
		self.seed = seed
		self.gamma = gamma
		self.learning_rate = learning_rate
		self.n_inputs = n_inputs
		self.n_actions = n_actions
		self.n_neurons = n_neurons
		self.inputs = keras.layers.Input(shape=(n_inputs,))
		self.common = keras.layers.Dense(n_neurons, activation="relu")(self.inputs)
		self.action = keras.layers.Dense(n_actions, activation="softmax")(self.common)
		self.critic = keras.layers.Dense(1)(self.common)
		self.model = keras.Model(inputs=self.inputs, outputs=[self.action, self.critic])
		self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
		self.loss_function = keras.losses.MeanAbsoluteError()
		self.action_probs_history = []
		self.critic_value_history = []
		self.rewards_history = []
		self.state = None

	def new_game(self):
		# Clear the loss and reward history
		self.action_probs_history.clear()
		self.critic_value_history.clear()
		self.rewards_history.clear()

	def move(self, game):
		game_state = self.get_state(game)
		game_state = tf.convert_to_tensor(game_state)
		game_state = tf.expand_dims(game_state, 0)
		# Predict action probabilities and estimated future rewards from game_state
		action_probs, critic_value = self.model(game_state)
		self.critic_value_history.append(critic_value[0, 0])
		# Sample action from action probability distribution
		action = np.random.choice(self.n_actions, p=np.squeeze(action_probs))
		self.action_probs_history.append(tf.math.log(action_probs[0, action]))
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
			game_state[2] = current_turn
			if current_turn > 0: # second turn and beyond
				game_state[0] = game.investor_gen[-1]
				game_state[1] = game.trustee_gen[-1]  if not np.isnan(game.trustee_gen[-1]) else -1
			else:  # first turn
				game_state[0] = 0
				game_state[1] = 0
		elif self.player == 'trustee':
			current_turn = len(game.investor_give)
			game_state[2] = current_turn
			game_state[0] = game.investor_gen[-1]
			if current_turn > 1:
				game_state[1] = game.trustee_gen[-1]  if not np.isnan(game.trustee_gen[-1]) else -1
			else:
				game_state[1] = 0
		return game_state

	def learn(self, game):
		# Calculate expected value from rewards
		# - At each timestep what was the total reward received after that timestep
		# - Rewards in the past are discounted by multiplying them with gamma
		# - These are the labels for our critic
		returns = []
		discounted_sum = 0
		rewards_history = game.investor_reward if self.player=='investor' else game.trustee_reward
		for r in rewards_history[::-1]:  # loop through reward history backwards
			discounted_sum = r + self.gamma * discounted_sum
			returns.insert(0, discounted_sum)  # append to beginning of list
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
			critic_losses.append(self.loss_function(tf.expand_dims(value, 0), tf.expand_dims(ret, 0)))
		loss_value = sum(actor_losses) + sum(critic_losses)
		grads = game.tape.gradient(loss_value, self.model.trainable_variables)
		self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))  # Backpropagation