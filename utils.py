import numpy as np
import random
import pandas as pd
import tensorflow as tf

class Game():
	def __init__(self, coins=10, match=3, turns=5, tape=tf.GradientTape()):
		self.coins = coins
		self.match = match
		self.turns = turns
		self.investor_give = []
		self.investor_keep = []
		self.investor_gen = []
		self.investor_reward = []
		self.investor_state = []
		self.trustee_give = []
		self.trustee_keep = []
		self.trustee_gen = []
		self.trustee_reward = []
		self.trustee_state = []
		self.tape = tape

def generosity(player, give, keep):
	return np.NaN if keep==0 and player=='trustee' else give/(give+keep)

def play_game(game, investor, trustee):
	assert investor.player == 'investor' and trustee.player == 'trustee', \
		f"invalid player assignments {investor.player, trustee.player}"
	investor.new_game()
	trustee.new_game()
	with game.tape:
		for t in range(game.turns):
			i_give, i_keep = investor.move(game)
			game.investor_give.append(i_give)
			game.investor_keep.append(i_keep)
			game.investor_gen.append(generosity('investor', i_give, i_keep))
			game.investor_state.append(investor.state)
			t_give, t_keep = trustee.move(game)
			game.trustee_give.append(t_give)
			game.trustee_keep.append(t_keep)
			game.trustee_gen.append(generosity('trustee', t_give, t_keep))
			game.trustee_state.append(trustee.state)
			game.investor_reward.append(i_keep+t_give)
			game.trustee_reward.append(t_keep)
		investor.learn(game)
		trustee.learn(game)