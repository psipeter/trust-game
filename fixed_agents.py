import numpy as np
import random

class t4t():
	def __init__(self, player, O=1, X=0.5, F=1.0, P=1.0, C=0.2, ID="t4t"):
		self.player = player
		self.ID = ID
		self.O = O  # initial agent state
		self.X = X  # expected generosity of opponent
		self.F = F  # rate of forgiveness (state increase with opponent generosity)
		self.P = P  # rate of punishment (state decrease with opponent greed)
		self.C = C  # comeback rate (state change if opponent had a forced skip last turn)
		self.M = 1 if player=="investor" else 0.5	 # maximum state of agent (prevents 100% generosity as trustee)
		self.state = O  if player=="investor" else O/2 # dynamic agent state
		assert F >= 0, "forgiveness rate must be positive or zero"
		assert P >= 0, "punishment rate must be positive or zero"

	def new_game(self):
		self.state = self.O  if self.player=="investor" else self.O/2
		self.M = 1 if self.player=="investor" else 0.5

	def move(self, game):
		self.update_state(game)
		coins = game.coins if self.player=='investor' else game.investor_give[-1]*game.match  # coins available on this turn
		action = coins * self.state
		give = int(np.clip(action, 0, coins))
		keep = int(coins - give)
		return give, keep

	def update_state(self, game):
		if self.player == "investor":
			if len(game.trustee_give)==0: return
			o_gen = game.trustee_gen[-1]
			if np.isnan(o_gen):
				# if opponent was skipped last turn, agent state goes from zero to self.C (*self.F)
				delta = self.C
			else:
				# delta proportional to generosity fraction minus expected generosity (self.X)
				delta = o_gen - self.X
		else:
			o_gen = game.investor_gen[-1]
			# delta proportional to generosity fraction minus expected generosity (self.X)
			delta = o_gen - self.X
		self.state += delta*self.F if delta>0 else delta*self.P
		self.state = np.clip(self.state, 0, self.M)

	def learn(self, game):
		pass