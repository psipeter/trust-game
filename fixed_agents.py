import numpy as np
import random
from utils import *

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

	def new_game(self, game):
		self.state = self.O if self.player=="investor" else self.O/2
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


class t4tv(t4t):
	def __init__(self, player, seed, minO=0.9, maxO=1, minX=0.5, maxX=0.6, minF=0.9, maxF=1, minP=0.9, maxP=1,C=0.2, ID="t4tv"):
		self.player = player
		self.ID = ID
		rng = np.random.RandomState(seed=seed)
		self.O = np.around(rng.uniform(minO, maxO), decimals=2) if minO<maxO else minO # initial state of the agent
		self.X = np.around(rng.uniform(minX, maxX), decimals=2) if minX<maxX else minX  # expected generosity of opponent (fraction of capital given, fraction of available money returned)
		self.F = np.around(rng.uniform(minF, maxF), decimals=2) if minF<maxF else minF  # rate of forgiveness (state increase with opponent generosity)
		self.P = np.around(rng.uniform(minP, maxP), decimals=2) if minP<maxP else minP  # rate of punishment (state decrease with opponent greed)
		self.C = C  # comeback rate (state change if opponent had a forced skip last turn)
		self.M = 1 if player=="investor" else 0.5	 # maximum state of agent (prevents 100% generosity as trustee)
		self.state = self.O if player=="investor" else self.O/2 # dynamic agent state
		assert self.F >= 0, "forgiveness rate must be positive or zero"
		assert self.P >= 0, "punishment rate must be positive or zero"


class adaptive():
	def __init__(self, player, ID, n_actions=11, thr_trustee=0.5, thr_investor=1.0):
		self.player = player
		self.ID = ID
		self.n_actions = n_actions
		self.state = 0
		self.thr_trustee = thr_trustee
		self.thr_investor = thr_investor

	def new_game(self, game):
		self.state = 0

	def move(self, game):
		self.update_state(game)
		give, keep, action_idx = action_to_coins(self.player, self.state, self.n_actions, game)
		return give, keep

	def update_state(self, game):
		t = len(game.investor_give) if self.player=='investor' else len(game.trustee_give)
		# optimal learned policy against this opponent indicated as [turn1, turn2...], [cooperate, defect, ...]
		if self.ID == "cooperate":
			if self.player == 'investor':  # [c,c,c,c,d]
				self.state = 1
				if np.any(np.array(game.trustee_gen)<self.thr_trustee): self.state = 0.1
				# if t==0: self.state = 1
				# if t>0 and game.trustee_gen[-1]<self.thr_trustee: self.state = 0.1
			elif self.player == 'trustee': # [c,c,c,c,c]
				self.state = 0.5
		if self.ID == "attrition":
			if self.player == 'investor':  # [d,d,d,d,d]
				self.state = 1  
			elif self.player == 'trustee':  # [d,d,d,d,d]
				self.state = 0
		if self.ID == "defect":
			if self.player == 'investor':  # [c,d,d,d,d]
				if t==0: self.state = 1
				if t==1 and game.trustee_gen[-1]<self.thr_trustee: self.state = 0.1
			elif self.player == 'trustee':  # [c,d,d,d,d]
				if t==0: self.state = 0.5
				if t>0: self.state = 0
		if self.ID == "gift":
			if self.player == 'investor':  # [d,c,d,d,d]
				if t==0: self.state = 1
				if t==1: self.state = 0.1
				if t==2 and game.trustee_gen[-1]>=self.thr_investor: self.state = 0.5
			elif self.player == 'trustee':  # [d,c,d,d,d]
				if t==0: self.state = 0
				if t==1 and game.investor_gen[-1]>=self.thr_investor: self.state = 0.5
				if t>1: self.state = 0

	def learn(self, game):
		pass