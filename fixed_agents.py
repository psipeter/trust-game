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


class test_forgive():
	# as trustee: greedy on turns 0,1, friendly on turns 2,3,4
	# as investor: greedy on turns 0,1, friendly on turns 2,3,4 IF trustee is generous on turns 1,2,3, else greedy
	def __init__(self, player, ID="test_forgive"):
		self.player = player
		self.ID = ID
		self.state = 0

	def new_game(self):
		self.state = 0

	def move(self, game):
		self.update_state(game)
		coins = game.coins if self.player=='investor' else game.investor_give[-1]*game.match  # coins available on this turn
		action = coins * self.state
		give = int(np.clip(action, 0, coins))
		keep = int(coins - give)
		return give, keep

	def update_state(self, game):
		if self.player == 'investor':
			if len(game.investor_give) < 2:
				self.state = 0.5
			elif len(game.investor_give) >= 2:
				if game.trustee_gen[-1] < 0.5:
					self.state -= 0.4
				else:
					self.state += 0.5
			self.state = np.clip(self.state, 0.1, 1)
		elif self.player == 'trustee':
			self.state = 0 if len(game.investor_give) <= 2 else 0.5  

	def learn(self, game):
		pass