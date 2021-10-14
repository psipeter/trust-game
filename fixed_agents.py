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


class adaptive():
	def __init__(self, player, ID, thr_trustee=0.5, thr_investor=1.0):
		self.player = player
		self.ID = ID
		self.state = 0
		self.thr_trustee = thr_trustee
		self.thr_investor = thr_investor

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
		if self.ID == "turn_based":
			if self.player == 'investor':
				if len(game.investor_gen)==0: self.state = 1
				if len(game.investor_gen)==1: self.state = 1  # trustee should learn to be greedy turn 0
				if len(game.investor_gen)==2 and game.trustee_gen[-1]<self.thr_trustee: self.state = 0.1  # trustee should learn to be generous turn 1
				if len(game.investor_gen)==3 and game.trustee_gen[-1]<self.thr_trustee: self.state = 0.1  # trustee should learn to be generous turn 2
				if len(game.investor_gen)==4: self.state = 1  # trustee should learn to be greedy turn 3 and 4
			elif self.player == 'trustee':
				if len(game.investor_gen)==1: self.state = 0  # investor should learn to be greedy turn 0
				if len(game.investor_gen)==2: self.state = 0.5  # investor should learn to be generous turn 1
				if len(game.investor_gen)==3: self.state = 0.5  # investor should learn to be generous turn 2
				if len(game.investor_gen)==4: self.state = 0  # investor should learn to be greedy turn 3
				if len(game.investor_gen)==5: self.state = 0  # investor should learn be greedy turn 4
		if self.ID == "cooperate":
			if self.player == 'investor':   #  trustee should learn to cooperate on turn 0-4
				self.state = 1 if (len(game.investor_gen)==0 or game.trustee_gen[-1]>=self.thr_trustee) else 0
			elif self.player == 'trustee':
				self.state = 0.5  #  investor should learn to cooperate on turn 0-4
		if self.ID == "defect":
			if self.player == 'investor':
				self.state = 1   #  trustee should learn to defect on turn 0-4
			elif self.player == 'trustee':
				if len(game.investor_gen)==1: self.state = 0.5  # investor should learn to cooperate turn 0
				if len(game.investor_gen)==2: self.state = 0.5  # investor should learn to cooperate turn 1
				if len(game.investor_gen)==3: self.state = 0.5  # investor should learn to cooperate turn 2
				if len(game.investor_gen)==4: self.state = 0  # investor should learn to defect turn 3
				if len(game.investor_gen)==5: self.state = 0  # investor should learn attrition turn 4
		if self.ID == "gift":
			if self.player == 'investor':
				if len(game.investor_gen)==0: self.state = 0.3  # agent begins with a small investment
				if len(game.investor_gen)==1 and game.trustee_gen[-1]>=self.thr_trustee: self.state += 0.3  # trustee should learn to offer gift turn 0
				if len(game.investor_gen)==2 and game.trustee_gen[-1]>=self.thr_trustee: self.state += 0.4  # trustee should learn to offer gift turn 1
				if len(game.investor_gen)==3: pass  # trustee should defect turn 2
				if len(game.investor_gen)==4: pass  # trustee should defect turn 3 and 4
			elif self.player == 'trustee':
				if len(game.investor_gen)==1: self.state = 0  # investor should learn to offer nothing turn 0
				if len(game.investor_gen)==2 and game.investor_gen[-1]>=self.thr_trustee: self.state += 0.45  # investor should learn to offer gift turn 1
				if len(game.investor_gen)==3 and game.investor_gen[-1]>=self.thr_trustee: self.state += 0.55  # investor should learn to offer gift turn 2
				if len(game.investor_gen)==4: pass  # investor should learn to cooperate turn 3
				if len(game.investor_gen)==5: pass  # investor should learn to cooperate turn 4
		if self.ID == "attrition":
			if self.player == 'investor':
				self.state = 0.9
			elif self.player == 'trustee':
				self.state = 0

	def learn(self, game):
		pass