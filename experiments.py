import numpy as np
import random
import pandas as pd
import time
from utils import *
from plots import *
from fixed_agents import *
from learning_agents import *

class Game():
	def __init__(self, coins=10, match=3, turns=5, train=True):
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
		self.train = train

def play_game(game, investor, trustee):
	assert investor.player == 'investor' and trustee.player == 'trustee', \
		f"invalid player assignments {investor.player, trustee.player}"
	start_time = time.time()
	investor.new_game(game)
	trustee.new_game(game)
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
	if game.train:  
		if isinstance(investor, SPA) or isinstance(trustee, SPA):  # extra turn for nengo learning
			i_give, i_keep = investor.move(game)
			t_give, t_keep = trustee.move(game)
		investor.learn(game)
		trustee.learn(game)
	end_time = time.time()
	print(f"execution time: {end_time-start_time:.3}")

def game_loop(investor, trustee, agent, g, dfs):
	columns = ('ID', 'opponent', 'player', 'game', 'turn', 'generosity', 'coins', 'orientation', 'gamma')
	game = Game()
	play_game(game, investor, trustee)
	for t in range(game.turns):
		dfs.append(pd.DataFrame([[investor.ID, trustee.ID, 'investor', g, t,
			game.investor_gen[t], game.investor_reward[t], agent.orientation, agent.gamma]], columns=columns))
		dfs.append(pd.DataFrame([[trustee.ID, investor.ID, 'trustee', g, t,
			game.trustee_gen[t], game.trustee_reward[t], agent.orientation, agent.gamma]], columns=columns))
	return dfs

def train(investors, trustees, player, games):
	agents = investors if player=='investor' else trustees
	opponents = trustees if player=='investor' else investors
	dfs = []
	for agent in agents:
		print(f"{agent.ID} vs {opponents[0].ID}")
		agent.reinitialize(player=player)
		for g in range(games):
			print(f"game {g}")
			if player=='investor':
				dfs = game_loop(agent, trustees[g], agent, g=g, dfs=dfs)
			elif player=='trustee':
				dfs = game_loop(investors[g], agent, agent, g=g, dfs=dfs)
	data = pd.concat([df for df in dfs], ignore_index=True)
	return data

def make_learners(agent, seed, N, randomize=False):
	if agent=="TQ":
		learners = [TQ('investor', ID="TQ"+str(n+seed), seed=n+seed, randomize=randomize) for n in range(N)]
	if agent=="DQN":
		learners = [DQN('investor', ID="DQN"+str(n+seed), seed=n+seed, randomize=randomize) for n in range(N)]
	if agent=="IBL":
		learners = [IBL('investor', ID="IBL"+str(n+seed), seed=n+seed, randomize=randomize) for n in range(N)]
	if agent=="SPA":
		learners = [SPA('investor', ID="SPA"+str(n+seed), seed=n+seed, randomize=randomize) for n in range(N)]
	return learners

def baseline(agent, N=10, games=100, seed=0, load=False):
	learners = make_learners(agent, seed, N)
	if load:
		data = pd.read_pickle(f'agent_data/{agent}_N={N}_games={games}_benchmark.pkl')
	else:
		dfs = []
		cooperate_trustee = [benchmark('trustee', 'cooperate') for _ in range(games)]
		cooperate_investor = [benchmark('investor', 'cooperate') for _ in range(games)]
		defect_trustee = [benchmark('trustee', 'defect') for _ in range(games)]
		defect_investor = [benchmark('investor', 'defect') for _ in range(games)]
		gift_trustee = [benchmark('trustee', 'gift') for _ in range(games)]
		gift_investor = [benchmark('investor', 'gift') for _ in range(games)]
		attrition_trustee = [benchmark('trustee', 'attrition') for _ in range(games)]
		attrition_investor = [benchmark('investor', 'attrition') for _ in range(games)]
		dfs.append(train(learners, cooperate_trustee, 'investor', games))
		dfs.append(train(cooperate_investor, learners, 'trustee', games))
		# dfs.append(train(learners, defect_trustee, 'investor', games))
		# dfs.append(train(defect_investor, learners, 'trustee', games))
		# dfs.append(train(learners, gift_trustee, 'investor', games))
		# dfs.append(train(gift_investor, learners, 'trustee', games))
		# dfs.append(train(learners, attrition_trustee, 'investor', games))
		# dfs.append(train(attrition_investor, learners, 'trustee', games))
		data = pd.concat(dfs, ignore_index=True)
		data.to_pickle(f'agent_data/{agent}_N={N}_games={games}_benchmark.pkl')
	# plot_trajectories_generosities_baseline(data, agent)
	plot_final_generosities_baseline(data, agent)

def svo(agent, N=10, games=100, seed=0, load=False):
	learners = make_learners(agent, seed, N, randomize=True)
	if load:
		data = pd.read_pickle(f'agent_data/{agent}_N={N}_games={games}_svo.pkl')
	else:
		dfs = []
		greedy_trustee = [T4T("trustee", seed=n, minO=0.1, maxO=0.3, minX=0.5, maxX=0.5, minF=0.0, maxF=0.1, minP=0.2, maxP=0.2, ID='greedy') for n in range(games)]
		greedy_investor = [T4T("investor", seed=n, minO=0.8, maxO=1.0, minX=0.5, maxX=0.5, minF=1.0, maxF=1.0, minP=0.1, maxP=0.3, ID='greedy') for n in range(games)]
		generous_trustee = [T4T("trustee", seed=n, minO=0.3, maxO=0.5, minX=0.5, maxX=0.5, minF=0.4, maxF=0.6, minP=1.0, maxP=1.0, ID='generous') for n in range(games)]
		generous_investor = [T4T("investor", seed=n, minO=0.6, maxO=0.8, minX=0.5, maxX=0.5, minF=0.8, maxF=1.0, minP=1.0, maxP=1.0, ID='generous') for n in range(games)]
		dfs.append(train(learners, greedy_trustee, 'investor', games))
		dfs.append(train(greedy_investor, learners, 'trustee', games))
		dfs.append(train(learners, generous_trustee, 'investor', games))
		dfs.append(train(generous_investor, learners, 'trustee', games))
		data = pd.concat(dfs, ignore_index=True)
		data.to_pickle(f'agent_data/{agent}_N={N}_games={games}_svo.pkl')
	plot_final_generosities_svo(data, agent)
