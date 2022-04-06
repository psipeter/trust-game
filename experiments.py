import numpy as np
import random
import pandas as pd
from utils import *
from plots import *
from fixed_agents import *
from learning_agents import *

class Game():
	def __init__(self, coins=10, match=3, turns=5, train=False):
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
		if isinstance(investor, NQ2) or isinstance(trustee, NQ2) or isinstance(investor, NQ3) or isinstance(trustee, NQ3):  # extra turn for nengo learning
			i_give, i_keep = investor.move(game)
			t_give, t_keep = trustee.move(game)
		investor.learn(game)
		trustee.learn(game)

def game_loop(investor, trustee, learner, train, g, dfs):
	columns = ('ID', 'opponent_ID', 'player', 'train', 'game', 'turn', 'generosity', 'coins', 'friendliness', 'gamma')
	game = Game(train=train)
	play_game(game, investor, trustee)
	for t in range(game.turns):
		dfs.append(pd.DataFrame([[investor.ID, trustee.ID, 'investor', train, g, t,
			game.investor_gen[t], game.investor_reward[t], learner.friendliness, learner.gamma]], columns=columns))
		dfs.append(pd.DataFrame([[trustee.ID, investor.ID, 'trustee', train, g, t,
			game.trustee_gen[t], game.trustee_reward[t], learner.friendliness, learner.gamma]], columns=columns))
	return dfs

def train_and_test(investors, trustees, learner_plays, n_train, learner_name, opponent_name):
	learners = investors if learner_plays=='investor' else trustees
	n_learners = len(learners)
	dfs = []
	for learner in learners:
		print(f"{learner_name} {learner.ID} vs {opponent_name}")
		learner.reinitialize(player=learner_plays, ID=learner.ID, seed=learner.seed)
		for g in range(n_train):
			print(f"game {g}")
			if learner_plays=='investor':
				dfs = game_loop(learner, trustees[g], learner, train=True, g=g, dfs=dfs)
			elif learner_plays=='trustee':
				dfs = game_loop(investors[g], learner, learner, train=True, g=g, dfs=dfs)
	data = pd.concat([df for df in dfs], ignore_index=True)
	return data
	# if learner_plays=='investor':
	# 	metrics_gen, metrics_score = process_data(data, investors)
	# if learner_plays=='trustee':
	# 	metrics_gen, metrics_score = process_data(data, trustees)
	# plot_metrics(metrics_gen, metrics_score, learner_plays=learner_plays, learner_name=learner_name, opponent_name=opponent_name)

def make_learners(learner_type, seed, n_learners):
	if learner_type=="tabular-q-learning":
		learners = [TabularQLearning('investor', ID=n+seed, seed=n+seed) for n in range(n_learners)]
	if learner_type=="normalized-q-learning":
		learners = [NormalizedQLearning('investor', ID=n+seed, seed=n+seed) for n in range(n_learners)]
	if learner_type=="tabular-actor-critic":
		learners = [TabularActorCritic('investor', ID=n+seed, seed=n+seed) for n in range(n_learners)]
	if learner_type=="deep-q-learning":
		learners = [DeepQLearning('investor', ID=n+seed, seed=n+seed) for n in range(n_learners)]
	if learner_type=="deep-actor-critic":
		learners = [DeepActorCritic('investor', ID=n+seed, seed=n+seed) for n in range(n_learners)]
	if learner_type=="instance-based":
		learners = [InstanceBased('investor', ID=n+seed, seed=n+seed) for n in range(n_learners)]
	if learner_type=="nengo-q-learning":
		learners = [NengoQLearning('investor', ID=n+seed, seed=n+seed) for n in range(n_learners)]
	if learner_type=="NQ2":
		learners = [NQ2('investor', ID=n+seed, seed=n+seed) for n in range(n_learners)]
	if learner_type=="NQ3":
		learners = [NQ3('investor', ID=n+seed, seed=n+seed) for n in range(n_learners)]
	return learners

def test_adaptivity(learner_type, n_learners=10, n_train=1000, seed=0, load=False):
	learners = make_learners(learner_type, seed, n_learners)
	learner_name = learners[0].__class__.__name__
	if load:
		df = pd.read_pickle(f'agent_data/{learner_name}_N={n_learners}_adaptivity.pkl')
	else:
		dfs = []
		cooperate_trustee = [adaptive('trustee', 'cooperate') for _ in range(n_train)]
		cooperate_investor = [adaptive('investor', 'cooperate') for _ in range(n_train)]
		defect_trustee = [adaptive('trustee', 'defect') for _ in range(n_train)]
		defect_investor = [adaptive('investor', 'defect') for _ in range(n_train)]
		gift_trustee = [adaptive('trustee', 'gift') for _ in range(n_train)]
		gift_investor = [adaptive('investor', 'gift') for _ in range(n_train)]
		attrition_trustee = [adaptive('trustee', 'attrition') for _ in range(n_train)]
		attrition_investor = [adaptive('investor', 'attrition') for _ in range(n_train)]
		dfs.append(train_and_test(learners, cooperate_trustee, 'investor', n_train, learner_name, "LearnToCooperate_Investor"))
		dfs.append(train_and_test(cooperate_investor, learners, 'trustee', n_train, learner_name, "LearnToCooperate_Trustee"))
		# dfs.append(train_and_test(learners, defect_trustee, 'investor', n_train, learner_name, "LearnToDefect_Investor"))
		# dfs.append(train_and_test(defect_investor, learners, 'trustee', n_train, learner_name, "LearnToDefect_Trustee"))
		# dfs.append(train_and_test(learners, gift_trustee, 'investor', n_train, learner_name, "LearnToGift_Investor"))
		# dfs.append(train_and_test(gift_investor, learners, 'trustee', n_train, learner_name, "LearnToGift_Trustee"))
		# dfs.append(train_and_test(learners, attrition_trustee, 'investor', n_train, learner_name, "LearnToAttrition_Investor"))
		# dfs.append(train_and_test(attrition_investor, learners, 'trustee', n_train, learner_name, "LearnToAttrition_Trustee"))
		df = pd.concat(dfs, ignore_index=True)
		df.to_pickle(f'agent_data/{learner_name}_N={n_learners}_adaptivity.pkl')
	plot_learning_and_policy_agent_adaptivity(df, learner_type)
	# plot_learning_and_coins_agent_adaptivity(df, learner_type)

def test_t4tv(learner_type, n_learners=100, n_train=1000, seed=0, load=False):
	learners = make_learners(learner_type, seed, n_learners)
	learner_name = learners[0].__class__.__name__
	if load:
		df = pd.read_pickle(f'agent_data/{learner_name}_N={n_learners}_friendliness.pkl')
	else:
		dfs = []
		greedy_trustee = [t4tv("trustee", seed=n, minO=0.1, maxO=0.3, minX=0.5, maxX=0.5, minF=0.0, maxF=0.1, minP=0.2, maxP=0.2, ID='GreedyT4T') for n in range(n_train)]
		greedy_investor = [t4tv("investor", seed=n, minO=0.8, maxO=1.0, minX=0.5, maxX=0.5, minF=1.0, maxF=1.0, minP=0.1, maxP=0.3, ID='GreedyT4T') for n in range(n_train)]
		generous_trustee = [t4tv("trustee", seed=n, minO=0.3, maxO=0.5, minX=0.5, maxX=0.5, minF=0.4, maxF=0.6, minP=1.0, maxP=1.0, ID='GenerousT4T') for n in range(n_train)]
		generous_investor = [t4tv("investor", seed=n, minO=0.6, maxO=0.8, minX=0.5, maxX=0.5, minF=0.8, maxF=1.0, minP=1.0, maxP=1.0, ID='GenerousT4T') for n in range(n_train)]
		dfs.append(train_and_test(learners, greedy_trustee, 'investor', n_train, learner_name, "GreedyT4T"))
		dfs.append(train_and_test(greedy_investor, learners, 'trustee', n_train, learner_name, "GreedyT4T"))
		dfs.append(train_and_test(learners, generous_trustee, 'investor', n_train, learner_name, "GenerousT4T"))
		dfs.append(train_and_test(generous_investor, learners, 'trustee', n_train, learner_name, "GenerousT4T"))
		df = pd.concat(dfs, ignore_index=True)
		df.to_pickle(f'agent_data/{learner_name}_N={n_learners}_friendliness.pkl')
	plot_learning_and_policy_agent_friendliness(df, learner_type)
	# plot_learning_and_coins_agent_friendliness(df, learner_type)