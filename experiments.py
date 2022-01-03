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
		investor.learn(game)
		trustee.learn(game)

def game_loop(investor, trustee, train, g, dfs):
	columns = ('ID', 'opponent_ID', 'player', 'train', 'game', 'turn', 'generosity', 'coins')
	print(f"game {g}")
	game = Game(train=train)
	play_game(game, investor, trustee)
	for t in range(game.turns):
		dfs.append(pd.DataFrame([[investor.ID, trustee.ID, 'investor', train,
			g, t, game.investor_gen[t], game.investor_reward[t]]], columns=columns))
		dfs.append(pd.DataFrame([[trustee.ID, investor.ID, 'trustee', train,
			g, t, game.trustee_gen[t], game.trustee_reward[t]]], columns=columns))
	return dfs

# def play_tournament(investors, trustees, tournament_type, learner_plays, n_train=100, n_test=100, seed=0):
# 	for investor in investors: assert investor.player == 'investor', "invalid investor assignment"
# 	for trustee in trustees: assert trustee.player == 'trustee', "invalid trustee assignment"
# 	rng = np.random.RandomState(seed=seed)

# 	if tournament_type == 'one_game':
# 		dfs = game_loop(investors[0], trustees[0], train=True, g=0, dfs=[])			
# 		data = pd.concat([df for df in dfs], ignore_index=True)
# 		plot_one_game(data)

# 	if tournament_type == 'many_games':
# 		dfs = []
# 		for g in range(n_train):
# 			dfs = game_loop(investors[0], trustees[0], train=True, g=g, dfs=dfs)			
# 		data = pd.concat([df for df in dfs], ignore_index=True)
# 		plot_many_games(data, learner_plays=learner_plays, name='many_games')

# 	if tournament_type == 'many_learners_one_opponent':
# 		dfs = []
# 		if learner_plays=='investor':
# 			for investor in investors:
# 				for g in range(n_train):
# 					dfs = game_loop(investor, trustees[0], train=True, g=g, dfs=dfs)
# 				for g in range(n_test):
# 					dfs = game_loop(investor, trustees[0], train=False, g=g, dfs=dfs)
# 		if learner_plays=='trustee':
# 			for trustee in trustees:
# 				for g in range(n_train):
# 					dfs = game_loop(investors[0], trustee, train=True, g=g, dfs=dfs)
# 				for g in range(n_test):
# 					dfs = game_loop(investors[0], trustee, train=False, g=g, dfs=dfs)
# 		data = pd.concat([df for df in dfs], ignore_index=True)
# 		plot_learning(data, learner_plays=learner_plays, name='all')
# 		plot_policy(data, learner_plays=learner_plays, name='all')
# 		if learner_plays=='investor':
# 			metrics_gen, metrics_score = process_data(data, investors)
# 		if learner_plays=='trustee':
# 			metrics_gen, metrics_score = process_data(data, trustees)
# 		plot_metrics(metrics_gen, metrics_score, learner_plays=learner_plays)

# 	if tournament_type == 'many_learners_many_opponents':
# 		dfs = []
# 		if learner_plays=='investor':
# 			for investor in investors:
# 				for g in range(n_train):
# 					trustee = rng.choice(trustees)
# 					dfs = game_loop(investor, trustee, train=True, g=g, dfs=dfs)
# 		if learner_plays=='trustee':
# 			for trustee in trustees:
# 				for g in range(n_train):
# 					investor = rng.choice(investors)
# 					dfs = game_loop(investor, trustee, train=True, g=g, dfs=dfs)
# 		data = pd.concat([df for df in dfs], ignore_index=True)
# 		plot_learning(data, learner_plays=learner_plays, name='all')
# 		plot_policy(data, learner_plays=learner_plays, name='all')

def train_and_test(investors, trustees, learner_plays, n_train, n_test, learner_name, opponent_name):
	learners = investors if learner_plays=='investor' else trustees
	n_learners = len(learners)
	# data = pd.read_pickle(f'agent_data/{learner_name}_as_{learner_plays}_versus_{opponent_name}_N={n_learners}.pkl')
	# plot_learning_and_policy_agent_friendliness(data, learners, learner_plays, learner_name, opponent_name)
	dfs = []
	for learner in learners:
		learner.reinitialize(player=learner_plays, ID=learner.ID, seed=learner.seed)
		for g in range(n_train):
			if learner_plays=='investor':
				dfs = game_loop(learner, trustees[g], train=True, g=g, dfs=dfs)
			elif learner_plays=='trustee':
				dfs = game_loop(investors[g], learner, train=True, g=g, dfs=dfs)
	data = pd.concat([df for df in dfs], ignore_index=True)
	# plot_learning(data, learner_plays=learner_plays, learner_name=learner_name, opponent_name=opponent_name)
	# plot_policy(data, learner_plays=learner_plays, learner_name=learner_name, opponent_name=opponent_name)
	# plot_learning_and_policy(data, learner_plays, learner_name, opponent_name)
	plot_learning_and_policy_agent_friendliness(data, learners, learner_plays, learner_name, opponent_name)
	data.to_pickle(f'agent_data/{learner_name}_as_{learner_plays}_versus_{opponent_name}_N={n_learners}.pkl')
	# if learner_plays=='investor':
	# 	metrics_gen, metrics_score = process_data(data, investors)
	# if learner_plays=='trustee':
	# 	metrics_gen, metrics_score = process_data(data, trustees)
	# plot_metrics(metrics_gen, metrics_score, learner_plays=learner_plays, learner_name=learner_name, opponent_name=opponent_name)

def make_learners(learner_type, seed, n_learners):
	if learner_type=="tabular-q-learning":
		learners = [TabularQLearning('investor', ID=n+seed, seed=n+seed) for n in range(n_learners)]
	if learner_type=="tabular-actor-critic":
		learners = [TabularActorCritic('investor', ID=n+seed, seed=n+seed) for n in range(n_learners)]
	if learner_type=="tabular-model-based":
		learners = [TabularModelBased('investor', ID=n+seed, seed=n+seed) for n in range(n_learners)]
	if learner_type=="deep-q-learning":
		learners = [DeepQLearning('investor', ID=n+seed, seed=n+seed) for n in range(n_learners)]
	if learner_type=="deep-actor-critic":
		learners = [DeepActorCritic('investor', ID=n+seed, seed=n+seed) for n in range(n_learners)]
	if learner_type=="instance-based":
		learners = [InstanceBased('investor', ID=n+seed, seed=n+seed) for n in range(n_learners)]
	if learner_type=="nengo-q-learning":
		learners = [NengoQLearning('investor', ID=n+seed, seed=n+seed) for n in range(n_learners)]
	if learner_type=="nengo-actor-critic":
		learners = [NengoActorCritic('investor', ID=n+seed, seed=n+seed) for n in range(n_learners)]
	return learners

def test_adaptivity(learner_type, n_learners=10, n_train=1000, n_test=100, seed=0):
	learners = make_learners(learner_type, seed, n_learners)
	learner_name = learners[0].__class__.__name__
	cooperate_trustee = [adaptive('trustee', 'cooperate') for _ in range(n_train)]
	cooperate_investor = [adaptive('investor', 'cooperate') for _ in range(n_train)]
	defect_trustee = [adaptive('trustee', 'defect') for _ in range(n_train)]
	defect_investor = [adaptive('investor', 'defect') for _ in range(n_train)]
	gift_trustee = [adaptive('trustee', 'gift') for _ in range(n_train)]
	gift_investor = [adaptive('investor', 'gift') for _ in range(n_train)]
	attrition_trustee = [adaptive('trustee', 'attrition') for _ in range(n_train)]
	attrition_investor = [adaptive('investor', 'attrition') for _ in range(n_train)]
	train_and_test(learners, cooperate_trustee, 'investor', n_train, n_test, learner_name, "LearnToCooperate")
	train_and_test(cooperate_investor, learners, 'trustee', n_train, n_test, learner_name, "LearnToCooperate")
	train_and_test(learners, defect_trustee, 'investor', n_train, n_test, learner_name, "LearnToDefect")
	train_and_test(defect_investor, learners, 'trustee', n_train, n_test, learner_name, "LearnToDefect")
	train_and_test(learners, gift_trustee, 'investor', n_train, n_test, learner_name, "LearnToGift")
	train_and_test(gift_investor, learners, 'trustee', n_train, n_test, learner_name, "LearnToGift")
	train_and_test(learners, attrition_trustee, 'investor', n_train, n_test, learner_name, "LearnToAttrition")
	train_and_test(attrition_investor, learners, 'trustee', n_train, n_test, learner_name, "LearnToAttrition")

def test_t4tv(learner_type, n_learners=100, n_train=1000, n_test=100, seed=0):
	learners = make_learners(learner_type, seed, n_learners)
	learner_name = learners[0].__class__.__name__
	greedy_trustee = [t4tv("trustee", seed=n, minO=0.1, maxO=0.3, minX=0.5, maxX=0.5, minF=0.0, maxF=0.1, minP=0.2, maxP=0.2) for n in range(n_train)]
	greedy_investor = [t4tv("investor", seed=n, minO=0.8, maxO=1.0, minX=0.5, maxX=0.5, minF=1.0, maxF=1.0, minP=0.1, maxP=0.3) for n in range(n_train)]
	generous_trustee = [t4tv("trustee", seed=n, minO=0.3, maxO=0.5, minX=0.5, maxX=0.5, minF=0.4, maxF=0.6, minP=1.0, maxP=1.0) for n in range(n_train)]
	generous_investor = [t4tv("investor", seed=n, minO=0.6, maxO=0.8, minX=0.5, maxX=0.5, minF=0.8, maxF=1.0, minP=1.0, maxP=1.0) for n in range(n_train)]
	train_and_test(learners, greedy_trustee, 'investor', n_train, n_test, learner_name, "GreedyT4T")
	train_and_test(greedy_investor, learners, 'trustee', n_train, n_test, learner_name, "GreedyT4T")
	train_and_test(learners, generous_trustee, 'investor', n_train, n_test, learner_name, "GenerousT4T")
	train_and_test(generous_investor, learners, 'trustee', n_train, n_test, learner_name, "GenerousT4T")
