import numpy as np
import random
import pandas as pd
from utils import *
from plots import *

def game_loop(investor, trustee, phase, g, dfs):
	columns = ('ID', 'opponent_ID', 'player', 'phase', 'game', 'turn', 'generosity', 'coins')
	game = Game()
	play_game(game, investor, trustee, phase)
	for t in range(game.turns):
		dfs.append(pd.DataFrame([[investor.ID, trustee.ID, 'investor', phase,
			g, t, game.investor_gen[t], game.investor_reward[t]]], columns=columns))			
		dfs.append(pd.DataFrame([[trustee.ID, investor.ID, 'trustee', phase,
			g, t, game.trustee_gen[t], game.trustee_reward[t]]], columns=columns))			
	return dfs

def play_tournament(investors, trustees, testers, tournament_type, learner_plays, n_train=100, n_test=100, seed=0):
	for investor in investors: assert investor.player == 'investor', "invalid investor assignment"
	for trustee in trustees: assert trustee.player == 'trustee', "invalid trustee assignment"
	rng = np.random.RandomState(seed=seed)

	if tournament_type == 'one_game':
		dfs = game_loop(investors[0], trustees[0], 'train', g=0, dfs=[])			
		data = pd.concat([df for df in dfs], ignore_index=True)
		plot_one_game(data)

	if tournament_type == 'many_games':
		dfs = []
		for g in range(n_train):
			dfs = game_loop(investors[0], trustees[0], 'train', g=g, dfs=dfs)			
		data = pd.concat([df for df in dfs], ignore_index=True)
		plot_many_games(data, learner_plays=learner_plays)

	if tournament_type == 'many_learners_one_opponent':
		dfs = []
		if learner_plays=='investor':
			for investor in investors:
				for g in range(n_train):
					dfs = game_loop(investor, trustees[0], 'train', g=g, dfs=dfs)
				for g in range(n_test):
					dfs = game_loop(investor, testers[0], 'test', g=g, dfs=dfs)
		if learner_plays=='trustee':
			for trustee in trustees:
				for g in range(n_train):
					dfs = game_loop(investors[0], trustee, 'train', g=g, dfs=dfs)
				for g in range(n_test):
					dfs = game_loop(testers[0], trustee, 'test', g=g, dfs=dfs)
		data = pd.concat([df for df in dfs], ignore_index=True)
		plot_learning(data, learner_plays=learner_plays, name='all')
		plot_policy(data, learner_plays=learner_plays, name='all')
		if learner_plays=='investor':
			# for investor in investors:
		# 		plot_learning(data, learner_plays=learner_plays, name=investor.ID)
		# 		plot_policy(data, learner_plays=learner_plays, name=investor.ID)
			metrics_gen, metrics_score = process_data(data, investors)
		if learner_plays=='trustee':
		# 	for trustee in trustees:
		# 		plot_learning(data, learner_plays=learner_plays, name=trustee.ID)
		# 		plot_policy(data, learner_plays=learner_plays, name=trustee.ID)
			metrics_gen, metrics_score = process_data(data, trustees)
		plot_metrics(metrics_gen, metrics_score, learner_plays=learner_plays)

	if tournament_type == 'many_learners_many_opponents':
		dfs = []
		if learner_plays=='investor':
			for investor in investors:
				for g in range(n_train):
					trustee = rng.choice(trustees)
					dfs = game_loop(investor, trustee, 'train', g=g, dfs=dfs)
		if learner_plays=='trustee':
			for trustee in trustees:
				for g in range(n_train):
					investor = rng.choice(investors)
					dfs = game_loop(investor, trustee, 'train', g=g, dfs=dfs)
		data = pd.concat([df for df in dfs], ignore_index=True)
		plot_learning(data, learner_plays=learner_plays, name='all')
		plot_policy(data, learner_plays=learner_plays, name='all')
