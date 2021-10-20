import numpy as np
import random
import pandas as pd
from utils import *
from plots import *
from fixed_agents import *
from learning_agents import *

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
					actor_loss, critic_loss = trustee.actor_losses[g], trustee.critic_losses[g]
					dfs_loss.append(pd.DataFrame([[trustee.ID, investor.ID, 'trustee', g, actor_loss, critic_loss]], columns=columns_loss))
				for g in range(n_test):
					dfs = game_loop(testers[0], trustee, 'test', g=g, dfs=dfs)
		data = pd.concat([df for df in dfs], ignore_index=True)

		dfs_loss = []
		columns_loss = ('ID', 'opponent_ID', 'player', 'game', 'actor_loss', 'critic_loss')
		if learner_plays=='investor':
			for i in range(len(investor.actor_losses)):
				g = investor.actor_losses[i][0]
				actor_loss = investor.actor_losses[i][1]
				critic_loss = investor.critic_losses[i][1]
				dfs_loss.append(pd.DataFrame([[investor.ID, trustee.ID, 'investor', g, actor_loss, critic_loss]], columns=columns_loss))
		if learner_plays=='trustee':
			for i in range(len(trustee.actor_losses)):
				g = trustee.actor_losses[i][0]
				actor_loss = trustee.actor_losses[i][1]
				critic_loss = trustee.critic_losses[i][1]
				dfs_loss.append(pd.DataFrame([[trustee.ID, investor.ID, 'trustee', g, actor_loss, critic_loss]], columns=columns_loss))
		data_loss = pd.concat([df for df in dfs_loss], ignore_index=True)

		plot_learning(data, data_loss, learner_plays=learner_plays, name='all')
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

def test_adaptivity(n_learners=100, learning_rate=1e-3, n_train=1000, n_test=100, n_inputs=3, seed=0):

	def train_and_test(investors, trustees, learner_plays, n_train, n_test, name):
		dfs = []
		if learner_plays=='investor':
			for investor in investors:
				for g in range(n_train):
					dfs = game_loop(investor, trustees[0], 'train', g=g, dfs=dfs)
				for g in range(n_test):
					dfs = game_loop(investor, trustees[0], 'test', g=g, dfs=dfs)
		if learner_plays=='trustee':
			for trustee in trustees:
				for g in range(n_train):
					dfs = game_loop(investors[0], trustee, 'train', g=g, dfs=dfs)
				for g in range(n_test):
					dfs = game_loop(investors[0], trustee, 'test', g=g, dfs=dfs)
		data = pd.concat([df for df in dfs], ignore_index=True)
		plot_learning(data, learner_plays=learner_plays, name=name)
		plot_policy(data, learner_plays=learner_plays, name=name)
		if learner_plays=='investor':
			metrics_gen, metrics_score = process_data(data, investors)
		if learner_plays=='trustee':
			metrics_gen, metrics_score = process_data(data, trustees)
		plot_metrics(metrics_gen, metrics_score, learner_plays=learner_plays, name=name)

	'''test turn-to-turn adaptivity, investor'''
	# investors = [ActorCritic('investor', ID=n, n_inputs=n_inputs, learning_rate=learning_rate) for n in range(n_learners)]
	investors = [SoftActorCritic('investor', ID=n, n_inputs=n_inputs, learning_rate=learning_rate) for n in range(n_learners)]
	trustees = [adaptive('trustee', 'turn_based')]
	train_and_test(investors, trustees, 'investor', n_train, n_test, "adapt")

	'''test turn-to-turn adaptivity, trustee'''
	# trustees = [ActorCritic('trustee', ID=n, n_inputs=n_inputs, learning_rate=learning_rate, n_actions=31) for n in range(n_learners)]
	trustees = [SoftActorCritic('trustee', ID=n, n_inputs=n_inputs, learning_rate=learning_rate, n_actions=31) for n in range(n_learners)]
	investors = [adaptive('investor', 'turn_based')]
	train_and_test(investors, trustees, 'trustee', n_train, n_test, "adapt")

	'''test cooperation, investor'''
	# investors = [ActorCritic('investor', ID=n, n_inputs=n_inputs, learning_rate=learning_rate) for n in range(n_learners)]
	investors = [SoftActorCritic('investor', ID=n, n_inputs=n_inputs, learning_rate=learning_rate) for n in range(n_learners)]
	trustees = [adaptive('trustee', 'cooperate')]
	train_and_test(investors, trustees, 'investor', n_train, n_test, "cooperate")

	'''test cooperation, trustee'''
	# trustees = [ActorCritic('trustee', ID=n, n_inputs=n_inputs, learning_rate=learning_rate, n_actions=31) for n in range(n_learners)]
	trustees = [SoftActorCritic('trustee', ID=n, n_inputs=n_inputs, learning_rate=learning_rate, n_actions=31) for n in range(n_learners)]
	investors = [adaptive('investor', 'cooperate')]
	train_and_test(investors, trustees, 'trustee', n_train, n_test, "cooperate")

	'''test defection, investor'''
	# investors = [ActorCritic('investor', ID=n, n_inputs=n_inputs, learning_rate=learning_rate) for n in range(n_learners)]
	investors = [SoftActorCritic('investor', ID=n, n_inputs=n_inputs, learning_rate=learning_rate) for n in range(n_learners)]
	trustees = [adaptive('trustee', 'defect')]
	train_and_test(investors, trustees, 'investor', n_train, n_test, "defect")

	'''test defection, trustee'''
	# trustees = [ActorCritic('trustee', ID=n, n_inputs=n_inputs, learning_rate=learning_rate, n_actions=31) for n in range(n_learners)]
	trustees = [SoftActorCritic('trustee', ID=n, n_inputs=n_inputs, learning_rate=learning_rate, n_actions=31) for n in range(n_learners)]
	investors = [adaptive('investor', 'defect')]
	train_and_test(investors, trustees, 'trustee', n_train, n_test, "defect")

	'''test gifting, investor'''
	# investors = [ActorCritic('investor', ID=n, n_inputs=n_inputs, learning_rate=learning_rate) for n in range(n_learners)]
	investors = [SoftActorCritic('investor', ID=n, n_inputs=n_inputs, learning_rate=learning_rate) for n in range(n_learners)]
	trustees = [adaptive('trustee', 'gift')]
	train_and_test(investors, trustees, 'investor', n_train, n_test, "gift")

	'''test gifting, trustee'''
	# trustees = [ActorCritic('trustee', ID=n, n_inputs=n_inputs, learning_rate=learning_rate, n_actions=31) for n in range(n_learners)]
	trustees = [SoftActorCritic('trustee', ID=n, n_inputs=n_inputs, learning_rate=learning_rate, n_actions=31) for n in range(n_learners)]
	investors = [adaptive('investor', 'gift')]
	train_and_test(investors, trustees, 'trustee', n_train, n_test, "gift")

	'''test attrition, investor'''
	# investors = [ActorCritic('investor', ID=n, n_inputs=n_inputs, learning_rate=learning_rate) for n in range(n_learners)]
	investors = [SoftActorCritic('investor', ID=n, n_inputs=n_inputs, learning_rate=learning_rate) for n in range(n_learners)]
	trustees = [adaptive('trustee', 'attrition')]
	train_and_test(investors, trustees, 'investor', n_train, n_test, "attrition")

	'''test attrition, trustee'''
	# trustees = [ActorCritic('trustee', ID=n, n_inputs=n_inputs, learning_rate=learning_rate, n_actions=31) for n in range(n_learners)]
	trustees = [SoftActorCritic('trustee', ID=n, n_inputs=n_inputs, learning_rate=learning_rate, n_actions=31) for n in range(n_learners)]
	investors = [adaptive('investor', 'attrition')]
	train_and_test(investors, trustees, 'trustee', n_train, n_test, "attrition")