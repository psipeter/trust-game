import numpy as np
import random
import pandas as pd
import torch
from scipy.stats import entropy, skew, kurtosis, normaltest

def process_data(raw, agents):

	metrics = (  			# analysis of each individual agent
		'agent',  			# agent ID 
		'player',  			# player 
		'mean',  		  	# mean gen/score on test set
		'std', 		  		# standard deviation of gen/score dist. on test set
		'skew',  		  	# skewness of gen/score dist., indicates deviation from normal dist on test set
		'kurtosis',  	  	# kurtosis of gen/score dist., indicates outliers in data (esp tails) on test set
		'learn',	  		# changes in gen/score between initial and final games on training set:
							# KL divergence(all initial moves, all final moves)
					 		# averages over turn-to-turn variance
		'speed',  			# number of iterations before learning is complete on training set
							# find earliest iteration when gen/score dist. is indistinguishable from final gen/score dist. 
		'adapt',  			# changes in gen/score between consecutive turns on test set,
							# mean of differences betwee gen/score now and gen/score next turn
					   		# ideal conditions: no game-to-game variance
					   		# changes will reflect turn-to-turn progression and responses to opponent behavior
		'cooperate',  		# probability of reciprocating a generous move, averaged across turns and games on test set
							# cooperation defined as the learner moving with gen above threshold after opponent moves with gen above threshold
		'defect',  			# probability of defecting on opponent, averaged across turns and games on test set
							# defect defined as learner moving with gen below threshold after opponent moves with gen above threshold
		'gift',  			# probability of offering a gift to the opponent, averaged across turns and games on test set
							# gifting defined as the learner moving with gen above threshold after opponent moves with gen below threshold 
		'attrition',  		# probability of reciprocating a greedy move, averaged across turns and games on test set
							# attrition defined as learner moving with gen below threshold after opponent moves with gen below threshold
		)

	metrics_score = ('agent', 'player', 'mean', 'std', 'skew', 'kurtosis', 'adapt')

	def chi_squared_distance(a,b,var):
		# turn arrays of generosities / scores into histograms
		bins = np.arange(0, 1.1, 0.1) if var=='generosity' else np.arange(0, 31, 1)
		A = np.histogram(a, bins=bins)[0] / len(a)
		B = np.histogram(b, bins=bins)[0] / len(b)
		if np.sum([A,B])==0:
			return 0
		else:
			return 0.5*np.sum(np.square(A-B)/np.sum([A,B]))

	def get_adaptivity(data, var, turns):
		deltas = []
		for g in np.array(data['game'].unique()):
			for t in range(turns-1):
				t_next=t+1
				var0 = data.query('game==@g & turn==@t')[var].to_numpy()[0]
				var1 = data.query('game==@g & turn==@t_next')[var].to_numpy()[0]
				deltas.append(np.abs(var1-var0))
		return np.mean(deltas)

	def get_learning(data, initial_games, final_games):
		dist_initial = np.array(data.query('game<@initial_games')['generosity'])
		dist_final = np.array(data.query('game>=@final_games')['generosity'])
		delta = chi_squared_distance(dist_initial, dist_final, 'generosity')
		return delta

	def get_speed(data, games, final_games, thr=0.98):
		dist_final = np.array(data.query('game>@final_games')['generosity'])
		similarities = []
		for g in range(games):
			dist_g = np.array(data.query('game==@g')['generosity'])
			similarities.append(1 - chi_squared_distance(dist_g, dist_final, 'generosity'))
		# find the game past which all generosity distributions resemble the final generosity distribution
		for g in range(len(similarities)-1):
			if np.all(np.array(similarities)[g:]>thr):
				return g/games  # normalize 0-1
		return 0  # if this criteria is never met, return zero

	def get_probabilities(raw, ID, player, turns, thr_investor=1, thr_trustee=0.5):
		data = raw.query('train==False')
		cooperates = 0
		defects = 0
		gifts = 0
		attritions = 0
		for g in np.array(data['game'].unique()):
			game = data.query('game==@g')
			for t in range(turns-1):
				if player=='investor':
					opponent_gen = game.query('turn==@t & player=="trustee" & opponent_ID==@ID')['generosity'].to_numpy()[0]
					learner_gen = game.query('turn==@t & player=="investor" & ID==@ID')['generosity'].to_numpy()[0]
					if np.isnan(opponent_gen) or np.isnan(learner_gen): continue
					elif opponent_gen>=thr_trustee and learner_gen>=thr_investor: cooperates += 1
					elif opponent_gen>=thr_trustee and learner_gen<thr_investor: defects += 1
					elif opponent_gen<thr_trustee and learner_gen>=thr_investor: gifts += 1
					elif opponent_gen<thr_trustee and learner_gen<thr_investor: attritions += 1
				else:
					opponent_gen = game.query('turn==@t & player=="investor" & opponent_ID==@ID')['generosity'].to_numpy()[0]
					learner_gen = game.query('turn==@t & player=="trustee" & ID==@ID')['generosity'].to_numpy()[0]
					if np.isnan(opponent_gen) or np.isnan(learner_gen): continue
					elif opponent_gen>=thr_investor and learner_gen>=thr_trustee: cooperates += 1
					elif opponent_gen>=thr_investor and learner_gen<thr_trustee: defects += 1
					elif opponent_gen<thr_investor and learner_gen>=thr_trustee: gifts += 1
					elif opponent_gen<thr_investor and learner_gen<thr_trustee: attritions += 1
		p_cooperate = cooperates / (cooperates+defects) if cooperates+defects>0 else None
		p_defect = defects / (cooperates+defects) if cooperates+defects>0 else None
		p_gift = gifts / (gifts+attritions) if gifts+attritions>0 else None
		p_attrition = attritions / (gifts+attritions) if gifts+attritions>0 else None
		return p_cooperate, p_defect, p_gift, p_attrition

	turns = np.max(raw['turn'])+1
	games = np.max(raw['game'])+1
	initial_games = int(0.2*games)
	final_games = int(0.8*games)

	dfs = []
	for agent in agents:
		ID = agent.ID
		player = agent.player
		data_train = raw.query('ID==@ID & player==@player & train==True')
		data_test = raw.query('ID==@ID & player==@player & train==False')
		mean = np.mean(data_test['generosity'])
		std = np.std(data_test['generosity'])
		skw = skew(data_test['generosity'])
		kurt = kurtosis(data_test['generosity'])
		learn = get_learning(data_train, initial_games, final_games)
		speed = get_speed(data_train, games, final_games)
		adapt = get_adaptivity(data_test, 'generosity', turns)
		cooperate, defect, gift, attrition = get_probabilities(raw, ID, player, turns)
		df = pd.DataFrame([[ID, player, mean, std, skw, kurt, learn, speed, adapt, cooperate, defect, gift, attrition]], columns=metrics)
		dfs.append(df)
	data_metrics_gen = pd.concat([df for df in dfs], ignore_index=True)

	dfs = []
	for agent in agents:
		ID = agent.ID
		player = agent.player
		data_train = raw.query('ID==@ID & player==@player & train==True')
		data_test = raw.query('ID==@ID & player==@player & train==False')
		mean = np.mean(data_test['coins'])
		std = np.std(data_test['coins'])
		skw = skew(data_test['coins'])
		kurt = kurtosis(data_test['coins'])
		adapt = get_adaptivity(data_test, 'coins', turns)
		df = pd.DataFrame([[ID, player, mean, std, skw, kurt, adapt]], columns=metrics_score)
		dfs.append(df)
	data_metrics_score = pd.concat([df for df in dfs], ignore_index=True)
	metrics_score = ('agent', 'player', 'mean', 'std', 'skew', 'kurtosis', 'adapt')

	return data_metrics_gen, data_metrics_score


def get_n_inputs(representation, player, n_actions, turns=5, coins=10, match=3, extra_turn=0):
	turns += extra_turn
	if representation=='turn':return turns
	if representation=='turn-coin':
		if player=='investor': return turns
		elif player=='trustee': return turns * (coins * match + 1)
	if representation=='turn-gen-opponent': return turns * (n_actions+1)
	if representation=='turn-gen-both': return turns * (n_actions+1)**2

def get_state(player, representation, game, return_type, n_actions=0, dim=0):
	index = 0
	t = len(game.investor_give) if player=='investor' else len(game.trustee_give)
	if representation == 'turn':
		index = t
	if representation == "turn-coin":
		if player=='investor': index = t
		elif player=='trustee': index = t * (game.coins*game.match+1) + game.investor_give[-1]*game.match
	if representation == "turn-gen-opponent":
		index = (n_actions+1) * t
		if player == 'investor':
			if t==0: opponent_gen = 0
			elif np.isnan(game.trustee_gen[-1]): opponent_gen = -1
			else: opponent_gen = game.trustee_gen[-1]
		elif player == 'trustee':
			opponent_gen = game.investor_gen[-1]
		gen_bins = np.linspace(0, 1, n_actions)
		for i in range(len(gen_bins)-1):
			if gen_bins[i] < opponent_gen <= gen_bins[i+1]:
				index += i
		if opponent_gen == -1:
			index += n_actions
	if representation == "turn-gen-both":
		index = (n_actions+1)**2 * t
		if player == 'investor':
			if t==0: my_gen = 0
			else: my_gen = game.investor_gen[-1]
			if t==0: opponent_gen = 0
			elif np.isnan(game.trustee_gen[-1]): opponent_gen = -1
			else: opponent_gen = game.trustee_gen[-1]
		elif player == 'trustee':
			opponent_gen = game.investor_gen[-1]
			if t==0: my_gen = 0
			elif np.isnan(game.trustee_gen[-1]): my_gen = -1
			else: my_gen = game.trustee_gen[-1]
		gen_bins = np.linspace(0, 1, n_actions)
		for i in range(len(gen_bins)-1):
			if gen_bins[i] < my_gen <= gen_bins[i+1]:
				index += n_actions * i
			if gen_bins[i] < opponent_gen <= gen_bins[i+1]:
				index += i
		if my_gen == -1:
			index += (n_actions+1)**2
		if opponent_gen == -1:
			index += n_actions+1
	if return_type=='index':
		return index
	if return_type=='one-hot':
		vector = np.zeros((dim))
		vector[index] = 1
		return vector
	if return_type=='tensor':
		vector = np.zeros((dim))
		vector[index] = 1
		return torch.FloatTensor(vector)

def action_to_coins(player, state, n_actions, game):
	available = game.coins if player=='investor' else game.investor_give[-1]*game.match  # coins available
	precise_give = state * available
	possible_actions = np.linspace(0, available, n_actions).astype(int)
	action_idx = (np.abs(possible_actions - precise_give)).argmin()
	action = possible_actions[action_idx]
	give = action
	keep = available - action
	return give, keep, action_idx

def generosity(player, give, keep):
	return np.NaN if give+keep==0 and player=='trustee' else give/(give+keep)