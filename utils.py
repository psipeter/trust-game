import numpy as np
import random
import pandas as pd
from tensorflow import GradientTape
from scipy.stats import entropy, skew, kurtosis, normaltest

class Game():
	def __init__(self, coins=10, match=3, turns=5, tape=GradientTape()):
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
		self.tape = tape

def generosity(player, give, keep):
	return np.NaN if give+keep==0 and player=='trustee' else give/(give+keep)

def play_game(game, investor, trustee, phase):
	assert investor.player == 'investor' and trustee.player == 'trustee', \
		f"invalid player assignments {investor.player, trustee.player}"
	investor.new_game()
	trustee.new_game()
	with game.tape:
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
		if phase=='train':
			investor.learn(game)
			trustee.learn(game)

def process_data(raw, agents):

	metrics = (  			# analysis of each individual agent
		'agent',  			# agent ID 
		'player',  			# player 
		'mean',  		  	# mean gen/score on test set
		'std', 		  		# standard deviation of gen/score dist. on test set
		# 'entropy',  	  	# entropy of gen/score dist. on test game
		'skew',  		  	# skewness of gen/score dist., indicates deviation from normal dist on test set
		'kurtosis',  	  	# kurtosis of gen/score dist., indicates outliers in data (esp tails) on test set
		# 'normality',	  	# test whether gen/score dist. differs from a normal dist. on test set
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

	def distribution_distance(a,b,var):
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
		delta = distribution_distance(dist_initial, dist_final, 'generosity')
		return delta

	def get_speed(data, games, final_games, thr=0.98):
		dist_final = np.array(data.query('game>@final_games')['generosity'])
		similarities = []
		for g in range(games):
			dist_g = np.array(data.query('game==@g')['generosity'])
			similarities.append(1 - distribution_distance(dist_g, dist_final, 'generosity'))
		# find the game past which all generosity distributions resemble the final generosity distribution
		for g in range(len(similarities)-1):
			if np.all(np.array(similarities)[g:]>thr):
				return g/games  # normalize 0-1
		return 0  # if this criteria is never met, return zero

	def get_probabilities(raw, ID, player, turns, thr_investor=1, thr_trustee=0.5):
		data = raw.query('phase=="test"')
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
		data_train = raw.query('ID==@ID & player==@player & phase=="train"')
		data_test = raw.query('ID==@ID & player==@player & phase=="test"')
		mean = np.mean(data_test['generosity'])
		std = np.std(data_test['generosity'])
		ent = entropy(data_test['generosity'])
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
		data_train = raw.query('ID==@ID & player==@player & phase=="train"')
		data_test = raw.query('ID==@ID & player==@player & phase=="test"')
		mean = np.mean(data_test['coins'])
		std = np.std(data_test['coins'])
		ent = entropy(data_test['coins'])
		skw = skew(data_test['coins'])
		kurt = kurtosis(data_test['coins'])
		adapt = get_adaptivity(data_test, 'coins', turns)
		df = pd.DataFrame([[ID, player, mean, std, skw, kurt, adapt]], columns=metrics_score)
		dfs.append(df)
	data_metrics_score = pd.concat([df for df in dfs], ignore_index=True)
	metrics_score = ('agent', 'player', 'mean', 'std', 'skew', 'kurtosis', 'adapt')

	# print(data_metrics_gen)
	# print(data_metrics_score)

	return data_metrics_gen, data_metrics_score