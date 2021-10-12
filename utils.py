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
		'mean',  		  	# mean gen/score on test games
		'std', 		  		# standard deviation of gen/score dist. on test games
		# 'entropy',  	  	# entropy of gen/score dist. on test game
		'skew',  		  	# skewness of gen/score dist., indicates deviation from normal dist on test games
		'kurtosis',  	  	# kurtosis of gen/score dist., indicates outliers in data (esp tails) on test games
		# 'normality',	  	# test whether gen/score dist. differs from a normal dist. on test games
		'adapt',  			# changes in gen/score between consecutive turns on test games,
							# mean of KL divergence(1,2),(2,3)(3,4)(4,5)
					   		# ideal conditions: no game-to-game variance
					   		# changes will reflect turn-to-turn progression and responses to opponent behavior
		'learn',	  		# changes in gen/score between initial and final games during training:
							# KL divergence(all initial moves, all final moves)
					 		# averages over turn-to-turn variance
		'speed',  			# number of iterations before learning is complete during training
							# find earliest iteration when gen/score dist. is indistinguishable from final gen/score dist. 
		'punish', 			# change in gen following a pair of turns when an opponent became more greedy in test games
							# find turns when opponent became more greedy, find learner's delta gen, average across all pairs
		'forgive', 			# change in gen following a pair of turns when an opponent became more generous in test games
							# find turns when opponent became more generous, find learner's delta gen, average across all pairs
		'gift',				# probability of increasing generosity following an opponent's greedy or NaN move in test games
		'defect',			# probability of decreasing generosity following an opponent's generous move in test games
		# 'trustee_end'		# mean generosity on the last turn when playing the trustee
		)

	metrics_score = ('agent', 'player', 'mean', 'std', 'skew', 'kurtosis', 'adapt')

	def get_variance(arr):
		if np.mean(arr)==0:
			return 0
		else:
			return np.max([np.abs(item-np.mean(arr))/np.mean(arr) for item in arr])

	def distribution_distance(a,b,var):
		# turn arrays of generosities / scores into histograms
		bins = np.arange(0, 1.1, 0.1) if var=='generosity' else np.arange(0, 31, 1)
		A = np.histogram(a, bins=bins)[0] / len(a)
		B = np.histogram(b, bins=bins)[0] / len(b)
		if np.sum([A,B])==0:
			return 0
		else:
			return 0.5*np.sum(np.square(A-B)/np.sum([A,B]))

	def get_adaptivity(data, var):
		dist0 = np.array(data.query('turn==0')[var])
		dist1 = np.array(data.query('turn==1')[var])
		dist2 = np.array(data.query('turn==2')[var])
		dist3 = np.array(data.query('turn==3')[var])
		dist4 = np.array(data.query('turn==4')[var])
		delta01 = distribution_distance(dist0, dist1, var)
		delta12 = distribution_distance(dist1, dist2, var)
		delta23 = distribution_distance(dist2, dist3, var)
		delta34 = distribution_distance(dist3, dist4, var)
		deltas = [delta01, delta12, delta23, delta34]
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
				return g
		return 0  # if this criteria is never met, return zero

	def get_punishment(raw, ID, player, turns):
		data = raw.query('phase=="test"')
		deltas = []
		for g in np.array(data['game'].unique()):
			game = data.query('game==@g')
			for t in range(turns-2):
				t_next = t+1
				t_next2 = t+2
				if player=='investor':
					opponent_gen_now = game.query('turn==@t & player=="trustee" & opponent_ID==@ID')['generosity'].to_numpy()[0]
					opponent_gen_next = game.query('turn==@t_next & player=="trustee" & opponent_ID==@ID')['generosity'].to_numpy()[0]
					if np.isnan(opponent_gen_now) or np.isnan(opponent_gen_next): continue
					if opponent_gen_next<opponent_gen_now:
						my_gen_before = game.query('turn==@t_next & player=="investor" & ID==@ID')['generosity'].to_numpy()[0]
						my_gen_after = game.query('turn==@t_next2 & player=="investor" & ID==@ID')['generosity'].to_numpy()[0]
						deltas.append(my_gen_after-my_gen_before)
				else:
					opponent_gen_now = game.query('turn==@t & player=="investor" & opponent_ID==@ID')['generosity'].to_numpy()[0]
					opponent_gen_next = game.query('turn==@t_next & player=="investor" & opponent_ID==@ID')['generosity'].to_numpy()[0]
					if opponent_gen_next<opponent_gen_now:
						my_gen_before = game.query('turn==@t & player=="trustee" & ID==@ID')['generosity'].to_numpy()[0]
						my_gen_after = game.query('turn==@t_next & player=="trustee" & ID==@ID')['generosity'].to_numpy()[0]
						deltas.append(my_gen_after-my_gen_before)
		# print('punishment', deltas)
		return np.mean(deltas) if len(deltas)>0 else None

	def get_forgiveness(raw, ID, player, turns):
		data = raw.query('phase=="test"')
		deltas = []
		for g in np.array(data['game'].unique()):
			game = data.query('game==@g')
			t_max = turns-2 if player=='investor' else turns-1
			for t in range(t_max):
				t_next = t+1
				if player=='investor':
					t_next2 = t+2
					opponent_gen_now = game.query('turn==@t & player=="trustee" & opponent_ID==@ID')['generosity'].to_numpy()[0]
					opponent_gen_next = game.query('turn==@t_next & player=="trustee" & opponent_ID==@ID')['generosity'].to_numpy()[0]
					# if np.isnan(opponent_gen_now) or np.isnan(opponent_gen_next): continue
					# code NaN generosity (trustee had 0 coins to give) as zero generosity
					if np.isnan(opponent_gen_now): opponent_gen_now=0
					if np.isnan(opponent_gen_next): opponent_gen_next=0
					# if opponent_gen_next>opponent_gen_now:
					if opponent_gen_next-opponent_gen_now>0.2:  # stronger constraint on increased generosity
						my_gen_before = game.query('turn==@t_next & player=="investor" & ID==@ID')['generosity'].to_numpy()[0]
						my_gen_after = game.query('turn==@t_next2 & player=="investor" & ID==@ID')['generosity'].to_numpy()[0]
						deltas.append(my_gen_after-my_gen_before)
				else:
					opponent_gen_now = game.query('turn==@t & player=="investor" & opponent_ID==@ID')['generosity'].to_numpy()[0]
					opponent_gen_next = game.query('turn==@t_next & player=="investor" & opponent_ID==@ID')['generosity'].to_numpy()[0]
					# if opponent_gen_next>opponent_gen_now:
					if opponent_gen_next-opponent_gen_now>0.2:  # stronger constraint on increased generosity
						my_gen_before = game.query('turn==@t & player=="trustee" & ID==@ID')['generosity'].to_numpy()[0]
						my_gen_after = game.query('turn==@t_next & player=="trustee" & ID==@ID')['generosity'].to_numpy()[0]
						deltas.append(my_gen_after-my_gen_before)
					print(t)
					print('investor', game.query('turn==@t & player=="investor" & opponent_ID==@ID')['generosity'].to_numpy()[0])
					print('trustee', game.query('turn==@t & player=="trustee" & ID==@ID')['generosity'].to_numpy()[0])
		print('forgiveness', deltas)
		return np.mean(deltas) if len(deltas)>0 else None

	def get_gift(raw, ID, player, turns, thr_investor=1, thr_trustee=0.5):
		data = raw.query('phase=="test"')
		increases = 0
		sames = 0
		decreases = 0
		for g in np.array(data['game'].unique()):
			game = data.query('game==@g')
			for t in range(turns-1):
				t1 = t+1
				if player=='investor':
					opponent_gen = np.array(game.query('turn==@t & player=="trustee" & opponent_ID==@ID')['generosity'])
					if not np.isnan(opponent_gen) and opponent_gen<thr_trustee:
						my_gen_before = np.array(game.query('turn==@t & player=="investor" & ID==@ID')['generosity'])
						my_gen_after = np.array(game.query('turn==@t1 & player=="investor" & ID==@ID')['generosity'])
						delta = my_gen_after-my_gen_before
						if delta>0: increases += 1
						if delta==0: sames += 1
						if delta<0: decreases += 1
				else:
					opponent_gen = np.array(game.query('turn==@t & player=="investor" & opponent_ID==@ID')['generosity'])
					if opponent_gen<thr_investor:
						my_gen_before = np.array(game.query('turn==@t & player=="trustee" & ID==@ID')['generosity'])
						my_gen_after = np.array(game.query('turn==@t1 & player=="trustee" & ID==@ID')['generosity'])
						delta = my_gen_after-my_gen_before
						if delta>0: increases += 1
						if delta==0: sames += 1
						if delta<0: decreases += 1
		# print('gift', increases, decreases, sames)
		return increases / (increases+sames+decreases) if (increases+sames+decreases)>0 else None

	def get_defection(raw, ID, player, turns, thr_investor=1, thr_trustee=0.5):
		data = raw.query('phase=="test"')
		increases = 0
		sames = 0
		decreases = 0
		for g in np.array(data['game'].unique()):
			game = data.query('game==@g')
			for t in range(turns-1):
				t1 = t+1
				if player=='investor':
					opponent_gen = np.array(game.query('turn==@t & player=="trustee" & opponent_ID==@ID')['generosity'])
					if not np.isnan(opponent_gen) and opponent_gen>=thr_trustee:
						my_gen_before = np.array(game.query('turn==@t & player=="investor" & ID==@ID')['generosity'])
						my_gen_after = np.array(game.query('turn==@t1 & player=="investor" & ID==@ID')['generosity'])
						delta = my_gen_after-my_gen_before
						if delta>0: increases += 1
						if delta==0: sames += 1
						if delta<0: decreases += 1
				else:
					opponent_gen = np.array(game.query('turn==@t & player=="investor" & opponent_ID==@ID')['generosity'])
					if opponent_gen>=thr_investor:
						my_gen_before = np.array(game.query('turn==@t & player=="trustee" & ID==@ID')['generosity'])
						my_gen_after = np.array(game.query('turn==@t1 & player=="trustee" & ID==@ID')['generosity'])
						delta = my_gen_after-my_gen_before
						if delta>0: increases += 1
						if delta==0: sames += 1
						if delta<0: decreases += 1
		# print('defections', increases, decreases, sames)
		return decreases / (increases+sames+decreases) if (increases+sames+decreases)>0 else None

	def get_trustee_end(raw, ID, player, turns):
		if player=='investor':
			return np.nan
		else:
			gens = []
			data = raw.query('phase=="test"')
			for g in np.array(data['game'].unique()):
				gens.append(data.query('game==@g & turn==@turns'))
			return np.mean(gens)

	turns = np.max(raw['turn'])+1
	games = np.max(raw['game'])+1
	initial_games = int(0.2*games)
	final_games = int(0.8*games)

	dfs = []
	for agent in agents:
		ID = agent.ID
		player = agent.player
		data = raw.query('ID==@ID & player==@player')
		data_test = raw.query('ID==@ID & player==@player & phase=="test"')
		mean = np.mean(data_test['generosity'])
		std = np.std(data_test['generosity'])
		ent = entropy(data_test['generosity'])
		skw = skew(data_test['generosity'])
		kurt = kurtosis(data_test['generosity'])
		# normality = normaltest(data_test[var])[1]  # the pvalue for the normaltest
		adapt = get_adaptivity(data_test, 'generosity')
		learn = get_learning(data, initial_games, final_games)
		speed = get_speed(data, games, final_games)
		punish = get_punishment(raw, ID, player, turns)
		forgive = get_forgiveness(raw, ID, player, turns)
		gift = get_gift(raw, ID, player, turns)
		defect = get_defection(raw, ID, player, turns)
		# trustee_end = get_trustee_end(raw, ID, player, turns)
		df = pd.DataFrame([[
			# ID, player, mean, std, ent, skw, kurt, normality, adapt, learn, speed, punish, forgive, gift, defect, trustee_end
			ID, player, mean, std, skw, kurt, adapt, learn, speed, punish, forgive, gift, defect,
			]], columns=metrics)
		dfs.append(df)
	data_metrics_gen = pd.concat([df for df in dfs], ignore_index=True)

	dfs = []
	for agent in agents:
		ID = agent.ID
		player = agent.player
		data = raw.query('ID==@ID & player==@player')
		mean = np.mean(data['coins'])
		std = np.std(data['coins'])
		ent = entropy(data['coins'])
		skw = skew(data['coins'])
		kurt = kurtosis(data['coins'])
		adapt = get_adaptivity(data, 'coins')
		df = pd.DataFrame([[ID, player, mean, std, skw, kurt, adapt]], columns=metrics_score)
		dfs.append(df)
	data_metrics_score = pd.concat([df for df in dfs], ignore_index=True)
	metrics_score = ('agent', 'player', 'mean', 'std', 'skew', 'kurtosis', 'adapt')

	# print(data_metrics_gen)
	# print(data_metrics_score)

	return data_metrics_gen, data_metrics_score