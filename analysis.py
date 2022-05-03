import numpy as np
import pandas as pd
import os
from scipy.stats import ks_2samp, ttest_ind, entropy

from utils import *
from plots import *

palette = sns.color_palette("colorblind")
sns.set(context='paper', style='white', font='CMU Serif', rc={'font.size':12, 'mathtext.fontset': 'cm'})
sns.set_palette(palette)

def ks_test_final_strategies(thr=0.1, games=3):
	dfs = []
	columns = ('agent', 'player', 'opponent', 'orientation', 'generosities')
	for agent in ["Human", "DQN", "IBL", "SPA"]:
		if agent=="Human": data = pd.read_pickle("user_data/all_users.pkl")
		if agent=="DQN": data = pd.read_pickle(f'agent_data/DeepQLearning_N=100_friendliness.pkl')
		if agent=="IBL": data = pd.read_pickle(f'agent_data/InstanceBased_N=100_friendliness.pkl')
		if agent=="SPA": data = pd.read_pickle(f'agent_data/NengoQLearning_N=100_friendliness.pkl')
		last_game = data['game'].unique().max()
		final_games = np.arange(last_game-(games-1), last_game+1)
		game_string = 'game in @final_games'
		for player in ["investor", "trustee"]:
			if player=="investor": player_string = 'player=="investor"'
			if player=="trustee": player_string = 'player=="trustee"'
			for opponent in ["greedy", "generous"]:
				if agent=="Human" and opponent=="greedy": opponent_string = 'opponent_ID=="greedyT4T"'
				if agent=="Human" and opponent=="generous": opponent_string = 'opponent_ID=="generousT4T"'
				if agent!="Human" and opponent=="greedy": opponent_string = 'opponent_ID=="GreedyT4T"'
				if agent!="Human" and opponent=="generous": opponent_string = 'opponent_ID=="GenerousT4T"'
				for orientation in ["proself", "prosocial"]:
					if agent=="Human" and orientation=="proself": orientation_string = 'orientation=="self"'
					if agent=="Human" and orientation=="prosocial": orientation_string = 'orientation=="social"'
					if agent!="Human" and orientation=="proself": orientation_string = 'friendliness<@thr'
					if agent!="Human" and orientation=="prosocial": orientation_string = 'friendliness>@thr'
					query_string = player_string + ' & ' + opponent_string + ' & ' + orientation_string + ' & ' + game_string
					generosities = data.query(query_string).dropna()['generosity'].to_numpy()
					dfs.append(pd.DataFrame([[agent, player, opponent, orientation, generosities]], columns=columns))
	data = pd.concat(dfs, ignore_index=True)
	# print(data)
	dfs2 = []
	columns2 = ('agent1', 'player1', 'opponent1', 'orientation1', 'agent2', 'player2', 'opponent2', 'orientation2', 'statistic', 'pvalue')
	i = 0
	for agent1 in ["Human", "DQN", "IBL", "SPA"]:
		for player1 in ["investor", "trustee"]:
			for opponent1 in ["greedy", "generous"]:
				for orientation1 in ["proself", "prosocial"]:
					generosities1 = data.query('agent==@agent1 & player==@player1 & opponent==@opponent1 & orientation==@orientation1')['generosities'].to_numpy()[0]
					for agent2 in ["Human", "DQN", "IBL", "SPA"]:
						for player2 in ["investor", "trustee"]:
							for opponent2 in ["greedy", "generous"]:
								for orientation2 in ["proself", "prosocial"]:
									generosities2 = data.query('agent==@agent2 & player==@player2 & opponent==@opponent2 & orientation==@orientation2')['generosities'].to_numpy()[0]
									statistic, pvalue = ks_2samp(generosities1, generosities2)
									dfs2.append(pd.DataFrame([[
										agent1, player1, opponent1, orientation1, agent2, player2, opponent2, orientation2, statistic, pvalue
										]], columns=columns2))
	ks_data = pd.concat(dfs2, ignore_index=True)
	ks_data.to_pickle("analysis_data/ks_data.pkl")
	# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
	# 	print(ks_data)

# ks_test_final_strategies()

def p_value_to_significance(p, stars=False):
	if stars:
		if p > 0.05: sig = "(ns)"
		elif 0.01 < p < 0.05: sig = "(*)"
		elif 0.001 < p < 0.01: sig = "(**)"
		elif 0.0001 < p < 0.001: sig = "(***)"
		else: sig = "(****)"
	else:
		if p > 0.05: sig = "(ns)"
		elif 0.01 < p < 0.05: sig = "(p<0.05)"
		elif 0.001 < p < 0.01: sig = "(p<0.01)"
		elif 0.0001 < p < 0.001: sig = "(p<0.001)"
		else: sig = "(p<0.0001)"
	return sig

def print_ks_pairs():
	ks_data = pd.read_pickle("analysis_data/ks_data.pkl")
	print("proself vs prosocial")
	for agent in ["Human", "DQN", "IBL", "SPA"]:
		for player in ["investor", "trustee"]:
			for opponent in ["greedy", "generous"]:
				orientation1 = "proself"
				orientation2 = "prosocial"
				data = ks_data.query("agent1==@agent & agent2==@agent & player1==@player & player2==@player \
					& opponent1==@opponent & opponent2==@opponent & orientation1==@orientation1 & orientation2==@orientation2")
				statistic = data['statistic'].to_numpy()[0]
				pvalue = p_value_to_significance(data['pvalue'].to_numpy()[0])
				print(agent+"\t"+player+" \t"+opponent+"  \t"+orientation1+" vs "+orientation2+f": \t {statistic:.3}  {pvalue}")

	print("greedy vs generous")
	for agent in ["Human", "DQN", "IBL", "SPA"]:
		for player in ["investor", "trustee"]:
			for orientation in ['proself', 'prosocial']:
				opponent1 = "greedy"
				opponent2 = "generous"
				data = ks_data.query("agent1==@agent & agent2==@agent & player1==@player & player2==@player \
					& opponent1==@opponent1 & opponent2==@opponent2 & orientation1==@orientation & orientation2==@orientation")
				statistic = data['statistic'].to_numpy()[0]
				pvalue = p_value_to_significance(data['pvalue'].to_numpy()[0])
				print(agent+"\t"+player+" \t"+orientation+"  \t"+opponent1+" vs "+opponent2+f": \t {statistic:.3}  {pvalue}")

	print("agents vs humans, proself and prosocial")
	for agent in ["DQN", "IBL", "SPA"]:
		for player in ["investor", "trustee"]:
			for opponent in ["greedy", "generous"]:
				for orientation1 in ['proself', 'prosocial']:
					for orientation2 in ['proself', 'prosocial']:
						data = ks_data.query("agent1==@agent & agent2=='Human' & player1==@player & player2==@player \
							& opponent1==@opponent & opponent2==@opponent & orientation1==@orientation1 & orientation2==@orientation2")
						statistic = data['statistic'].to_numpy()[0]
						pvalue = p_value_to_significance(data['pvalue'].to_numpy()[0])
						print(agent+"\t"+player+" \t"+opponent+"  \tAgent "+orientation1+"\t\tHuman "+orientation2+f":  \t {statistic:.3}  {pvalue}")

print_ks_pairs()