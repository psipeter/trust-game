import numpy as np
import pandas as pd
import os
from scipy.stats import ks_2samp, ttest_ind, entropy
from scipy.ndimage import histogram
from scipy.spatial.distance import jensenshannon

from utils import *
from plots import *

palette = sns.color_palette("colorblind")
sns.set(context='paper', style='white', font='CMU Serif', rc={'font.size':12, 'mathtext.fontset': 'cm'})
sns.set_palette(palette)

def ks_test_final_strategies(agents, games=3):
	dfs = []
	columns = ('agent', 'player', 'opponent', 'orientation', 'generosities')
	agents_plus_human = [agent for agent in agents]
	agents_plus_human.append("Human")
	for agent in agents_plus_human:
		if agent=="Human": data = pd.read_pickle("human_data/human_data.pkl")
		if agent=="TQ": data = pd.read_pickle("agent_data/TQ_N=100_games=200_svo.pkl")
		if agent=="DQN": data = pd.read_pickle(f'agent_data/DQN_N=300_games=400_svo.pkl')
		if agent=="IBL": data = pd.read_pickle(f'agent_data/IBL_N=300_games=200_svo.pkl')
		last_game = data['game'].unique().max()
		final_games = np.arange(last_game-(games-1), last_game+1)
		game_string = 'game in @final_games'
		for player in ["investor", "trustee"]:
			for opponent in ["greedy", "generous"]:
				for orientation in ["proself", "prosocial"]:
					query_string = "player==@player & opponent==@opponent & orientation==@orientation & game in @final_games"
					generosities = data.query(query_string).dropna()['generosity'].to_numpy()
					dfs.append(pd.DataFrame([[agent, player, opponent, orientation, generosities]], columns=columns))
	data = pd.concat(dfs, ignore_index=True)
	# print(data)
	dfs2 = []
	columns2 = ('agent1', 'player1', 'opponent1', 'orientation1', 'agent2', 'player2', 'opponent2', 'orientation2', 'statistic', 'pvalue')
	i = 0
	for agent1 in agents_plus_human:
		for player1 in ["investor", "trustee"]:
			for opponent1 in ["greedy", "generous"]:
				for orientation1 in ["proself", "prosocial"]:
					generosities1 = data.query('agent==@agent1 & player==@player1 & opponent==@opponent1 & orientation==@orientation1')['generosities'].to_numpy()[0]
					for agent2 in agents_plus_human:
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

	# print("proself vs prosocial")
	# for agent in ["Human", "DQN", "IBL", "SPA"]:
	# 	for player in ["investor", "trustee"]:
	# 		for opponent in ["greedy", "generous"]:
	# 			orientation1 = "proself"
	# 			orientation2 = "prosocial"
	# 			data = ks_data.query("agent1==@agent & agent2==@agent & player1==@player & player2==@player \
	# 				& opponent1==@opponent & opponent2==@opponent & orientation1==@orientation1 & orientation2==@orientation2")
	# 			statistic = data['statistic'].to_numpy()[0]
	# 			pvalue = p_value_to_significance(data['pvalue'].to_numpy()[0])
	# 			print(agent+"\t"+player+" \t"+opponent+"  \t"+orientation1+" vs "+orientation2+f": \t {statistic:.3}  {pvalue}")

	# print("greedy vs generous")
	# for agent in ["Human", "DQN", "IBL", "SPA"]:
	# 	for player in ["investor", "trustee"]:
	# 		for orientation in ['proself', 'prosocial']:
	# 			opponent1 = "greedy"
	# 			opponent2 = "generous"
	# 			data = ks_data.query("agent1==@agent & agent2==@agent & player1==@player & player2==@player \
	# 				& opponent1==@opponent1 & opponent2==@opponent2 & orientation1==@orientation & orientation2==@orientation")
	# 			statistic = data['statistic'].to_numpy()[0]
	# 			pvalue = p_value_to_significance(data['pvalue'].to_numpy()[0])
	# 			print(agent+"\t"+player+" \t"+orientation+"  \t"+opponent1+" vs "+opponent2+f": \t {statistic:.3}  {pvalue}")

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

# print_ks_pairs()

def KS_similarity_metric(agents):
	ks_data = pd.read_pickle("analysis_data/ks_data.pkl")
	for orientation_human in ['proself', 'prosocial']:
		proself_agent_is_better_match = 0
		proself_agent_is_better_match_conditions = []
		prosocial_agent_is_better_match = 0
		prosocial_agent_is_better_match_conditions = []
		for agent in agents:
			for player in ["investor", "trustee"]:
				for opponent in ["greedy", "generous"]:
					proself_data = ks_data.query("agent1==@agent & agent2=='Human' & player1==@player & player2==@player \
							& opponent1==@opponent & opponent2==@opponent & orientation1=='proself' & orientation2==@orientation_human")
					prosocial_data = ks_data.query("agent1==@agent & agent2=='Human' & player1==@player & player2==@player \
							& opponent1==@opponent & opponent2==@opponent & orientation1=='prosocial' & orientation2==@orientation_human")
					proself_similarity = 1 - proself_data['statistic'].to_numpy()[0]
					prosocial_similarity = 1 - prosocial_data['statistic'].to_numpy()[0]
					if proself_similarity > prosocial_similarity:
						proself_agent_is_better_match += 1
						proself_agent_is_better_match_conditions.append(f"{player} {opponent}")
					else:
						prosocial_agent_is_better_match += 1
						prosocial_agent_is_better_match_conditions.append(f"{player} {opponent}")
		n_conditions = proself_agent_is_better_match + prosocial_agent_is_better_match
		print(f"For {orientation_human} humans")
		print(f"proself agents are better fits in {proself_agent_is_better_match}/{n_conditions} conditions ({proself_agent_is_better_match_conditions})")
		print(f"prosocial agents are better fits in {prosocial_agent_is_better_match}/{n_conditions} conditions ({prosocial_agent_is_better_match_conditions})")

# KS_similarity_metric()

# def rename_human_data():
# 	data = pd.read_pickle("user_data/all_users.pkl")
# 	data = data.rename(columns={"opponent_ID": "opponent"})
# 	data = data.replace('greedyT4T', "greedy")
# 	data = data.replace('generousT4T', "generous")
# 	data = data.replace('social', "prosocial")
# 	data = data.replace('self', "proself")
# 	data.to_pickle("human_data/human_data.pkl")

# rename_human_data()

# data = pd.read_pickle("human_data/human_data.pkl")
# plot_final_generosities_svo(data, "Human")



# def similarity_metric(agents, games=3, test="mean"):
# 	dfs = []
# 	columns = ('agent', 'player', 'opponent', 'orientation_human', 'orientation_agent', 'metric')
# 	human_data_raw = pd.read_pickle("human_data/human_data.pkl")
# 	last_game = human_data_raw['game'].unique().max()
# 	final_games = np.arange(last_game-(games-1), last_game+1)
# 	human_data = human_data_raw.query("game in @final_games")
# 	for agent in agents:
# 		if agent=="TQ": agent_data_raw = pd.read_pickle("agent_data/TQ_N=100_games=200_svo.pkl")
# 		if agent=="DQN": agent_data_raw = pd.read_pickle(f'agent_data/DQN_N=300_games=400_svo.pkl')
# 		if agent=="IBL": agent_data_raw = pd.read_pickle(f'agent_data/IBL_N=300_games=200_svo.pkl')
# 		last_game = agent_data_raw['game'].unique().max()
# 		final_games = np.arange(last_game-(games-1), last_game+1)
# 		agent_data = agent_data_raw.query("game in @final_games")
# 		for player in ["investor", "trustee"]:
# 			for opponent in ["greedy", "generous"]:
# 				for orientation_human in ["proself", "prosocial"]:
# 					for orientation_agent in ["proself", "prosocial"]:
# 						metric = 0
# 						for turn in range(5):
# 							human_gens = human_data.query("player==@player & opponent==@opponent & orientation==@orientation_human & turn==@turn").dropna()['generosity'].to_numpy()
# 							agent_gens = agent_data.query("player==@player & opponent==@opponent & orientation==@orientation_agent & turn==@turn").dropna()['generosity'].to_numpy()
# 							if test == "median":
# 								metric += np.abs(np.median(human_gens) - np.median(agent_gens))
# 							if test == "mean":
# 								metric += np.abs(np.mean(human_gens) - np.mean(agent_gens))
# 							if test == "KS":
# 								metric += ks_2samp(human_gens, agent_gens)[0]
# 							if test == "JS":
# 								human_histogram = histogram(human_gens, min=0, max=1, bins=5) / len(human_gens)
# 								agent_histogram = histogram(agent_gens, min=0, max=1, bins=5) / len(agent_gens)
# 								metric = jensenshannon(human_histogram, agent_histogram)
# 						dfs.append(pd.DataFrame([[agent, player, opponent, orientation_human, orientation_agent, metric]], columns=columns))
# 	data = pd.concat(dfs, ignore_index=True)
# 	print(data)
# 	for orientation_human in ["proself", "prosocial"]:
# 		proself_agent_is_better_match = 0
# 		prosocial_agent_is_better_match = 0
# 		for player in ["investor", "trustee"]:
# 			for opponent in ["greedy", "generous"]:
# 				for agent in agents:
# 					proself_query = "player==@player & opponent==@opponent & orientation_human==@orientation_human & agent==@agent & orientation_agent=='proself'"
# 					prosocial_query = "player==@player & opponent==@opponent & orientation_human==@orientation_human & agent==@agent & orientation_agent=='prosocial'"
# 					proself_metric = data.query(proself_query)['metric'].to_numpy()
# 					prosocial_metric = data.query(prosocial_query)['metric'].to_numpy()
# 					if proself_metric < prosocial_metric:
# 						proself_agent_is_better_match += 1
# 					else:
# 						prosocial_agent_is_better_match += 1
# 		n_conditions = proself_agent_is_better_match + prosocial_agent_is_better_match
# 		if orientation_human=="proself":
# 			print(f"For {orientation_human} humans, proself agents are better fits in {proself_agent_is_better_match}/{n_conditions} conditions")
# 		else:
# 			print(f"For {orientation_human} humans, prosocial agents are better fits in {prosocial_agent_is_better_match}/{n_conditions} conditions")


def similarity_metric(agents, games=3, test="mean"):
	dfs = []
	columns = ('agent', 'player', 'opponent', 'orientation_human', 'orientation_agent', 'turn', 'metric')
	human_data_raw = pd.read_pickle("human_data/human_data.pkl")
	last_game = human_data_raw['game'].unique().max()
	final_games = np.arange(last_game-(games-1), last_game+1)
	human_data = human_data_raw.query("game in @final_games")
	for agent in agents:
		if agent=="TQ": agent_data_raw = pd.read_pickle("agent_data/TQ_N=100_games=200_svo.pkl")
		if agent=="DQN": agent_data_raw = pd.read_pickle(f'agent_data/DQN_N=250_games=400_svo.pkl')
		if agent=="IBL": agent_data_raw = pd.read_pickle(f'agent_data/IBL_N=250_games=200_svo.pkl')
		last_game = agent_data_raw['game'].unique().max()
		final_games = np.arange(last_game-(games-1), last_game+1)
		agent_data = agent_data_raw.query("game in @final_games")
		for player in ["investor", "trustee"]:
			for opponent in ["greedy", "generous"]:
				for orientation_human in ["proself", "prosocial"]:
					for orientation_agent in ["proself", "prosocial"]:
						for turn in range(5):
							human_gens = human_data.query("player==@player & opponent==@opponent & orientation==@orientation_human & turn==@turn").dropna()['generosity'].to_numpy()
							agent_gens = agent_data.query("player==@player & opponent==@opponent & orientation==@orientation_agent & turn==@turn").dropna()['generosity'].to_numpy()
							if test == "median":
								metric = np.abs(np.median(human_gens) - np.median(agent_gens))
							if test == "mean":
								metric = np.abs(np.mean(human_gens) - np.mean(agent_gens))
							if test == "KS":
								metric = ks_2samp(human_gens, agent_gens)[0]
							if test == "JS":
								human_histogram = histogram(human_gens, min=0, max=1, bins=5) / len(human_gens)
								agent_histogram = histogram(agent_gens, min=0, max=1, bins=5) / len(agent_gens)
								metric = jensenshannon(human_histogram, agent_histogram)
							if test == "5x5":
								human_histogram = histogram(human_gens, min=0, max=1, bins=5) / len(human_gens)
								agent_histogram = histogram(agent_gens, min=0, max=1, bins=5) / len(agent_gens)
								metric = 0
								for gen_bin in range(len(human_histogram)):
									metric += np.abs(human_histogram[gen_bin] - agent_histogram[gen_bin])
							dfs.append(pd.DataFrame([[agent, player, opponent, orientation_human, orientation_agent, turn, metric]], columns=columns))
	data = pd.concat(dfs, ignore_index=True)
	with pd.option_context('display.max_rows', None):
		print(data)

	for orientation_human in ["proself", "prosocial"]:
		proself_agent_is_better_match = 0
		prosocial_agent_is_better_match = 0
		for player in ["investor", "trustee"]:
			for opponent in ["greedy", "generous"]:
				for agent in agents:
					proself_query = "player==@player & opponent==@opponent & orientation_human==@orientation_human & agent==@agent & orientation_agent=='proself'"
					prosocial_query = "player==@player & opponent==@opponent & orientation_human==@orientation_human & agent==@agent & orientation_agent=='prosocial'"
					proself_metric = 0
					prosocial_metric = 0
					for turn in range(5):
						proself_metric += data.query(proself_query+f" & turn==@turn")['metric'].to_numpy()
						prosocial_metric += data.query(prosocial_query+f" & turn==@turn")['metric'].to_numpy()
					if proself_metric < prosocial_metric:
						proself_agent_is_better_match += 1
					else:
						prosocial_agent_is_better_match += 1
		n_conditions = proself_agent_is_better_match + prosocial_agent_is_better_match
		if orientation_human=="proself":
			print(f"For {orientation_human} humans, proself agents are better fits in {proself_agent_is_better_match}/{n_conditions} conditions")
		else:
			print(f"For {orientation_human} humans, prosocial agents are better fits in {prosocial_agent_is_better_match}/{n_conditions} conditions")
