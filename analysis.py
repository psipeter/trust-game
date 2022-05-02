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









def test_human_learning():
	human_data = pd.read_pickle("user_data/all_users.pkl")
	initial_games = [0,1,2]
	final_games = [12,13,14]

	human_investor_greedyT4T_self = human_data.query('player=="investor" & opponent_ID=="greedyT4T" & orientation=="self"').dropna()
	human_investor_greedyT4T_social = human_data.query('player=="investor" & opponent_ID=="greedyT4T" & orientation=="social"').dropna()
	human_investor_generousT4T_self = human_data.query('player=="investor" & opponent_ID=="generousT4T" & orientation=="self"').dropna()
	human_investor_generousT4T_social = human_data.query('player=="investor" & opponent_ID=="generousT4T" & orientation=="social"').dropna()
	human_trustee_greedyT4T_self = human_data.query('player=="trustee" & opponent_ID=="greedyT4T" & orientation=="self"').dropna()
	human_trustee_greedyT4T_social = human_data.query('player=="trustee" & opponent_ID=="greedyT4T" & orientation=="social"').dropna()
	human_trustee_generousT4T_self = human_data.query('player=="trustee" & opponent_ID=="generousT4T" & orientation=="self"').dropna()
	human_trustee_generousT4T_social = human_data.query('player=="trustee" & opponent_ID=="generousT4T" & orientation=="social"').dropna()

	ksval_investor_greedyT4T_self, pval_investor_greedyT4T_self = ks_2samp(
		human_investor_greedyT4T_self.query("game in @initial_games")['generosity'].to_numpy(),
		human_investor_greedyT4T_self.query("game in @final_games")['generosity'].to_numpy())
	ksval_investor_greedyT4T_social, pval_investor_greedyT4T_social = ks_2samp(
		human_investor_greedyT4T_social.query("game in @initial_games")['generosity'].to_numpy(),
		human_investor_greedyT4T_social.query("game in @final_games")['generosity'].to_numpy())
	ksval_investor_generousT4T_self, pval_investor_generousT4T_self = ks_2samp(
		human_investor_generousT4T_self.query("game in @initial_games")['generosity'].to_numpy(),
		human_investor_generousT4T_self.query("game in @final_games")['generosity'].to_numpy())
	ksval_investor_generousT4T_social, pval_investor_generousT4T_social = ks_2samp(
		human_investor_generousT4T_social.query("game in @initial_games")['generosity'].to_numpy(),
		human_investor_generousT4T_social.query("game in @final_games")['generosity'].to_numpy())
	ksval_trustee_greedyT4T_self, pval_itrusteegreedyT4T_self = ks_2samp(
		human_trustee_greedyT4T_self.query("game in @initial_games")['generosity'].to_numpy(),
		human_trustee_greedyT4T_self.query("game in @final_games")['generosity'].to_numpy())
	ksval_trustee_greedyT4T_social, pval_itrusteegreedyT4T_social = ks_2samp(
		human_trustee_greedyT4T_social.query("game in @initial_games")['generosity'].to_numpy(),
		human_trustee_greedyT4T_social.query("game in @final_games")['generosity'].to_numpy())
	ksval_trustee_generousT4T_self, pval_itrusteegenerousT4T_self = ks_2samp(
		human_trustee_generousT4T_self.query("game in @initial_games")['generosity'].to_numpy(),
		human_trustee_generousT4T_self.query("game in @final_games")['generosity'].to_numpy())
	ksval_trustee_generousT4T_social, pval_itrusteegenerousT4T_social = ks_2samp(
		human_trustee_generousT4T_social.query("game in @initial_games")['generosity'].to_numpy(),
		human_trustee_generousT4T_social.query("game in @final_games")['generosity'].to_numpy())

	print("\nIs the initial distribution of generosities different than the final distribution for participants?")
	print("Run a two-sample Kolmogorov-Smirnov on these distributions, report KS statistic and p_value")

	print(f"Investor vs GreedyT4T, self-oriented \t \t: KS={ksval_investor_greedyT4T_self:.4}, p={pval_investor_greedyT4T_self:.4}")
	print(f"Investor vs GreedyT4T, socially-oriented \t: KS={ksval_investor_greedyT4T_social:.4}, p={pval_investor_greedyT4T_social:.4}")
	print(f"Investor vs GenerousT4T, self-oriented \t \t: KS={ksval_investor_generousT4T_self:.4}, p={pval_investor_generousT4T_self:.4}")
	print(f"Investor vs GenerousT4T, socially-oriented \t: KS={ksval_investor_generousT4T_social:.4}, p={pval_investor_generousT4T_social:.4}")
	print(f"Trustee vs GreedyT4T, self-oriented \t \t: KS={ksval_trustee_greedyT4T_self:.4}, p={pval_itrusteegreedyT4T_self:.4}")
	print(f"Trustee vs GreedyT4T, socially-oriented \t: KS={ksval_trustee_greedyT4T_social:.4}, p={pval_itrusteegreedyT4T_social:.4}")
	print(f"Trustee vs GenerousT4T, self-oriented \t \t: KS={ksval_trustee_generousT4T_self:.4}, p={pval_itrusteegenerousT4T_self:.4}")
	print(f"Trustee vs GenerousT4T, socially-oriented \t: KS={ksval_trustee_generousT4T_social:.4}, p={pval_itrusteegenerousT4T_social:.4}")

def test_human_orientation():
	human_data = pd.read_pickle("user_data/all_users.pkl")
	initial_games = [0,1,2]
	final_games = [12,13,14]

	human_investor_greedyT4T_self = human_data.query('player=="investor" & opponent_ID=="greedyT4T" & orientation=="self"').dropna()
	human_investor_greedyT4T_social = human_data.query('player=="investor" & opponent_ID=="greedyT4T" & orientation=="social"').dropna()
	human_investor_generousT4T_self = human_data.query('player=="investor" & opponent_ID=="generousT4T" & orientation=="self"').dropna()
	human_investor_generousT4T_social = human_data.query('player=="investor" & opponent_ID=="generousT4T" & orientation=="social"').dropna()
	human_trustee_greedyT4T_self = human_data.query('player=="trustee" & opponent_ID=="greedyT4T" & orientation=="self"').dropna()
	human_trustee_greedyT4T_social = human_data.query('player=="trustee" & opponent_ID=="greedyT4T" & orientation=="social"').dropna()
	human_trustee_generousT4T_self = human_data.query('player=="trustee" & opponent_ID=="generousT4T" & orientation=="self"').dropna()
	human_trustee_generousT4T_social = human_data.query('player=="trustee" & opponent_ID=="generousT4T" & orientation=="social"').dropna()

	ksval_investor_greedyT4T, pval_investor_greedyT4T = ks_2samp(
		human_investor_greedyT4T_self.query("game in @final_games")['generosity'].to_numpy(),
		human_investor_greedyT4T_social.query("game in @final_games")['generosity'].to_numpy())
	ksval_investor_generousT4T, pval_investor_generousT4T = ks_2samp(
		human_investor_generousT4T_self.query("game in @final_games")['generosity'].to_numpy(),
		human_investor_generousT4T_social.query("game in @final_games")['generosity'].to_numpy())
	ksval_trustee_greedyT4T, pval_trustee_greedyT4T = ks_2samp(
		human_trustee_greedyT4T_self.query("game in @final_games")['generosity'].to_numpy(),
		human_trustee_greedyT4T_social.query("game in @final_games")['generosity'].to_numpy())
	ksval_trustee_generousT4T, pval_trustee_generousT4T = ks_2samp(
		human_trustee_generousT4T_self.query("game in @final_games")['generosity'].to_numpy(),
		human_trustee_generousT4T_social.query("game in @final_games")['generosity'].to_numpy())


	print("\nIs the final distribution of generosities different for self-oriented vs socially-oriented participants?")
	print("Run a two-sample Kolmogorov-Smirnov on these distributions, report KS statistic and p_value")

	print(f"Investor vs GreedyT4T: \t \t KS={ksval_investor_greedyT4T:.4}, p={pval_investor_greedyT4T:.4}")
	print(f"Investor vs GenerousT4T: \t KS={ksval_investor_generousT4T:.4}, p={pval_investor_generousT4T:.4}")
	print(f"Trustee vs GreedyT4T: \t \t KS={ksval_trustee_greedyT4T:.4}, p={pval_trustee_greedyT4T:.4}")
	print(f"Trustee vs GenerousT4T: \t KS={ksval_trustee_generousT4T:.4}, p={pval_trustee_generousT4T:.4}")


def test_human_generosity():
	human_data = pd.read_pickle("user_data/all_users.pkl")
	final_games = [12,13,14]

	human_self = human_data.query('game in @final_games & orientation=="self"').dropna()
	human_social = human_data.query('game in @final_games & orientation=="social"').dropna()

	human_self_greedy = human_self.query('generosity < 0.5').size
	human_self_generous = human_self.query('generosity >= 0.5').size
	human_social_greedy = human_social.query('generosity < 0.5').size
	human_social_generous = human_social.query('generosity >= 0.5').size

	print(f"\nSelf-oriented participants make greedy moves (generosity<0.5) {human_self_greedy/(human_self_greedy+human_self_generous):.2f}% of the time in the final games")
	print(f"Socially-oriented participants make greedy moves (generosity<0.5) {human_social_greedy/(human_social_greedy+human_social_generous):.2f}% of the time in the final games")

	print("\nAre the means of the generosity distributions different for self- versus socially-oriented participants?")
	print("Run a Welsch's T-test, on these distributions, report t-statistic and p_value")

	tval, pval = ttest_ind(human_self['generosity'].to_numpy(), human_social['generosity'].to_numpy())
	print(f't-statistic={tval} \t p={pval}')
	print(f"mean for self-oriented={np.mean(human_self['generosity'].to_numpy()):.2} \t mean for socially-oriented={np.mean(human_social['generosity'].to_numpy()):.2}")


def test_human_behaviors():
	human_data = pd.read_pickle("user_data/all_users.pkl")
	final_games = [12,13,14]

	human_investor_greedyT4T_self = human_data.query('player=="investor" & opponent_ID=="greedyT4T" & orientation=="self" & game in @final_games').dropna()
	human_investor_greedyT4T_social = human_data.query('player=="investor" & opponent_ID=="greedyT4T" & orientation=="social" & game in @final_games').dropna()
	human_investor_generousT4T_self = human_data.query('player=="investor" & opponent_ID=="generousT4T" & orientation=="self" & game in @final_games').dropna()
	human_investor_generousT4T_social = human_data.query('player=="investor" & opponent_ID=="generousT4T" & orientation=="social" & game in @final_games').dropna()

	human_investor_greedyT4T_self_keep_all = human_investor_greedyT4T_self.query('generosity==0').size / human_investor_greedyT4T_self.size
	human_investor_greedyT4T_social_keep_all = human_investor_greedyT4T_social.query('generosity==0').size / human_investor_greedyT4T_social.size
	human_investor_generousT4T_self_keep_all = human_investor_generousT4T_self.query('generosity==1').size / human_investor_generousT4T_self.size
	human_investor_generousT4T_social_keep_all = human_investor_generousT4T_social.query('generosity==1').size / human_investor_generousT4T_social.size

	print(f'\nWhen playing the investor against a greedyT4T opponent, what percentate of participants keep all their coins?')
	print(f'{human_investor_greedyT4T_self_keep_all:.2}% of self-oriented participants and {human_investor_greedyT4T_social_keep_all:.2}% of socially-oriented participants')

	print(f'\nWhen playing the investor against a generousT4T opponent, what percentate of participants invest all their coins?')
	print(f'{human_investor_generousT4T_self_keep_all:.2}% of self-oriented participants and {human_investor_generousT4T_social_keep_all:.2}% of socially-oriented participants')



# test_human_learning()
# test_human_orientation()
# test_human_generosity()
# test_human_behaviors()



def test_agent_learning():
	initial_games = [0,1,2,3,4,5,6,7,8,9]
	final_games = [140,141,142,143,144,145,146,147,148,149]

	f_thr = 0.1
	dqn_data = pd.read_pickle(f'agent_data/DeepQLearning_N=100_friendliness.pkl')
	ibl_data = pd.read_pickle(f'agent_data/InstanceBased_N=100_friendliness.pkl')
	nef_data = pd.read_pickle(f'agent_data/NengoQLearning_N=100_friendliness.pkl')

	dqn_investor_greedyT4T_self = dqn_data.query('player=="investor" & opponent_ID=="GreedyT4T" & friendliness<@f_thr').drop_na()
	dqn_investor_greedyT4T_social = dqn_data.query('player=="investor" & opponent_ID=="GreedyT4T" & friendliness>=@f_thr').drop_na()
	dqn_investor_generousT4T_self = dqn_data.query('player=="investor" & opponent_ID=="GenerousT4T" & friendliness<@f_thr').drop_na()
	dqn_investor_generousT4T_social = dqn_data.query('player=="investor" & opponent_ID=="GenerousT4T" & friendliness>=@f_thr').drop_na()
	dqn_trustee_greedyT4T_self = dqn_data.query('player=="trustee" & opponent_ID=="GreedyT4T" & friendliness<@f_thr').drop_na()
	dqn_trustee_greedyT4T_social = dqn_data.query('player=="trustee" & opponent_ID=="GreedyT4T" & friendliness>=@f_thr').drop_na()
	dqn_trustee_generousT4T_self = dqn_data.query('player=="trustee" & opponent_ID=="GenerousT4T" & friendliness<@f_thr').drop_na()
	dqn_trustee_generousT4T_social = dqn_data.query('player=="trustee" & opponent_ID=="GenerousT4T" & friendliness>=@f_thr').drop_na()

	ibl_investor_greedyT4T_self = ibl_data.query('player=="investor" & opponent_ID=="GreedyT4T" & friendliness<@f_thr').drop_na()
	ibl_investor_greedyT4T_social = ibl_data.query('player=="investor" & opponent_ID=="GreedyT4T" & friendliness>=@f_thr').drop_na()
	ibl_investor_generousT4T_self = ibl_data.query('player=="investor" & opponent_ID=="GenerousT4T" & friendliness<@f_thr').drop_na()
	ibl_investor_generousT4T_social = ibl_data.query('player=="investor" & opponent_ID=="GenerousT4T" & friendliness>=@f_thr').drop_na()
	ibl_trustee_greedyT4T_self = ibl_data.query('player=="trustee" & opponent_ID=="GreedyT4T" & friendliness<@f_thr').drop_na()
	ibl_trustee_greedyT4T_social = ibl_data.query('player=="trustee" & opponent_ID=="GreedyT4T" & friendliness>=@f_thr').drop_na()
	ibl_trustee_generousT4T_self = ibl_data.query('player=="trustee" & opponent_ID=="GenerousT4T" & friendliness<@f_thr').drop_na()
	ibl_trustee_generousT4T_social = ibl_data.query('player=="trustee" & opponent_ID=="GenerousT4T" & friendliness>=@f_thr').drop_na()

	nef_investor_greedyT4T_self = nef_data.query('player=="investor" & opponent_ID=="GreedyT4T" & friendliness<@f_thr').drop_na()
	nef_investor_greedyT4T_social = nef_data.query('player=="investor" & opponent_ID=="GreedyT4T" & friendliness>=@f_thr').drop_na()
	nef_investor_generousT4T_self = nef_data.query('player=="investor" & opponent_ID=="GenerousT4T" & friendliness<@f_thr').drop_na()
	nef_investor_generousT4T_social = nef_data.query('player=="investor" & opponent_ID=="GenerousT4T" & friendliness>=@f_thr').drop_na()
	nef_trustee_greedyT4T_self = nef_data.query('player=="trustee" & opponent_ID=="GreedyT4T" & friendliness<@f_thr').drop_na()
	nef_trustee_greedyT4T_social = nef_data.query('player=="trustee" & opponent_ID=="GreedyT4T" & friendliness>=@f_thr').drop_na()
	nef_trustee_generousT4T_self = nef_data.query('player=="trustee" & opponent_ID=="GenerousT4T" & friendliness<@f_thr').drop_na()
	nef_trustee_generousT4T_social = nef_data.query('player=="trustee" & opponent_ID=="GenerousT4T" & friendliness>=@f_thr').drop_na()


	ksval_investor_greedyT4T_self, pval_investor_greedyT4T_self = ks_2samp(
		dqn_investor_greedyT4T_self.query("game in @initial_games")['generosity'].to_numpy(),
		dqn_investor_greedyT4T_self.query("game in @final_games")['generosity'].to_numpy())
	ksval_investor_greedyT4T_social, pval_investor_greedyT4T_social = ks_2samp(
		dqn_investor_greedyT4T_social.query("game in @initial_games")['generosity'].to_numpy(),
		dqn_investor_greedyT4T_social.query("game in @final_games")['generosity'].to_numpy())
	ksval_investor_generousT4T_self, pval_investor_generousT4T_self = ks_2samp(
		dqn_investor_generousT4T_self.query("game in @initial_games")['generosity'].to_numpy(),
		dqn_investor_generousT4T_self.query("game in @final_games")['generosity'].to_numpy())
	ksval_investor_generousT4T_social, pval_investor_generousT4T_social = ks_2samp(
		dqn_investor_generousT4T_social.query("game in @initial_games")['generosity'].to_numpy(),
		dqn_investor_generousT4T_social.query("game in @final_games")['generosity'].to_numpy())
	ksval_trustee_greedyT4T_self, pval_trustee_greedyT4T_self = ks_2samp(
		dqn_trustee_greedyT4T_self.query("game in @initial_games")['generosity'].to_numpy(),
		dqn_trustee_greedyT4T_self.query("game in @final_games")['generosity'].to_numpy())
	ksval_trustee_greedyT4T_social, pval_trustee_greedyT4T_social = ks_2samp(
		dqn_trustee_greedyT4T_social.query("game in @initial_games")['generosity'].to_numpy(),
		dqn_trustee_greedyT4T_social.query("game in @final_games")['generosity'].to_numpy())
	ksval_trustee_generousT4T_self, pval_trustee_generousT4T_self = ks_2samp(
		dqn_trustee_generousT4T_self.query("game in @initial_games")['generosity'].to_numpy(),
		dqn_trustee_generousT4T_self.query("game in @final_games")['generosity'].to_numpy())
	ksval_trustee_generousT4T_social, pval_trustee_generousT4T_social = ks_2samp(
		dqn_trustee_generousT4T_social.query("game in @initial_games")['generosity'].to_numpy(),
		dqn_trustee_generousT4T_social.query("game in @final_games")['generosity'].to_numpy())

	print("\nIs the initial distribution of generosities different than the final distribution for DQN agents?")
	print("Run a two-sample Kolmogorov-Smirnov on these distributions, report KS statistic and p_value")

	print(f"Investor vs GreedyT4T, self-oriented \t \t: KS={ksval_investor_greedyT4T_self:.4}, p={pval_investor_greedyT4T_self:.4}")
	print(f"Investor vs GreedyT4T, socially-oriented \t: KS={ksval_investor_greedyT4T_social:.4}, p={pval_investor_greedyT4T_social:.4}")
	print(f"Investor vs GenerousT4T, self-oriented \t \t: KS={ksval_investor_generousT4T_self:.4}, p={pval_investor_generousT4T_self:.4}")
	print(f"Investor vs GenerousT4T, socially-oriented \t: KS={ksval_investor_generousT4T_social:.4}, p={pval_investor_generousT4T_social:.4}")
	print(f"Trustee vs GreedyT4T, self-oriented \t \t: KS={ksval_trustee_greedyT4T_self:.4}, p={pval_trustee_greedyT4T_self:.4}")
	print(f"Trustee vs GreedyT4T, socially-oriented \t: KS={ksval_trustee_greedyT4T_social:.4}, p={pval_trustee_greedyT4T_social:.4}")
	print(f"Trustee vs GenerousT4T, self-oriented \t \t: KS={ksval_trustee_generousT4T_self:.4}, p={pval_trustee_generousT4T_self:.4}")
	print(f"Trustee vs GenerousT4T, socially-oriented \t: KS={ksval_trustee_generousT4T_social:.4}, p={pval_trustee_generousT4T_social:.4}")

	ksval_investor_greedyT4T_self, pval_investor_greedyT4T_self = ks_2samp(
		ibl_investor_greedyT4T_self.query("game in @initial_games")['generosity'].to_numpy(),
		ibl_investor_greedyT4T_self.query("game in @final_games")['generosity'].to_numpy())
	ksval_investor_greedyT4T_social, pval_investor_greedyT4T_social = ks_2samp(
		ibl_investor_greedyT4T_social.query("game in @initial_games")['generosity'].to_numpy(),
		ibl_investor_greedyT4T_social.query("game in @final_games")['generosity'].to_numpy())
	ksval_investor_generousT4T_self, pval_investor_generousT4T_self = ks_2samp(
		ibl_investor_generousT4T_self.query("game in @initial_games")['generosity'].to_numpy(),
		ibl_investor_generousT4T_self.query("game in @final_games")['generosity'].to_numpy())
	ksval_investor_generousT4T_social, pval_investor_generousT4T_social = ks_2samp(
		ibl_investor_generousT4T_social.query("game in @initial_games")['generosity'].to_numpy(),
		ibl_investor_generousT4T_social.query("game in @final_games")['generosity'].to_numpy())
	ksval_trustee_greedyT4T_self, pval_trustee_greedyT4T_self = ks_2samp(
		ibl_trustee_greedyT4T_self.query("game in @initial_games")['generosity'].to_numpy(),
		ibl_trustee_greedyT4T_self.query("game in @final_games")['generosity'].to_numpy())
	ksval_trustee_greedyT4T_social, pval_trustee_greedyT4T_social = ks_2samp(
		ibl_trustee_greedyT4T_social.query("game in @initial_games")['generosity'].to_numpy(),
		ibl_trustee_greedyT4T_social.query("game in @final_games")['generosity'].to_numpy())
	ksval_trustee_generousT4T_self, pval_trustee_generousT4T_self = ks_2samp(
		ibl_trustee_generousT4T_self.query("game in @initial_games")['generosity'].to_numpy(),
		ibl_trustee_generousT4T_self.query("game in @final_games")['generosity'].to_numpy())
	ksval_trustee_generousT4T_social, pval_trustee_generousT4T_social = ks_2samp(
		ibl_trustee_generousT4T_social.query("game in @initial_games")['generosity'].to_numpy(),
		ibl_trustee_generousT4T_social.query("game in @final_games")['generosity'].to_numpy())

	print("\nIs the initial distribution of generosities different than the final distribution for IBL agents?")
	print("Run a two-sample Kolmogorov-Smirnov on these distributions, report KS statistic and p_value")

	print(f"Investor vs GreedyT4T, self-oriented \t \t: KS={ksval_investor_greedyT4T_self:.4}, p={pval_investor_greedyT4T_self:.4}")
	print(f"Investor vs GreedyT4T, socially-oriented \t: KS={ksval_investor_greedyT4T_social:.4}, p={pval_investor_greedyT4T_social:.4}")
	print(f"Investor vs GenerousT4T, self-oriented \t \t: KS={ksval_investor_generousT4T_self:.4}, p={pval_investor_generousT4T_self:.4}")
	print(f"Investor vs GenerousT4T, socially-oriented \t: KS={ksval_investor_generousT4T_social:.4}, p={pval_investor_generousT4T_social:.4}")
	print(f"Trustee vs GreedyT4T, self-oriented \t \t: KS={ksval_trustee_greedyT4T_self:.4}, p={pval_trustee_greedyT4T_self:.4}")
	print(f"Trustee vs GreedyT4T, socially-oriented \t: KS={ksval_trustee_greedyT4T_social:.4}, p={pval_trustee_greedyT4T_social:.4}")
	print(f"Trustee vs GenerousT4T, self-oriented \t \t: KS={ksval_trustee_generousT4T_self:.4}, p={pval_trustee_generousT4T_self:.4}")
	print(f"Trustee vs GenerousT4T, socially-oriented \t: KS={ksval_trustee_generousT4T_social:.4}, p={pval_trustee_generousT4T_social:.4}")

	ksval_investor_greedyT4T_self, pval_investor_greedyT4T_self = ks_2samp(
		nef_investor_greedyT4T_self.query("game in @initial_games")['generosity'].to_numpy(),
		nef_investor_greedyT4T_self.query("game in @final_games")['generosity'].to_numpy())
	ksval_investor_greedyT4T_social, pval_investor_greedyT4T_social = ks_2samp(
		nef_investor_greedyT4T_social.query("game in @initial_games")['generosity'].to_numpy(),
		nef_investor_greedyT4T_social.query("game in @final_games")['generosity'].to_numpy())
	ksval_investor_generousT4T_self, pval_investor_generousT4T_self = ks_2samp(
		nef_investor_generousT4T_self.query("game in @initial_games")['generosity'].to_numpy(),
		nef_investor_generousT4T_self.query("game in @final_games")['generosity'].to_numpy())
	ksval_investor_generousT4T_social, pval_investor_generousT4T_social = ks_2samp(
		nef_investor_generousT4T_social.query("game in @initial_games")['generosity'].to_numpy(),
		nef_investor_generousT4T_social.query("game in @final_games")['generosity'].to_numpy())
	ksval_trustee_greedyT4T_self, pval_trustee_greedyT4T_self = ks_2samp(
		nef_trustee_greedyT4T_self.query("game in @initial_games")['generosity'].to_numpy(),
		nef_trustee_greedyT4T_self.query("game in @final_games")['generosity'].to_numpy())
	ksval_trustee_greedyT4T_social, pval_trustee_greedyT4T_social = ks_2samp(
		nef_trustee_greedyT4T_social.query("game in @initial_games")['generosity'].to_numpy(),
		nef_trustee_greedyT4T_social.query("game in @final_games")['generosity'].to_numpy())
	ksval_trustee_generousT4T_self, pval_trustee_generousT4T_self = ks_2samp(
		nef_trustee_generousT4T_self.query("game in @initial_games")['generosity'].to_numpy(),
		nef_trustee_generousT4T_self.query("game in @final_games")['generosity'].to_numpy())
	ksval_trustee_generousT4T_social, pval_trustee_generousT4T_social = ks_2samp(
		nef_trustee_generousT4T_social.query("game in @initial_games")['generosity'].to_numpy(),
		nef_trustee_generousT4T_social.query("game in @final_games")['generosity'].to_numpy())

	print("\nIs the initial distribution of generosities different than the final distribution for NEF agents?")
	print("Run a two-sample Kolmogorov-Smirnov on these distributions, report KS statistic and p_value")

	print(f"Investor vs GreedyT4T, self-oriented \t \t: KS={ksval_investor_greedyT4T_self:.4}, p={pval_investor_greedyT4T_self:.4}")
	print(f"Investor vs GreedyT4T, socially-oriented \t: KS={ksval_investor_greedyT4T_social:.4}, p={pval_investor_greedyT4T_social:.4}")
	print(f"Investor vs GenerousT4T, self-oriented \t \t: KS={ksval_investor_generousT4T_self:.4}, p={pval_investor_generousT4T_self:.4}")
	print(f"Investor vs GenerousT4T, socially-oriented \t: KS={ksval_investor_generousT4T_social:.4}, p={pval_investor_generousT4T_social:.4}")
	print(f"Trustee vs GreedyT4T, self-oriented \t \t: KS={ksval_trustee_greedyT4T_self:.4}, p={pval_trustee_greedyT4T_self:.4}")
	print(f"Trustee vs GreedyT4T, socially-oriented \t: KS={ksval_trustee_greedyT4T_social:.4}, p={pval_trustee_greedyT4T_social:.4}")
	print(f"Trustee vs GenerousT4T, self-oriented \t \t: KS={ksval_trustee_generousT4T_self:.4}, p={pval_trustee_generousT4T_self:.4}")
	print(f"Trustee vs GenerousT4T, socially-oriented \t: KS={ksval_trustee_generousT4T_social:.4}, p={pval_trustee_generousT4T_social:.4}")

def test_entropy_model_human():
	bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
	human_data = pd.read_pickle("user_data/all_users.pkl")
	final_games = [12,13,14]

	human_investor_greedyT4T_self = human_data.query('game in @final_games & player=="investor" & opponent_ID=="greedyT4T" & orientation=="self"').dropna()
	human_investor_greedyT4T_social = human_data.query('game in @final_games & player=="investor" & opponent_ID=="greedyT4T" & orientation=="social"').dropna()
	human_investor_generousT4T_self = human_data.query('game in @final_games & player=="investor" & opponent_ID=="generousT4T" & orientation=="self"').dropna()
	human_investor_generousT4T_social = human_data.query('game in @final_games & player=="investor" & opponent_ID=="generousT4T" & orientation=="social"').dropna()
	human_trustee_greedyT4T_self = human_data.query('game in @final_games & player=="trustee" & opponent_ID=="greedyT4T" & orientation=="self"').dropna()
	human_trustee_greedyT4T_social = human_data.query('game in @final_games & player=="trustee" & opponent_ID=="greedyT4T" & orientation=="social"').dropna()
	human_trustee_generousT4T_self = human_data.query('game in @final_games & player=="trustee" & opponent_ID=="generousT4T" & orientation=="self"').dropna()
	human_trustee_generousT4T_social = human_data.query('game in @final_games & player=="trustee" & opponent_ID=="generousT4T" & orientation=="social"').dropna()

	human_investor_greedyT4T_self_prob_dist = np.histogram(human_investor_greedyT4T_self['generosity'].to_numpy(), bins=bins)[0]
	human_investor_greedyT4T_social_prob_dist = np.histogram(human_investor_greedyT4T_social['generosity'].to_numpy(), bins=bins)[0]
	human_investor_generousT4T_self_prob_dist = np.histogram(human_investor_generousT4T_self['generosity'].to_numpy(), bins=bins)[0]
	human_investor_generousT4T_social_prob_dist = np.histogram(human_investor_generousT4T_social['generosity'].to_numpy(), bins=bins)[0]
	human_trustee_greedyT4T_self_prob_dist = np.histogram(human_trustee_greedyT4T_self['generosity'].to_numpy(), bins=bins)[0]
	human_trustee_greedyT4T_social_prob_dist = np.histogram(human_trustee_greedyT4T_social['generosity'].to_numpy(), bins=bins)[0]
	human_trustee_generousT4T_self_prob_dist = np.histogram(human_trustee_generousT4T_self['generosity'].to_numpy(), bins=bins)[0]
	human_trustee_generousT4T_social_prob_dist = np.histogram(human_trustee_generousT4T_social['generosity'].to_numpy(), bins=bins)[0]

	final_games = [140,141,142,143,144,145,146,147,148,149]

	f_thr = 0.1
	dqn_data = pd.read_pickle(f'agent_data/DeepQLearning_N=100_friendliness.pkl')
	ibl_data = pd.read_pickle(f'agent_data/InstanceBased_N=100_friendliness.pkl')
	nef_data = pd.read_pickle(f'agent_data/NengoQLearning_N=100_friendliness.pkl')

	dqn_investor_greedyT4T_self = dqn_data.query('game in @final_games & player=="investor" & opponent_ID=="GreedyT4T" & friendliness<@f_thr').dropna()
	dqn_investor_greedyT4T_social = dqn_data.query('game in @final_games & player=="investor" & opponent_ID=="GreedyT4T" & friendliness>=@f_thr').dropna()
	dqn_investor_generousT4T_self = dqn_data.query('game in @final_games & player=="investor" & opponent_ID=="GenerousT4T" & friendliness<@f_thr').dropna()
	dqn_investor_generousT4T_social = dqn_data.query('game in @final_games & player=="investor" & opponent_ID=="GenerousT4T" & friendliness>=@f_thr').dropna()
	dqn_trustee_greedyT4T_self = dqn_data.query('game in @final_games & player=="trustee" & opponent_ID=="GreedyT4T" & friendliness<@f_thr').dropna()
	dqn_trustee_greedyT4T_social = dqn_data.query('game in @final_games & player=="trustee" & opponent_ID=="GreedyT4T" & friendliness>=@f_thr').dropna()
	dqn_trustee_generousT4T_self = dqn_data.query('game in @final_games & player=="trustee" & opponent_ID=="GenerousT4T" & friendliness<@f_thr').dropna()
	dqn_trustee_generousT4T_social = dqn_data.query('game in @final_games & player=="trustee" & opponent_ID=="GenerousT4T" & friendliness>=@f_thr').dropna()

	dqn_investor_greedyT4T_self_prob_dist = np.histogram(dqn_investor_greedyT4T_self['generosity'].to_numpy(), bins=bins)[0]
	dqn_investor_greedyT4T_social_prob_dist = np.histogram(dqn_investor_greedyT4T_social['generosity'].to_numpy(), bins=bins)[0]
	dqn_investor_generousT4T_self_prob_dist = np.histogram(dqn_investor_generousT4T_self['generosity'].to_numpy(), bins=bins)[0]
	dqn_investor_generousT4T_social_prob_dist = np.histogram(dqn_investor_generousT4T_social['generosity'].to_numpy(), bins=bins)[0]
	dqn_trustee_greedyT4T_self_prob_dist = np.histogram(dqn_trustee_greedyT4T_self['generosity'].to_numpy(), bins=bins)[0]
	dqn_trustee_greedyT4T_social_prob_dist = np.histogram(dqn_trustee_greedyT4T_social['generosity'].to_numpy(), bins=bins)[0]
	dqn_trustee_generousT4T_self_prob_dist = np.histogram(dqn_trustee_generousT4T_self['generosity'].to_numpy(), bins=bins)[0]
	dqn_trustee_generousT4T_social_prob_dist = np.histogram(dqn_trustee_generousT4T_social['generosity'].to_numpy(), bins=bins)[0]

	ibl_investor_greedyT4T_self = ibl_data.query('game in @final_games & player=="investor" & opponent_ID=="GreedyT4T" & friendliness<@f_thr').dropna()
	ibl_investor_greedyT4T_social = ibl_data.query('game in @final_games & player=="investor" & opponent_ID=="GreedyT4T" & friendliness>=@f_thr').dropna()
	ibl_investor_generousT4T_self = ibl_data.query('game in @final_games & player=="investor" & opponent_ID=="GenerousT4T" & friendliness<@f_thr').dropna()
	ibl_investor_generousT4T_social = ibl_data.query('game in @final_games & player=="investor" & opponent_ID=="GenerousT4T" & friendliness>=@f_thr').dropna()
	ibl_trustee_greedyT4T_self = ibl_data.query('game in @final_games & player=="trustee" & opponent_ID=="GreedyT4T" & friendliness<@f_thr').dropna()
	ibl_trustee_greedyT4T_social = ibl_data.query('game in @final_games & player=="trustee" & opponent_ID=="GreedyT4T" & friendliness>=@f_thr').dropna()
	ibl_trustee_generousT4T_self = ibl_data.query('game in @final_games & player=="trustee" & opponent_ID=="GenerousT4T" & friendliness<@f_thr').dropna()
	ibl_trustee_generousT4T_social = ibl_data.query('game in @final_games & player=="trustee" & opponent_ID=="GenerousT4T" & friendliness>=@f_thr').dropna()

	ibl_investor_greedyT4T_self_prob_dist = np.histogram(ibl_investor_greedyT4T_self['generosity'].to_numpy(), bins=bins)[0]
	ibl_investor_greedyT4T_social_prob_dist = np.histogram(ibl_investor_greedyT4T_social['generosity'].to_numpy(), bins=bins)[0]
	ibl_investor_generousT4T_self_prob_dist = np.histogram(ibl_investor_generousT4T_self['generosity'].to_numpy(), bins=bins)[0]
	ibl_investor_generousT4T_social_prob_dist = np.histogram(ibl_investor_generousT4T_social['generosity'].to_numpy(), bins=bins)[0]
	ibl_trustee_greedyT4T_self_prob_dist = np.histogram(ibl_trustee_greedyT4T_self['generosity'].to_numpy(), bins=bins)[0]
	ibl_trustee_greedyT4T_social_prob_dist = np.histogram(ibl_trustee_greedyT4T_social['generosity'].to_numpy(), bins=bins)[0]
	ibl_trustee_generousT4T_self_prob_dist = np.histogram(ibl_trustee_generousT4T_self['generosity'].to_numpy(), bins=bins)[0]
	ibl_trustee_generousT4T_social_prob_dist = np.histogram(ibl_trustee_generousT4T_social['generosity'].to_numpy(), bins=bins)[0]

	nef_investor_greedyT4T_self = nef_data.query('game in @final_games & player=="investor" & opponent_ID=="GreedyT4T" & friendliness<@f_thr').dropna()
	nef_investor_greedyT4T_social = nef_data.query('game in @final_games & player=="investor" & opponent_ID=="GreedyT4T" & friendliness>=@f_thr').dropna()
	nef_investor_generousT4T_self = nef_data.query('game in @final_games & player=="investor" & opponent_ID=="GenerousT4T" & friendliness<@f_thr').dropna()
	nef_investor_generousT4T_social = nef_data.query('game in @final_games & player=="investor" & opponent_ID=="GenerousT4T" & friendliness>=@f_thr').dropna()
	nef_trustee_greedyT4T_self = nef_data.query('game in @final_games & player=="trustee" & opponent_ID=="GreedyT4T" & friendliness<@f_thr').dropna()
	nef_trustee_greedyT4T_social = nef_data.query('game in @final_games & player=="trustee" & opponent_ID=="GreedyT4T" & friendliness>=@f_thr').dropna()
	nef_trustee_generousT4T_self = nef_data.query('game in @final_games & player=="trustee" & opponent_ID=="GenerousT4T" & friendliness<@f_thr').dropna()
	nef_trustee_generousT4T_social = nef_data.query('game in @final_games & player=="trustee" & opponent_ID=="GenerousT4T" & friendliness>=@f_thr').dropna()

	nef_investor_greedyT4T_self_prob_dist = np.histogram(nef_investor_greedyT4T_self['generosity'].to_numpy(), bins=bins)[0]
	nef_investor_greedyT4T_social_prob_dist = np.histogram(nef_investor_greedyT4T_social['generosity'].to_numpy(), bins=bins)[0]
	nef_investor_generousT4T_self_prob_dist = np.histogram(nef_investor_generousT4T_self['generosity'].to_numpy(), bins=bins)[0]
	nef_investor_generousT4T_social_prob_dist = np.histogram(nef_investor_generousT4T_social['generosity'].to_numpy(), bins=bins)[0]
	nef_trustee_greedyT4T_self_prob_dist = np.histogram(nef_trustee_greedyT4T_self['generosity'].to_numpy(), bins=bins)[0]
	nef_trustee_greedyT4T_social_prob_dist = np.histogram(nef_trustee_greedyT4T_social['generosity'].to_numpy(), bins=bins)[0]
	nef_trustee_generousT4T_self_prob_dist = np.histogram(nef_trustee_generousT4T_self['generosity'].to_numpy(), bins=bins)[0]
	nef_trustee_generousT4T_social_prob_dist = np.histogram(nef_trustee_generousT4T_social['generosity'].to_numpy(), bins=bins)[0]

	print('\nHow similar are agent generosities in the final games to human generosities?')

	print("Investor plays GreedyT4T, self-oriented")
	print(f"DQN: {entropy(dqn_investor_greedyT4T_self_prob_dist, human_investor_greedyT4T_self_prob_dist):.3}")
	print(f"IBL: {entropy(ibl_investor_greedyT4T_self_prob_dist, human_investor_greedyT4T_self_prob_dist):.3}")
	print(f"NEF: {entropy(nef_investor_greedyT4T_self_prob_dist, human_investor_greedyT4T_self_prob_dist):.3}")

	print("Investor plays GreedyT4T, socially-oriented")
	print(f"DQN: {entropy(dqn_investor_greedyT4T_social_prob_dist, human_investor_greedyT4T_social_prob_dist):.3}")
	print(f"IBL: {entropy(ibl_investor_greedyT4T_social_prob_dist, human_investor_greedyT4T_social_prob_dist):.3}")
	print(f"NEF: {entropy(nef_investor_greedyT4T_social_prob_dist, human_investor_greedyT4T_social_prob_dist):.3}")

	print("Investor plays GenerousT4T, self-oriented")
	print(f"DQN: {entropy(dqn_investor_generousT4T_self_prob_dist, human_investor_generousT4T_self_prob_dist):.3}")
	print(f"IBL: {entropy(ibl_investor_generousT4T_self_prob_dist, human_investor_generousT4T_self_prob_dist):.3}")
	print(f"NEF: {entropy(nef_investor_generousT4T_self_prob_dist, human_investor_generousT4T_self_prob_dist):.3}")

	print("Investor plays GenerousT4T, socially-oriented")
	print(f"DQN: {entropy(dqn_investor_generousT4T_social_prob_dist, human_investor_generousT4T_social_prob_dist):.3}")
	print(f"IBL: {entropy(ibl_investor_generousT4T_social_prob_dist, human_investor_generousT4T_social_prob_dist):.3}")
	print(f"NEF: {entropy(nef_investor_generousT4T_social_prob_dist, human_investor_generousT4T_social_prob_dist):.3}")

	print("Trustee plays GreedyT4T, self-oriented")
	print(f"DQN: {entropy(dqn_trustee_greedyT4T_self_prob_dist, human_trustee_greedyT4T_self_prob_dist):.3}")
	print(f"IBL: {entropy(ibl_trustee_greedyT4T_self_prob_dist, human_trustee_greedyT4T_self_prob_dist):.3}")
	print(f"NEF: {entropy(nef_trustee_greedyT4T_self_prob_dist, human_trustee_greedyT4T_self_prob_dist):.3}")

	print("Trustee plays GreedyT4T, socially-oriented")
	print(f"DQN: {entropy(dqn_trustee_greedyT4T_social_prob_dist, human_trustee_greedyT4T_social_prob_dist):.3}")
	print(f"IBL: {entropy(ibl_trustee_greedyT4T_social_prob_dist, human_trustee_greedyT4T_social_prob_dist):.3}")
	print(f"NEF: {entropy(nef_trustee_greedyT4T_social_prob_dist, human_trustee_greedyT4T_social_prob_dist):.3}")

	print("Trustee plays GenerousT4T, self-oriented")
	print(f"DQN: {entropy(dqn_trustee_generousT4T_self_prob_dist, human_trustee_generousT4T_self_prob_dist):.3}")
	print(f"IBL: {entropy(ibl_trustee_generousT4T_self_prob_dist, human_trustee_generousT4T_self_prob_dist):.3}")
	print(f"NEF: {entropy(nef_trustee_generousT4T_self_prob_dist, human_trustee_generousT4T_self_prob_dist):.3}")

	print("Trustee plays GenerousT4T, socially-oriented")
	print(f"DQN: {entropy(dqn_trustee_generousT4T_social_prob_dist, human_trustee_generousT4T_social_prob_dist):.3}")
	print(f"IBL: {entropy(ibl_trustee_generousT4T_social_prob_dist, human_trustee_generousT4T_social_prob_dist):.3}")
	print(f"NEF: {entropy(nef_trustee_generousT4T_social_prob_dist, human_trustee_generousT4T_social_prob_dist):.3}")

	dqn_list = [
		entropy(dqn_investor_greedyT4T_self_prob_dist, human_investor_greedyT4T_self_prob_dist),
		entropy(dqn_investor_greedyT4T_social_prob_dist, human_investor_greedyT4T_social_prob_dist),
		entropy(dqn_investor_generousT4T_self_prob_dist, human_investor_generousT4T_self_prob_dist),
		entropy(dqn_investor_generousT4T_social_prob_dist, human_investor_generousT4T_social_prob_dist),
		entropy(dqn_trustee_greedyT4T_self_prob_dist, human_trustee_greedyT4T_self_prob_dist),
		entropy(dqn_trustee_greedyT4T_social_prob_dist, human_trustee_greedyT4T_social_prob_dist),
		entropy(dqn_trustee_generousT4T_self_prob_dist, human_trustee_generousT4T_self_prob_dist),
		entropy(dqn_trustee_generousT4T_social_prob_dist, human_trustee_generousT4T_social_prob_dist)]
	print(f"\nDQN similarity to human data across 8 groups: mean={np.mean(dqn_list):.3}, std={np.std(dqn_list):.3}")

	ibl_list = [
		entropy(ibl_investor_greedyT4T_self_prob_dist, human_investor_greedyT4T_self_prob_dist),
		entropy(ibl_investor_greedyT4T_social_prob_dist, human_investor_greedyT4T_social_prob_dist),
		entropy(ibl_investor_generousT4T_self_prob_dist, human_investor_generousT4T_self_prob_dist),
		entropy(ibl_investor_generousT4T_social_prob_dist, human_investor_generousT4T_social_prob_dist),
		entropy(ibl_trustee_greedyT4T_self_prob_dist, human_trustee_greedyT4T_self_prob_dist),
		entropy(ibl_trustee_greedyT4T_social_prob_dist, human_trustee_greedyT4T_social_prob_dist),
		entropy(ibl_trustee_generousT4T_self_prob_dist, human_trustee_generousT4T_self_prob_dist),
		entropy(ibl_trustee_generousT4T_social_prob_dist, human_trustee_generousT4T_social_prob_dist)]
	print(f"\nIBL similarity to human data across 8 groups: mean={np.mean(ibl_list):.3}, std={np.std(ibl_list):.3}")

	nef_list = [
		entropy(nef_investor_greedyT4T_self_prob_dist, human_investor_greedyT4T_self_prob_dist),
		entropy(nef_investor_greedyT4T_social_prob_dist, human_investor_greedyT4T_social_prob_dist),
		entropy(nef_investor_generousT4T_self_prob_dist, human_investor_generousT4T_self_prob_dist),
		entropy(nef_investor_generousT4T_social_prob_dist, human_investor_generousT4T_social_prob_dist),
		entropy(nef_trustee_greedyT4T_self_prob_dist, human_trustee_greedyT4T_self_prob_dist),
		entropy(nef_trustee_greedyT4T_social_prob_dist, human_trustee_greedyT4T_social_prob_dist),
		entropy(nef_trustee_generousT4T_self_prob_dist, human_trustee_generousT4T_self_prob_dist),
		entropy(nef_trustee_generousT4T_social_prob_dist, human_trustee_generousT4T_social_prob_dist)]
	print(f"\nNEF similarity to human data across 8 groups: mean={np.mean(nef_list):.3}, std={np.std(nef_list):.3}")

	dqn_list = [
		entropy(dqn_investor_greedyT4T_self_prob_dist, human_investor_greedyT4T_self_prob_dist),
		entropy(dqn_investor_greedyT4T_social_prob_dist, human_investor_greedyT4T_social_prob_dist),
		entropy(dqn_investor_generousT4T_self_prob_dist, human_investor_generousT4T_self_prob_dist),
		entropy(dqn_investor_generousT4T_social_prob_dist, human_investor_generousT4T_social_prob_dist)]
	print(f"\nDQN similarity to human data across investor groups: mean={np.mean(dqn_list):.3}, std={np.std(dqn_list):.3}")

	ibl_list = [
		entropy(ibl_investor_greedyT4T_self_prob_dist, human_investor_greedyT4T_self_prob_dist),
		entropy(ibl_investor_greedyT4T_social_prob_dist, human_investor_greedyT4T_social_prob_dist),
		entropy(ibl_investor_generousT4T_self_prob_dist, human_investor_generousT4T_self_prob_dist),
		entropy(ibl_investor_generousT4T_social_prob_dist, human_investor_generousT4T_social_prob_dist)]
	print(f"\nIBL similarity to human data across investor groups: mean={np.mean(ibl_list):.3}, std={np.std(ibl_list):.3}")

	nef_list = [
		entropy(nef_investor_greedyT4T_self_prob_dist, human_investor_greedyT4T_self_prob_dist),
		entropy(nef_investor_greedyT4T_social_prob_dist, human_investor_greedyT4T_social_prob_dist),
		entropy(nef_investor_generousT4T_self_prob_dist, human_investor_generousT4T_self_prob_dist),
		entropy(nef_investor_generousT4T_social_prob_dist, human_investor_generousT4T_social_prob_dist)]
	print(f"\nNEF similarity to human data across investor groups: mean={np.mean(nef_list):.3}, std={np.std(nef_list):.3}")

	dqn_list = [
		entropy(dqn_trustee_greedyT4T_self_prob_dist, human_trustee_greedyT4T_self_prob_dist),
		entropy(dqn_trustee_greedyT4T_social_prob_dist, human_trustee_greedyT4T_social_prob_dist),
		entropy(dqn_trustee_generousT4T_self_prob_dist, human_trustee_generousT4T_self_prob_dist),
		entropy(dqn_trustee_generousT4T_social_prob_dist, human_trustee_generousT4T_social_prob_dist)]
	print(f"\nDQN similarity to human data across trustee groups: mean={np.mean(dqn_list):.3}, std={np.std(dqn_list):.3}")

	ibl_list = [
		entropy(ibl_trustee_greedyT4T_self_prob_dist, human_trustee_greedyT4T_self_prob_dist),
		entropy(ibl_trustee_greedyT4T_social_prob_dist, human_trustee_greedyT4T_social_prob_dist),
		entropy(ibl_trustee_generousT4T_self_prob_dist, human_trustee_generousT4T_self_prob_dist),
		entropy(ibl_trustee_generousT4T_social_prob_dist, human_trustee_generousT4T_social_prob_dist)]
	print(f"\nIBL similarity to human data across trustee groups: mean={np.mean(ibl_list):.3}, std={np.std(ibl_list):.3}")

	nef_list = [
		entropy(nef_trustee_greedyT4T_self_prob_dist, human_trustee_greedyT4T_self_prob_dist),
		entropy(nef_trustee_greedyT4T_social_prob_dist, human_trustee_greedyT4T_social_prob_dist),
		entropy(nef_trustee_generousT4T_self_prob_dist, human_trustee_generousT4T_self_prob_dist),
		entropy(nef_trustee_generousT4T_social_prob_dist, human_trustee_generousT4T_social_prob_dist)]
	print(f"\nNEF similarity to human data across trustee groups: mean={np.mean(nef_list):.3}, std={np.std(nef_list):.3}")

def test_agent_orientation():
	final_games = [140,141,142,143,144,145,146,147,148,149]
	f_thr = 0.1
	dqn_data = pd.read_pickle(f'agent_data/DeepQLearning_N=100_friendliness.pkl')
	ibl_data = pd.read_pickle(f'agent_data/InstanceBased_N=100_friendliness.pkl')
	nef_data = pd.read_pickle(f'agent_data/NengoQLearning_N=100_friendliness.pkl')

	dqn_investor_greedyT4T_self = dqn_data.query('game in @final_games & player=="investor" & opponent_ID=="GreedyT4T" & friendliness<@f_thr').dropna()
	dqn_investor_greedyT4T_social = dqn_data.query('game in @final_games & player=="investor" & opponent_ID=="GreedyT4T" & friendliness>=@f_thr').dropna()
	dqn_investor_generousT4T_self = dqn_data.query('game in @final_games & player=="investor" & opponent_ID=="GenerousT4T" & friendliness<@f_thr').dropna()
	dqn_investor_generousT4T_social = dqn_data.query('game in @final_games & player=="investor" & opponent_ID=="GenerousT4T" & friendliness>=@f_thr').dropna()
	dqn_trustee_greedyT4T_self = dqn_data.query('game in @final_games & player=="trustee" & opponent_ID=="GreedyT4T" & friendliness<@f_thr').dropna()
	dqn_trustee_greedyT4T_social = dqn_data.query('game in @final_games & player=="trustee" & opponent_ID=="GreedyT4T" & friendliness>=@f_thr').dropna()
	dqn_trustee_generousT4T_self = dqn_data.query('game in @final_games & player=="trustee" & opponent_ID=="GenerousT4T" & friendliness<@f_thr').dropna()
	dqn_trustee_generousT4T_social = dqn_data.query('game in @final_games & player=="trustee" & opponent_ID=="GenerousT4T" & friendliness>=@f_thr').dropna()

	ibl_investor_greedyT4T_self = ibl_data.query('game in @final_games & player=="investor" & opponent_ID=="GreedyT4T" & friendliness<@f_thr').dropna()
	ibl_investor_greedyT4T_social = ibl_data.query('game in @final_games & player=="investor" & opponent_ID=="GreedyT4T" & friendliness>=@f_thr').dropna()
	ibl_investor_generousT4T_self = ibl_data.query('game in @final_games & player=="investor" & opponent_ID=="GenerousT4T" & friendliness<@f_thr').dropna()
	ibl_investor_generousT4T_social = ibl_data.query('game in @final_games & player=="investor" & opponent_ID=="GenerousT4T" & friendliness>=@f_thr').dropna()
	ibl_trustee_greedyT4T_self = ibl_data.query('game in @final_games & player=="trustee" & opponent_ID=="GreedyT4T" & friendliness<@f_thr').dropna()
	ibl_trustee_greedyT4T_social = ibl_data.query('game in @final_games & player=="trustee" & opponent_ID=="GreedyT4T" & friendliness>=@f_thr').dropna()
	ibl_trustee_generousT4T_self = ibl_data.query('game in @final_games & player=="trustee" & opponent_ID=="GenerousT4T" & friendliness<@f_thr').dropna()
	ibl_trustee_generousT4T_social = ibl_data.query('game in @final_games & player=="trustee" & opponent_ID=="GenerousT4T" & friendliness>=@f_thr').dropna()

	nef_investor_greedyT4T_self = nef_data.query('game in @final_games & player=="investor" & opponent_ID=="GreedyT4T" & friendliness<@f_thr').dropna()
	nef_investor_greedyT4T_social = nef_data.query('game in @final_games & player=="investor" & opponent_ID=="GreedyT4T" & friendliness>=@f_thr').dropna()
	nef_investor_generousT4T_self = nef_data.query('game in @final_games & player=="investor" & opponent_ID=="GenerousT4T" & friendliness<@f_thr').dropna()
	nef_investor_generousT4T_social = nef_data.query('game in @final_games & player=="investor" & opponent_ID=="GenerousT4T" & friendliness>=@f_thr').dropna()
	nef_trustee_greedyT4T_self = nef_data.query('game in @final_games & player=="trustee" & opponent_ID=="GreedyT4T" & friendliness<@f_thr').dropna()
	nef_trustee_greedyT4T_social = nef_data.query('game in @final_games & player=="trustee" & opponent_ID=="GreedyT4T" & friendliness>=@f_thr').dropna()
	nef_trustee_generousT4T_self = nef_data.query('game in @final_games & player=="trustee" & opponent_ID=="GenerousT4T" & friendliness<@f_thr').dropna()
	nef_trustee_generousT4T_social = nef_data.query('game in @final_games & player=="trustee" & opponent_ID=="GenerousT4T" & friendliness>=@f_thr').dropna()


	print("\nIs the final distribution of generosities different for friendly versus unfriendly agents?")
	print("Run a two-sample Kolmogorov-Smirnov on these distributions, report KS statistic and p_value")

	ksval_investor_greedyT4T, pval_investor_greedyT4T = ks_2samp(
		dqn_investor_greedyT4T_self['generosity'].to_numpy(),
		dqn_investor_greedyT4T_social['generosity'].to_numpy())
	ksval_investor_generousT4T, pval_investor_generousT4T = ks_2samp(
		dqn_investor_generousT4T_self['generosity'].to_numpy(),
		dqn_investor_generousT4T_social['generosity'].to_numpy())
	ksval_trustee_greedyT4T, pval_trustee_greedyT4T = ks_2samp(
		dqn_trustee_greedyT4T_self['generosity'].to_numpy(),
		dqn_trustee_greedyT4T_social['generosity'].to_numpy())
	ksval_trustee_generousT4T, pval_trustee_generousT4T = ks_2samp(
		dqn_trustee_generousT4T_self['generosity'].to_numpy(),
		dqn_trustee_generousT4T_social['generosity'].to_numpy())

	print("\nDQN")
	print(f"Investor vs GreedyT4T: \t \t KS={ksval_investor_greedyT4T:.4}, p={pval_investor_greedyT4T:.4}")
	print(f"Investor vs GenerousT4T: \t KS={ksval_investor_generousT4T:.4}, p={pval_investor_generousT4T:.4}")
	print(f"Trustee vs GreedyT4T: \t \t KS={ksval_trustee_greedyT4T:.4}, p={pval_trustee_greedyT4T:.4}")
	print(f"Trustee vs GenerousT4T: \t KS={ksval_trustee_generousT4T:.4}, p={pval_trustee_generousT4T:.4}")

	ksval_investor_greedyT4T, pval_investor_greedyT4T = ks_2samp(
		ibl_investor_greedyT4T_self['generosity'].to_numpy(),
		ibl_investor_greedyT4T_social['generosity'].to_numpy())
	ksval_investor_generousT4T, pval_investor_generousT4T = ks_2samp(
		ibl_investor_generousT4T_self['generosity'].to_numpy(),
		ibl_investor_generousT4T_social['generosity'].to_numpy())
	ksval_trustee_greedyT4T, pval_trustee_greedyT4T = ks_2samp(
		ibl_trustee_greedyT4T_self['generosity'].to_numpy(),
		ibl_trustee_greedyT4T_social['generosity'].to_numpy())
	ksval_trustee_generousT4T, pval_trustee_generousT4T = ks_2samp(
		ibl_trustee_generousT4T_self['generosity'].to_numpy(),
		ibl_trustee_generousT4T_social['generosity'].to_numpy())

	print("\nIBL")
	print(f"Investor vs GreedyT4T: \t \t KS={ksval_investor_greedyT4T:.4}, p={pval_investor_greedyT4T:.4}")
	print(f"Investor vs GenerousT4T: \t KS={ksval_investor_generousT4T:.4}, p={pval_investor_generousT4T:.4}")
	print(f"Trustee vs GreedyT4T: \t \t KS={ksval_trustee_greedyT4T:.4}, p={pval_trustee_greedyT4T:.4}")
	print(f"Trustee vs GenerousT4T: \t KS={ksval_trustee_generousT4T:.4}, p={pval_trustee_generousT4T:.4}")

	ksval_investor_greedyT4T, pval_investor_greedyT4T = ks_2samp(
		nef_investor_greedyT4T_self['generosity'].to_numpy(),
		nef_investor_greedyT4T_social['generosity'].to_numpy())
	ksval_investor_generousT4T, pval_investor_generousT4T = ks_2samp(
		nef_investor_generousT4T_self['generosity'].to_numpy(),
		nef_investor_generousT4T_social['generosity'].to_numpy())
	ksval_trustee_greedyT4T, pval_trustee_greedyT4T = ks_2samp(
		nef_trustee_greedyT4T_self['generosity'].to_numpy(),
		nef_trustee_greedyT4T_social['generosity'].to_numpy())
	ksval_trustee_generousT4T, pval_trustee_generousT4T = ks_2samp(
		nef_trustee_generousT4T_self['generosity'].to_numpy(),
		nef_trustee_generousT4T_social['generosity'].to_numpy())

	print("\nNEF")
	print(f"Investor vs GreedyT4T: \t \t KS={ksval_investor_greedyT4T:.4}, p={pval_investor_greedyT4T:.4}")
	print(f"Investor vs GenerousT4T: \t KS={ksval_investor_generousT4T:.4}, p={pval_investor_generousT4T:.4}")
	print(f"Trustee vs GreedyT4T: \t \t KS={ksval_trustee_greedyT4T:.4}, p={pval_trustee_greedyT4T:.4}")
	print(f"Trustee vs GenerousT4T: \t KS={ksval_trustee_generousT4T:.4}, p={pval_trustee_generousT4T:.4}")



def test_agent_behaviors():
	human_data = pd.read_pickle("user_data/all_users.pkl")
	final_games = [12,13,14]

	human_investor_generousT4T_self = human_data.query('game in @final_games & player=="investor" & opponent_ID=="generousT4T" & orientation=="self"').dropna()
	human_investor_generousT4T_social = human_data.query('game in @final_games & player=="investor" & opponent_ID=="generousT4T" & orientation=="social"').dropna()
	human_trustee_greedyT4T_self = human_data.query('game in @final_games & player=="trustee" & opponent_ID=="greedyT4T" & orientation=="self"').dropna()
	human_trustee_greedyT4T_social = human_data.query('game in @final_games & player=="trustee" & opponent_ID=="greedyT4T" & orientation=="social"').dropna()

	human_investor_generousT4T_self_percent = human_investor_generousT4T_self.query('generosity==1').size / human_investor_generousT4T_self.size
	human_investor_generousT4T_social_percent = human_investor_generousT4T_social.query('generosity==1').size / human_investor_generousT4T_social.size
	human_trustee_greedyT4T_self_percent = human_trustee_greedyT4T_self.query('generosity>=0.5').size / human_trustee_greedyT4T_self.size
	human_trustee_greedyT4T_social_percent = human_trustee_greedyT4T_social.query('generosity>=0.5').size / human_trustee_greedyT4T_social.size

	print(f'\nWhen playing the investor against a generousT4T opponent, what percentate of human participants invest all their coins?')
	print(f'{human_investor_generousT4T_self_percent:.2}% of unfriendly agents and {human_investor_generousT4T_social_percent:.2}% of friendly agents')
	print(f'\nWhen playing the trustee against a greedyT4T opponent, what percentate of human participants returned coins fairly?')
	print(f'{human_trustee_greedyT4T_self_percent:.2}% of unfriendly agents and {human_trustee_greedyT4T_social_percent:.2}% of friendly agents')

	final_games = [140,141,142,143,144,145,146,147,148,149]
	f_thr = 0.1
	dqn_data = pd.read_pickle(f'agent_data/DeepQLearning_N=100_friendliness.pkl')
	ibl_data = pd.read_pickle(f'agent_data/InstanceBased_N=100_friendliness.pkl')
	nef_data = pd.read_pickle(f'agent_data/NengoQLearning_N=100_friendliness.pkl')

	ibl_investor_generousT4T_self = ibl_data.query('game in @final_games & player=="investor" & opponent_ID=="GenerousT4T" & friendliness<@f_thr').dropna()
	ibl_investor_generousT4T_social = ibl_data.query('game in @final_games & player=="investor" & opponent_ID=="GenerousT4T" & friendliness>=@f_thr').dropna()
	ibl_trustee_greedyT4T_self = ibl_data.query('game in @final_games & player=="trustee" & opponent_ID=="GreedyT4T" & friendliness<@f_thr').dropna()
	ibl_trustee_greedyT4T_social = ibl_data.query('game in @final_games & player=="trustee" & opponent_ID=="GreedyT4T" & friendliness>=@f_thr').dropna()
	ibl_investor_generousT4T_self_percent = ibl_investor_generousT4T_self.query('generosity==1').size / ibl_investor_generousT4T_self.size
	ibl_investor_generousT4T_social_percent = ibl_investor_generousT4T_social.query('generosity==1').size / ibl_investor_generousT4T_social.size
	ibl_trustee_greedyT4T_self_percent = ibl_trustee_greedyT4T_self.query('generosity>=0.5').size / ibl_trustee_greedyT4T_self.size
	ibl_trustee_greedyT4T_social_percent = ibl_trustee_greedyT4T_social.query('generosity>=0.5').size / ibl_trustee_greedyT4T_social.size

	nef_investor_generousT4T_self = nef_data.query('game in @final_games & player=="investor" & opponent_ID=="GenerousT4T" & friendliness<@f_thr').dropna()
	nef_investor_generousT4T_social = nef_data.query('game in @final_games & player=="investor" & opponent_ID=="GenerousT4T" & friendliness>=@f_thr').dropna()
	nef_trustee_greedyT4T_self = nef_data.query('game in @final_games & player=="trustee" & opponent_ID=="GreedyT4T" & friendliness<@f_thr').dropna()
	nef_trustee_greedyT4T_social = nef_data.query('game in @final_games & player=="trustee" & opponent_ID=="GreedyT4T" & friendliness>=@f_thr').dropna()
	nef_investor_generousT4T_self_percent = nef_investor_generousT4T_self.query('generosity==1').size / nef_investor_generousT4T_self.size
	nef_investor_generousT4T_social_percent = nef_investor_generousT4T_social.query('generosity==1').size / nef_investor_generousT4T_social.size
	nef_trustee_greedyT4T_self_percent = nef_trustee_greedyT4T_self.query('generosity>=0.5').size / nef_trustee_greedyT4T_self.size
	nef_trustee_greedyT4T_social_percent = nef_trustee_greedyT4T_social.query('generosity>=0.5').size / nef_trustee_greedyT4T_social.size

	print(f'\nWhen playing the investor against a generousT4T opponent, what percentate of IBL agents invest all their coins?')
	print(f'{ibl_investor_generousT4T_self_percent:.2}% of unfriendly agents and {ibl_investor_generousT4T_social_percent:.2}% of friendly agents')
	print(f'\nWhen playing the trustee against a greedyT4T opponent, what percentate of IBL agents returned coins fairly?')
	print(f'{ibl_trustee_greedyT4T_self_percent:.2}% of unfriendly agents and {ibl_trustee_greedyT4T_social_percent:.2}% of friendly agents')

	print(f'\nWhen playing the investor against a generousT4T opponent, what percentate of NEF agents invest all their coins?')
	print(f'{nef_investor_generousT4T_self_percent:.5}% of unfriendly agents and {nef_investor_generousT4T_social_percent:.5}% of friendly agents')
	print(f'\nWhen playing the trustee against a greedyT4T opponent, what percentate of NEF agents returned coins fairly?')
	print(f'{nef_trustee_greedyT4T_self_percent:.2}% of unfriendly agents and {nef_trustee_greedyT4T_social_percent:.2}% of friendly agents')


# test_agent_learning()
# test_entropy_model_human()
# test_agent_orientation()
# test_agent_behaviors()


