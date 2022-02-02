import numpy as np
import pandas as pd
import os
from scipy.stats import ks_2samp, ttest_ind, entropy

from utils import *
from plots import *

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


def test_human_returns():
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
# test_human_returns()



def test_agent_learning():
	initial_games = [0,1,2,3,4,5,6,7,8,9]
	final_games = [140,141,142,143,144,145,146,147,148,149]

	f_thr = 0.1
	dqn_data = pd.read_pickle(f'agent_data/DeepQLearning_N=100_friendliness.pkl')
	ibl_data = pd.read_pickle(f'agent_data/InstanceBased_N=100_friendliness.pkl')
	nef_data = pd.read_pickle(f'agent_data/NengoQLearning_N=3_friendliness.pkl')

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
	nef_data = pd.read_pickle(f'agent_data/NengoQLearning_N=3_friendliness.pkl')

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
	print(f"DQN: {entropy(dqn_investor_greedyT4T_self_prob_dist, human_investor_greedyT4T_self_prob_dist):.3}")
	print(f"IBL: {entropy(ibl_investor_greedyT4T_self_prob_dist, human_investor_greedyT4T_self_prob_dist):.3}")
	print(f"NEF: {entropy(nef_investor_greedyT4T_self_prob_dist, human_investor_greedyT4T_self_prob_dist):.3}")

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
	print(f"DQN: {entropy(dqn_trustee_greedyT4T_self_prob_dist, human_trustee_greedyT4T_self_prob_dist):.3}")
	print(f"IBL: {entropy(ibl_trustee_greedyT4T_self_prob_dist, human_trustee_greedyT4T_self_prob_dist):.3}")
	print(f"NEF: {entropy(nef_trustee_greedyT4T_self_prob_dist, human_trustee_greedyT4T_self_prob_dist):.3}")

	print("Trustee plays GenerousT4T, socially-oriented")
	print(f"DQN: {entropy(dqn_trustee_generousT4T_social_prob_dist, human_trustee_generousT4T_social_prob_dist):.3}")
	print(f"IBL: {entropy(ibl_trustee_generousT4T_social_prob_dist, human_trustee_generousT4T_social_prob_dist):.3}")
	print(f"NEF: {entropy(nef_trustee_generousT4T_social_prob_dist, human_trustee_generousT4T_social_prob_dist):.3}")



# test_agent_learning()
test_entropy_model_human()
