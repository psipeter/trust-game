import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import os
from scipy.stats import ks_2samp, ttest_ind, entropy

palette = sns.color_palette("colorblind")
sns.set(context='paper', style='white', font='CMU Serif', rc={'font.size':12, 'mathtext.fontset': 'cm', 'axes.linewidth': 0.1})
sns.set_palette(palette)

def plot_trajectories_generosities_baseline(data, agent):
	last_game = data['game'].unique().max()
	fig, axes = plt.subplots(nrows=2, ncols=4, figsize=((7,4)), sharey=True, sharex=True)
	for i, player in enumerate(["investor", "trustee"]):
		for j, opponent in enumerate(["cooperate", "defect", "gift", "attrition"]):
			query_string = "player==@player & opponent==@opponent"
			subdata = data.query(query_string)
			sns.kdeplot(data=subdata, x='game', y='generosity', bw_method=0.1, levels=6, thresh=0.05, fill=True, ax=axes[i][j])
			axes[i][j].set(xticks=((1, last_game+1)), yticks=((0,1)), ylim=((0,1)), title=f"{player} vs {opponent}")
	plt.tight_layout()
	fig.savefig(f"plots/plot_trajectories_generosities_baseline_{agent}.pdf", bbox_inches="tight", pad_inches=0)

def plot_final_generosities_baseline(data, agent, games=3, turn=True):
	last_game = data['game'].unique().max()
	final_games = np.arange(last_game-(games-1), last_game+1)
	fig, axes = plt.subplots(nrows=2, ncols=4, figsize=((7,4)), sharey=True, sharex=True)
	for i, player in enumerate(["investor", "trustee"]):
		for j, opponent in enumerate(["cooperate", "defect", "gift", "attrition"]):
			query_string = "player==@player & opponent==@opponent & game in @final_games"
			subdata = data.query(query_string)
			if turn:
				sns.histplot(subdata, x='turn', y='generosity', stat='density', binwidth=[1, 0.2], binrange=[[0,5],[0,1]], thresh=0.05, ax=axes[i][j])
				axes[i][j].set(xticks=((1,2,3,4,5)), xlim=((0,5)), yticks=((0,1)), ylim=((0,1)), title=f"{player} vs {opponent}")
				for x in [0,1,2,3,4]: axes[i][j].axvline(x, color='k', alpha=0.5, linewidth=0.5)	
				for y in [0,0.2,0.4,0.6,0.8,1.0]: axes[i][j].axhline(y, color='k', alpha=0.5, linewidth=0.5)	
			else:
				sns.histplot(data=subdata, x='generosity', stat="percent", fill=False, lw=2, element="poly", binwidth=0.1, binrange=[0,1], ax=axes[i][j])
				axes[i][j].set(xticks=((0,1)), yticks=((0, 100)), title=f"{player} vs {opponent}")
	plt.tight_layout()
	fig.savefig(f"plots/plot_final_generosities_baseline_{agent}.pdf", bbox_inches="tight", pad_inches=0)

def plot_final_generosities_svo(data, agent, games=3, turn=True):
	last_game = data['game'].unique().max()
	final_games = np.arange(last_game-(games-1), last_game+1)
	fig, axes = plt.subplots(nrows=2, ncols=4, figsize=((7,4)), sharey=True, sharex=True)
	for i, orientation in enumerate(["proself", "prosocial"]):
		for j, player in enumerate(["investor", "trustee"]):
			for k, opponent in enumerate(["greedy", "generous"]):
				query_string = "player==@player & opponent==@opponent & orientation==@orientation & game in @final_games"
				subdata = data.query(query_string)
				if turn:
					sns.histplot(subdata, x='turn', y='generosity', stat='density', binwidth=[1, 0.2], binrange=[[0,5],[0,1]], thresh=0.05, ax=axes[i][2*j+k])
					axes[i][2*j+k].set(xticks=((1,2,3,4,5)), xlim=((0,5)), yticks=((0,1)), ylim=((0,1)), title=f"{player} vs {opponent}")
					for x in [0,1,2,3,4]: axes[i][2*j+k].axvline(x, color='k', alpha=0.5, linewidth=0.5)	
					for y in [0,0.2,0.4,0.6,0.8,1.0]: axes[i][2*j+k].axhline(y, color='k', alpha=0.5, linewidth=0.5)							
				else:
					sns.histplot(data=subdata, x='generosity', stat="percent", fill=False, lw=2, element="poly", binwidth=0.1, binrange=[0,1], ax=axes[i][2*j+k])
					axes[i][2*j+k].set(xticks=((0,1)), yticks=((0, 100)), title=f"{player} vs {opponent}")
	plt.tight_layout()
	fig.savefig(f"plots/plot_final_generosities_svo_{agent}.pdf", bbox_inches="tight", pad_inches=0)

def histograms_final_generosity_all(games=3, thr_friendliness=0.1):

	# plot human data in background
	human_data = pd.read_pickle("user_data/all_users.pkl")
	last_game = human_data['game'].unique().max()
	human_final_games = np.arange(last_game-(games-1), last_game+1)
	human_proself_investor_vs_greedy = human_data.query('player=="investor" & opponent_ID=="greedyT4T" & orientation=="self" & game in @human_final_games').dropna()
	human_prosocial_investor_vs_greedy = human_data.query('player=="investor" & opponent_ID=="greedyT4T" & orientation=="social" & game in @human_final_games').dropna()
	human_proself_investor_vs_generous = human_data.query('player=="investor" & opponent_ID=="generousT4T" & orientation=="self" & game in @human_final_games').dropna()
	human_prosocial_investor_vs_generous = human_data.query('player=="investor" & opponent_ID=="generousT4T" & orientation=="social" & game in @human_final_games').dropna()
	human_proself_trustee_vs_greedy = human_data.query('player=="trustee" & opponent_ID=="greedyT4T" & orientation=="self" & game in @human_final_games').dropna()
	human_prosocial_trustee_vs_greedy = human_data.query('player=="trustee" & opponent_ID=="greedyT4T" & orientation=="social" & game in @human_final_games').dropna()
	human_proself_trustee_vs_generous = human_data.query('player=="trustee" & opponent_ID=="generousT4T" & orientation=="self" & game in @human_final_games').dropna()
	human_prosocial_trustee_vs_generous = human_data.query('player=="trustee" & opponent_ID=="generousT4T" & orientation=="social" & game in @human_final_games').dropna()

	fig, axes = plt.subplots(nrows=2, ncols=4, figsize=((7,4)), sharey=False, sharex=True)
	sns.histplot(data=human_proself_investor_vs_greedy, x='generosity', stat="percent", fill=True, lw=0.1, element="poly", binwidth=0.1, binrange=[0,1], ax=axes[0][0], color='k', label="Human")
	sns.histplot(data=human_proself_investor_vs_generous, x='generosity', stat="percent", fill=True, lw=0.1, element="poly", binwidth=0.1, binrange=[0,1], ax=axes[0][1], color='k')
	sns.histplot(data=human_proself_trustee_vs_greedy, x='generosity', stat="percent", fill=True, lw=0.1, element="poly", binwidth=0.1, binrange=[0,1], ax=axes[0][2], color='k')
	sns.histplot(data=human_proself_trustee_vs_generous, x='generosity', stat="percent", fill=True, lw=0.1, element="poly", binwidth=0.1, binrange=[0,1], ax=axes[0][3], color='k')
	sns.histplot(data=human_prosocial_investor_vs_greedy, x='generosity', stat="percent", fill=True, lw=0.1, element="poly", binwidth=0.1, binrange=[0,1], ax=axes[1][0], color='k')
	sns.histplot(data=human_prosocial_investor_vs_generous, x='generosity', stat="percent", fill=True, lw=0.1, element="poly", binwidth=0.1, binrange=[0,1], ax=axes[1][1], color='k')
	sns.histplot(data=human_prosocial_trustee_vs_greedy, x='generosity', stat="percent", fill=True, lw=0.1, element="poly", binwidth=0.1, binrange=[0,1], ax=axes[1][2], color='k')
	sns.histplot(data=human_prosocial_trustee_vs_generous, x='generosity', stat="percent", fill=True, lw=0.1, element="poly", binwidth=0.1, binrange=[0,1], ax=axes[1][3], color='k')
	i = 0
	for agent in ["DQN", "IBL", "SPA"]:
		if agent=="DQN": data = pd.read_pickle(f'agent_data/DeepQLearning_N=100_friendliness.pkl')
		if agent=="IBL": data = pd.read_pickle(f'agent_data/InstanceBased_N=100_friendliness.pkl')
		if agent=="SPA": data = pd.read_pickle(f'agent_data/NengoQLearning_N=100_friendliness.pkl')
		last_game = data['game'].unique().max()
		final_games = np.arange(last_game-(games-1), last_game+1)
		proself_investor_vs_greedy = data.query('player=="investor" & opponent_ID=="GreedyT4T" & friendliness<@thr_friendliness & game in @final_games').dropna()
		prosocial_investor_vs_greedy = data.query('player=="investor" & opponent_ID=="GreedyT4T" & friendliness>@thr_friendliness & game in @final_games').dropna()
		proself_investor_vs_generous = data.query('player=="investor" & opponent_ID=="GenerousT4T" & friendliness<@thr_friendliness & game in @final_games').dropna()
		prosocial_investor_vs_generous = data.query('player=="investor" & opponent_ID=="GenerousT4T" & friendliness>@thr_friendliness & game in @final_games').dropna()
		proself_trustee_vs_greedy = data.query('player=="trustee" & opponent_ID=="GreedyT4T" & friendliness<@thr_friendliness & game in @final_games').dropna()
		prosocial_trustee_vs_greedy = data.query('player=="trustee" & opponent_ID=="GreedyT4T" & friendliness>@thr_friendliness & game in @final_games').dropna()
		proself_trustee_vs_generous = data.query('player=="trustee" & opponent_ID=="GenerousT4T" & friendliness<@thr_friendliness & game in @final_games').dropna()
		prosocial_trustee_vs_generous = data.query('player=="trustee" & opponent_ID=="GenerousT4T" & friendliness>@thr_friendliness & game in @final_games').dropna()
		sns.histplot(data=proself_investor_vs_greedy, x='generosity', stat="percent", fill=False, lw=2, element="poly", binwidth=0.1, binrange=[0,1], ax=axes[0][0], color=palette[i], label=agent)
		sns.histplot(data=proself_investor_vs_generous, x='generosity', stat="percent", fill=False, lw=2, element="poly", binwidth=0.1, binrange=[0,1], ax=axes[0][1], color=palette[i])
		sns.histplot(data=proself_trustee_vs_greedy, x='generosity', stat="percent", fill=False, lw=2, element="poly", binwidth=0.1, binrange=[0,1], ax=axes[0][2], color=palette[i])
		sns.histplot(data=proself_trustee_vs_generous, x='generosity', stat="percent", fill=False, lw=2, element="poly", binwidth=0.1, binrange=[0,1], ax=axes[0][3], color=palette[i])
		sns.histplot(data=prosocial_investor_vs_greedy, x='generosity', stat="percent", fill=False, lw=2, element="poly", binwidth=0.1, binrange=[0,1], ax=axes[1][0], color=palette[i])
		sns.histplot(data=prosocial_investor_vs_generous, x='generosity', stat="percent", fill=False, lw=2, element="poly", binwidth=0.1, binrange=[0,1], ax=axes[1][1], color=palette[i])
		sns.histplot(data=prosocial_trustee_vs_greedy, x='generosity', stat="percent", fill=False, lw=2, element="poly", binwidth=0.1, binrange=[0,1], ax=axes[1][2], color=palette[i])
		sns.histplot(data=prosocial_trustee_vs_generous, x='generosity', stat="percent", fill=False, lw=2, element="poly", binwidth=0.1, binrange=[0,1], ax=axes[1][3], color=palette[i])
		i += 1
	# ylims = plt.gca().get_ylim()
	# fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
	# yticks = mtick.FormatStrFormatter(fmt)
	# axes[0][0].yaxis.set_major_formatter(yticks)
	# axes[0][1].yaxis.set_major_formatter(yticks)
	# axes[0][2].yaxis.set_major_formatter(yticks)
	# axes[0][3].yaxis.set_major_formatter(yticks)
	# axes[1][0].yaxis.set_major_formatter(yticks)
	# axes[1][1].yaxis.set_major_formatter(yticks)
	# axes[1][2].yaxis.set_major_formatter(yticks)
	# axes[1][3].yaxis.set_major_formatter(yticks)
	axes[0][0].set(yticks=(()), xticks=((0,1)), ylabel="Proself\nPercent", title="Investor vs. Greedy")
	axes[0][1].set(yticks=(()), xticks=((0,1)), ylabel=None, title="Investor vs. Generous")
	axes[0][2].set(yticks=(()), xticks=((0,1)), ylabel=None, title="Trustee vs. Greedy")
	axes[0][3].set(yticks=(()), xticks=((0,1)), ylabel=None, title="Trustee vs. Generous")
	axes[1][0].set(yticks=(()), xticks=((0,1)), ylabel="Prosocial\nPercent")
	axes[1][1].set(yticks=(()), xticks=((0,1)), ylabel=None)
	axes[1][2].set(yticks=(()), xticks=((0,1)), ylabel=None)
	axes[1][3].set(yticks=(()), xticks=((0,1)), ylabel=None)
	axes[0][0].legend()
	plt.tight_layout()
	# sns.despine(left=True, right=True, top=True)
	fig.savefig(f"plots/histograms_final_generosity_all.svg")
	fig.savefig(f"plots/histograms_final_generosity_all.pdf")

# histograms_final_generosity_all()

def lineplots_learning_similarity(load=False, games=3, thr=0.1):
	if not load:
		dfs = []
		columns = ('agent', 'ID', 'player', 'opponent', 'orientation', 'game', 'generosities', 'sim_final')
		for agent in ["Human", "DQN", "IBL", "SPA"]:
			if agent=="Human": data = pd.read_pickle("user_data/all_users.pkl")
			if agent=="DQN": data = pd.read_pickle(f'agent_data/DeepQLearning_N=100_friendliness.pkl')
			if agent=="IBL": data = pd.read_pickle(f'agent_data/InstanceBased_N=100_friendliness.pkl')
			if agent=="SPA": data = pd.read_pickle(f'agent_data/NengoQLearning_N=100_friendliness.pkl')
			last_game = data['game'].unique().max()
			final_games = np.arange(last_game-(games-1), last_game+1)
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
						group_string = player_string + ' & ' + opponent_string + ' & ' + orientation_string
						group_data = data.query(group_string)
						individuals = group_data['ID'].unique()
						for ID in individuals:
							id_string = "ID == @ID"
							individual_data = group_data.query(id_string)
							final_games_string = "game in @final_games"
							generosities_final = individual_data.query(final_games_string).dropna()['generosity'].to_numpy()
							for game in range(last_game+1):
								game_string = "game == @game"
								game_data = individual_data.query(game_string)
								generosities = game_data.dropna()['generosity'].to_numpy()
								sim_final = ks_2samp(generosities, generosities_final)[0]
								rescaled_game = game / (last_game+1)
								dfs.append(pd.DataFrame([[
									agent, ID, player, opponent, orientation, rescaled_game, generosities, sim_final]],
									columns=columns))
		data = pd.concat(dfs, ignore_index=True)
		data.to_pickle("analysis_data/dynamics.pkl")
		print(data)
	else:
		data = pd.read_pickle("analysis_data/dynamics.pkl")

	fig, axes = plt.subplots(nrows=2, ncols=4, figsize=((7,4)), sharey=True, sharex=True)
	subdata = data.query("agent=='Human'")
	sns.lineplot(data=subdata.query("player=='investor' & opponent=='greedy' & orientation=='proself'"), x="game", y='sim_final', ax=axes[0][0], color='k', label="Human")
	sns.lineplot(data=subdata.query("player=='investor' & opponent=='generous' & orientation=='proself'"), x="game", y='sim_final', ax=axes[0][1], color='k')
	sns.lineplot(data=subdata.query("player=='trustee' & opponent=='greedy' & orientation=='proself'"), x="game", y='sim_final', ax=axes[0][2], color='k')
	sns.lineplot(data=subdata.query("player=='trustee' & opponent=='generous' & orientation=='proself'"), x="game", y='sim_final', ax=axes[0][3], color='k')
	sns.lineplot(data=subdata.query("player=='investor' & opponent=='greedy' & orientation=='prosocial'"), x="game", y='sim_final', ax=axes[1][0], color='k')
	sns.lineplot(data=subdata.query("player=='investor' & opponent=='generous' & orientation=='prosocial'"), x="game", y='sim_final', ax=axes[1][1], color='k')
	sns.lineplot(data=subdata.query("player=='trustee' & opponent=='greedy' & orientation=='prosocial'"), x="game", y='sim_final', ax=axes[1][2], color='k')
	sns.lineplot(data=subdata.query("player=='trustee' & opponent=='generous' & orientation=='prosocial'"), x="game", y='sim_final', ax=axes[1][3], color='k')
	i = 0
	for agent in ["DQN", "IBL", "SPA"]:
		subdata = data.query("agent==@agent")
		sns.lineplot(data=subdata.query("player=='investor' & opponent=='greedy' & orientation=='proself'"), x="game", y='sim_final', ax=axes[0][0], color=palette[i], label=agent)
		sns.lineplot(data=subdata.query("player=='investor' & opponent=='generous' & orientation=='proself'"), x="game", y='sim_final', ax=axes[0][1], color=palette[i])
		sns.lineplot(data=subdata.query("player=='trustee' & opponent=='greedy' & orientation=='proself'"), x="game", y='sim_final', ax=axes[0][2], color=palette[i])
		sns.lineplot(data=subdata.query("player=='trustee' & opponent=='generous' & orientation=='proself'"), x="game", y='sim_final', ax=axes[0][3], color=palette[i])
		sns.lineplot(data=subdata.query("player=='investor' & opponent=='greedy' & orientation=='prosocial'"), x="game", y='sim_final', ax=axes[1][0], color=palette[i])
		sns.lineplot(data=subdata.query("player=='investor' & opponent=='generous' & orientation=='prosocial'"), x="game", y='sim_final', ax=axes[1][1], color=palette[i])
		sns.lineplot(data=subdata.query("player=='trustee' & opponent=='greedy' & orientation=='prosocial'"), x="game", y='sim_final', ax=axes[1][2], color=palette[i])
		sns.lineplot(data=subdata.query("player=='trustee' & opponent=='generous' & orientation=='prosocial'"), x="game", y='sim_final', ax=axes[1][3], color=palette[i])
		i += 1
	axes[0][0].set(yticks=((0, 1)), xticks=((0, 1)), xlabel='game', ylabel="Proself\nSimilarity to Final Gen.", title="Investor vs. Greedy")
	axes[0][1].set(yticks=((0, 1)), xticks=((0, 1)), xlabel='game', ylabel=None, title="Investor vs. Generous")
	axes[0][2].set(yticks=((0, 1)), xticks=((0, 1)), xlabel='game', ylabel=None, title="Trustee vs. Greedy")
	axes[0][3].set(yticks=((0, 1)), xticks=((0, 1)), xlabel='game', ylabel=None, title="Trustee vs. Generous")
	axes[1][0].set(yticks=((0, 1)), xticks=((0, 1)), xlabel='game', ylabel="Prosocial\nSimilarity to Final Gen.")
	axes[1][1].set(yticks=((0, 1)), xticks=((0, 1)), xlabel='game', ylabel=None)
	axes[1][2].set(yticks=((0, 1)), xticks=((0, 1)), xlabel='game', ylabel=None)
	axes[1][3].set(yticks=((0, 1)), xticks=((0, 1)), xlabel='game', ylabel=None)
	axes[0][0].set_xticklabels(['first', 'last'])
	axes[0][1].set_xticklabels(['first', 'last'])
	axes[0][2].set_xticklabels(['first', 'last'])
	axes[0][3].set_xticklabels(['first', 'last'])
	axes[1][0].set_xticklabels(['first', 'last'])
	axes[1][1].set_xticklabels(['first', 'last'])
	axes[1][2].set_xticklabels(['first', 'last'])
	axes[1][3].set_xticklabels(['first', 'last'])
	axes[0][0].legend()
	plt.tight_layout()
	fig.savefig(f"plots/lineplots_learning_similarity.svg", bbox_inches="tight", pad_inches=0)
	fig.savefig(f"plots/lineplots_learning_similarity.pdf", bbox_inches="tight", pad_inches=0)


# lineplots_learning_similarity(False)
