import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import os
from scipy.stats import ks_2samp, ttest_ind, entropy
from scipy.ndimage import histogram
from scipy.spatial.distance import jensenshannon

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
	axes[0][0].set(ylabel="Proself\ngenerosity")
	axes[1][0].set(ylabel="Prosocial\ngenerosity")
	plt.tight_layout()
	fig.savefig(f"plots/plot_final_generosities_svo_{agent}.pdf", bbox_inches="tight", pad_inches=0)

def compare_final_generosities(agents=['Human'], games=3):
	fmt = '%.0f%%'
	yticks = mtick.FormatStrFormatter(fmt)
	fig, axes = plt.subplots(nrows=2, ncols=4, figsize=((7,4)), sharey=True, sharex=True)
	for agent in agents:
		if agent=="Random": continue
		if agent=="Human":
			data = pd.read_pickle("human_data/human_data.pkl")
			fill = True
			lw = 0.1
			color = 'k'
		if agent=="DQN":
			data = pd.read_pickle(f'agent_data/DQN_N=300_games=500_svo.pkl')
			fill = False
			lw = 2
			color = palette[0]
		if agent=="IBL":
			data = pd.read_pickle(f'agent_data/IBL_N=300_games=200_svo.pkl')
			fill = False
			lw = 2
			color = palette[1]
		if agent=="TQ":
			data = pd.read_pickle(f'agent_data/TQ_N=300_games=200_svo.pkl')
			fill = False
			lw = 2
			color = palette[1]
		last_game = data['game'].unique().max()
		final_games = np.arange(last_game-(games-1), last_game+1)
		for i, orientation in enumerate(["proself", "prosocial"]):
			for j, player in enumerate(["investor", "trustee"]):
				for k, opponent in enumerate(["greedy", "generous"]):
					query_string = "player==@player & opponent==@opponent & orientation==@orientation & game in @final_games"
					subdata = data.query(query_string)
					sns.histplot(data=subdata, x='generosity', stat="percent", color=color, label=agent, ax=axes[i][2*j+k],
						fill=fill, lw=lw, element="poly", binwidth=0.1, binrange=[0,1])
					axes[i][2*j+k].set(xticks=((0,1)), yticks=((0, 100)))
					if i==0: axes[i][2*j+k].set(title=f"{player} vs {opponent}")
	axes[0][0].legend()
	axes[0][0].set(ylabel="proself")
	axes[1][0].set(ylabel="prosocial")
	axes[0][0].yaxis.set_major_formatter(yticks)
	axes[0][1].yaxis.set_major_formatter(yticks)
	plt.tight_layout()
	fig.savefig(f"plots/compare_final_generosities.pdf", bbox_inches="tight", pad_inches=0)

def compare_convergence(agents=['Human'], load=False, last_n_games=3, metric='KS'):
	if not load:
		dfs = []
		columns = ('agent', 'ID', 'player', 'opponent', 'orientation', 'game', 'sim_final')
		for agent in agents:
			if agent=="Human": data = pd.read_pickle("human_data/human_data.pkl")
			if agent=="Random": data = pd.read_pickle("agent_data/DQN_N=100_games=100_svo_random.pkl")
			if agent=="TQ": data = pd.read_pickle("agent_data/TQ_N=300_games=200_svo.pkl")
			if agent=="DQN": data = pd.read_pickle(f'agent_data/DQN_N=300_games=500_svo.pkl')
			if agent=="IBL": data = pd.read_pickle(f'agent_data/IBL_N=300_games=200_svo.pkl')
			last_game = data['game'].unique().max()
			final_games = np.arange(last_game-(last_n_games-1), last_game+1)
			for player in ["investor", "trustee"]:
				for opponent in ["greedy", "generous"]:
					for orientation in ["proself", "prosocial"]:
						query_string = "player==@player & opponent==@opponent & orientation==@orientation"
						group_data = data.query(query_string)
						individuals = group_data['ID'].unique()
						for ID in individuals:
							individual_data = group_data.query("ID == @ID")
							final_games_string = "game in @final_games"
							final_gens = individual_data.query(final_games_string).dropna()['generosity'].to_numpy()
							if metric=='JS':
								final_histogram = histogram(final_gens, min=0, max=1, bins=11) / len(final_gens)
							for game in range(last_game+1):
								game_data = individual_data.query("game == @game")
								current_gens = game_data.dropna()['generosity'].to_numpy()
								if metric=='JS':
									current_histogram = histogram(current_gens, min=0, max=1, bins=11) / len(current_gens)
									sim_final = 1 - jensenshannon(current_histogram, final_histogram)
								if metric=='KS':
									sim_final = 1 - ks_2samp(current_gens, final_gens)[0]
								rescaled_game = game / (last_game+1)
								dfs.append(pd.DataFrame([[agent, ID, player, opponent, orientation, rescaled_game, sim_final]], columns=columns))
		data = pd.concat(dfs, ignore_index=True)
		data.to_pickle(f"analysis_data/convergence.pkl")
		print(data)
	else:
		data = pd.read_pickle(f"analysis_data/convergence.pkl")

	fig, axes = plt.subplots(nrows=2, ncols=4, figsize=((7,4)), sharey=True, sharex=True)
	for agent in ["Human", "TQ", "DQN", "IBL", "Random"]:
		if agent=="Human": color = 'k'
		if agent=="DQN": color = palette[0]
		if agent=="IBL": color = palette[1]
		if agent=="TQ": color = palette[3]
		if agent=="Random": color = palette[4]
		for i, orientation in enumerate(["proself", "prosocial"]):
			for j, player in enumerate(["investor", "trustee"]):
				for k, opponent in enumerate(["greedy", "generous"]):
					query_string = "player==@player & opponent==@opponent & orientation==@orientation & agent==@agent"
					subdata = data.query(query_string)
					sns.lineplot(data=subdata, x='game', y='sim_final', color=color, label=agent, ax=axes[i][2*j+k])
					axes[i][2*j+k].set(xticks=((0,1)), yticks=((0, 1)))
					axes[i][2*j+k].set_xticklabels(['first', 'last'])
					if i==0: axes[i][2*j+k].set(title=f"{player} vs {opponent}")
	axes[0][0].legend()
	axes[0][1].get_legend().remove()
	axes[0][2].get_legend().remove()
	axes[0][3].get_legend().remove()
	axes[1][0].get_legend().remove()
	axes[1][1].get_legend().remove()
	axes[1][2].get_legend().remove()
	axes[1][3].get_legend().remove()
	axes[0][0].set(ylabel="proself\nsimilarity to final gen.")
	axes[1][0].set(ylabel="prosocial\nsimilarity to final gen.")
	plt.tight_layout()
	fig.savefig(f"plots/compare_convergence.pdf", bbox_inches="tight", pad_inches=0)