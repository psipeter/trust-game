import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='paper', style='white', font='CMU Serif', rc={'font.size':12, 'mathtext.fontset': 'cm'})

def plot_one_game(data):
	fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2, figsize=((6, 2)))
	sns.lineplot(data=data, x='turn', y='generosity', hue='player', ax=ax)
	sns.lineplot(data=data, x='turn', y='coins', hue='player', ax=ax2)
	ax.set(ylim=((0, 1.05)), yticks=((0, 1)))
	ax2.get_legend().remove()
	plt.tight_layout()
	sns.despine(top=True, right=True)
	fig.savefig('plots/one_game.pdf')

def plot_many_games(data, learner_plays):
	fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2, figsize=((6, 2)))
	sns.lineplot(data=data, x='game', y='generosity', hue='player', ax=ax)
	sns.lineplot(data=data, x='game', y='coins', hue='player', ax=ax2)
	# sns.lineplot(data=data, x='game', y='generosity', hue='player', style='ID', ax=ax)
	# sns.lineplot(data=data, x='game', y='coins', hue='player', style='ID', ax=ax2)
	ax.set(ylim=((0, 1)), yticks=((0, 1)))
	ax2.set(ylim=((0, 15)), yticks=((0,5,10,15)))
	# ax.get_legend().remove()
	ax2.get_legend().remove()
	plt.tight_layout()
	sns.despine(top=True, right=True)
	fig.savefig(f'plots/many_games_learner_plays_{learner_plays}.pdf')

def plot_learning(data, learner_plays, name):
	if name=='all':
		data_train = data.query("phase=='train'")
	else:
		data_train = data.query("phase=='train' & ID==@name")
	fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2, figsize=((6, 2)))
	sns.kdeplot(data=data_train.query("player==@learner_plays"), x='game', y='generosity', bw_method=0.1, levels=5, thresh=0.2, fill=True, ax=ax)
	sns.kdeplot(data=data_train.query("player==@learner_plays"), x='game', y='coins', bw_method=0.1, levels=5, thresh=0.2, fill=True, ax=ax2)
	ax.set(ylim=((-0.1, 1.1)), yticks=((0, 1)))
	if learner_plays=='investor':
		ax2.set(ylim=((-1, 16)), yticks=((0,5,10,15)))
	else:
		ax2.set(ylim=((-1, 31)), yticks=((0,5,10,15,20,25,30)))		
	plt.tight_layout()
	sns.despine(top=True, right=True)
	fig.savefig(f'plots/learning_{learner_plays}_{name}.pdf')

def plot_policy(data, learner_plays, name):
	if name=='all':
		data_test = data.query("phase=='test'")
	else:
		data_test = data.query("phase=='test' & ID==@name")
	# dfs = []
	# columns = ('ID', 'opponent_ID', 'player', 'turn', 'my_generosity', 'opponent_generosity')
	# for index, row in data_test.iterrows():
	# 	player = row['player']
	# 	if row['turn']>0 and player==learner_plays:
	# 		ID = row['ID']
	# 		opponent_ID = row['opponent_ID']
	# 		game = row['game']
	# 		turn = row['turn']
	# 		last_turn = turn-1
	# 		my_generosity = row['generosity']
	# 		opponent_generosity = data.query('phase=="test" & opponent_ID==@opponent_ID & game==@game & turn==@last_turn')['generosity'].to_numpy()[0]
	# 		df = pd.DataFrame([[ID, opponent_ID, player, turn, my_generosity, opponent_generosity]], columns=columns)
	# 		dfs.append(df)
	# forgive_punish_data = pd.concat([df for df in dfs], ignore_index=True)

	if np.std(data_test.query('player==@learner_plays')['generosity'].to_numpy())>0.05:
		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=((2, 2)))
		sns.kdeplot(data=data_test.query('player==@learner_plays'), x='turn', y='generosity', bw_method=0.1, levels=5, thresh=0.1, fill=True, ax=ax)
		ax.set(xlim=((-0.5, 4.5)), xticks=((0,1,2,3,4)), ylim=((-0.1, 1.1)), yticks=((0, 1)))
		plt.tight_layout()
		sns.despine(top=True, right=True)
		fig.savefig(f'plots/turn_policy_{learner_plays}_{name}.pdf')
	else:
		bins = np.arange(0, 1.1, 0.1)
		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=((2, 2)))
		sns.histplot(data=data_test.query('player==@learner_plays'), x='generosity', bins=bins, stat='probability', ax=ax)
		ax.set(xticks=((0, 1)), xlim=((0, 1)), ylim=((0, 1)), yticks=((0, 1)))
		plt.tight_layout()
		sns.despine(top=True, right=True)
		fig.savefig(f'plots/turn_policy_{learner_plays}_{name}.pdf')

	# try:
	# 	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=((2, 2)))
	# 	sns.kdeplot(data=forgive_punish_data, x='opponent_generosity', y='my_generosity', bw_method=0.1, levels=5, fill=True, ax=ax)
	# 	ax.set(xlim=((-0.1, 1.1)), xticks=((0, 1)), ylim=((-0.1, 1.1)), yticks=((0, 1)))
	# 	plt.tight_layout()
	# 	sns.despine(top=True, right=True)
	# 	fig.savefig(f'plots/responsiveness_policy_{learner_plays}_{name}.pdf')
	# except:
	# 	bins = np.arange(0, 1.1, 0.1)
	# 	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=((2, 2)))
	# 	sns.histplot(data=forgive_punish_data, x='opponent_generosity', bins=bins, stat='probability', ax=ax)
	# 	ax.set(xlim=((-0.1, 1.1)), xticks=((0, 1)), ylim=((0, 1)), yticks=((0, 1)))
	# 	plt.tight_layout()
	# 	sns.despine(top=True, right=True)
	# 	fig.savefig(f'plots/responsiveness_policy_{learner_plays}_{name}.pdf')

def plot_metrics(data_gen, data_score, learner_plays):
	metrics_gen = data_gen.columns[2:]
	metrics_score = data_score.columns[2:]

	fig, axes = plt.subplots(nrows=1, ncols=len(metrics_gen), figsize=((len(metrics_gen)*2, 2)), sharey=True)
	axes[0].set(ylim=((0, 1)), yticks=((0, 1)))
	for i, metric in enumerate(metrics_gen):
		if metric in ['mean', 'speed', 'adapt', 'cooperate', 'defect', 'gift', 'attrition']:
			sns.histplot(data=data_gen, x=metric, bins=np.arange(0, 1.1, 0.1), stat='probability', ax=axes[i])
			axes[i].set(title=metric, xlim=((0, 1)), xticks=((0, 0.5, 1)))
		elif metric in ['std']:
			sns.histplot(data=data_gen, x=metric, bins=np.arange(0, 0.55, 0.05), stat='probability', ax=axes[i])
			axes[i].set(title=metric, xlim=((0, 0.5)), xticks=((0, 0.5)))
		elif metric in ['skew']:
			sns.histplot(data=data_gen, x=metric, bins=np.arange(-0.4, 0.4, 0.04), stat='probability', ax=axes[i])
			axes[i].set(title=metric, xlim=((-0.4, 0.4)), xticks=((-0.4, 0, 0.4)))
		elif metric in ['kurtosis']:
			sns.histplot(data=data_gen, x=metric, bins=np.arange(-2, 2, 0.2), stat='probability', ax=axes[i])
			axes[i].set(title=metric, xlim=((-2, 2)), xticks=((-2, 0, 2)))
		else:
			sns.histplot(data=data_gen, x=metric, stat='probability', ax=axes[i])
			axes[i].set(title=metric)
	fig.savefig(f'plots/generosity_metrics_{learner_plays}.pdf')

	fig, axes = plt.subplots(nrows=1, ncols=len(metrics_score), figsize=((len(metrics_score)*2, 2)), sharey=True)
	axes[0].set(ylim=((0, 1)), yticks=((0, 1)))
	for i, metric in enumerate(metrics_score):
		if metric in ['mean', 'adapt']:
			sns.histplot(data=data_score, x=metric, bins=np.arange(0, 31, 1), stat='probability', ax=axes[i])
			axes[i].set(title=metric, xticks=((0, 10, 15, 30)), xlim=((0, 30)))
		else:
			sns.histplot(data=data_score, x=metric, bins=10, stat='probability', ax=axes[i])
			axes[i].set(title=metric)
	fig.savefig(f'plots/score_metrics_{learner_plays}.pdf')