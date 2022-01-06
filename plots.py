import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
palette = sns.color_palette("deep")
sns.set(context='paper', style='white', font='CMU Serif', rc={'font.size':12, 'mathtext.fontset': 'cm'})
sns.set_palette(palette)

# def plot_one_game(data):
# 	fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2, figsize=((6, 2)))
# 	sns.lineplot(data=data, x='turn', y='generosity', hue='player', ax=ax)
# 	sns.lineplot(data=data, x='turn', y='coins', hue='player', ax=ax2)
# 	ax.set(ylim=((0, 1.05)), yticks=((0, 1)))
# 	ax2.get_legend().remove()
# 	plt.tight_layout()
# 	sns.despine(top=True, right=True)
# 	fig.savefig('plots/one_game.pdf')

# def plot_many_games(data, learner_plays, name):
# 	fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2, figsize=((6, 2)))
# 	sns.lineplot(data=data, x='game', y='generosity', hue='player', ax=ax)
# 	sns.lineplot(data=data, x='game', y='coins', hue='player', ax=ax2)
# 	# sns.lineplot(data=data, x='game', y='generosity', hue='player', style='ID', ax=ax)
# 	# sns.lineplot(data=data, x='game', y='coins', hue='player', style='ID', ax=ax2)
# 	if learner_plays=='investor':
# 		ax2.set(ylim=((-1, 16)), yticks=((0,5,10,15)))
# 	else:
# 		ax2.set(ylim=((-1, 31)), yticks=((0,5,10,15,20,25,30)))	
# 	# ax.get_legend().remove()
# 	ax2.get_legend().remove()
# 	plt.tight_layout()
# 	sns.despine(top=True, right=True)
# 	fig.savefig(f'plots/{name}_{learner_plays}.pdf')

def plot_learning_and_policy(data, learner_plays, learner_name, opponent_name, human=False):
	palette = sns.color_palette('deep')
	data = data.query("player==@learner_plays")
	data_learning = data.query("train==True") if not human else data
	final_games = np.max(data.query('train==True')['game']) - int(np.max(data.query('train==True')['game']) * 0.1)
	data_final = data.query("train==True & game>=@final_games").dropna() if not human else data.query('game>=@final_games').dropna()
	generosities = data_final.query('player==@learner_plays')['generosity'].to_numpy()
	n_learners = len(data['ID'].unique())
	n_total = len(generosities)
	n_nans = np.count_nonzero(np.isnan(generosities))  # count "nan" generosities (which indicate a skipped turn) in the data
	generosities = generosities[~np.isnan(generosities)]  # remove these entries from the data
	ylim = ((-0.1, 1.1))# if learner_plays=='investor' else ((-0.1, 0.6))
	yticks = ((0, 1))# if learner_plays=='investor' else ((0, 0.5))
	fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2, figsize=((6, 2)), sharey=False)
	sns.kdeplot(data=data_learning, x='game', y='generosity', color=palette[0],
		bw_method=0.15, levels=5, thresh=0.2, fill=True, ax=ax)
	ax.set(ylim=ylim, yticks=yticks, title="learning")
	sns.histplot(data_final, x='turn', y='generosity', stat='density', color=palette[1],
		binwidth=[1, 0.2], binrange=[[0,5],[0,1]], thresh=0.05, ax=ax2)
	ax2.set(xlim=((0, 5)), xticks=((1,2,3,4,5)), ylim=((0, 1)), yticks=((0, 1)))
	for x in np.arange(5): ax2.axvline(x, color='gray', alpha=0.5, linewidth=0.5)
	for y in np.arange(0, 1.1, 0.2): ax2.axhline(y, color='gray', alpha=0.5, linewidth=0.5)
	plt.tight_layout()
	fig.savefig(f'plots/{learner_name}_as_{learner_plays}_versus_{opponent_name}_N={n_learners}.pdf')
	plt.close('all')

# def plot_learning_and_policy_agent_friendliness(data, learners, learner_plays, learner_name, opponent_name, human=False):
# 	final_games = np.max(data['game']) - int(np.max(data['game']) * 0.1)
# 	data_final = data.query("game>=@final_games").dropna()
# 	for friendliness in ['self', 'social']:
# 		group = []
# 		for learner in learners:
# 			if friendliness=='self' and learner.friendliness==0:
# 				group.append(learner.ID)
# 			elif friendliness=='social' and learner.friendliness>0:
# 				group.append(learner.ID)
# 		data_group = data.query('ID.isin(@group)')
# 		data_final_group = data_final.query('ID.isin(@group)')
# 		generosities = data_final_group['generosity'].to_numpy()
# 		n_learners = len(data_group['ID'].unique())
# 		n_total = len(generosities)
# 		n_nans = np.count_nonzero(np.isnan(generosities))  # count "nan" generosities (which indicate a skipped turn) in the data
# 		generosities = generosities[~np.isnan(generosities)]  # remove these entries from the data
# 		ylim = ((-0.1, 1.1))# if learner_plays=='investor' else ((-0.1, 0.6))
# 		yticks = ((0, 1))# if learner_plays=='investor' else ((0, 0.5))
# 		fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2, figsize=((6, 2)), sharey=False)
# 		sns.kdeplot(data=data_group, x='game', y='generosity', bw_method=0.15, levels=5, thresh=0.2, fill=True, ax=ax)
# 		ax.set(ylim=ylim, yticks=yticks, title="learning")
# 		if np.std(generosities)>1e-1:
# 			sns.kdeplot(data=data_final_group, x='turn', y='generosity', bw_method=0.15, levels=5, thresh=0.1, fill=True, ax=ax2)
# 			ax2.set(xlim=((-0.5, 4.5)), xticks=((0,1,2,3,4)), ylim=ylim, yticks=yticks, title="final policy")
# 		else:
# 			bins = np.arange(0, 1.1, 0.1)# if learner_plays=='investor' else np.arange(0, 0.6, 0.1)
# 			sns.histplot(data=data_final_group, x='generosity', bins=bins, stat='probability', ax=ax2)
# 			ax2.set(xticks=((0, 1)), xlim=((0, 1)), ylim=((0, ylim[1])), yticks=(()), ylabel='', title="final policy")
# 		sns.despine(top=True, right=True)
# 		plt.tight_layout()
# 		fig.savefig(f'plots/{friendliness}_{learner_name}_as_{learner_plays}_versus_{opponent_name}_N={n_learners}.pdf')
# 		plt.close('all')

# def plot_learning_and_policy_agent_friendliness(data, learners):
# 	n_games = np.max(data['game']) + 1
# 	final_games = n_games - int(n_games * 0.1)
# 	fig, axes = plt.subplots(nrows=4, ncols=4, figsize=((8, 8)))
# 	bins = np.arange(0, 1.1, 0.1)
# 	xlim1 = ((-100, 1100))
# 	xlim2 = ((-1, 5))
# 	ylim = ((-0.1, 1.1))
# 	yticks = ((0, 1))
# 	palette = sns.color_palette('deep')
# 	greedy_background = '#edffef'
# 	generous_background = '#faebe8'
# 	for i, opponent in enumerate(['GreedyT4T', 'GenerousT4T']):
# 		for j, player in enumerate(['investor', 'trustee']):
# 			for k, friendliness in enumerate(['self', 'social']):
# 				friendliness_group = []
# 				for learner in learners:
# 					if friendliness=='self' and learner.friendliness==0:
# 						friendliness_group.append(learner.ID)
# 					elif friendliness=='social' and learner.friendliness>0:
# 						friendliness_group.append(learner.ID)
# 				ax = axes[2*i+j][2*k]
# 				ax2 = axes[2*i+j][2*k+1]
# 				if j==1:
# 					ax.fill_between(xlim1, ylim[0], ylim[1], hatch='...', edgecolor='#bfbfbf', facecolor=greedy_background if i==0 else generous_background)
# 					ax2.fill_between(xlim2, ylim[0], ylim[1], hatch='...', edgecolor='#bfbfbf', facecolor=greedy_background if i==0 else generous_background)
# 				data_group = data.query('player==@player & opponent==@opponent & ID.isin(@friendliness_group)')
# 				data_final_group = data_group.query('game>=@final_games').dropna()
# 				n_learners = len(data_group['ID'].unique())
# 				generosities = data_final_group['generosity'].to_numpy()
# 				generosities = generosities[~np.isnan(generosities)]  # remove these entries from the data
# 				sns.kdeplot(data=data_group, x='game', y='generosity', color=palette[0], bw_method=0.15, levels=5, thresh=0.2, fill=True, ax=ax)
# 				ax.set(xlim=xlim1, xticks=(()), xlabel='', ylim=ylim, yticks=(()), ylabel='', facecolor=greedy_background if i==0 else generous_background)
# 				if np.std(generosities)>0:
# 					sns.kdeplot(data=data_final_group, x='turn', y='generosity', color=palette[1], bw_method=0.15, levels=5, thresh=0.1, fill=True, ax=ax2)
# 				else:
# 					ax2.scatter([0,1,2,3,4], np.mean(generosities)*np.ones((5)), color=palette[1], s=50)
# 				ax2.set(xlim=xlim2, xticks=(()), xlabel='', ylim=ylim, yticks=(()), ylabel='', facecolor=greedy_background if i==0 else generous_background)
# 	axes[0][0].set(ylabel='generosity', yticks=((0, 1)), title='self-oriented')
# 	axes[0][2].set(title='socially-oriented')
# 	axes[1][0].set(ylabel='generosity', yticks=((0, 1)))
# 	axes[2][0].set(ylabel='generosity', yticks=((0, 1)))
# 	axes[3][0].set(ylabel='generosity', yticks=((0, 1)), xlabel='game', xticks=((0, n_games)))
# 	axes[3][2].set(xlabel='game', xticks=((0, n_games)))
# 	axes[3][1].set(xlabel='turn', xticks=((0,1,2,3,4)))
# 	axes[3][3].set(xlabel='turn', xticks=((0,1,2,3,4)))
# 	plt.tight_layout()
# 	fig.savefig(f'plots/agent_friendliness_learning_policy.pdf')
# 	fig.savefig(f'plots/agent_friendliness_learning_policy.svg')
# 	plt.close('all')


def plot_learning_and_policy_agent_friendliness(data, learners, learner_type, thr_friendliness=0.1):
	n_games = np.max(data['game']) + 1
	final_games = n_games - int(n_games * 0.1)
	fig, axes = plt.subplots(nrows=4, ncols=4, figsize=((6.5, 6.5)))
	xlim1 = ((-100, n_games+100))
	xlim2 = ((0, 5))
	ylim = ((-0.1, 1.1))
	ylim2 = ((0, 1))
	yticks = ((0, 1))
	palette = sns.color_palette('deep')
	greedy_background = '#edffef'
	generous_background = '#faebe8'
	xbins = np.arange(5)
	ybins = np.arange(0, 1.1, 0.2)
	for i, opponent in enumerate(['GreedyT4T', 'GenerousT4T']):
		for j, player in enumerate(['investor', 'trustee']):
			for k, friendliness in enumerate(['self', 'social']):
				friendliness_group = []
				for learner in learners:
					if friendliness=='self' and learner.friendliness<=thr_friendliness:
						friendliness_group.append(learner.ID)
					elif friendliness=='social' and learner.friendliness>thr_friendliness:
						friendliness_group.append(learner.ID)
				ax = axes[2*i+j][2*k]
				ax2 = axes[2*i+j][2*k+1]
				plot_color = palette[0] if friendliness=='self' else palette[1]
				frame_color = palette[3] if opponent=='GreedyT4T' else palette[2]
				background_color = '' if player=='investor' else '..'
				ax.fill_between(xlim1, ylim[0], ylim[1], hatch=background_color, edgecolor='#bfbfbf', facecolor='white')
				ax2.fill_between(xlim2, ylim[0], ylim[1], hatch=background_color, edgecolor='#bfbfbf', facecolor='white')
				for spine in ax.spines.values():
					spine.set_linewidth(2)
					spine.set_edgecolor(frame_color)
				for spine in ax2.spines.values():
					spine.set_linewidth(2)
					spine.set_edgecolor(frame_color)
				data_group = data.query('player==@player & opponent==@opponent & ID.isin(@friendliness_group)')
				data_final_group = data_group.query('game>=@final_games').dropna()
				n_learners = len(data_group['ID'].unique())
				sns.kdeplot(data=data_group, x='game', y='generosity', color=plot_color,
					bw_method=0.1, levels=6, thresh=0.05, fill=True, ax=ax)
				ax.set(xlim=xlim1, xticks=(()), xlabel='', ylim=ylim, yticks=(()), ylabel='')
				sns.histplot(data_final_group, x='turn', y='generosity', stat='density', color=plot_color,
					binwidth=[1, 0.2], binrange=[[0,5],[0,1]], thresh=0.05, ax=ax2)
				ax2.set(xlim=xlim2, xticks=(()), xlabel='', ylim=ylim2, yticks=(()), ylabel='')
				for x in xbins: ax2.axvline(x, color='gray', alpha=0.5, linewidth=0.5)
				for y in ybins: ax2.axhline(y, color='gray', alpha=0.5, linewidth=0.5)

	axes[0][0].set(ylabel='generosity', yticks=((0, 1)))
	axes[1][0].set(ylabel='generosity', yticks=((0, 1)))
	axes[2][0].set(ylabel='generosity', yticks=((0, 1)))
	axes[3][0].set(ylabel='generosity', yticks=((0, 1)), xlabel='game', xticks=((0, n_games)))
	axes[3][2].set(xlabel='game', xticks=((0, n_games)))
	axes[3][1].set(xlabel='turn', xticks=((1,2,3,4,5)))
	axes[3][3].set(xlabel='turn', xticks=((1,2,3,4,5)))
	plt.tight_layout()
	fig.savefig(f'plots/{learner_type}_friendliness_learning_policy.pdf')
	fig.savefig(f'plots/{learner_type}_friendliness_learning_policy.svg')
	plt.close('all')


def plot_learning_and_policy_human_friendliness(data):
	n_games = np.max(data['game']) + 1
	final_games = n_games - int(n_games * 0.1)
	fig, axes = plt.subplots(nrows=4, ncols=4, figsize=((6.5, 6.5)))
	xlim1 = ((-2, 16))
	xlim2 = ((0, 5))
	ylim = ((-0.1, 1.1))
	ylim2 = ((0, 1))
	yticks = ((0, 1))
	palette = sns.color_palette('deep')
	greedy_background = '#edffef'
	generous_background = '#faebe8'
	xbins = np.arange(5)
	ybins = np.arange(0, 1.1, 0.2)
	for i, opponent in enumerate(['greedyT4T', 'generousT4T']):
		for j, player in enumerate(['investor', 'trustee']):
			for k, orientation in enumerate(['self', 'social']):
				ax = axes[2*i+j][2*k]
				ax2 = axes[2*i+j][2*k+1]
				plot_color = palette[0] if orientation=='self' else palette[1]
				frame_color = palette[3] if opponent=='greedyT4T' else palette[2]
				background_color = '' if player=='investor' else '..'
				ax.fill_between(xlim1, ylim[0], ylim[1], hatch=background_color, edgecolor='#bfbfbf', facecolor='white')
				ax2.fill_between(xlim2, ylim[0], ylim[1], hatch=background_color, edgecolor='#bfbfbf', facecolor='white')
				for spine in ax.spines.values():
					spine.set_linewidth(2)
					spine.set_edgecolor(frame_color)
				for spine in ax2.spines.values():
					spine.set_linewidth(2)
					spine.set_edgecolor(frame_color)
				data_group = data.query('player==@player & opponent_ID==@opponent & orientation==@orientation')
				data_final_group = data_group.query('game>=@final_games').dropna()
				n_learners = len(data_group['ID'].unique())
				sns.kdeplot(data=data_group, x='game', y='generosity', color=plot_color,
					bw_method=0.1, levels=6, thresh=0.05, fill=True, ax=ax)
				ax.set(xlim=xlim1, xticks=(()), xlabel='', ylim=ylim, yticks=(()), ylabel='')
				sns.histplot(data_final_group, x='turn', y='generosity', stat='density', color=plot_color,
					binwidth=[1, 0.2], binrange=[[0,5],[0,1]], thresh=0.05, ax=ax2)
				ax2.set(xlim=xlim2, xticks=(()), xlabel='', ylim=ylim2, yticks=(()), ylabel='')
				for x in xbins: ax2.axvline(x, color='gray', alpha=0.5, linewidth=0.5)
				for y in ybins: ax2.axhline(y, color='gray', alpha=0.5, linewidth=0.5)
	axes[0][0].set(ylabel='generosity', yticks=((0, 1)))
	axes[1][0].set(ylabel='generosity', yticks=((0, 1)))
	axes[2][0].set(ylabel='generosity', yticks=((0, 1)))
	axes[3][0].set(ylabel='generosity', yticks=((0, 1)), xlabel='game', xticks=((0, max(data['game']))))
	axes[3][2].set(xlabel='game', xticks=((0, max(data['game']))))
	axes[3][1].set(xlabel='turn', xticks=((1,2,3,4,5)))
	axes[3][3].set(xlabel='turn', xticks=((1,2,3,4,5)))
	plt.tight_layout()
	fig.savefig(f'plots/human_friendliness_learning_policy.pdf')
	fig.savefig(f'plots/human_friendliness_learning_policy.svg')
	plt.close('all')

# def plot_learning(data, learner_plays, learner_name, opponent_name, human=False):
# 	data_train = data.query("train==True") if not human else data
# 	fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2, figsize=((6, 2)))
# 	sns.kdeplot(data=data_train.query("player==@learner_plays"), x='game', y='generosity',
# 		bw_method=0.1, levels=5, thresh=0.2, fill=True, ax=ax)
# 	sns.kdeplot(data=data_train.query("player==@learner_plays"), x='game', y='coins',
# 		bw_method=0.1, levels=5, thresh=0.2, fill=True, ax=ax2)
# 	ax.set(ylim=((-0.1, 1.1)), yticks=((0, 1)))
# 	if learner_plays=='investor':
# 		ax2.set(ylim=((-1, 16)), yticks=((0,5,10,15)))
# 	else:
# 		ax2.set(ylim=((-1, 31)), yticks=((0,5,10,15,20,25,30)))	
# 	plt.tight_layout()
# 	fig.savefig(f'plots/{learner_name}_as_{learner_plays}_versus_{opponent_name}_learning.pdf')

# def plot_policy(data, learner_plays, learner_name, opponent_name, human=False):
# 	data_test = data.query("train==False") if not human else data.query('game>=12')
# 	generosities = data_test.query('player==@learner_plays')['generosity'].to_numpy()
# 	n_total = len(generosities)
# 	n_nans = np.count_nonzero(np.isnan(generosities))  # count "nan" generosities (which indicate a skipped turn) in the data
# 	generosities = generosities[~np.isnan(generosities)]  # remove these entries from the data
# 	data = data_test.query('player==@learner_plays').dropna()
# 	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=((2, 2)))
# 	if np.std(data_test.query('player==@learner_plays')['generosity'].to_numpy())>1e-5:
# 		sns.kdeplot(data=data, x='turn', y='generosity',
# 			bw_method=0.1, levels=5, thresh=0.1, fill=True, ax=ax)
# 		ax.set(xlim=((-0.5, 4.5)), xticks=((0,1,2,3,4)), ylim=((-0.1, 1.1)), yticks=((0, 1)), title=f'N={n_total-n_nana}')
# 		# title=f'skips: {n_nans}/{n_total}'
# 	else:
# 		bins = np.arange(0, 1.1, 0.1)
# 		sns.histplot(data=data_test.query('player==@learner_plays'), x='generosity', bins=bins, stat='probability', ax=ax)
# 		ax.set(xticks=((0, 1)), xlim=((0, 1)), ylim=((0, 1)), yticks=((0, 1)), title=f'N={n_total-n_nana}')
# 	plt.tight_layout()
# 	sns.despine(top=True, right=True)
# 	fig.savefig(f'plots/{learner_name}_as_{learner_plays}_versus_{opponent_name}_policy.pdf')

# def plot_metrics(data_gen, data_score, learner_plays, learner_name, opponent_name):
# 	metrics_gen = data_gen.columns[2:]
# 	metrics_score = data_score.columns[2:]
# 	metrics_main = ['mean', 'std', 'adapt', 'cooperate', 'defect', 'gift', 'attrition']
# 	metrics_etc = ['skew', 'kurtosis', 'learn', 'speed']

# 	fig, axes = plt.subplots(nrows=1, ncols=len(metrics_main), figsize=((len(metrics_main)*2, 2)), sharey=True)
# 	for i, metric in enumerate(metrics_main):
# 		if metric in ['mean', 'adapt', 'cooperate', 'defect', 'gift', 'attrition']:
# 			sns.histplot(data=data_gen, x=metric, bins=np.arange(0, 1.1, 0.1), stat='probability', ax=axes[i])
# 			axes[i].set(title=metric, xlim=((0, 1)), xticks=((0, 0.5, 1)))
# 		elif metric in ['std']:
# 			sns.histplot(data=data_gen, x=metric, bins=np.arange(0, 0.55, 0.05), stat='probability', ax=axes[i])
# 			axes[i].set(title=metric, xlim=((0, 0.5)), xticks=((0, 0.5)))
# 	fig.savefig(f'plots/{learner_name}_as_{learner_plays}_versus_{opponent_name}_metrics.pdf')
# 	plt.close('all')