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

def plot_many_games(data):
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
	fig.savefig('plots/many_games.pdf')