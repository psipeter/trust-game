import numpy as np
import pandas as pd
import os

from utils import *
from plots import *
from experiments import make_learners

def plot_user_data():
	dfs = []
	for filename in os.listdir("user_data"):
		human = filename.split("_")[0]
		player = filename.split("_")[1].split(".")[0]
		df = pd.read_pickle(f"user_data/{filename}")
		opponent = df['opponent_ID'][0]
		orientation = df['orientation'][0]
		dfs.append(df)
		# plot_learning(df, learner_plays=player, learner_name=human, opponent_name=opponent, human=True)
		# plot_policy(df, learner_plays=player, learner_name=human, opponent_name=opponent, human=True)
		# plt.close('all')
	df = pd.concat([df for df in dfs], ignore_index=True)
	plot_learning_and_policy_human_friendliness(df)
	# print(df.to_string())
	# for player in ['investor', 'trustee']:
	# 	for opponent in ['greedyT4T', 'generousT4T']:
	# 		for orientation in ['social', 'self']:
	# 			df_group = df.query('player==@player & opponent_ID==@opponent & orientation==@orientation')
	# 			# print(df_group.to_string())
	# 			plot_learning_and_policy(df_group, player, f"{orientation}Human", opponent, True)

def plot_agent_data(filename, learner_type, seed, n_learners):
	# generates IDs that are used when querying the dataframe in plot()
	learners = make_learners(learner_type, seed, n_learners)
	df = pd.read_pickle(filename)
	plot_learning_and_policy_agent_friendliness(df, learners)