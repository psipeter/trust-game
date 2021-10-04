import numpy as np
import random
import pandas as pd
import tensorflow as tf
from utils import *
from plots import *

def play_tournament(investors, trustees, tournament_type):
	for investor in investors: assert investor.player == 'investor', "invalid investor assignment"
	for trustee in trustees: assert trustee.player == 'trustee', "invalid trustee assignment"

	if tournament_type == 'one_game':
		dfs = []
		columns = ('player', 'turn', 'generosity', 'coins')
		game = Game()
		investor = investors[0]
		trustee = trustees[0]
		play_game(game, investor, trustee)
		for t in range(game.turns):
			dfs.append(pd.DataFrame([['investor', t, game.investor_gen[t], game.investor_reward[t]]], columns=columns))			
			dfs.append(pd.DataFrame([['trustee', t, game.trustee_gen[t], game.trustee_reward[t]]], columns=columns))			
		data = pd.concat([df for df in dfs], ignore_index=True)
		plot_one_game(data)

	if tournament_type == 'many_games':
		nGames = 200
		investor = investors[0]
		trustee = trustees[0]
		dfs = []
		columns = ('ID', 'player', 'game', 'turn', 'generosity', 'coins')
		for g in range(nGames):
			game = Game()
			play_game(game, investor, trustee)
			for t in range(game.turns):
				dfs.append(pd.DataFrame([[investor.ID, 'investor', g, t, game.investor_gen[t], game.investor_reward[t]]], columns=columns))			
				dfs.append(pd.DataFrame([[trustee.ID, 'trustee', g, t, game.trustee_gen[t], game.trustee_reward[t]]], columns=columns))			
		data = pd.concat([df for df in dfs], ignore_index=True)
		plot_many_games(investor, trustee, data)

	if tournament_type == 'identical_learning_investors_vs_t4t':
		nGames = 300
		trustee = trustees[0]
		dfs = []
		columns = ('ID', 'player', 'game', 'turn', 'generosity', 'coins')
		for investor in investors:
			for g in range(nGames):
				game = Game()
				play_game(game, investor, trustee)
				for t in range(game.turns):
					dfs.append(pd.DataFrame([[investor.ID, 'investor', g, t, game.investor_gen[t], game.investor_reward[t]]], columns=columns))			
					dfs.append(pd.DataFrame([[trustee.ID, 'trustee', g, t, game.trustee_gen[t], game.trustee_reward[t]]], columns=columns))			
		data = pd.concat([df for df in dfs], ignore_index=True)
		plot_many_games(data)

	if tournament_type == 'identical_learning_trustees_vs_t4t':
		nGames = 300
		investor = investors[0]
		dfs = []
		columns = ('ID', 'player', 'game', 'turn', 'generosity', 'coins')
		for trustee in trustees:
			for g in range(nGames):
				game = Game()
				play_game(game, investor, trustee)
				for t in range(game.turns):
					dfs.append(pd.DataFrame([[investor.ID, 'investor', g, t, game.investor_gen[t], game.investor_reward[t]]], columns=columns))			
					dfs.append(pd.DataFrame([[trustee.ID, 'trustee', g, t, game.trustee_gen[t], game.trustee_reward[t]]], columns=columns))			
		data = pd.concat([df for df in dfs], ignore_index=True)
		plot_many_games(data)