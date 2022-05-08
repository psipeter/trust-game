import numpy as np
import random
import pandas as pd
import torch
import nengo
from scipy.stats import entropy, skew, kurtosis, normaltest

def get_state(player, game, agent, dim=0, turn_basis=None, coin_basis=None, representation="one-hot"):
	t = len(game.investor_give) if player=='investor' else len(game.trustee_give)
	if agent=='TQ':
		index = t if player=='investor' else t * (game.coins*game.match+1) + game.investor_give[-1]*game.match
		return index
	if agent=='DQN':
		if representation=="one-hot":
			index = t if player=='investor' else t * (game.coins*game.match+1) + game.investor_give[-1]*game.match
			vector = np.zeros((dim))
			vector[index] = 1
			return torch.FloatTensor(vector)
		elif representation=="ssp":
			c = game.coins if player=='investor' else game.investor_give[-1]*game.match
			vector = encode_state(t, c, turn_basis, coin_basis)
			return torch.FloatTensor(vector)
	if agent=="IBL":
		if player=="investor":
			return t, game.coins
		else:
			return t, game.investor_give[-1]*game.match
	if agent=="SPA":
		if representation=="ssp":
			c = game.coins if player=='investor' else game.investor_give[-1]*game.match
			vector = encode_state(t, c, turn_basis, coin_basis) if t<5 else np.zeros((dim))
			return vector

def action_to_coins(player, state, n_actions, game):
	available = game.coins if player=='investor' else game.investor_give[-1]*game.match  # coins available
	precise_give = state * available
	possible_actions = np.linspace(0, available, n_actions).astype(int)
	action_idx = (np.abs(possible_actions - precise_give)).argmin()
	action = possible_actions[action_idx]
	give = action
	keep = available - action
	return give, keep, action_idx

def generosity(player, give, keep):
	return np.NaN if give+keep==0 and player=='trustee' else give/(give+keep)

def encode_state(t, c, turn_basis, coin_basis, turn_exp=1.0, coin_exp=1.0):
	return np.fft.ifft(turn_basis**(t*turn_exp) * coin_basis**(c*coin_exp)).real.squeeze()

def make_unitary(v):
	return v/np.absolute(v)

def measure_sparsity(spikes1, spikes2):
	n_neurons = spikes1.shape[0]
	diff = []
	quiet = 0
	for n in range(n_neurons):
		if spikes1[n]+spikes2[n]>0:
			diff.append((spikes1[n]-spikes2[n]) / (spikes1[n]+spikes2[n]))
		else:
			quiet += 1
	diff = np.array(diff)
	quiet = quiet / n_neurons
	pdiff = (np.histogram(diff)[0][0] + np.histogram(diff)[0][-1]) / diff.shape[0]
	return 100*pdiff, 100*quiet

def measure_similarity(ssp1, ssp2, mode="cosine"):
    if mode=="dot":
        return np.sum(ssp1 * ssp2)
    elif mode=="cosine":
        return np.sum(ssp1 * ssp2) / (np.linalg.norm(ssp1, ord=2) * np.linalg.norm(ssp2, ord=2))

def get_rewards(player, svo, game, normalize, gamma):
	rewards_self = game.investor_reward if player=='investor' else game.trustee_reward
	rewards_other = game.trustee_reward if player=='investor' else game.investor_reward
	rewards = np.array(rewards_self) + svo*np.array(rewards_other)
	if normalize:
		rewards = rewards / (game.coins * game.match)
		rewards[:-1] = (1-gamma)*rewards[:-1]
	return rewards