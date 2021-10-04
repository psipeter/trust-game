from fixed_agents import *
from learning_agents import *
from utils import *
from experiments import *

# tournament_type = "single_game"
# tournament_type = "many_games"
tournament_type = "identical_learning_investors_vs_t4t"
# tournament_type = "identical_learning_trustees_vs_t4t"
# investors = [t4t(player='investor', O=0.9, P=1)]
# investors = [ActorCritic('investor')]
investors = [ActorCritic('investor', ID=n) for n in range(10)]
trustees = [t4t(player='trustee', O=1, F=1)]
# investors = [t4t(player='investor', O=1, F=1)]
# trustees = [ActorCritic('trustee', ID=n, n_actions=31) for n in range(10)]
play_tournament(investors, trustees, tournament_type)