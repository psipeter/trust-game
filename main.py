from fixed_agents import *
from learning_agents import *
from utils import *
from experiments import *

# tournament_type = "one_game"
# tournament_type = "many_games"
tournament_type = "many_learners_one_opponent"
# tournament_type = "many_learners_many_opponents"

# learner_plays = 'investor'
learner_plays = 'trustee'

n_learners = 3
# n_opponents = 200
n_train = 100
n_test = 10

# investors = [ActorCritic('investor', ID=n, learning_rate=1e-3) for n in range(n_learners)]
# investors = [t4t(player='investor', O=0.9, P=1)]
# investors = [t4tv(player='investor', seed=n, minO=0.8, maxO=1.0, minX=0.5, maxX=0.5, minF=1.0, maxF=1.0, minP=0.1, maxP=0.3) for n in range(n_opponents)]
# investors = [t4tv(player='investor', seed=n, minO=0.6, maxO=0.8, minX=0.5, maxX=0.5, minF=0.8, maxF=1.0, minP=1.0, maxP=1.0) for n in range(n_opponents)]
investors = [adaptive('investor', 'turn_based')]

trustees = [ActorCritic('trustee', ID=n, n_actions=31, learning_rate=1e-3) for n in range(n_learners)]
# trustees = [t4t(player='trustee', O=1, F=1)]
# trustees = [t4tv(player='trustee', seed=n, minO=0.1, maxO=0.3, minX=0.5, maxX=0.5, minF=0.0, maxF=0.1, minP=0.2, maxP=0.2) for n in range(n_opponents)]
# trustees = [t4tv(ID=f't4tv{n}', player='trustee', seed=n, minO=0.3, maxO=0.5, minX=0.5, maxX=0.5, minF=0.4, maxF=0.6, minP=1.0, maxP=1.0) for n in range(n_opponents)]
# trustees = [adaptive('trustee', 'turn_based')]

# testers = [t4tv(ID=f't4tv{n}', player='trustee', seed=n, minO=0.3, maxO=0.5, minX=0.5, maxX=0.5, minF=0.4, maxF=0.6, minP=1.0, maxP=1.0) for n in range(n_opponents)]
testers = [adaptive('investor', 'turn_based')]
# testers = [adaptive('trustee', 'turn_based')]

# play_tournament(investors, trustees, testers, tournament_type, learner_plays, n_train, n_test)


test_adaptivity(n_learners=10, learning_rate=1e-3, n_train=5000, n_test=100, n_inputs=15, seed=0)