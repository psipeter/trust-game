from fixed_agents import *
from learning_agents import *
from utils import *
from experiments import *
from analysis import *

# plot_user_data()
# plot_agent_data("agent_data/DeepQLearning_N=3_friendliness.pkl", learner_type='deep-q-learning', seed=0, n_learners=3)

# tournament_type = "one_game"
# tournament_type = "many_games"
# tournament_type = "many_learners_one_opponent"
# tournament_type = "many_learners_many_opponents"

# learner_plays = 'investor'
# learner_plays = 'trustee'

# n_learners = 5
# n_opponents = 200
# n_train = 300

# investors = [ActorCritic('investor', ID=n, seed=n) for n in range(n_learners)]
# investors = [InstanceBased('investor', ID=n, seed=n) for n in range(n_learners)]
# investors = [t4t(player='investor', O=0.9, P=1)]
# investors = [t4tv(player='investor', seed=n, minO=0.8, maxO=1.0, minX=0.5, maxX=0.5, minF=1.0, maxF=1.0, minP=0.1, maxP=0.3) for n in range(n_opponents)]
# investors = [t4tv(player='investor', seed=n, minO=0.6, maxO=0.8, minX=0.5, maxX=0.5, minF=0.8, maxF=1.0, minP=1.0, maxP=1.0) for n in range(n_opponents)]
# investors = [adaptive('investor', 'attrition')]

# trustees = [ActorCritic('trustee', ID=n, seed=n) for n in range(n_learners)]
# trustees = [InstanceBased('trustee', ID=n, seed=n) for n in range(n_learners)]
# trustees = [t4t(player='trustee', O=1, F=1)]
# trustees = [t4tv(player='trustee', seed=n, minO=0.1, maxO=0.3, minX=0.5, maxX=0.5, minF=0.0, maxF=0.1, minP=0.2, maxP=0.2) for n in range(n_opponents)]
# trustees = [t4tv(ID=f't4tv{n}', player='trustee', seed=n, minO=0.3, maxO=0.5, minX=0.5, maxX=0.5, minF=0.4, maxF=0.6, minP=1.0, maxP=1.0) for n in range(n_opponents)]
# trustees = [adaptive('trustee', 'cooperate')]

# play_tournament(investors, trustees, tournament_type, learner_plays, n_train)

load = False

# test_adaptivity(learner_type="tabular-q-learning", n_learners=2, n_train=1000, n_test=10, seed=0)
# test_adaptivity(learner_type="tabular-actor-critic", n_learners=2, n_train=2000, n_test=10, seed=0)
# test_adaptivity(learner_type="deep-q-learning", n_learners=1, n_train=100, seed=0, load=load)
# test_adaptivity(learner_type="deep-actor-critic", n_learners=3, n_train=1500, n_test=10, seed=0)
# test_adaptivity(learner_type="instance-based", n_learners=10, n_train=150, seed=0)  # deday.998
test_adaptivity(learner_type="nengo-q-learning", n_learners=10, n_train=150, seed=0)
# test_adaptivity(learner_type="nengo-actor-critic", n_learners=1, n_train=150, seed=0)

# test_t4tv(learner_type="tabular-q-learning", n_learners=100, n_train=15, seed=0, load=load)
# test_t4tv(learner_type="tabular-actor-critic", n_learners=2, n_train=1000, seed=0)
# test_t4tv(learner_type="deep-q-learning", n_learners=10, n_train=150, seed=0, load=load)
# test_t4tv(learner_type="deep-actor-critic", n_learners=2, n_train=1000, seed=0)
# test_t4tv(learner_type="instance-based", n_learners=100, n_train=150, seed=0, load=load)
# test_t4tv(learner_type="nengo-q-learning", n_learners=10, n_train=150, seed=0)
# test_t4tv(learner_type="nengo-actor-critic", n_learners=1, n_train=150, seed=0)