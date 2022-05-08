# trust-game
Behavioral and neural models of social decision making in the Trust Game

pip install numpy scipy matplotlib seaborn pandas nengo nengo-spa torch
 - note: torch requires python < 3.8

- main.py describes which experiment to run, with which agents. Execute this file to run the program (python main.py)
- experiments.py details the experiments called by main.py, saves data to pandas dataframes, and calls plotting and analysis routies
- utils.py contains the core game code and analysis functions
- fixed_agents.py specifies the agents against which the learning agents play
- learning_agents.py specifies the learning agents. learning_agents must implement the following methods
  - reinitialize: resets the agent to the default state, allowing them to be trained from scratch
  - newgame: resets any memory the learner had of the previous game
  - get_state: converts information about the state of the current game (including the current turn and the move history of each agent) into an internal state used by the agent
  - move: calls get_state() on the current game state, runs the core logic of the learner, and returns the number of coins to give/keep
  - learn: update parameters of the agent. Called after each game has concluded.

