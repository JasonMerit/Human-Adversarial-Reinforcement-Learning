# Tron Changelog

## 1.0.0 - Initial Release
- Heuristics based AI opponent intended for gathering human play data for training a RNN RL agent.
- Adversary AI uses a Voronoi heuristic to evaluate the resulting state of each possible move, assuming the player stands still. I suspect implementation is flawed, will have to look up theory.


## 1.1.0 - Adversary update
- Simplified the adversary's decision-making by sticking to Voronoi heuristic only.
- Lowered the tick rate and now I check how often you guys make mistakes so I can tune accordingly.

