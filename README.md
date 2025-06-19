# Tic Tac Toe 

## About
This project is a Tic Tac Toe game where AI agents learn to play using Q-Learning. You can train the AI, play against it, or watch two AI agents compete.

## How to Use
1. Run the game by executing `main.py`.
2. Choose an option:
   - Train the AI agents.
   - Play against the AI.
   - Watch AI vs AI.
   - Save the trained agents.
   - Exit the game.

## Files
- `main.py`: Trains the model and we  can play with model.
- `agent_X.pkl` and `agent_O.pkl`: Saved AI data.

## Notes
- The AI gets better with training.
- You can tweak the AI settings in the code.

## Code Summary
I have used  Q-Learning, a reinforcement learning algorithm, to train AI agents for Tic Tac Toe
The Q-Learning model updates a Q-table and stores history of an episode and updates all the q values at once.
The agents learn optimal strategies by exploring moves and improving their Q-values over time. This approach ensures the AI becomes smarter with more training episodes.

Enjoy playing Tic Tac Toe with smart AI!
