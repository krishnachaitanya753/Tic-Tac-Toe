import random
import pickle
import os
from typing import Dict, List, Tuple

class Board:
    def __init__(self):
        self.state = ['-' for _ in range(9)]
        self.current_player = 'X'

    def get_state_str(self) -> str:
        return ''.join(self.state)

    def get_valid_actions(self) -> List[int]:
        return [i for i, cell in enumerate(self.state) if cell == '-']

    def apply_action(self, action: int) -> None:
        if self.state[action] != '-':
            raise ValueError(f"Invalid move at position {action}")
        self.state[action] = self.current_player

    def print_board(self) -> None:
        print("\n   0   1   2")
        for i in range(3):
            row = self.state[i*3:(i+1)*3]
            print(f"{i}  {' | '.join(row)}")
            if i < 2:
                print("  -----------")
        print()

    def check_winner(self, player: str) -> bool:
        winning_lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
            [0, 4, 8], [2, 4, 6]              # diagonals
        ]
        return any(all(self.state[i] == player for i in line) for line in winning_lines)

    def is_draw(self) -> bool:
        return '-' not in self.state

    def switch_player(self) -> None:
        self.current_player = 'O' if self.current_player == 'X' else 'X'

    def reset(self) -> None:
        self.state = ['-' for _ in range(9)]
        self.current_player = 'X'


class QLearningAgent:
    def __init__(self, symbol: str, alpha: float = 0.1, gamma: float = 0.9, 
                 epsilon: float = 0.8, epsilon_min: float = 0.01, decay: float = 0.9995):
        self.symbol = symbol
        self.q_table: Dict[str, Dict[int, float]] = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay = decay
        self.game_history: List[Tuple[str, int, float]] = []

    def get_q_value(self, state: str, action: int) -> float:
        return self.q_table.setdefault(state, {}).get(action, 0.0)

    def set_q_value(self, state: str, action: int, value: float) -> None:
        self.q_table.setdefault(state, {})[action] = value

    def choose_action(self, board: Board, training: bool = True) -> int:
        valid_actions = board.get_valid_actions()
        state_str = board.get_state_str()

        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)

        return max(valid_actions, key=lambda action: self.get_q_value(state_str, action))

    def add_to_history(self, state: str, action: int, reward: float) -> None:
        self.game_history.append((state, action, reward))

    def learn_from_game(self, final_reward: float) -> None:
        for i, (state, action, immediate_reward) in enumerate(self.game_history):
            future_reward = sum((self.gamma ** (j - i)) * self.game_history[j][2] for j in range(i + 1, len(self.game_history)))
            future_reward += (self.gamma ** (len(self.game_history) - i)) * final_reward
            old_q = self.get_q_value(state, action)
            new_q = old_q + self.alpha * (immediate_reward + future_reward - old_q)
            self.set_q_value(state, action, new_q)
        self.game_history.clear()

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay)

    def save_q_table(self, filename: str) -> None:
        with open(filename, 'wb') as f:
            pickle.dump({'q_table': self.q_table, 'epsilon': self.epsilon, 'symbol': self.symbol}, f)

    def load_q_table(self, filename: str) -> None:
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data['q_table']
                self.epsilon = data['epsilon']
                print(f"Loaded Q-table for {self.symbol}: {len(self.q_table)} states")

class TicTacToeGame:
    def __init__(self):
        self.board = Board()
        self.agent_X = QLearningAgent('X')
        self.agent_O = QLearningAgent('O')

    def train_agents(self, episodes: int = 50000, save_every: int = 10000) -> None:
        print(f"🎯 Starting training for {episodes} episodes...")
        wins_X, wins_O, draws = 0, 0, 0

        for episode in range(episodes):
            self.board.reset()
            current_agent, opponent = (self.agent_X, self.agent_O) if random.random() < 0.5 else (self.agent_O, self.agent_X)
            self.board.current_player = current_agent.symbol

            game_result = self._play_training_game(current_agent, opponent)

            if game_result == 'X':
                wins_X += 1
            elif game_result == 'O':
                wins_O += 1
            else:
                draws += 1

            self.agent_X.decay_epsilon()
            self.agent_O.decay_epsilon()

            if (episode + 1) % 5000 == 0:
                total_games = episode + 1
                print(f"Episode {total_games:,}: X wins: {wins_X / total_games:.1%}, O wins: {wins_O / total_games:.1%}, Draws: {draws / total_games:.1%} (ε: {self.agent_X.epsilon:.3f})")

            if (episode + 1) % save_every == 0:
                self.save_agents()

        print("🎉 Training completed!")
        self.save_agents()

    def _play_training_game(self, first_agent: QLearningAgent, second_agent: QLearningAgent) -> str:
        current_agent = first_agent

        while True:
            state_str = self.board.get_state_str()
            action = current_agent.choose_action(self.board, training=True)
            current_agent.add_to_history(state_str, action, -0.01)
            self.board.apply_action(action)

            if self.board.check_winner(current_agent.symbol):
                current_agent.learn_from_game(1.0)
                second_agent.learn_from_game(-1.0)
                return current_agent.symbol
            elif self.board.is_draw():
                first_agent.learn_from_game(0.0)
                second_agent.learn_from_game(0.0)
                return 'Draw'

            self.board.switch_player()
            current_agent = second_agent if current_agent == first_agent else first_agent

    def play_vs_human(self, human_symbol: str = 'X') -> None:
        self.board.reset()
        ai_agent = self.agent_O if human_symbol == 'X' else self.agent_X

        print(f"🎮 You are {human_symbol}, AI is {ai_agent.symbol}")
        print("Enter position (0-8) or 'quit' to exit")

        while True:
            self.board.print_board()

            if self.board.current_player == human_symbol:
                try:
                    user_input = input(f"Your move ({human_symbol}): ").strip().lower()
                    if user_input == 'quit':
                        print("Thanks for playing!")
                        return

                    action = int(user_input)
                    if action not in self.board.get_valid_actions():
                        print("❌ Invalid move! Try again.")
                        continue

                    self.board.apply_action(action)
                except (ValueError, IndexError):
                    print("❌ Please enter a number between 0-8!")
                    continue
            else:
                action = ai_agent.choose_action(self.board, training=False)
                print(f"🤖 AI plays position {action}")
                self.board.apply_action(action)

            if self.board.check_winner(self.board.current_player):
                self.board.print_board()
                winner = "You" if self.board.current_player == human_symbol else "AI"
                print(f"🏆 {winner} win{'s' if winner == 'AI' else ''}!")
                break
            elif self.board.is_draw():
                self.board.print_board()
                print("🤝 It's a draw!")
                break

            self.board.switch_player()

    def play_ai_vs_ai(self, num_games: int = 10) -> None:
        wins_X, wins_O, draws = 0, 0, 0

        for game in range(num_games):
            print(f"\n🎯 Game {game + 1}/{num_games}")
            self.board.reset()
            current_agent = self.agent_X

            while True:
                action = current_agent.choose_action(self.board, training=False)
                print(f"Player {current_agent.symbol} plays position {action}")
                self.board.apply_action(action)
                self.board.print_board()

                if self.board.check_winner(current_agent.symbol):
                    print(f"🏆 Player {current_agent.symbol} wins!")
                    if current_agent.symbol == 'X':
                        wins_X += 1
                    else:
                        wins_O += 1
                    break
                elif self.board.is_draw():
                    print("🤝 Draw!")
                    draws += 1
                    break

                self.board.switch_player()
                current_agent = self.agent_O if current_agent == self.agent_X else self.agent_X

                input("Press Enter for next move...")

        print(f"\n📊 Final Results: X: {wins_X}, O: {wins_O}, Draws: {draws}")

    def save_agents(self) -> None:
        self.agent_X.save_q_table('agent_X.pkl')
        self.agent_O.save_q_table('agent_O.pkl')
        print("💾 Agents saved!")

    def load_agents(self) -> None:
        self.agent_X.load_q_table('agent_X.pkl')
        self.agent_O.load_q_table('agent_O.pkl')

def main():
    game = TicTacToeGame()

    print("🎮 Welcome to Advanced Tic Tac Toe with Q-Learning!")
    print("=" * 50)

    game.load_agents()

    while True:
        print("\nOptions:")
        print("1. Train AI agents")
        print("2. Play vs AI")
        print("3. Watch AI vs AI")
        print("4. Save agents")
        print("5. Exit")

        choice = input("\nEnter choice (1-5): ").strip()

        if choice == '1':
            episodes = int(input("Enter number of training episodes (default 50000): ") or 50000)
            game.train_agents(episodes)
        elif choice == '2':
            symbol = input("Choose your symbol (X/O, default X): ").upper() or 'X'
            if symbol not in ['X', 'O']:
                symbol = 'X'
            game.play_vs_human(symbol)
        elif choice == '3':
            num_games = int(input("Number of games to watch (default 5): ") or 5)
            game.play_ai_vs_ai(num_games)
        elif choice == '4':
            game.save_agents()
        elif choice == '5':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice!")

if __name__ == "__main__":
    main()
