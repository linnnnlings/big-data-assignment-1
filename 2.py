import numpy as np
import tkinter as tk
import random

# Game Settings
GRID_SIZE = 10  # Size of the grid (10x10)
CELL_SIZE = 30  # Size of each cell in pixels
ACTIONS = ['up', 'down', 'left', 'right']  # Possible actions
LEARNING_RATE = 0.1  # Learning rate
DISCOUNT_FACTOR = 0.9  # Discount factor
EPSILON = 0.1  # Epsilon-greedy exploration rate
EPISODES = 500  # Number of training episodes

# Initialize Q-Table
q_table = {}

# Helper functions for Q-learning and game mechanics
def state_to_string(state):
    """Convert state to string for Q-table indexing."""
    return str(state)

def next_position(head, action):
    """Calculate the next position of the snake's head based on the action."""
    x, y = head
    if action == 'up':
        return (x - 1, y)
    elif action == 'down':
        return (x + 1, y)
    elif action == 'left':
        return (x, y - 1)
    elif action == 'right':
        return (x, y + 1)

def wrap_position(position, grid_size):
    """Handle wrapping of the position when the snake crosses the wall."""
    x, y = position
    return (x % grid_size, y % grid_size)

def is_collision(position, snake):
    """Check if the position collides with the snake's body."""
    return position in snake

def get_reward(new_head, food, collision):
    """Define the reward function."""
    if collision:
        return -10  # Penalty for collision with itself
    elif new_head == food:
        return 10  # Reward for eating food
    else:
        return -1  # Small penalty for each step

def choose_action(state, current_direction):
    """Epsilon-greedy policy to choose an action, with direction constraints."""
    # Define opposite directions to prevent moving backward
    opposite_directions = {
        'up': 'down',
        'down': 'up',
        'left': 'right',
        'right': 'left'
    }
    invalid_action = opposite_directions[current_direction]

    if random.uniform(0, 1) < EPSILON:
        # Explore: Choose a random action excluding the invalid one
        valid_actions = [action for action in ACTIONS if action != invalid_action]
        return random.choice(valid_actions)
    else:
        # Exploit: Choose the best action from the Q-table, excluding the invalid one
        state_str = state_to_string(state)
        if state_str not in q_table:
            q_table[state_str] = np.zeros(len(ACTIONS))
        # Find the best action, ignoring the invalid action
        valid_indices = [i for i, action in enumerate(ACTIONS) if action != invalid_action]
        best_action_index = valid_indices[np.argmax(q_table[state_str][valid_indices])]
        return ACTIONS[best_action_index]

def update_q_table(state, action, reward, next_state):
    """Update the Q-table using the Q-learning formula."""
    state_str = state_to_string(state)
    next_state_str = state_to_string(next_state)
    if state_str not in q_table:
        q_table[state_str] = np.zeros(len(ACTIONS))
    if next_state_str not in q_table:
        q_table[next_state_str] = np.zeros(len(ACTIONS))
    action_index = ACTIONS.index(action)
    max_future_q = np.max(q_table[next_state_str])
    q_table[state_str][action_index] += LEARNING_RATE * (
        reward + DISCOUNT_FACTOR * max_future_q - q_table[state_str][action_index]
    )

# Create Tkinter-based UI for the Snake Game
class SnakeGame(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Q-Learning Snake Game")
        self.geometry(f"{GRID_SIZE * CELL_SIZE}x{GRID_SIZE * CELL_SIZE}")
        self.resizable(False, False)
        self.canvas = tk.Canvas(self, width=GRID_SIZE * CELL_SIZE, height=GRID_SIZE * CELL_SIZE, bg="black")
        self.canvas.pack()
        self.snake = [(5, 5)]
        self.food = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
        while self.food in self.snake:
            self.food = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
        self.running = True
        self.current_direction = 'right'
        self.episode = 0
        self.score = 0
        self.run_game()

    def draw_grid(self):
        """Draw the game grid, snake, and food."""
        self.canvas.delete("all")
        # Draw food
        fx, fy = self.food
        self.canvas.create_rectangle(
            fy * CELL_SIZE, fx * CELL_SIZE,
            (fy + 1) * CELL_SIZE, (fx + 1) * CELL_SIZE,
            fill="red"
        )
        # Draw snake
        for segment in self.snake:
            sx, sy = segment
            self.canvas.create_rectangle(
                sy * CELL_SIZE, sx * CELL_SIZE,
                (sy + 1) * CELL_SIZE, (sx + 1) * CELL_SIZE,
                fill="green"
            )

    def run_game(self):
        """Main game loop."""
        if not self.running:
            return
        state = (self.snake, self.food)
        action = choose_action(state, self.current_direction)
        new_head = next_position(self.snake[0], action)
        # Wrap position if it crosses the wall
        new_head = wrap_position(new_head, GRID_SIZE)
        collision = is_collision(new_head, self.snake)
        reward = get_reward(new_head, self.food, collision)
        next_snake = self.snake[:]
        next_snake.insert(0, new_head)  # Add new head
        if not collision:
            if new_head == self.food:
                # Snake eats food, generate a new food position
                self.score += 1
                self.food = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
                while self.food in next_snake:
                    self.food = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            else:
                # Remove tail if no food is eaten
                next_snake.pop()
        update_q_table(state, action, reward, (next_snake, self.food))
        if collision:
            self.running = False
            self.show_message("Game Over!")
            self.after(1000, self.reset_game)
            return
        self.snake = next_snake
        self.current_direction = action  # Update the current direction
        self.draw_grid()
        self.after(200, self.run_game)

    def reset_game(self):
        """Reset the game state for the next episode."""
        self.episode += 1
        if self.episode < EPISODES:
            self.snake = [(5, 5)]
            self.food = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            while self.food in self.snake:
                self.food = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            self.running = True
            self.score = 0
            self.current_direction = 'right'
            self.run_game()
        else:
            self.show_message("Training Complete!")

    def show_message(self, message):
        """Display a message at the end of the game."""
        self.canvas.create_text(
            GRID_SIZE * CELL_SIZE // 2, GRID_SIZE * CELL_SIZE // 2,
            text=message, fill="white", font=("Arial", 24)
        )

# Run the Snake Game
if __name__ == "__main__":
    game = SnakeGame()
    game.mainloop()