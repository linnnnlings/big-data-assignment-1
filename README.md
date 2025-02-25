# big-data-assignment-1
# Q-Learning Snake Game

## Overview

This project implements a Snake game using the Q-Learning algorithm for decision-making. The game allows the snake to learn how to collect food while avoiding collisions with itself, demonstrating the principles of reinforcement learning in a fun and interactive way.

## Features

- **Grid-Based Gameplay**: The game is played on a 10x10 grid.
- **Q-Learning Algorithm**: The snake uses Q-Learning to choose its actions and optimize its strategy over time.
- **Random Food Generation**: Food is generated randomly within the grid, and the snake grows when it eats.
- **Collision Detection**: The game ends if the snake collides with its own body.
- **Multiple Episodes**: The snake learns over multiple episodes, resetting its state after each game.

## Installation

To run this project, ensure you have Python 3.x installed along with the required libraries. You can install the required libraries using pip:

```bash
pip install numpy tkinter
