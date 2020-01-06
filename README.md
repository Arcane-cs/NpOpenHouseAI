# Overflow Open House 2020 - AI Demo
Demos made for open house 2020

## Intro
This repository contains the demos made for NP Open House 2020, to demostrate AI @ Overflow.
The demos showcase applying reinforcement learning (Q-Learning) to train a agent to play simple game.

## Demos

| Demo | Description | Run |
| --- | --- | --- |
| Maze | Agent trained using Q-learning to navigate a maze | `python Maze/Maze.py`
| Space Invaders | Agent trained using Q-Learning play space invaders | `python Space\ Invaders/Spaceinvaders.py` |

## Setup
### Bare Metal
Setup:
1. Install anaconda/miniconda
2. Create conda env with packages `conda create --name overflow-oh-ai` 
3. `conda activate overflow-oh-ai`
4. Run the demos

Additional setup for specific Demos:
- Maze demo 
    - `cd deps/gym-maze/ && python setup.py install`
