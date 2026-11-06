"""Gói core: môi trường mê cung, Q-Learning, BFS/Random và đánh giá."""
from .maze import (ACTION_ARROWS, ACTION_NAMES, PRESET_LAYOUTS, Maze,
                   generate_random_layout)
from .qlearning import (QConfig, TrainResult, greedy_rollout,
                        train_q_learning)
from .baselines import bfs_shortest_path, bfs_metrics, random_episode
from .evaluation import (compare_agents, evaluate_multi_seed,
                         summarize_multi_seed, convergence_episode)
from .render import render_maze_html
