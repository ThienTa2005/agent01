"""
baselines.py – Hai tác tử dùng để so sánh với Q-Learning:
    1. BFS (tìm kiếm theo chiều rộng): luôn cho đường đi NGẮN NHẤT (biết trước bản đồ).
    2. Tác tử ngẫu nhiên: mỗi bước chọn bừa 1 trong 4 hành động.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import numpy as np

from .maze import Maze, N_ACTIONS, REWARD_GOAL, REWARD_STEP


def bfs_shortest_path(maze: Maze) -> Optional[List[int]]:
    """Tìm đường ngắn nhất từ S đến G bằng BFS.

    Trả về danh sách trạng thái từ S đến G (gồm cả S và G),
    hoặc None nếu không có đường đi.
    """
    start, goal = maze.start_state, maze.goal_state
    parent = {start: None}          # lưu ô cha để dựng lại đường đi
    queue = deque([start])
    while queue:
        s = queue.popleft()
        if s == goal:
            break
        for a in range(N_ACTIONS):
            if maze.hit[s, a]:       # bỏ qua hướng đi va tường
                continue
            ns = int(maze.next_state[s, a])
            if ns not in parent:
                parent[ns] = s
                queue.append(ns)
    if goal not in parent:
        return None
    # Dựng lại đường đi từ G ngược về S
    path, cur = [], goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return path[::-1]


def bfs_metrics(maze: Maze) -> Dict[str, float]:
    """Chỉ số của BFS (xác định, không ngẫu nhiên, không va tường)."""
    path = bfs_shortest_path(maze)
    if path is None:
        return dict(success_rate=0.0, avg_steps=float("nan"),
                    avg_reward=float("nan"), avg_collisions=0.0)
    steps = len(path) - 1
    # (steps-1) bước thường bị -1, bước cuối đến đích được +50
    reward = REWARD_STEP * (steps - 1) + REWARD_GOAL
    return dict(success_rate=1.0, avg_steps=float(steps),
                avg_reward=float(reward), avg_collisions=0.0)


def random_episode(maze: Maze, rng: np.random.Generator, max_steps: int) -> Dict:
    """Chạy 1 episode với tác tử ngẫu nhiên."""
    s = maze.start_state
    path, total, collisions, success = [s], 0.0, 0, False
    for _ in range(max_steps):
        a = int(rng.integers(N_ACTIONS))
        s, r, done, hit = maze.step(s, a)
        total += r
        collisions += int(hit)
        path.append(s)
        if done:
            success = True
            break
    return dict(path=path, reward=total, steps=len(path) - 1,
                collisions=collisions, success=success)
