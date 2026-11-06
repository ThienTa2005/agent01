"""
qlearning.py – Thuật toán Q-Learning (dạng bảng) cài đặt bằng NumPy.

Công thức cập nhật (Bellman):
    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

Trong đó:
    alpha  : tốc độ học (learning rate)
    gamma  : hệ số chiết khấu (discount factor)
    epsilon: xác suất khám phá (epsilon-greedy)
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional

import numpy as np

from .maze import Maze, N_ACTIONS


@dataclass
class QConfig:
    """Các tham số huấn luyện (có thể chỉnh trên giao diện)."""
    alpha: float = 0.5            # tốc độ học
    gamma: float = 0.95           # hệ số chiết khấu
    epsilon: float = 1.0          # epsilon ban đầu
    epsilon_decay: float = 0.995  # nhân epsilon với hệ số này sau mỗi episode (1.0 = không giảm)
    epsilon_min: float = 0.05     # epsilon tối thiểu
    episodes: int = 500           # số episode huấn luyện
    max_steps: int = 100          # số bước tối đa mỗi episode
    seed: int = 42                # random seed (để tái lập kết quả)
    n_snapshots: int = 8          # số episode lưu lại để phát lại hoạt ảnh


@dataclass
class TrainResult:
    """Kết quả huấn luyện: bảng Q + lịch sử theo từng episode."""
    q_table: np.ndarray
    rewards: np.ndarray       # tổng phần thưởng mỗi episode
    steps: np.ndarray         # số bước mỗi episode
    collisions: np.ndarray    # số lần va tường mỗi episode
    successes: np.ndarray     # 1 nếu đến đích, 0 nếu không
    epsilons: np.ndarray      # epsilon tại mỗi episode
    snapshots: Dict[int, Dict] = field(default_factory=dict)  # episode -> quỹ đạo + Q tại thời điểm đó
    config: Optional[QConfig] = None


def choose_action(q_row: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    """Chính sách epsilon-greedy.

    - Với xác suất epsilon: chọn ngẫu nhiên (khám phá).
    - Ngược lại: chọn hành động có Q lớn nhất (khai thác); nếu hòa thì chọn ngẫu nhiên
      trong các hành động hòa để tránh luôn thiên về "Lên".
    """
    if rng.random() < epsilon:
        return int(rng.integers(N_ACTIONS))
    best = np.flatnonzero(q_row == q_row.max())
    return int(best[rng.integers(len(best))])


def train_q_learning(
    maze: Maze,
    cfg: QConfig,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> TrainResult:
    """Huấn luyện tác tử Q-Learning trên mê cung."""
    rng = np.random.default_rng(cfg.seed)
    q = np.zeros((maze.n_states, N_ACTIONS))  # khởi tạo bảng Q = 0

    n = cfg.episodes
    rewards = np.zeros(n)
    steps = np.zeros(n, dtype=int)
    collisions = np.zeros(n, dtype=int)
    successes = np.zeros(n, dtype=int)
    epsilons = np.zeros(n)

    # Chọn trước các episode cần lưu "ảnh chụp" để phát lại hoạt ảnh
    snap_eps = set(np.unique(np.linspace(0, n - 1, max(2, cfg.n_snapshots)).astype(int)).tolist())
    snapshots: Dict[int, Dict] = {}

    eps = cfg.epsilon
    for ep in range(n):
        s = maze.start_state
        total, n_hit, success = 0.0, 0, False
        path = [s] if ep in snap_eps else None

        for t in range(cfg.max_steps):
            a = choose_action(q[s], eps, rng)                 # 1. chọn hành động
            s2, r, done, hit = maze.step(s, a)                # 2. tương tác môi trường
            # 3. cập nhật bảng Q theo công thức Bellman
            target = r if done else r + cfg.gamma * q[s2].max()
            q[s, a] += cfg.alpha * (target - q[s, a])
            s = s2
            total += r
            n_hit += int(hit)
            if path is not None:
                path.append(s)
            if done:
                success = True
                break

        rewards[ep] = total
        steps[ep] = t + 1
        collisions[ep] = n_hit
        successes[ep] = int(success)
        epsilons[ep] = eps

        if path is not None:
            snapshots[ep] = dict(path=path, reward=total, collisions=n_hit,
                                 success=success, epsilon=eps, q=q.copy())

        # Giảm epsilon dần: ban đầu khám phá nhiều, về sau khai thác nhiều
        eps = max(cfg.epsilon_min, eps * cfg.epsilon_decay)

        if progress_callback is not None and (ep % max(1, n // 100) == 0 or ep == n - 1):
            progress_callback(ep + 1, n)

    return TrainResult(q, rewards, steps, collisions, successes, epsilons, snapshots, cfg)


def greedy_rollout(maze: Maze, q: np.ndarray, max_steps: int = 100) -> Dict:
    """Cho tác tử đi theo chính sách tham lam (epsilon = 0) – dùng để đánh giá/hiển thị."""
    s = maze.start_state
    path, total, collisions, success = [s], 0.0, 0, False
    visited_count: Dict[int, int] = {}
    for _ in range(max_steps):
        row = q[s]
        a = int(np.argmax(row))           # tie-break cố định để kết quả xác định
        s, r, done, hit = maze.step(s, a)
        total += r
        collisions += int(hit)
        path.append(s)
        if done:
            success = True
            break
        # Phát hiện lặp vô hạn (chính sách chưa học xong): dừng sớm
        visited_count[s] = visited_count.get(s, 0) + 1
        if visited_count[s] > 3:
            break
    return dict(path=path, reward=total, steps=len(path) - 1,
                collisions=collisions, success=success)


def q_epsilon_episode(maze: Maze, q: np.ndarray, epsilon: float,
                      rng: np.random.Generator, max_steps: int) -> Dict:
    """Chạy 1 episode đánh giá với chính sách epsilon-greedy cố định (epsilon nhỏ)."""
    s = maze.start_state
    path, total, collisions, success = [s], 0.0, 0, False
    for _ in range(max_steps):
        a = choose_action(q[s], epsilon, rng)
        s, r, done, hit = maze.step(s, a)
        total += r
        collisions += int(hit)
        path.append(s)
        if done:
            success = True
            break
    return dict(path=path, reward=total, steps=len(path) - 1,
                collisions=collisions, success=success)


def config_to_dict(cfg: QConfig) -> Dict:
    return asdict(cfg)
