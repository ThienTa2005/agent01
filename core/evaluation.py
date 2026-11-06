"""
evaluation.py – So sánh Q-Learning / BFS / Random và thống kê nhiều seed.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List

import numpy as np
import pandas as pd

from .baselines import bfs_metrics, bfs_shortest_path, random_episode
from .maze import Maze
from .qlearning import (QConfig, greedy_rollout, q_epsilon_episode,
                        train_q_learning)

METRIC_COLS = ["success_rate", "avg_steps", "avg_reward", "avg_collisions"]


def _summarize(episodes: List[Dict]) -> Dict[str, float]:
    """Gộp danh sách episode thành các chỉ số trung bình."""
    ok = [e for e in episodes if e["success"]]
    return dict(
        success_rate=float(np.mean([e["success"] for e in episodes])),
        # Số bước trung bình chỉ tính trên các lần THÀNH CÔNG (nếu không có lần nào -> NaN)
        avg_steps=float(np.mean([e["steps"] for e in ok])) if ok else float("nan"),
        avg_reward=float(np.mean([e["reward"] for e in episodes])),
        avg_collisions=float(np.mean([e["collisions"] for e in episodes])),
    )


def convergence_episode(successes: np.ndarray, window: int = 50, threshold: float = 0.9) -> float:
    """Episode đầu tiên mà tỉ lệ thành công trượt (cửa sổ `window`) đạt ngưỡng. NaN nếu chưa đạt."""
    if len(successes) < window:
        return float("nan")
    roll = np.convolve(successes, np.ones(window) / window, mode="valid")
    idx = np.flatnonzero(roll >= threshold)
    return float(idx[0] + window) if len(idx) else float("nan")


# ---------------------------------------------------------------------------
# So sánh 3 tác tử trên 1 lần huấn luyện
# ---------------------------------------------------------------------------
def compare_agents(maze: Maze, q: np.ndarray, runs: int = 500, seed: int = 0,
                   max_steps: int = 100, eval_epsilon: float = 0.0) -> Dict:
    """So sánh Q-Learning (đã huấn luyện) với BFS và tác tử ngẫu nhiên."""
    rng = np.random.default_rng(seed)

    # Q-Learning: chạy tham lam 1 lần (xác định) để lấy đường đi + chạy `runs` lần để lấy thống kê
    q_greedy = greedy_rollout(maze, q, max_steps)
    if eval_epsilon > 0:
        q_eps = [q_epsilon_episode(maze, q, eval_epsilon, rng, max_steps) for _ in range(runs)]
    else:
        q_eps = [q_greedy]
    q_stats = _summarize(q_eps)

    # Random: chạy `runs` episode
    rand_eps = [random_episode(maze, rng, max_steps) for _ in range(runs)]
    rand_stats = _summarize(rand_eps)

    # BFS: xác định
    bfs_path = bfs_shortest_path(maze)
    bfs_stats = bfs_metrics(maze)

    table = pd.DataFrame(
        [q_stats, bfs_stats, rand_stats],
        index=["Q-Learning", "BFS", "Random"],
    )
    return dict(table=table, q_path=q_greedy["path"], q_success=q_greedy["success"],
                bfs_path=bfs_path, random_path=rand_eps[0]["path"],
                random_success=rand_eps[0]["success"])


# ---------------------------------------------------------------------------
# Thống kê nhiều random seed
# ---------------------------------------------------------------------------
def evaluate_multi_seed(maze: Maze, cfg: QConfig, n_seeds: int = 10,
                        eval_runs: int = 100, eval_epsilon: float = 0.05) -> pd.DataFrame:
    """Huấn luyện lại Q-Learning với `n_seeds` seed khác nhau rồi đánh giá.

    Với mỗi seed:
      * Q-Learning: huấn luyện từ đầu, rồi chạy `eval_runs` episode đánh giá
        (epsilon-greedy với epsilon nhỏ = eval_epsilon để các lần chạy có độ biến thiên).
      * Random   : chạy `eval_runs` episode.
      * BFS      : kết quả cố định (không phụ thuộc seed).
    Trả về DataFrame dạng "dài": mỗi dòng = (seed, agent).
    """
    rows = []
    bfs_stats = bfs_metrics(maze)
    for i in range(n_seeds):
        seed = cfg.seed + i
        res = train_q_learning(maze, replace(cfg, seed=seed, n_snapshots=2))
        rng = np.random.default_rng(seed + 10_000)

        q_eps = [q_epsilon_episode(maze, res.q_table, eval_epsilon, rng, cfg.max_steps)
                 for _ in range(eval_runs)]
        q_stats = _summarize(q_eps)
        q_stats["converge_episode"] = convergence_episode(res.successes)
        q_stats["final_train_success"] = float(res.successes[-max(1, cfg.episodes // 10):].mean())
        rows.append(dict(seed=seed, agent="Q-Learning", **q_stats))

        rand_eps = [random_episode(maze, rng, cfg.max_steps) for _ in range(eval_runs)]
        rows.append(dict(seed=seed, agent="Random", **_summarize(rand_eps)))
        rows.append(dict(seed=seed, agent="BFS", **bfs_stats))
    return pd.DataFrame(rows)


def summarize_multi_seed(df: pd.DataFrame) -> pd.DataFrame:
    """Bảng trung bình ± độ lệch chuẩn theo từng tác tử."""
    agg = df.groupby("agent")[METRIC_COLS].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reindex(["Q-Learning", "BFS", "Random"])
