"""Kiểm thử phần backend. Chạy:  python -m unittest discover -s tests -v"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (PRESET_LAYOUTS, Maze, QConfig, bfs_metrics, bfs_shortest_path,
                  compare_agents, evaluate_multi_seed, generate_random_layout,
                  greedy_rollout, summarize_multi_seed, train_q_learning)
from core.maze import REWARD_GOAL, REWARD_STEP, REWARD_WALL


class TestMaze(unittest.TestCase):
    def test_presets_are_8x8_and_solvable(self):
        for name, layout in PRESET_LAYOUTS.items():
            m = Maze(layout)
            self.assertEqual((m.n_rows, m.n_cols), (8, 8), name)
            self.assertIsNotNone(bfs_shortest_path(m), name)

    def test_rewards(self):
        m = Maze(PRESET_LAYOUTS["Mẫu 1 – Cổ điển"])
        # Từ S đi lên -> ra ngoài biên -> -10, đứng yên
        s2, r, done, hit = m.step(m.start_state, 0)
        self.assertEqual((s2, r, done, hit), (m.start_state, REWARD_WALL, False, True))
        # Từ S đi phải -> ô hợp lệ -> -1
        s2, r, done, hit = m.step(m.start_state, 3)
        self.assertEqual((r, done, hit), (REWARD_STEP, False, False))
        # Bước vào G -> +50
        gr, gc = m.goal
        s2, r, done, hit = m.step(m.to_state(gr, gc - 1), 3)
        self.assertEqual((s2, r, done), (m.goal_state, REWARD_GOAL, True))

    def test_random_layout_solvable(self):
        for seed in range(5):
            self.assertIsNotNone(bfs_shortest_path(Maze(generate_random_layout(8, 0.3, seed))))


class TestQLearning(unittest.TestCase):
    def test_learns_optimal_path(self):
        for name, layout in PRESET_LAYOUTS.items():
            m = Maze(layout)
            res = train_q_learning(m, QConfig(episodes=800, seed=1))
            g = greedy_rollout(m, res.q_table, 100)
            opt = len(bfs_shortest_path(m)) - 1
            self.assertTrue(g["success"], name)
            self.assertEqual(g["steps"], opt, name)   # Q-Learning tìm được đường ngắn nhất
            self.assertEqual(g["collisions"], 0, name)

    def test_reproducible(self):
        m = Maze(PRESET_LAYOUTS["Mẫu 1 – Cổ điển"])
        a = train_q_learning(m, QConfig(episodes=100, seed=7))
        b = train_q_learning(m, QConfig(episodes=100, seed=7))
        np.testing.assert_array_equal(a.q_table, b.q_table)

    def test_history_shapes(self):
        m = Maze(PRESET_LAYOUTS["Mẫu 1 – Cổ điển"])
        r = train_q_learning(m, QConfig(episodes=50, n_snapshots=5))
        for arr in (r.rewards, r.steps, r.collisions, r.successes, r.epsilons):
            self.assertEqual(len(arr), 50)
        self.assertIn(0, r.snapshots)
        self.assertIn(49, r.snapshots)


class TestEvaluation(unittest.TestCase):
    def test_compare_and_multiseed(self):
        m = Maze(PRESET_LAYOUTS["Mẫu 1 – Cổ điển"])
        res = train_q_learning(m, QConfig(episodes=600, seed=3))
        out = compare_agents(m, res.q_table, runs=100, seed=0)
        t = out["table"]
        self.assertEqual(t.loc["Q-Learning", "success_rate"], 1.0)
        self.assertEqual(t.loc["BFS", "avg_steps"], len(out["bfs_path"]) - 1)
        self.assertLess(t.loc["Random", "success_rate"], 1.0 + 1e-9)

        df = evaluate_multi_seed(m, QConfig(episodes=300), n_seeds=3, eval_runs=20)
        self.assertEqual(len(df), 9)
        s = summarize_multi_seed(df)
        self.assertEqual(list(s.index), ["Q-Learning", "BFS", "Random"])


if __name__ == "__main__":
    unittest.main()
