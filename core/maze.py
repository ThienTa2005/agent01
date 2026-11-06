"""
maze.py – Môi trường mê cung 8x8 cho tác tử Q-Learning.

Ký hiệu trong layout (danh sách chuỗi):
    'S' : điểm bắt đầu (Start)
    'G' : đích (Goal)
    '#' : tường
    '.' : đường đi

Phần thưởng (theo đề bài):
    - Đến đích        : +50
    - Bước hợp lệ     : -1
    - Va tường/biên   : -10  (tác tử đứng yên tại chỗ)

Để huấn luyện nhanh, toàn bộ "luật chơi" được tính trước thành các bảng NumPy
(next_state, reward, done, hit). Khi đó mỗi bước chỉ cần tra bảng O(1).
"""
from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Hằng số: hành động và phần thưởng
# ---------------------------------------------------------------------------
# 4 hành động: 0 = Lên, 1 = Xuống, 2 = Trái, 3 = Phải
ACTION_NAMES = ["Lên", "Xuống", "Trái", "Phải"]
ACTION_ARROWS = ["↑", "↓", "←", "→"]
# Độ dời (dòng, cột) tương ứng từng hành động
ACTION_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N_ACTIONS = 4

REWARD_GOAL = 50.0   # đến đích
REWARD_STEP = -1.0   # bước đi hợp lệ
REWARD_WALL = -10.0  # va tường hoặc ra ngoài biên

# ---------------------------------------------------------------------------
# Các mê cung mẫu 8x8 (đều đảm bảo có đường đi từ S đến G – xem tests/)
# ---------------------------------------------------------------------------
PRESET_LAYOUTS = {
    "Mẫu 1 – Cổ điển": [
        "S...#...",
        ".##.#.#.",
        ".#..#.#.",
        ".#.##.#.",
        ".#....#.",
        ".####.#.",
        "......#.",
        ".####..G",
    ],
    "Mẫu 2 – Nhiều ngã rẽ": [
        "S.#.....",
        ".##.###.",
        "....#...",
        ".####.#.",
        ".#....#.",
        ".#.####.",
        ".#......",
        "...####G",
    ],
    "Mẫu 3 – Đường vòng": [
        "S.......",
        "#######.",
        "........",
        ".#######",
        "........",
        "#######.",
        "........",
        ".######G",
    ],
}


class Maze:
    """Mê cung dạng lưới + các bảng chuyển trạng thái đã tính sẵn."""

    def __init__(self, layout: List[str]):
        self.layout = [str(row) for row in layout]
        self.n_rows = len(self.layout)
        self.n_cols = len(self.layout[0])
        if any(len(r) != self.n_cols for r in self.layout):
            raise ValueError("Các dòng của mê cung phải có độ dài bằng nhau.")

        # grid: 1 = tường, 0 = ô đi được
        self.grid = np.zeros((self.n_rows, self.n_cols), dtype=np.int8)
        self.start: Optional[Tuple[int, int]] = None
        self.goal: Optional[Tuple[int, int]] = None
        for r, row in enumerate(self.layout):
            for c, ch in enumerate(row):
                if ch == "#":
                    self.grid[r, c] = 1
                elif ch == "S":
                    self.start = (r, c)
                elif ch == "G":
                    self.goal = (r, c)
        if self.start is None or self.goal is None:
            raise ValueError("Mê cung phải có đúng một ô 'S' và một ô 'G'.")

        self.n_states = self.n_rows * self.n_cols
        self.start_state = self.to_state(*self.start)
        self.goal_state = self.to_state(*self.goal)
        self._build_tables()

    # ------------------------------------------------------------------
    # Chuyển đổi giữa (dòng, cột) và chỉ số trạng thái
    # ------------------------------------------------------------------
    def to_state(self, r: int, c: int) -> int:
        return r * self.n_cols + c

    def to_pos(self, s: int) -> Tuple[int, int]:
        return divmod(int(s), self.n_cols)

    def is_wall(self, r: int, c: int) -> bool:
        return bool(self.grid[r, c] == 1)

    # ------------------------------------------------------------------
    # Xây dựng bảng chuyển trạng thái
    # ------------------------------------------------------------------
    def _build_tables(self) -> None:
        S, A = self.n_states, N_ACTIONS
        self.next_state = np.zeros((S, A), dtype=np.int64)  # trạng thái kế tiếp
        self.reward = np.zeros((S, A), dtype=np.float64)    # phần thưởng
        self.done = np.zeros((S, A), dtype=bool)            # đã đến đích?
        self.hit = np.zeros((S, A), dtype=bool)             # có va tường?

        for s in range(S):
            r, c = self.to_pos(s)
            for a, (dr, dc) in enumerate(ACTION_DELTAS):
                nr, nc = r + dr, c + dc
                out_of_bound = not (0 <= nr < self.n_rows and 0 <= nc < self.n_cols)
                if out_of_bound or self.grid[nr, nc] == 1:
                    # Va tường / ra ngoài biên: đứng yên và bị phạt -10
                    self.next_state[s, a] = s
                    self.reward[s, a] = REWARD_WALL
                    self.hit[s, a] = True
                else:
                    ns = self.to_state(nr, nc)
                    self.next_state[s, a] = ns
                    if ns == self.goal_state:
                        self.reward[s, a] = REWARD_GOAL
                        self.done[s, a] = True
                    else:
                        self.reward[s, a] = REWARD_STEP

    # ------------------------------------------------------------------
    def step(self, s: int, a: int):
        """Thực hiện hành động a tại trạng thái s.

        Trả về (trạng_thái_mới, phần_thưởng, đã_đến_đích, có_va_tường).
        """
        return (
            int(self.next_state[s, a]),
            float(self.reward[s, a]),
            bool(self.done[s, a]),
            bool(self.hit[s, a]),
        )

    def free_states(self) -> List[int]:
        """Danh sách các trạng thái không phải tường."""
        return [s for s in range(self.n_states) if self.grid[self.to_pos(s)] == 0]


# ---------------------------------------------------------------------------
# Sinh mê cung ngẫu nhiên (luôn đảm bảo có đường đi)
# ---------------------------------------------------------------------------
def _solvable(layout: List[str]) -> bool:
    from .baselines import bfs_shortest_path  # import trễ để tránh vòng lặp import
    try:
        return bfs_shortest_path(Maze(layout)) is not None
    except ValueError:
        return False


def generate_random_layout(size: int = 8, wall_prob: float = 0.25, seed: int = 0) -> List[str]:
    """Sinh mê cung ngẫu nhiên: S ở góc trên trái, G ở góc dưới phải.

    Thử nhiều lần cho đến khi có đường đi từ S đến G (BFS kiểm tra).
    Nếu không thành công thì dùng mẫu mặc định.
    """
    rng = np.random.default_rng(seed)
    for _ in range(500):
        grid = (rng.random((size, size)) < wall_prob)
        layout = []
        for r in range(size):
            row = ""
            for c in range(size):
                if (r, c) == (0, 0):
                    row += "S"
                elif (r, c) == (size - 1, size - 1):
                    row += "G"
                else:
                    row += "#" if grid[r, c] else "."
            layout.append(row)
        if _solvable(layout):
            return layout
    return list(PRESET_LAYOUTS["Mẫu 1 – Cổ điển"])
