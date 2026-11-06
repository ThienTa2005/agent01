"""
render.py – Vẽ mê cung bằng HTML/CSS thuần (không cần thư viện đồ họa).

Mỗi ô là một <td>. Có thể hiển thị:
    * tác tử (🤖), vệt đường đi (chấm xanh),
    * độ "nóng" của giá trị Q (ô càng xanh lá đậm = max Q càng cao),
    * mũi tên chính sách tốt nhất tại mỗi ô.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .maze import ACTION_ARROWS, Maze


def _q_color(value: float, vmin: float, vmax: float) -> str:
    """Chuyển giá trị Q thành màu nền (trắng -> xanh lá)."""
    if vmax - vmin < 1e-9:
        return "#f5f6fa"
    t = (value - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    # nội suy từ #f5f6fa (nhạt) đến #55c57a (xanh)
    r = int(245 + (85 - 245) * t)
    g = int(246 + (197 - 246) * t)
    b = int(250 + (122 - 250) * t)
    return f"rgb({r},{g},{b})"


def render_maze_html(
    maze: Maze,
    agent: Optional[int] = None,
    path: Optional[Sequence[int]] = None,
    q: Optional[np.ndarray] = None,
    show_arrows: bool = True,
    cell: int = 46,
    title: str = "",
) -> str:
    """Trả về chuỗi HTML của mê cung."""
    path_set = set(path) if path is not None else set()

    # Chuẩn hóa màu theo max Q của các ô đã được cập nhật
    vmin = vmax = 0.0
    if q is not None:
        free = maze.free_states()
        vals = np.array([q[s].max() for s in free])
        vmin, vmax = float(vals.min()), float(vals.max())

    rows_html = []
    for r in range(maze.n_rows):
        tds = []
        for c in range(maze.n_cols):
            s = maze.to_state(r, c)
            bg, content, color = "#f5f6fa", "", "#2d3436"

            if maze.grid[r, c] == 1:
                bg = "#2d3436"                                   # tường
            else:
                if q is not None:
                    bg = _q_color(q[s].max(), vmin, vmax)
                if s in path_set:
                    content = '<span style="color:#0984e3;font-size:20px">●</span>'
                if q is not None and show_arrows and np.any(q[s] != 0):
                    content = (f'<span style="color:#2d3436;font-size:18px;opacity:.75">'
                               f'{ACTION_ARROWS[int(np.argmax(q[s]))]}</span>')
                    if s in path_set:
                        bg = "#a5d8ff"
                elif s in path_set:
                    bg = "#a5d8ff"
                    content = ""
            if s == maze.start_state:
                bg, content, color = "#00b894", "S", "#ffffff"
            elif s == maze.goal_state:
                bg, content, color = "#fdcb6e", "G", "#2d3436"
            if agent is not None and s == agent:
                content = "🤖"

            tds.append(
                f'<td style="width:{cell}px;height:{cell}px;background:{bg};color:{color};'
                f'text-align:center;vertical-align:middle;font-weight:700;font-size:18px;'
                f'border:1px solid #b2bec3;padding:0">{content}</td>'
            )
        rows_html.append("<tr>" + "".join(tds) + "</tr>")

    caption = f'<div style="text-align:center;font-weight:600;margin-bottom:4px">{title}</div>' if title else ""
    return (
        f'{caption}<table style="border-collapse:collapse;margin:0 auto;table-layout:fixed">'
        + "".join(rows_html) + "</table>"
    )
