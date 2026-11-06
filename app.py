"""
app.py – Giao diện web Streamlit cho dự án "AI Q-Learning tìm đường trong mê cung".

Chạy local:   streamlit run app.py
Cấu trúc:
    Sidebar : chọn mê cung + chỉnh tham số (Alpha, Gamma, Epsilon, Episode...)
    Tab 1   : mê cung & hoạt ảnh tác tử học/di chuyển
    Tab 2   : biểu đồ huấn luyện (Reward, Success Rate, số bước, va chạm)
    Tab 3   : so sánh Q-Learning với BFS và Random
    Tab 4   : thống kê nhiều lần chạy / nhiều random seed
    Tab 5   : xuất dữ liệu kết quả
"""
import io
import json
import time
import zipfile
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import streamlit as st

from core import (ACTION_NAMES, PRESET_LAYOUTS, Maze, QConfig, compare_agents,
                  evaluate_multi_seed, generate_random_layout, greedy_rollout,
                  render_maze_html, summarize_multi_seed, train_q_learning)
from core.evaluation import convergence_episode

st.set_page_config(page_title="AI Q-Learning – Mê cung", page_icon="🤖", layout="wide")


# ===========================================================================
# Hàm tiện ích
# ===========================================================================
@st.cache_data(show_spinner=False)
def cached_multi_seed(layout, cfg_dict, n_seeds, eval_runs, eval_epsilon):
    """Chạy đánh giá nhiều seed (có cache để không phải chạy lại khi đổi tab)."""
    maze = Maze(list(layout))
    cfg = QConfig(**cfg_dict)
    return evaluate_multi_seed(maze, cfg, n_seeds, eval_runs, eval_epsilon)


def history_dataframe(res, window: int) -> pd.DataFrame:
    """Tạo bảng lịch sử huấn luyện + các đường trung bình trượt (moving average)."""
    df = pd.DataFrame({
        "episode": np.arange(1, len(res.rewards) + 1),
        "reward": res.rewards,
        "success": res.successes,
        "steps": res.steps,
        "collisions": res.collisions,
        "epsilon": res.epsilons,
    })
    for col in ["reward", "success", "steps", "collisions"]:
        df[f"{col}_ma"] = df[col].rolling(window, min_periods=1).mean()
    return df


def q_table_dataframe(maze: Maze, q: np.ndarray) -> pd.DataFrame:
    """Bảng Q dạng DataFrame (chỉ các ô không phải tường)."""
    rows = []
    for s in maze.free_states():
        r, c = maze.to_pos(s)
        row = {"state": s, "row": r, "col": c}
        for a, name in enumerate(ACTION_NAMES):
            row[name] = round(float(q[s, a]), 4)
        row["best_action"] = ACTION_NAMES[int(np.argmax(q[s]))] if np.any(q[s] != 0) else "-"
        rows.append(row)
    return pd.DataFrame(rows)


def show_maze(placeholder, maze, **kw):
    placeholder.markdown(render_maze_html(maze, **kw), unsafe_allow_html=True)


# ===========================================================================
# SIDEBAR: chọn mê cung + tham số
# ===========================================================================
st.sidebar.header("⚙️ Thiết lập")

st.sidebar.subheader("1. Mê cung 8×8")
maze_choice = st.sidebar.selectbox("Chọn mê cung", list(PRESET_LAYOUTS.keys()) + ["Ngẫu nhiên"])
if maze_choice == "Ngẫu nhiên":
    maze_seed = st.sidebar.number_input("Seed mê cung", 0, 10_000, 0, step=1)
    wall_prob = st.sidebar.slider("Mật độ tường", 0.05, 0.40, 0.25, 0.05)
    layout = generate_random_layout(8, wall_prob, int(maze_seed))
else:
    layout = PRESET_LAYOUTS[maze_choice]
maze = Maze(layout)

st.sidebar.subheader("2. Tham số Q-Learning")
alpha = st.sidebar.slider("Alpha (tốc độ học)", 0.01, 1.0, 0.5, 0.01)
gamma = st.sidebar.slider("Gamma (chiết khấu)", 0.0, 0.999, 0.95, 0.001)
epsilon = st.sidebar.slider("Epsilon ban đầu", 0.0, 1.0, 1.0, 0.01)
eps_decay = st.sidebar.slider("Hệ số giảm Epsilon / episode", 0.90, 1.0, 0.995, 0.001,
                              help="1.0 = giữ nguyên epsilon, không giảm.")
eps_min = st.sidebar.slider("Epsilon tối thiểu", 0.0, 0.5, 0.05, 0.01)
episodes = st.sidebar.slider("Số episode", 50, 5000, 500, 50)
max_steps = st.sidebar.slider("Số bước tối đa / episode", 20, 500, 100, 10)
seed = st.sidebar.number_input("Random seed", 0, 1_000_000, 42, step=1)

cfg = QConfig(alpha=alpha, gamma=gamma, epsilon=epsilon, epsilon_decay=eps_decay,
              epsilon_min=min(eps_min, epsilon), episodes=int(episodes),
              max_steps=int(max_steps), seed=int(seed))

# "Chữ ký" của cấu hình hiện tại: dùng để biết kết quả đã lưu có còn khớp hay không
signature = (tuple(layout), json.dumps(asdict(cfg), sort_keys=True))

st.sidebar.subheader("3. Điều khiển")
c1, c2 = st.sidebar.columns(2)
train_clicked = c1.button("🚀 Huấn luyện", type="primary")
reset_clicked = c2.button("🔄 Reset")

# Reset: xóa toàn bộ kết quả đã lưu
if reset_clicked:
    for k in ["result", "sig", "eval_df", "eval_sig"]:
        st.session_state.pop(k, None)
    st.rerun()

# Huấn luyện (chạy lại bất cứ lúc nào bằng cách bấm lại nút)
if train_clicked:
    bar = st.sidebar.progress(0.0, text="Đang huấn luyện...")
    result = train_q_learning(maze, cfg, lambda i, n: bar.progress(i / n, text=f"Episode {i}/{n}"))
    bar.empty()
    st.session_state["result"] = result
    st.session_state["sig"] = signature

result = st.session_state.get("result")
stale = result is not None and st.session_state.get("sig") != signature


# ===========================================================================
# TRANG CHÍNH
# ===========================================================================
st.title("🤖 AI Q-Learning tìm đường trong mê cung")
st.caption("Tác tử học cách đi từ **S** đến **G** trên lưới 8×8 bằng Q-Learning. "
           "Phần thưởng: đến đích **+50**, bước hợp lệ **−1**, va tường **−10**.")

if result is None:
    st.info("👈 Chỉnh tham số ở thanh bên rồi bấm **🚀 Huấn luyện** để bắt đầu.")
elif stale:
    st.warning("Tham số hoặc mê cung đã thay đổi so với lần huấn luyện gần nhất. "
               "Bấm **🚀 Huấn luyện** để cập nhật kết quả.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧭 Mê cung & hoạt ảnh", "📈 Biểu đồ huấn luyện", "⚔️ So sánh thuật toán",
    "🎲 Thống kê nhiều seed", "💾 Xuất dữ liệu"])

# ---------------------------------------------------------------------------
# TAB 1: Mê cung & hoạt ảnh
# ---------------------------------------------------------------------------
with tab1:
    left, right = st.columns([1, 1])
    with right:
        st.markdown("**Chú giải**")
        st.markdown("- 🟩 **S**: điểm bắt đầu  \n- 🟨 **G**: đích  \n- ⬛ tường  \n"
                    "- 🤖 tác tử  \n- Ô **xanh lá** càng đậm = giá trị Q càng cao  \n"
                    "- Mũi tên = hành động tốt nhất mà tác tử đã học  \n"
                    "- Ô **xanh dương nhạt** = đường tác tử đã đi")
        if result is not None:
            # Danh sách các "đoạn phim" có thể phát lại
            options = {f"Episode {ep + 1}  (ε={snap['epsilon']:.2f}, "
                       f"{'✅ tới đích' if snap['success'] else '❌ chưa tới đích'})": ep
                       for ep, snap in sorted(result.snapshots.items())}
            options["🏁 Đường đi tối ưu sau khi học xong (greedy)"] = "final"
            choice = st.selectbox("Chọn đoạn để phát lại", list(options.keys()),
                                  index=len(options) - 1)
            speed = st.slider("Tốc độ (giây / bước)", 0.02, 1.0, 0.15, 0.01)
            play = st.button("▶️ Phát hoạt ảnh")
        else:
            play = False

    with left:
        placeholder = st.empty()

    if result is None:
        # Chưa huấn luyện: chỉ hiển thị mê cung trống
        show_maze(placeholder, maze, agent=maze.start_state)
    else:
        key = options[choice]
        if key == "final":
            traj = greedy_rollout(maze, result.q_table, cfg.max_steps)
            q_show = result.q_table
            info = (f"Đường đi greedy: **{traj['steps']} bước**, "
                    f"phần thưởng **{traj['reward']:.0f}**, va tường **{traj['collisions']}**, "
                    f"{'✅ đến đích' if traj['success'] else '❌ chưa đến đích'}.")
        else:
            snap = result.snapshots[key]
            traj = dict(path=snap["path"], reward=snap["reward"], collisions=snap["collisions"],
                        steps=len(snap["path"]) - 1, success=snap["success"])
            q_show = snap["q"]          # bảng Q tại đúng thời điểm episode đó -> thấy được quá trình học
            info = (f"Episode {key + 1}: **{traj['steps']} bước**, phần thưởng **{traj['reward']:.0f}**, "
                    f"va tường **{traj['collisions']}**, "
                    f"{'✅ đến đích' if traj['success'] else '❌ chưa đến đích'}.")
        path = traj["path"]

        if play:
            # Phát từng khung hình: tác tử di chuyển, vệt đường đi dài dần
            for i, s in enumerate(path):
                show_maze(placeholder, maze, agent=s, path=path[: i + 1], q=q_show)
                time.sleep(speed)
        else:
            show_maze(placeholder, maze, agent=path[-1], path=path, q=q_show)
        right.info(info)

        with right.expander("Xem bảng Q (các ô không phải tường)"):
            st.dataframe(q_table_dataframe(maze, result.q_table), hide_index=True)

# ---------------------------------------------------------------------------
# TAB 2: Biểu đồ huấn luyện
# ---------------------------------------------------------------------------
with tab2:
    if result is None:
        st.info("Chưa có dữ liệu. Hãy huấn luyện trước.")
    else:
        window = st.slider("Cửa sổ trung bình trượt (episode)", 1, 200, 25)
        hist = history_dataframe(result, window)
        last = max(1, len(hist) // 10)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Reward TB (10% cuối)", f"{hist['reward'].tail(last).mean():.1f}")
        m2.metric("Success rate (10% cuối)", f"{hist['success'].tail(last).mean() * 100:.0f}%")
        m3.metric("Số bước TB (10% cuối)", f"{hist['steps'].tail(last).mean():.1f}")
        conv = convergence_episode(result.successes)
        m4.metric("Hội tụ (SR≥90%)", "chưa đạt" if np.isnan(conv) else f"ep {int(conv)}")

        ca, cb = st.columns(2)
        with ca:
            st.markdown("**Tổng phần thưởng mỗi episode**")
            st.line_chart(hist.set_index("episode")[["reward", "reward_ma"]])
            st.markdown("**Số bước mỗi episode**")
            st.line_chart(hist.set_index("episode")[["steps", "steps_ma"]])
        with cb:
            st.markdown("**Tỉ lệ thành công (Success Rate, trung bình trượt)**")
            st.line_chart(hist.set_index("episode")[["success_ma"]])
            st.markdown("**Số lần va tường mỗi episode**")
            st.line_chart(hist.set_index("episode")[["collisions", "collisions_ma"]])
        st.markdown("**Epsilon theo episode**")
        st.line_chart(hist.set_index("episode")[["epsilon"]])

# ---------------------------------------------------------------------------
# TAB 3: So sánh Q-Learning / BFS / Random
# ---------------------------------------------------------------------------
with tab3:
    if result is None:
        st.info("Chưa có dữ liệu. Hãy huấn luyện trước.")
    else:
        cc1, cc2, cc3 = st.columns(3)
        runs = cc1.slider("Số lần chạy mỗi tác tử ngẫu nhiên", 50, 2000, 500, 50)
        eval_eps = cc2.slider("Epsilon khi đánh giá Q-Learning", 0.0, 0.3, 0.0, 0.01,
                              help="0 = hoàn toàn tham lam (greedy).")
        cmp_seed = cc3.number_input("Seed đánh giá", 0, 100_000, 0)
        cmp = compare_agents(maze, result.q_table, runs=int(runs), seed=int(cmp_seed),
                             max_steps=cfg.max_steps, eval_epsilon=eval_eps)

        pretty = cmp["table"].rename(columns={
            "success_rate": "Tỉ lệ thành công", "avg_steps": "Số bước TB (khi thành công)",
            "avg_reward": "Phần thưởng TB", "avg_collisions": "Va tường TB"})
        st.dataframe(pretty.round(2))

        g1, g2, g3 = st.columns(3)
        g1.markdown(render_maze_html(maze, path=cmp["q_path"], q=result.q_table, show_arrows=False,
                                     title=f"Q-Learning ({len(cmp['q_path']) - 1} bước)"),
                    unsafe_allow_html=True)
        if cmp["bfs_path"]:
            g2.markdown(render_maze_html(maze, path=cmp["bfs_path"],
                                         title=f"BFS – tối ưu ({len(cmp['bfs_path']) - 1} bước)"),
                        unsafe_allow_html=True)
        g3.markdown(render_maze_html(maze, path=cmp["random_path"],
                                     title=f"Random – 1 lần chạy mẫu ({len(cmp['random_path']) - 1} bước, "
                                           f"{'tới đích' if cmp['random_success'] else 'thất bại'})"),
                    unsafe_allow_html=True)

        b1, b2 = st.columns(2)
        b1.markdown("**Tỉ lệ thành công**")
        b1.bar_chart(cmp["table"][["success_rate"]])
        b2.markdown("**Phần thưởng trung bình**")
        b2.bar_chart(cmp["table"][["avg_reward"]])
        st.caption("BFS biết trước toàn bộ bản đồ nên luôn tối ưu – đây là mốc để đánh giá Q-Learning. "
                   "Tác tử Random chỉ là mốc nền (baseline) thấp nhất.")

# ---------------------------------------------------------------------------
# TAB 4: Thống kê nhiều seed
# ---------------------------------------------------------------------------
with tab4:
    st.markdown("Huấn luyện lại Q-Learning với **nhiều random seed** khác nhau (dùng tham số ở thanh bên) "
                "để đo độ ổn định của thuật toán.")
    s1, s2, s3 = st.columns(3)
    n_seeds = s1.slider("Số seed", 3, 30, 10)
    eval_runs = s2.slider("Số lần chạy đánh giá / seed", 20, 500, 100, 20)
    eval_e = s3.slider("Epsilon đánh giá", 0.0, 0.3, 0.05, 0.01)
    if st.button("▶️ Chạy thống kê nhiều seed"):
        with st.spinner("Đang huấn luyện nhiều seed..."):
            df = cached_multi_seed(tuple(layout), asdict(cfg), int(n_seeds), int(eval_runs), float(eval_e))
        st.session_state["eval_df"] = df
        st.session_state["eval_sig"] = signature

    eval_df = st.session_state.get("eval_df")
    if eval_df is None:
        st.info("Bấm nút phía trên để chạy thống kê.")
    else:
        if st.session_state.get("eval_sig") != signature:
            st.warning("Kết quả này được tạo với mê cung/tham số cũ.")
        summ = summarize_multi_seed(eval_df)
        st.markdown("**Trung bình (mean) và độ lệch chuẩn (std) theo tác tử**")
        st.dataframe(summ.round(3))

        q_rows = eval_df[eval_df["agent"] == "Q-Learning"]
        k1, k2, k3 = st.columns(3)
        k1.metric("Q-Learning: success rate TB", f"{q_rows['success_rate'].mean() * 100:.1f}%")
        k2.metric("Seed đạt 100% thành công", f"{int((q_rows['success_rate'] == 1).sum())}/{len(q_rows)}")
        k3.metric("Episode hội tụ TB", f"{q_rows['converge_episode'].mean():.0f}"
                  if q_rows["converge_episode"].notna().any() else "chưa đạt")

        metric_label = {"success_rate": "Tỉ lệ thành công", "avg_steps": "Số bước TB",
                        "avg_reward": "Phần thưởng TB", "avg_collisions": "Va tường TB"}
        metric = st.selectbox("Chỉ số theo từng seed", list(metric_label), format_func=metric_label.get)
        pivot = eval_df.pivot(index="seed", columns="agent", values=metric)
        st.bar_chart(pivot)
        with st.expander("Xem toàn bộ dữ liệu từng seed"):
            st.dataframe(eval_df.round(3), hide_index=True)

# ---------------------------------------------------------------------------
# TAB 5: Xuất dữ liệu
# ---------------------------------------------------------------------------
with tab5:
    if result is None:
        st.info("Chưa có dữ liệu để xuất. Hãy huấn luyện trước.")
    else:
        hist_all = history_dataframe(result, 25)
        qdf = q_table_dataframe(maze, result.q_table)
        cfg_json = json.dumps({"maze": layout, "config": asdict(cfg)}, ensure_ascii=False, indent=2)

        d1, d2, d3 = st.columns(3)
        d1.download_button("⬇️ Lịch sử huấn luyện (CSV)", hist_all.to_csv(index=False).encode("utf-8-sig"),
                           "training_history.csv", "text/csv")
        d2.download_button("⬇️ Bảng Q (CSV)", qdf.to_csv(index=False).encode("utf-8-sig"),
                           "q_table.csv", "text/csv")
        d3.download_button("⬇️ Cấu hình (JSON)", cfg_json.encode("utf-8"),
                           "config.json", "application/json")

        # Gói tất cả vào 1 file zip
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("training_history.csv", hist_all.to_csv(index=False))
            zf.writestr("q_table.csv", qdf.to_csv(index=False))
            zf.writestr("config.json", cfg_json)
            cmp_all = compare_agents(maze, result.q_table, runs=500, seed=0, max_steps=cfg.max_steps)
            zf.writestr("comparison.csv", cmp_all["table"].to_csv())
            if st.session_state.get("eval_df") is not None:
                zf.writestr("multi_seed_results.csv", st.session_state["eval_df"].to_csv(index=False))
        st.download_button("📦 Tải tất cả (ZIP)", buf.getvalue(), "ket_qua_q_learning.zip", "application/zip",
                           type="primary")
        st.caption("File ZIP gồm: lịch sử huấn luyện, bảng Q, cấu hình, bảng so sánh "
                   "và (nếu đã chạy) thống kê nhiều seed.")
        st.dataframe(hist_all.head(50), hide_index=True)
