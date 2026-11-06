# 🤖 AI Q-Learning tìm đường trong mê cung

Ứng dụng web mô phỏng **tác tử AI học cách tìm đường tối ưu trong mê cung 8×8** bằng thuật toán
**Q-Learning**. Backend viết bằng **Python + NumPy**, giao diện bằng **Streamlit**.

> 🌐 **URL công khai:** _(điền sau khi deploy)_  
> 💻 **GitHub:** _(điền link repo của bạn)_

## ✨ Chức năng

| #   | Yêu cầu                                                                                         | Nơi thực hiện                                   |
| --- | ----------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| 1   | Mê cung 8×8 gồm S, G, tường, đường đi (3 mẫu + mê cung ngẫu nhiên luôn có lời giải)             | `core/maze.py`                                  |
| 2   | Q-Learning với 4 hành động: lên, xuống, trái, phải                                              | `core/qlearning.py`                             |
| 3   | Phần thưởng: đến đích **+50**, bước hợp lệ **−1**, va tường **−10**                             | `core/maze.py`                                  |
| 4   | Chỉnh Alpha, Gamma, Epsilon (+ hệ số giảm, epsilon tối thiểu), số episode, số bước tối đa, seed | Sidebar trong `app.py`                          |
| 5   | Hoạt ảnh tác tử di chuyển; xem lại các episode tiêu biểu với bảng Q tại đúng thời điểm đó       | Tab "Mê cung & hoạt ảnh"                        |
| 6   | So sánh Q-Learning với **BFS** và **tác tử ngẫu nhiên**                                         | `core/baselines.py`, Tab "So sánh"              |
| 7   | Biểu đồ Reward, Success Rate, số bước, số lần va chạm (có trung bình trượt)                     | Tab "Biểu đồ huấn luyện"                        |
| 8   | Thống kê qua nhiều lần chạy và nhiều random seed (mean ± std)                                   | `core/evaluation.py`, Tab "Thống kê nhiều seed" |
| 9   | Chạy lại, Reset, xuất CSV/JSON/ZIP                                                              | Sidebar + Tab "Xuất dữ liệu"                    |

## 🗂️ Cấu trúc thư mục

```
qlearning_maze/
├── app.py                 # Giao diện Streamlit
├── core/
│   ├── maze.py            # Môi trường mê cung, hành động, phần thưởng
│   ├── qlearning.py       # Thuật toán Q-Learning (epsilon-greedy)
│   ├── baselines.py       # BFS và tác tử ngẫu nhiên
│   ├── evaluation.py      # So sánh & thống kê nhiều seed
│   └── render.py          # Vẽ mê cung bằng HTML
├── tests/test_core.py     # Kiểm thử backend
├── requirements.txt
├── render.yaml            # Cấu hình deploy Render
└── .streamlit/config.toml
```

## ▶️ Chạy trên máy

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Mở trình duyệt tại http://localhost:8501.

Chạy kiểm thử backend: `python -m unittest discover -s tests -v`

## 🧠 Cách hoạt động

**Cập nhật Q-Learning (Bellman):**

```
Q(s,a) ← Q(s,a) + α · [ r + γ · max_a' Q(s',a') − Q(s,a) ]
```

- **α (Alpha)**: tốc độ học – lớn thì học nhanh nhưng dễ dao động.
- **γ (Gamma)**: mức coi trọng phần thưởng tương lai – gần 1 thì tác tử "nhìn xa".
- **ε (Epsilon)**: xác suất khám phá ngẫu nhiên. Mặc định bắt đầu 1.0 rồi giảm dần
  (× 0.995 mỗi episode) xuống tối thiểu 0.05: đầu tiên khám phá, về sau khai thác.
- **Phần thưởng**: đích +50, bước hợp lệ −1 (khuyến khích đường ngắn), va tường/ra biên −10
  (tác tử đứng yên tại chỗ).

**So sánh:** BFS biết trước bản đồ nên luôn cho đường ngắn nhất (mốc tối ưu). Tác tử ngẫu nhiên
là mốc thấp nhất. Q-Learning không biết bản đồ, chỉ học qua thử-sai nhưng sau huấn luyện thường
đạt số bước bằng BFS.

**Thống kê nhiều seed:** với mỗi seed, huấn luyện lại từ đầu rồi chạy nhiều episode đánh giá
(epsilon-greedy với ε nhỏ) để tính tỉ lệ thành công, số bước, phần thưởng, va chạm và episode hội tụ
(lần đầu tỉ lệ thành công trượt 50 episode ≥ 90%).

## ☁️ Deploy

### Cách 1 – Streamlit Community Cloud (đơn giản nhất)

1. Đẩy mã nguồn lên GitHub:
   ```bash
   git init
   git add .
   git commit -m "first commit"
   git branch -M main
   git remote add origin https://github.com/ThienTa2005/agent01.git
   git push -u origin main
   ```
2. Vào https://share.streamlit.io → đăng nhập bằng GitHub → **New app**.
3. Chọn repo, nhánh `main`, **Main file path** = `app.py` → **Deploy**.
4. Sau ~1–2 phút bạn có URL dạng `https://<tên-app>.streamlit.app`. Dán URL này vào đầu README.

### Cách 2 – Render

1. Đẩy mã lên GitHub như trên.
2. Vào https://render.com → **New +** → **Blueprint** (dùng file `render.yaml`)
   hoặc **Web Service** với:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
3. Chọn gói Free → Deploy. Render cấp URL dạng `https://qlearning-maze.onrender.com`
   (gói free sẽ "ngủ" khi không có truy cập, lần mở đầu hơi chậm).

## 📝 Gợi ý khi báo cáo / demo

- Thử `Alpha` rất nhỏ (0.01) hoặc `Episode` rất ít (50) để thấy tác tử học chưa xong.
- Đặt hệ số giảm Epsilon = 1.0 và Epsilon = 0.3 để thấy tác tử luôn khám phá ngẫu nhiên.
- Thử "Mẫu 3 – Đường vòng" hoặc mê cung ngẫu nhiên để kiểm tra khả năng tổng quát.
- Phát lại **Episode 1** và **Đường đi tối ưu (greedy)** để so sánh trước/sau khi học.
