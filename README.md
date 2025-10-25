#  Chess Engine with Neural Network (PyTorch)

##  Mục tiêu dự án
- Chuyển bàn cờ thành ma trận đầu vào cho mạng neural (13 kênh).
- Huấn luyện mô hình dự đoán nước đi hợp lý tiếp theo.
- Xây dựng engine chơi cờ có thể tự học từ dữ liệu thực hoặc tự chơi (self-play).

##  Cấu trúc thư mục
```
CHESS-ENGINE-MAINS/
│
├── data/
│   ├── pgn/                     # Thư mục chứa các file dữ liệu ván cờ (.pgn)
│   └── info.txt                 # Thông tin mô tả dữ liệu
│
├── engines/
│   ├── tensorflow/
│   │   └── train_and_predict.ipynb    # Notebook huấn luyện và dự đoán bằng TensorFlow ( cài này để chạy test thôi ạ, elo cờ vua khá thấp-khoảng 500-600)
│   │
│   └── torch/
│       ├── __pycache__/               # Thư mục cache tự động của Python
│       ├── auxiliary_func.py          # Các hàm tiện ích hỗ trợ xử lý bàn cờ
│       ├── dataset.py                 # Chuẩn bị dữ liệu huấn luyện từ file PGN
│       ├── main.ipynb                 # Notebook chạy thử hoặc demo AI cờ vua (AI elo 1400-1500)
│       ├── model.py                   # Định nghĩa kiến trúc mạng neural (PyTorch)
│       └── train.ipynb                # Notebook huấn luyện mô hình PyTorch
│
├── images/                            # Ảnh quân cờ cho giao diện hiển thị
│   ├── b.png, b1.png
│   ├── k.png, k1.png
│   ├── n.png, n1.png
│   ├── p.png, p1.png
│   ├── q.png, q1.png
│   └── r.png, r1.png
│
├── models/
│                 
│   ├── move_to_int                    # Ánh xạ nước đi ra số nguyên
│   └── TORCH_100EPOCHS.pth            # Trọng số mô hình huấn luyện PyTorch
│
├── .gitignore                         # Bỏ qua các file/thư mục khi commit Git
├── README.md                          # Mô tả dự án 
├── requirements.txt                   # Danh sách thư viện Python cần thiết
└── setup.cfg                          # Cấu hình cho dự án hoặc package
```

## Mô tả bài toán và thuật toán

### Dữ liệu
Lấy từ https://database.nikonoel.fr/ (1 web có các thông tin và data về hàng trăm và hàng triệu game từ các trận đấu từ 2000+ elo) file bọn mình chọn ở phần mô tả là "As a direct download (582 Mb, .7z format which you can open natively in linux or with 7-zip on windows)",do không thể commit data/pgn lên github được nên sau khi clone về phải tải dữ liệu này về và thêm mục pgn trong data chứa các file pgn dữ liệu ván đầu vào 

### Huấn luyện mô hình (Model Training)

Mô hình được huấn luyện bằng **mạng nơ-ron học sâu (Deep Neural Network – DNN)** xây dựng bằng **PyTorch**, thực hiện theo hướng **học có giám sát (supervised learning)**.  
Quá trình huấn luyện sử dụng **bộ dữ liệu cờ vua (ChessDataset)**
Mô hình được triển khai trên **GPU (nếu có)** để tăng tốc tính toán

Các thành phần chính trong quá trình huấn luyện:
- **Hàm mất mát (Loss function):** `CrossEntropyLoss()` – đo độ sai lệch giữa đầu ra mô hình và nhãn thực tế.
- 
- **Bộ tối ưu hóa (Optimizer):** `Adam` với `learning_rate = 0.0001`, giúp điều chỉnh trọng số thông qua lan truyền ngược.  
- **Số vòng lặp huấn luyện (Epochs):** `50`.  
- **Cắt gradient (Gradient Clipping):** `torch.nn.utils.clip_grad_norm_` được áp dụng để tránh hiện tượng gradient explosion.

Sau mỗi epoch, chương trình hiển thị:
- **Chỉ số mất mát trung bình (Average Loss)**  
- **Thời gian huấn luyện mỗi epoch**, giúp theo dõi hiệu suất mô hình.

Mục tiêu của quá trình này là tối ưu hóa trọng số của mạng nơ-ron nhằm **giảm sai số dự đoán và nâng cao độ chính xác** trong việc dự đoán kết quả bàn cờ.

### Kết quả và đánh giá mô hình 

Sau khi huấn luyện, mô hình AI Chess đạt **mức ELO trung bình khoảng 1400–1500**, như là **người chơi cờ trung cấp**.  
Mô hình thể hiện khả năng:
- **Đưa ra nước đi hợp lý trong các thế cờ cơ bản**  
- **Phản ứng khá tốt với các tình huống chiến thuật ngắn hạn**  
- **Tự điều chỉnh chiến lược thông qua quá trình huấn luyện có giám sát**

Hiệu suất của mô hình vẫn có thể được **cải thiện đáng kể** thông qua các hướng phát triển sau:
- 🔹 **Tăng kích thước và đa dạng của tập dữ liệu huấn luyện**, đặc biệt là từ các ván cờ có ELO cao.  
- 🔹 **Cải tiến kiến trúc mạng nơ-ron** 
- 🔹 **Kết hợp học tăng cường (Reinforcement Learning)** như AlphaZero để giúp AI tự học thông qua trải nghiệm thi đấu.  
- 🔹 **Điều chỉnh siêu tham số (hyperparameters)** như learning rate, số lớp ẩn, số neuron, hoặc batch size để tối ưu hóa hiệu năng.


## Hướng dẫn chơi trực tiếp với AI Chess bằng giao diện Pygame

### 1. Giới thiệu

Người chơi có thể lựa chọn màu quân (trắng hoặc đen) và thi đấu với **AI Chess**.  
AI sẽ dự đoán nước đi tốt nhất dựa trên trạng thái bàn cờ hiện tại, được mô hình học sâu phân tích.

---

### 2. Cấu hình ban đầu

Mã nguồn sử dụng:
- `pygame` để tạo giao diện đồ họa và xử lý sự kiện người chơi.  
- `python-chess` để quản lý logic bàn cờ, nước đi và luật chơi.  
- `torch` để nạp mô hình học sâu đã huấn luyện (`TORCH_100EPOCHS.pth`).  
- `pickle` để tải ánh xạ nước đi (`move_to_int` và `int_to_move`).  

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChessModel(num_classes=len(move_to_int))
model.load_state_dict(torch.load("models/TORCH_100EPOCHS.pth", map_location=device))
model.to(device)
model.eval()
```

---

### 3. Giao diện bàn cờ

- Bàn cờ có kích thước **480x480 px**, mỗi ô vuông được vẽ xen kẽ màu trắng và xanh.  
- Các quân cờ được nạp từ thư mục `images/` và hiển thị bằng Pygame:  
- Hàm `draw_board()` chịu trách nhiệm vẽ bàn cờ và cập nhật mỗi khi có nước đi mới.


### 4. Cơ chế dự đoán nước đi AI

AI sử dụng hàm `predict_move(board)` để chọn nước đi hợp lệ có **xác suất cao nhất** trong số các nước có thể:
Sau đó, hàm `ai_move(board)` sẽ **thực hiện nước đi đó** trên bàn cờ.
### 5. Cách chơi
Chạy code ở phần cuối trong mục main.ipynb
- Nếu chọn **white**, bạn sẽ đi trước.  
- Nếu chọn **black**, AI sẽ khởi động và đi trước.  

Người chơi chọn quân và di chuyển bằng **chuột trái**.  
AI sẽ tự động phản hồi sau mỗi lượt của bạn.

## Hướng dẫn cài đặt (Installation Guide)

### 1. Yêu cầu hệ thống
Để chạy được dự án **AI Chess**, máy tính của bạn cần:

- **Python 3.8+**  
- **Pip** đã được cài đặt sẵn  
- **GPU (khuyến nghị)** với **CUDA** nếu muốn huấn luyện hoặc chạy nhanh hơn  
- Hệ điều hành: Windows / macOS / Linux  

---

### 2. Tải dự án về máy
Clone hoặc tải dự án từ GitHub:

### 3. Cài đặt các thư viện cần thiết
Chạy lệnh sau trong terminal (nên sử dụng môi trường ảo `venv` hoặc `conda`):
```bash
pip install -r requirements.txt
```

### 4. Chuẩn bị mô hình và dữ liệu
Tải trọng số mô hình đã huấn luyện vào thư mục:

models/TORCH_100EPOCHS.pth
models/move_to_int



### 5. Kiểm tra cài đặt
Kiểm tra PyTorch đã nhận GPU hay chưa:

```python
import torch
print(torch.cuda.is_available())
```
Nếu trả về `True`, nghĩa là mô hình có thể chạy trên GPU.
  

---

## Contributors
- **Lê Tuấn Anh** – Xây dựng mô hình PyTorch, thiết kế cấu trúc dữ liệu và huấn luyện mạng neural  
- **Nguyễn Minh Đức** – Thu thập dữ liệu, Giao diện chơi cờ bằng Pygame, tích hợp mô hình AI  









