# Introduce-To-AI

**HDSD trước khi dùng:**

Lưu ý cần cài đặt thư viện *GraphViz* từ đường dẫn: https://graphviz.org/download/

Mở cmd và clone github:
```bash
git clone https://github.com/danielway2k3/Introduce-To-AI.git
```
Di chuyển vào thư mục:
```bash
cd Introduce-To-AI
cd TH_AI_Tuan 2
```
Tải các thư viện cần thiết
```bash
pip install -r requirements.txt
```
Khởi tạo không gian trạng thái(độ sâu là 20):
```bash
python generate_full_space_tree.py -d 20
```
Xây dựng tìm kiếm bằng thuật toán DFS:
```bash
python main.py -m dfs -l true
```
Xây dựng tìm kiếm bằng thuật toán BFS:
```bash
python main.py -m bfs -l true
```

