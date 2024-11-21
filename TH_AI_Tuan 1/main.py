from collections import defaultdict
from queue import Queue, PriorityQueue
from collections import deque

class Graph:
    def __init__(self, matrix=None):
        self.adj_matrix = matrix if matrix is not None else []
        self.adj_list = defaultdict(list)
        self.is_weighted = False

    # đọc dữ liệu từ file txt và khởi tạo ma trận kề
    @classmethod
    def from_file(cls, file):
        size = int(file.readline())
        start, goal = [int(num) for num in file.readline().split()]
        matrix = [[int(num) for num in line.split()] for line in file]
        return cls(matrix), start, goal

    # chuyển ma trận kề thành danh sách kề
    def convert_to_adj_list(self):
        self.adj_list = defaultdict(list)
        for i in range(len(self.adj_matrix)):
            for j in range(len(self.adj_matrix[i])):
                if self.adj_matrix[i][j] == 1:
                    self.adj_list[i].append(j)
        self.is_weighted = False

    # chuyển ma trận có trọng số thành danh sách kề với trọng số
    def convert_to_weighted_adj_list(self):
        self.adj_list = defaultdict(list)
        for i in range(len(self.adj_matrix)):
            for j in range(len(self.adj_matrix[i])):
                if self.adj_matrix[i][j] != 0:
                    self.adj_list[i].append((j, self.adj_matrix[i][j]))
        self.is_weighted = True

    # trả về ma trận kề
    def get_adj_matrix(self):
        return self.adj_matrix

    # trả về danh sách kề
    def get_adj_list(self):
        return self.adj_list

    # Thuật toán BFS
    def BFS(self, start, end):
        if not self.adj_list:
            self.convert_to_adj_list()

        # Khởi tạo queue và visited
        frontier = Queue()
        frontier.put(start)
        visited = set()

        parent = dict()
        parent[start] = None
        
        path_found = False

        while True:
            if frontier.empty():
                raise Exception("No way Exception")

            current_node = frontier.get()
            visited.add(current_node)

            if current_node == end:
                path_found = True
                break
                
            for node in self.adj_list[current_node]:
                if node not in visited:
                    frontier.put(node)                 
                    parent[node] = current_node
                    
        path = []
        if path_found:
            while end is not None:
                path.append(end)
                end = parent[end]
            path.reverse()

        return path

       
    # Thuật toán DFS
    def DFS(self, start, end):
        if not self.adj_list:
            self.convert_to_adj_list()

        frontier = []
        visited = set()

        frontier.append(start)

        parent = dict()
        parent[start] = None

        path_found = False

        while True:
            if frontier == []:
                raise Exception("No way Exception")

            current_node = frontier.pop()
            visited.add(current_node)

            if current_node == end:
                path_found = True
                break

            for node in self.adj_list[current_node]:
                if node not in visited:
                    frontier.append(node)
                    parent[node] = current_node

        path = []
        if path_found:
            while end is not None:
                path.append(end)
                end = parent[end]
            path.reverse()

        return path

    # Thuật toán UCS
    def UCS(self, start, end):
        if not self.adj_list:
            self.convert_to_weighted_adj_list()

        visited = set()
        frontier = PriorityQueue()

        parent = {start: None}
        frontier.put((0, start))
        cost_so_far = {start: 0} #thêm dictionary để lưu trữ chi phí
        path_found = False

        while True:
            if frontier.empty():
               raise Exception("No way Exception")
            
            current_w, current_node = frontier.get()
            visited.add(current_node)

            if current_node == end:
                path_found = True
                break

            for nodei in self.adj_list[current_node]:
                node, weight = nodei
                new_cost = current_w + weight

                if node not in cost_so_far or new_cost < cost_so_far[node]:
                    cost_so_far[node] = new_cost
                    frontier.put((new_cost, node))
                    parent[node] = current_node

        path = []
        if path_found:
            while end is not None:
                path.append(end)
                end = parent[end]
            path.reverse()

        return cost_so_far[path[-1]], path
    
if __name__ == "__main__":
    # đọc File Input.txt và InputUCS.txt
    file1 = open("Input.txt", "r")
    file2 = open("InputUCS.txt", "r")
    graph_1, start_1, goal_1 = Graph.from_file(file1)
    graph_2, start_2, goal_2 = Graph.from_file(file2)
    file1.close()
    file2.close()

    # Thực thi thuật toán BFS:
    result_bfs = graph_1.BFS(start_1, goal_1)
    print("Kết quả sử dụng thuật toán BFS: \n", result_bfs)
    # Thực thi thuật toán DFS:
    result_dfs = graph_1.DFS(start_1, goal_1)
    print("Kết quả sử dụng thuật toán DFS: \n", result_dfs)
    # Thực thi thuật toán UCS:
    cost, result_ucs = graph_2.UCS(start_2, goal_2)
    print("Kết quả sử dụng thuật toán UCS: \n", result_ucs, "với tổng chi phí là: ", cost)