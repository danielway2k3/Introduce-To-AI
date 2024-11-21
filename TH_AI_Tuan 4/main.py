from collections import defaultdict
from queue import PriorityQueue, Queue
import math
from matplotlib import pyplot as plt

class Point(object):
    def __init__(self, x, y, polygon_id=-1):
        self.x = x
        self.y = y
        self.polygon_id = polygon_id
        self.g = 0
        self.pre = None
    
    def rel(self, other, line):
        return line.d(self) * line.d(other) >= 0
    
    def can_see(self, other, line):
        l1 = self.line_to(line.p1)
        l2 = self.line_to(line.p2)
        d3 = line.d(self) * line.d(other) < 0
        d1 = other.rel(line.p2, l1)
        d2 = other.rel(line.p1, l2)
        return not (d1 and d2 and d3)
    
    def line_to(self, other):
        return Edge(self, other)
    
    def heuristic(self, other):
        return euclid_distance(self, other)
    
    def __eq__(self, point):
        return point and self.x == point.x and self.y == point.y
    
    def __ne__(self, point):
        return not self.__eq__(point)
    
    def __lt__(self, point):
        return hash(self) < hash(point)
    
    def __str__(self):
        return "(%d, %d)" % (self.x, self.y)
    
    def __hash__(self):
        return self.x.__hash__() ^ self.y.__hash__()
    
    def __repr__(self):
        return "(%d, %d)" % (self.x, self.y)
    
class Edge(object):
    def __init__(self, point1, point2):
        self.p1 = point1
        self.p2 = point2
    
    def get_adjacent(self, point):
        if point == self.p1:
            return self.p2
        if point == self.p2:
            return self.p1
        
    def d(self, point):
        vect_a = Point(self.p2.x - self.p1.x, self.p2.y - self.p1.y)
        vect_n = Point(-vect_a.y, vect_a.x)
        return vect_n.x * (point.x - self.p1.x) + vect_n.y * (point.y - self.p1.y)

    def __str__(self):
        return "({}, {})".format(self.p1, self.p2)

    def __contains__(self, point):
        return self.p1 == point or self.p2 == point

    def __hash__(self):
        return self.p1.__hash__() ^ self.p2.__hash__()

    def __repr__(self):
        return "Edge({!r}, {!r})".format(self.p1, self.p2)    
    
class Graph:
    def __init__(self, polygons):
        self.graph = defaultdict(set)
        self.edges = set()
        self.polygons = defaultdict(set)
        pid = 0
        for polygon in polygons:
            if len(polygon) == 2:
                polygon.pop()
            if polygon[0] == polygon[-1]:
                self.add_point(polygon[0])
            else:
                for i, point in enumerate(polygon):
                    neighbor_point = polygon[(i + 1) % len(polygon)]
                    edge = Edge(point, neighbor_point)
                    if len(polygon) > 2:
                        point.polygon_id = pid
                        neighbor_point.polygon_id = pid
                        self.polygons[pid].add(edge)
                    self.add_edge(edge)
                if len(polygon) > 2:
                    pid += 1

    def get_adjacent_points(self, point):
        return list(filter(None.__ne__, [edge.get_adjacent(point) for edge in self.edges]))
    
    def can_see(self, start):
        see_list = list()
        
        # Nếu điểm start thuộc một đa giác
        if start.polygon_id != -1:
            # Thêm các điểm kề với start trong cùng đa giác
            current_polygon_points = self.get_polygon_points(start.polygon_id)
            for point in self.get_adjacent_points(start):
                if point in current_polygon_points:
                    see_list.append(point)
            
            # Kiểm tra tầm nhìn tới các điểm của đa giác khác
            for point in self.get_points():
                if point != start and point.polygon_id != start.polygon_id:
                    path_clear = True
                    path_line = Edge(start, point)
                    
                    # Kiểm tra giao cắt với tất cả các đa giác
                    for polygon in self.polygons.values():
                        for edge in polygon:
                            if (edge.p1 != start and edge.p2 != start and 
                                edge.p1 != point and edge.p2 != point):
                                if do_edges_intersect(path_line, edge):
                                    path_clear = False
                                    break
                        if not path_clear:
                            break
                    
                    if path_clear:
                        see_list.append(point)
        else:
            # Nếu điểm start không thuộc đa giác nào (điểm start hoặc goal)
            for point in self.get_points():
                if point != start:
                    path_clear = True
                    path_line = Edge(start, point)
                    
                    for polygon in self.polygons.values():
                        for edge in polygon:
                            if (edge.p1 != start and edge.p2 != start and 
                                edge.p1 != point and edge.p2 != point):
                                if do_edges_intersect(path_line, edge):
                                    path_clear = False
                                    break
                        if not path_clear:
                            break
                    
                    if path_clear:
                        see_list.append(point)
        
        return see_list

        
    def get_polygon_points(self, index):
        point_set = set()
        for edge in self.polygons[index]:
            point_set.add(edge.p1)
            point_set.add(edge.p2)
        return point_set
    
    def get_points(self):
        return list(self.graph)
    
    def get_edges(self):
        return list(self.edges)
    
    def add_point(self, point):
        self.graph[point].add(point)

    def add_edge(self, edge):
        self.graph[edge.p1].add(edge)
        self.graph[edge.p2].add(edge)
        self.edges.add(edge)

    def __contains__(self, item):
        if isinstance(item, Point):
            return item in self.graph
        if isinstance(item, Edge):
            return item in self.edges
        return False
    
    def __getitem__(self, point):
        if point in self.graph:
            return self.graph[point]
        return set()
        
    def __str__(self):
        res = ""
        for point in self.graph:
            res += "\n" + str(point) + ": "
            for edge in self.graph[point]:
                res += str(edge)
        return res
    
    def __repr__(self):
        return self.__str__()

    def h(self, point):
        heuristic = getattr(self, 'heuristic', None)
        if heuristic:
            return heuristic[point]
        else:
            return -1

def euclid_distance(point1, point2):
    return round(float(math.sqrt((point2.x - point1.x)**2 + (point2.y - point1.y)**2)), 3)

# kiểm tra 2 đường thằng giao nhau
def do_edges_intersect(edge1, edge2):
    # xác định vị trí tương đối của điểm r với đoạn thẳng pq
    def orientation(p, q, r):
        val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
        if val == 0:
            return 0
        return 1 if val > 0 else 2
    # kiểm tra xem 3 điểm có thẳng hàng không
    def on_segment(p, q, r):
        return (q.x <= max(p.x, r.x) and q.x >= min(p.x, r.x) and
                q.y <= max(p.y, r.y) and q.y >= min(p.y, r.y))

    p1, q1 = edge1.p1, edge1.p2
    p2, q2 = edge2.p1, edge2.p2

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # Trường hợp tổng quát
    if o1 != o2 and o3 != o4:
        return True

    # Trường hợp đặc biệt
    if o1 == 0 and on_segment(p1, p2, q1): return True
    if o2 == 0 and on_segment(p1, q2, q1): return True
    if o3 == 0 and on_segment(p2, p1, q2): return True
    if o4 == 0 and on_segment(p2, q1, q2): return True

    return False

def search(graph, start, goal, func):
    closed = set()
    queue = PriorityQueue()
    queue.put((0 + func(graph, start), start))
    if start not in closed:
        closed.add(start)
    while not queue.empty():
        cost, node = queue.get()
        if node == goal:
            return node
        for i in graph.can_see(node):
            new_cost = node.g + euclid_distance(node, i)
            if i not in closed or new_cost < i.g:
                closed.add(i)
                i.g = new_cost
                i.pre = node
                new_cost = func(graph, i)
                queue.put((new_cost, i))
    return node
a_star = lambda graph, i: i.g + graph.h(i)
greedy = lambda graph, i: graph.h(i)

def BFS(graph, start, end):
    frontier = Queue()
    frontier.put(start)
    visited = {start}
    parent = {start: None}

    while not frontier.empty():
        current = frontier.get()
        
        if current == end:
            path = []
            while current:
                path.append(current)
                current = parent[current]
            return path[::-1]

        for next_node in graph.can_see(current):
            if next_node not in visited:
                frontier.put(next_node)
                visited.add(next_node)
                parent[next_node] = current
    return []

def DFS(graph, start, end):
    frontier = [(start, 0)]
    visited = {start}
    parent = {start: None}

    while frontier:
        current, _ = frontier.pop()
        
        if current == end:
            path = []
            while current:
                path.append(current)
                current = parent[current]
            return path[::-1]

        visible_nodes = []
        for next_node in graph.can_see(current):
            if next_node not in visited:
                dist = euclid_distance(next_node, end)
                visible_nodes.append((next_node, dist))
        
        visible_nodes.sort(key=lambda x: x[1], reverse=True)
        
        for next_node, _ in visible_nodes:
            frontier.append((next_node, _))
            visited.add(next_node)
            parent[next_node] = current
    return []

def UCS(graph, start, end):
    frontier = PriorityQueue()
    frontier.put((0, start))
    visited = set()
    parent = {start: None}
    cost_so_far = {start: 0}

    while not frontier.empty():
        current_cost, current = frontier.get()
        
        if current == end:
            path = []
            while current:
                path.append(current)
                current = parent[current]
            return current_cost, path[::-1]

        if current in visited:
            continue

        visited.add(current)

        for next_node in graph.can_see(current):
            if next_node not in visited:
                new_cost = current_cost + euclid_distance(current, next_node)
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    frontier.put((new_cost, next_node))
                    parent[next_node] = current

    return float('inf'), []

def is_point_in_polygon(point, polygon):
    """
    Kiểm tra một điểm có nằm trong polygon không sử dụng ray casting algorithm
    """
    if len(polygon) < 3:
        return False
    
    # Đếm số lần tia từ điểm cắt các cạnh của polygon nếu là lẻ thì điểm nằm trong polygon
    intersect_count = 0
    for i in range(len(polygon)):
        j = (i + 1) % len(polygon)
        if ((polygon[i].y > point.y) != (polygon[j].y > point.y) and
            point.x < (polygon[j].x - polygon[i].x) * (point.y - polygon[i].y) /
            (polygon[j].y - polygon[i].y) + polygon[i].x):
            intersect_count += 1
    
    return intersect_count % 2 == 1

def main():
    n_polygon = 0
    poly_list = list(list())
    x = list()
    y = list()
    with open('Input.txt', 'r') as f:
        line = f.readline()
        line = line.strip()
        line = line.split()
        line = list(map(int, line))
        n_polygon = line[0]
        start = Point(line[1], line[2])
        goal = Point(line[3], line[4])
        poly_list.append([start])
        for line in f:
            point_list = list()
            line = line.split()
            n_vertex = int(line[0])
            for j in range(0, 2 * n_vertex, 2):
                point_list.append(Point(int(line[j + 1]), int(line[j + 2])))
            poly_list.append(point_list[:])
        poly_list.append([goal])
        
        # Kiểm tra xem goal có nằm trong bất kỳ polygon nào không
        goal_inside = False
        for polygon in poly_list[1:-1]: 
            if is_point_in_polygon(goal, polygon):
                goal_inside = True
                break
                
        if goal_inside:
            print("Goal is inside a polygon. No path possible.")
            return
            
        graph = Graph(poly_list)
        graph.heuristic = {point: point.heuristic(goal) for point in graph.get_points()}

    print("Chọn thuật toán:")
    print("1. A*")
    print("2. Greedy")
    print("3. BFS")
    print("4. DFS")
    print("5. UCS")
    option = int(input("Nhập số (1-5): "))

    result = []
    if option == 1:
        a = search(graph, start, goal, a_star)
        while a:
            result.append(a)
            a = a.pre
        result.reverse()
    elif option == 2:
        a = search(graph, start, goal, greedy)
        while a:
            result.append(a)
            a = a.pre
        result.reverse()
    elif option == 3:
        result = BFS(graph, start, goal)
    elif option == 4:
        result = DFS(graph, start, goal)
    elif option == 5:
        _, result = UCS(graph, start, goal)
    else:
        print("Lựa chọn không hợp lệ!")
        return

    if not result:
        print("Goal is inside a polygon. No path possible.")
        return

    # In kết quả
    print_res = [[point, point.polygon_id] for point in result]
    print(*print_res, sep=' ->')

    # Vẽ đồ thị
    plt.figure()
    plt.plot([start.x], [start.y], 'ro')
    plt.plot([goal.x], [goal.y], 'ro')

    for point in graph.get_points():
        x.append(point.x)
        y.append(point.y)
    plt.plot(x, y, 'ro')
    for i in range(1, len(poly_list) - 1):
        coord = list()
        for point in poly_list[i]:
            coord.append([point.x, point.y])
        coord.append(coord[0])
        xs, ys = zip(*coord) # create lists ò x and y values
        plt.plot(xs, ys)
    x = list()
    y = list()
    for point in result:
        x.append(point.x)
        y.append(point.y)
    plt.plot(x, y, 'b', linewidth=2.0)
    plt.show()


if __name__ == "__main__":
    main()


