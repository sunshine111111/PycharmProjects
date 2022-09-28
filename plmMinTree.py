#普里姆算法：在连通网中查找最小生成树
#普里姆算法查找最小生成树的过程，采用了贪心算法的思想。对于包含 N 个顶点的连通网，普里姆算法每次从连通网中找出一个权值最小的边，这样的操作重复 N-1 次，由 N-1 条权值最小的边组成的生成树就是最小生成树。

V = 6  # 图中顶点的个数
cost = [[0] * V for i in range(V)]
print("输入图（顶点到顶点的路径和权值)：")
while True:
    li = input().split()

    p1 = int(li[0])
    p2 = int(li[1])
    if p1 == -1 and p2 == -1:
        break
    wight = int(li[2])
    cost[p1 - 1][p2 - 1] = wight
    cost[p2 - 1][p1 - 1] = wight

# 查找权值最小的、尚未被选择的顶点，key 列表记录了各顶点之间的权值数据，visited列表记录着各个顶点是否已经被选择的信息
def min_Key(key, visited):
    # 遍历 key 列表使用，min 记录最小的权值，min_index 记录最小权值关联的顶点
    min = float('inf')
    min_index = 0
    # 遍历 key 列表
    for v in range(V):
        # 如果当前顶点为被选择，且对应的权值小于 min 值
        if visited[v] == False and key[v] < min:
            # 更新  min 的值并记录该顶点的位置
            min = key[v]
            min_index = v
    # 返回最小权值的顶点的位置
    return min_index


# 输出最小生成树
def print_MST(parent, cost):
    minCost = 0
    print("最小生成树为：")
    # 遍历 parent 列表
    for i in range(1, V):
        # parent 列表下标值表示各个顶点，各个下标对应的值为该顶点的父节点
        print("%d - %d wight:%d" % (parent[i] + 1, i + 1, cost[i][parent[i]]))
        # 统计最小生成树的总权值
        minCost = minCost + cost[i][parent[i]];
    print("总权值为：%d" % (minCost))


# 根据用户提供了图的信息（存储在 cost 列表中），寻找最小生成树
def find_MST(cost):
    # key 列表用于记录 B 类顶点到 A 类顶点的权值
    # parent 列表用于记录最小生成树中各个顶点父节点的位置，便于最终生成最小生成树
    # visited 列表用于记录各个顶点属于 A 类还是 B 类
    parent = [-1] * V
    key = [float('inf')] * V
    visited = [False] * V
    # 选择 key 列表中第一个顶点，开始寻找最小生成树
    key[0] = 0
    parent[0] = -1
    # 对于 V 个顶点的图，最需选择 V-1 条路径，即可构成最小生成树
    for x in range(V - 1):
        # 从 key 列表中找到权值最小的顶点所在的位置
        u = min_Key(key, visited)
        visited[u] = True
        # 由于新顶点加入 A 类，因此需要更新 key 列表中的数据
        for v in range(V):
            # 如果类 B 中存在到下标为 u 的顶点的权值比 key 列表中记录的权值还小，表明新顶点的加入，使得类 B 到类 A 顶点的权值有了更好的选择
            if cost[u][v] != 0 and visited[v] == False and cost[u][v] < key[v]:
                # 更新 parent 列表记录的各个顶点父节点的信息
                parent[v] = u
                # 更新 key 列表
                key[v] = cost[u][v]
    # 根据 parent 记录的各个顶点父节点的信息，输出寻找到的最小生成树
    print_MST(parent, cost);


find_MST(cost)