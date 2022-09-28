#克鲁斯卡尔算法:在连通网中查找最小生成树
#克鲁斯卡尔算法查找最小生成树的方法是：将连通网中所有的边按照权值大小做升序排序，从权值最小的边开始选择，只要此边不和已选择的边一起构成环路，就可以选择它组成最小生成树。


N = 9   #图中边的数量
P = 6   #图中顶点的数量
#构建表示边的结构体
class edge:
    #一条边有 2 个顶点
    initial = 0
    end = 0
    #边的权值
    weight = 0
    def __init__(self,initial,end,weight):
        self.initial = initial
        self.end = end
        self.weight = weight

edges = []   # 用于保存用户输入的图各条边的信息
minTree=[]   # 保存最小生成数各个边的信息
#输入 N 条边的信息
for i in range(N):
    li = input().split()
    initial = int(li[0])
    end = int(li[1])
    weight = int(li[2])
    edges.append(edge(initial,end,weight))
# 根据 weight 给 edges 列表排序
def cmp(elem):
    return elem.weight
#克鲁斯卡尔算法寻找最小生成树
def kruskal_MinTree():
    #记录选择边的数量
    num = 0
    #为每个顶点配置一个不同的标记
    assists = [i for i in range(P)]
    #对 edges 列表进行排序
    edges.sort(key = cmp)
    #遍历 N 条边，从重选择可组成最小生成树的边
    for i in range(N):
        #找到当前边的两个顶点在 assists 数组中的位置下标
        initial = edges[i].initial -1
        end = edges[i].end-1
        # 如果顶点位置存在且顶点的标记不同，说明不在一个集合中，不会产生回路
        if assists[initial] != assists[end]:
            # 记录该边，作为最小生成树的组成部分
            minTree.append(edges[i])
            #计数+1
            num = num+1
            #将新加入生成树的顶点标记全部改为一样的
            elem = assists[end]
            for k in range(P):
                if assists[k] == elem:
                    assists[k]= assists[initial]
            #如果选择的边的数量和顶点数相差1，证明最小生成树已经形成，退出循环
            if num == P-1:
                break
def display():
    cost = 0
    print("最小生成树为:")
    for i in range(P-1):
        print("%d-%d  权值：%d"%(minTree[i].initial, minTree[i].end, minTree[i].weight))
        cost = cost + minTree[i].weight
    print("总权值为:%d"%(cost))

kruskal_MinTree()
display()
