#迷宫问题：用回溯算法解决
#指定地图的行数和列数
ROW = 5
COL = 5
#初始化地图
maze =[['1','0','1','1','1'],
       ['1','1','1','0','1'],
       ['1','0','0','1','1'],
       ['1','0','0','1','0'],
       ['1','0','0','1','1']]
#假设当前迷宫中没有起点到终点的路线
find = False
#回溯算法查找可行路线
def maze_puzzle(maze,row,col,outrow,outcol):
    global find
    maze[row][col] = 'Y'
    #如果行走至终点，表明有从起点到终点的路线
    if row == outrow and col == outcol:
        find = True
        print("成功走出迷宫,路线图为：")
        printmaze(maze)
        return
    if canMove(maze,row-1,col):
        maze_puzzle(maze, row - 1, col, outrow, outcol)
        #如果程序不结束，表明此路不通，恢复该区域的标记
        maze[row - 1][col] = '1'
    if canMove(maze, row, col - 1):
        maze_puzzle(maze, row, col - 1, outrow, outcol)
        #如果程序不结束，表明此路不通，恢复该区域的标记
        maze[row][col - 1] = '1'
    #尝试向下移动
    if canMove(maze, row + 1, col):
        maze_puzzle(maze, row + 1, col, outrow, outcol)
        #如果程序不结束，表明此路不通，恢复该区域的标记
        maze[row + 1][col] = '1'
    #尝试向右移动
    if canMove(maze, row, col + 1):
        maze_puzzle(maze, row, col + 1, outrow, outcol)
        #如果程序不结束，表明此路不通，恢复该区域的标记
        maze[row][col + 1] = '1'

#判断(row,col)区域是否可以移动
def canMove(maze,row,col):
    return row >= 0 and row <= ROW - 1 and col >= 0 and col <= COL - 1 and maze[row][col] != '0' and maze[row][col] != 'Y'

#输出行走路线
def printmaze(maze):
    for i in range(0,ROW):
        for j in range(0,COL):
            print(maze[i][j],end=" ")
        print()

maze_puzzle(maze,0,0,ROW-1,COL-1)
if find == False:
    print("未找到可行路线")
