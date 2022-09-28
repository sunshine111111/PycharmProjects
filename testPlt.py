import matplotlib.pyplot as plt
import numpy as np

#plot(x,y),对应x轴和y轴的坐标；
#如果只传递一个数据数组，则会将传递的数据视为y，x为[0,1,2,....n]
# plt.plot([1,2,3,4])
# plt.show()

# x=np.array([1,2,3,4,5])
# y=np.array([100,200,300,400,500])
# plt.plot(x,y)
# plt.show()

#0,1,2,3,4
# x=np.arange(5)
# y=np.arange(start=100,stop=600,step=100)
# plt.plot(x,y)
# plt.show()

# volume=30
# x=np.arange(volume)
# y=[x_i+np.random.randn(1) for x_i in x]
# print(np.random.randn(1))
# a,b=np.polyfit(x,y,1)
# _=plt.plot(x,y,'o',np.arange(volume),a*np.arange(volume)+b,'-')
#plt.show()

#在0-10之间，分成100等份
x=np.linspace(0,10,100)
# plt.plot(x,np.sin(x))
#plt.show()

# plt.plot(x,np.cos(x))
#plt.show()

# plt.plot(x,np.arctan(x))
# plt.title('title for the gragh')
# plt.xlabel('label for x-axis')
# plt.ylabel('label for y-axis')
# #添加网格
# plt.grid(True)
# plt.show()

#指定图表尺寸（缩小了）
# plt.figure(figsize=(4,4))
#
# x=np.linspace(0,2*np.pi)
# plt.plot(x,np.sin(x))
# plt.title('title for the gragh')
# plt.xlabel('label for x-axis')
# plt.ylabel('label for y-axis')
# #添加网格
# plt.grid(True)
# positions = [0,np.pi/2,np.pi,np.pi*3/2,np.pi*2]
# labels = ['0','90','180','270','360']
# #在xticks(显示位置,显示字符)中设置刻度
# plt.xticks(positions,labels)
# plt.show()

########散点图
# x=np.random.rand(100)
# y=np.random.rand(100)
# #绘制散点图
# #subplot(行数，列数，从左上角开始的位置)
# plt
# .subplot(221)
# #s是点的大小，c是点的颜色；alpha是点的透明度；linewidths是点绘制时的线宽；edgecolors是点边缘的颜色
# plt.scatter(x,y,s=600,c="pink",alpha=0.5,linewidths=2,edgecolors="red")
# # plt.grid(True)
# # plt.show()
# plt.subplot(222)
# plt.scatter(x,y,s=300,c=y,cmap="Greens")
# plt.subplot(223)
# plt.scatter(x,y,s=600,c="white",alpha=0.5,linewidths=2,edgecolors="yellow")
# plt.subplot(224)
# plt.scatter(x,y,s=300,c=y,cmap="Blues")
# # plt.title("title gose here")
# # plt.xlabel("x axis")
# # plt.ylabel("y axis")
# plt.grid(True)
# plt.show()

#####三维散点图
np.random.seed(0)
random_x=np.random.randn(100)
random_y=np.random.randn(100)
random_z=np.random.randn(100)

fig=plt.figure(figsize=(8,8))

ax=fig.add_subplot(1,1,1,projection="3d")

x=np.ravel(random_x)
y=np.ravel(random_y)
z=np.ravel(random_z)

ax.scatter3D(x,y,z,s=300,c="r",marker="^")
plt.show()