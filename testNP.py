import numpy as np

a = np.array([1,2,3])
b = np.array([1,1,4])
c=a+b
#a=a*3
#a=a+2
#内积
d=np.dot(a,b)

c=np.array([[1,2],[3,4]])
d=np.array([[3,4],[1,2]])
e=np.dot(c,d)
#print(e)

g=np.arange(9).reshape((3,3))
print(g)

zeros = np.zeros((3,4))
print(zeros)

ones = np.ones((3,3))
print(ones)

p=np.empty((2,3))
print(p)

k=np.matrix([[0,1,2],[3,4,5],[6,7,8]])
l=np.matrix([[1,1,1],[1,1,1],[1,1,1]])
m=np.dot(k,l)
print(m.shape)
print(m.ndim)
print(m.dtype)
print(m.itemsize)
print(m.size)

q=np.arange(1000000).reshape(1000,1000)
print(q)

x=np.arange(start=100,stop=600,step=100)
print(x)

x_float = x.astype('float32')
print('x_float.dtype.name',x_float.dtype.name)
print(x_float)

