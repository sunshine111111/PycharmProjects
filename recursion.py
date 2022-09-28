#递归算法，设计递归函数时，我们必须为它设置一个结束递归的“出口”，否则函数会一直调用自身（死循环），直至运行崩溃
#用递归方式求n!
def factorial(n):
    if n==1 or n==0:
        return 1
    return n*factorial(n-1)

print(factorial(4))