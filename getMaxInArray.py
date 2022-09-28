#分治算法实现
def get_max(arr,left,right):
    #列表中没有数据
    if len(arr) == 0:
        return -1
    #如果查找范围中仅有2个数字，则直接比较即可
    if right - left <= 1:
        if arr[left] >= arr[right]:
            return arr[left]
        return arr[right]
   #等量划分为2个区域
    middle = int((right-left)/2+left)
    max_left = get_max(arr,left,middle)
    max_right = get_max(arr,middle+1,right)
    if max_left >= max_right:
        return max_left;
    else:
        return max_right

arr = [3,7,2,1,8,9,10,20]
max = get_max(arr,0,7)
print("最大值： ",max,sep='')