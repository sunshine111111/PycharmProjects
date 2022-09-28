import cv2
import cv2.cv2
import matplotlib.pyplot as plt
import numpy as np

#获取的数据顺序是BGR,而不是RGB，所以直接显示不正常
img_bgr=cv2.imread('E:/algorithm/kawashima01.jpg')
#将python数组转换为Numpy数组
x=np.array(img_bgr)
print(x.shape)

#将BGR顺序的数据转为RGB
img_rgb=cv2.cvtColor(img_bgr,cv2.cv2.COLOR_BGR2RGB)

size= img_rgb.shape
print(size) #(320,320,3)，即长和宽为320像素，具有3个RGB通道的数据
# img_hsv=cv2.cvtColor(img_bgr,cv2.cv2.COLOR_BGR2HSV)
# img_gray=cv2.cvtColor(img_rgb,cv2.cv2.COLOR_RGB2GRAY)
#保存图片
# cv2.imwrite('E:/algorithm/rgb_kawashima01.jpg',img_rgb)
# plt.imshow(img_rgb)

#裁剪图片左上角的部分
#new_img=img_rgb[:size[0]//2,:size[1]//2]

#裁剪图片右下角的部分
# new_img=img_rgb[size[0]//2:,size[1]//2:]
# plt.imshow(new_img)
plt.subplot(211)
plt.imshow(img_rgb)
#放大
#resized_img = cv2.resize(img_rgb,(img_rgb.shape[0]*2,img_rgb.shape[1]*1))
#缩小
# resized_img = cv2.resize(img_rgb,(img_rgb.shape[0]//4,img_rgb.shape[1]//4))
# print(resized_img.shape)
#旋转
# mat = cv2.getRotationMatrix2D(tuple(np.array(img_rgb.shape[:2])/2),45,1.0)
# result_img=cv2.warpAffine(img_rgb,mat,img_rgb.shape[:2])

#图像二值化,四个参数分别表示图片数据、阈值、最大值、阈值类型
# retval,result_img=cv2.threshold(img_rgb,95,128,cv2.THRESH_TOZERO)
# print(result_img)
#result_img=cv2.GaussianBlur(img_rgb,(15,15),0)
#降噪
#result_img=cv2.fastNlMeansDenoisingColored(img_rgb)
#filt = np.array([[0,1,0],[1,0,1],[0,1,0]],np.uint8)
#缩放：dilate膨胀处理；erode收缩处理
#result_img=cv2.erode(img_rgb,filt)
img_gray=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
retval,thresh = cv2.threshold(img_gray,88,255,0)
img,contours,hierarchy=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

result_img=cv2.drawContours(img,contours,-1,(0,0,255),3)
print(thresh)
plt.subplot(212)
plt.imshow(thresh)
plt.show()

