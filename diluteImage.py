import os
import cv2
import numpy as np


#稀释图像数据
def make_image(input_img):
    #图像大小
    img_size = input_img.shape
    filter_one = np.ones((3,3))
    #反转用
    #旋转：旋转中心、旋转角度、放大倍率
    mat1 = cv2.getRotationMatrix2D(tuple(np.array(input_img.shape[:2])/2),23,1)
    mat2 = cv2.getRotationMatrix2D(tuple(np.array(input_img.shape[:2])/2),144,0.8)

    #用于填充方法的函数->lambda 参数：函数
    fake_method_array=np.array([
        lambda image: cv2.warpAffine(image,mat1,image.shape[:2]),
        lambda image: cv2.warpAffine(image,mat2,image.shape[:2]),
        lambda image: cv2.threshold(image,100,255,cv2.THRESH_TOZERO)[1],
        lambda image: cv2.GaussianBlur(image,(5,5),0),
        lambda image: cv2.resize(cv2.resize(image,(img_size[1]//5,img_size[0]//5)),(img_size[1],img_size[0])),
        lambda image: cv2.erode(image,filter_one),
        lambda image: cv2.flip(image,1)
    ])

    #执行图像转换过程
    images=[]

    for method in fake_method_array:
        faked_img = method(input_img)
        images.append(faked_img)

    return images

#载入图像
target_img = cv2.imread('E:/algorithm/kawashima01.jpg')

#稀释图像数据
fake_images = make_image(target_img)

#创建保存图像的文件夹
if not os.path.exists("E:/algorithm/fake_images"):
    os.mkdir("E:/algorithm/fake_images")

for number,img in enumerate(fake_images):
    #首先，指定要保存的目录“E:/algorithm/fake_images/”并将其编号
    cv2.imwrite("E:/algorithm/fake_images/"+str(number)+".jpg",img)


