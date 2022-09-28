import numpy as np
import cv2
import matplotlib.pyplot as plt
import diluteImage as imageData

#设置列数
NUM_COLUMNS = 4
#载入图像
target_img = cv2.imread('E:/algorithm/kawashima01.jpg')
#稀释图像数据
fake_images = imageData.make_image(target_img)
#行
ROWS_COUNT = len(fake_images)%NUM_COLUMNS
#列
COLUMS_COUNT = NUM_COLUMNS

# 用于保留图表对象
subfig = []
#确定figure(配置）对象创建大小
fig = plt.figure(figsize=(12,9))
fake_imagesPath = "E:/algorithm/fake_images/"
for i in range(1,len(fake_images)+1):
    # 按顺序添加第i个subfig对象
    subfig.append(fig.add_subplot(ROWS_COUNT,COLUMS_COUNT,i))

    img_bgr = cv2.imread(fake_imagesPath+str(i-1)+'.jpg')
    img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
    subfig[i-1].imshow(img_rgb)

#图表之间横向和纵向间隙的调整
fig.subplots_adjust(wspace=0.3,hspace=0.3)

plt.show()
