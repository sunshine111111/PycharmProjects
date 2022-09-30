import keras
import keras.backend
from keras.datasets import mnist
from keras import backend as Keras

#######TensorFlow+Keras+MNIST手写数字识别
#批量大小
BATCH_SIZE = 128
#学习次数
NUM_CLASSES = 10
EPOCHS = 10

#图像的垂直和水平大小
IMG_ROWS,IMG_COLS = 28,28

handwritten_number_names = ['0','1','2','3','4','5','6','7','8','9']
(train_data,train_teacher_labels),(test_data,test_teacher_labels) = mnist.load_data()
print('Channel调整变换前的训练数据train_data shape:',train_data.shape)
print('Channel调整变换前的测试数据test_data shape:',test_data.shape)


if Keras.image_data_format() == 'channels_first':
    #reshape():将train_data设置为（size,channels,rows,cols)的四阶张量（四维数组），将通道移到rows和cols之前
    train_data = train_data.reshape(train_data.shape[0],1,IMG_ROWS,IMG_COLS)
    test_data = test_data.reshape(test_data.shape[0],1,IMG_ROWS,IMG_COLS)

    input_shape = (1,IMG_ROWS,IMG_COLS)
else:
    # reshape():将train_data设置为（size,rows,cols,channels)的四阶张量（四维数组），将通道移到rows和cols之后
    train_data = train_data.reshape(train_data.shape[0],IMG_ROWS,IMG_COLS,1)
    test_data = test_data.reshape(test_data.shape[0],IMG_ROWS,IMG_COLS,1)
    input_shape = (IMG_ROWS,IMG_COLS,1)
print('Channel调整变换后的训练数据train_data shape:',train_data.shape)
print('Channel调整变换后的测试数据test_data shape:',test_data.shape)

#将数据转为float32
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')

#数据集数据分布在0-255，除以255，将数据转换为0-1.0分布
train_data /= 255
test_data /= 255

print('训练数据train_data shape:',train_data.shape)
print(train_data.shape[0],'训练样本')
print('测试数据test_data shape:',test_data.shape)
print(test_data.shape[0],'测试样本')

#将教师标签数据转换为One-Hot向量
print('Keras用于转换前训练的教师标签数据train_teacher_labels shape:',train_teacher_labels.shape)

train_teacher_labels = keras.utils.to_categorical(train_teacher_labels,NUM_CLASSES)
print('Keras用于转换后训练的教师标签数据train_teacher_labels shape:',train_teacher_labels.shape)

#将用于测试的教师标签数据转换为One-Hot向量
print('Keras用于转换前测试的教师标签数据test_teacher_labels shape:',test_teacher_labels.shape)
print(test_teacher_labels)
test_teacher_labels = keras.utils.to_categorical(test_teacher_labels,NUM_CLASSES)
print('Keras用于转换后测试的教师标签数据test_teacher_labels shape:',test_teacher_labels.shape)
print(test_teacher_labels)







