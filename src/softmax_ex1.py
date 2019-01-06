#encoding=utf-8

#目的，进行mnist训练


#得到训练数据
import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#运行TensorFlow的InteractiveSession
import tensorflow as tf
sess = tf.InteractiveSession()

#构建Softmax 回归模型

## 占位符
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

## 变量
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

## 初始化　　　
sess.run(tf.initialize_all_variables())

##　定义模型
y = tf.nn.softmax(tf.matmul(x,W) + b)

## 定义损失函数
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

## 定义训练方法： 最速下降法让交叉熵下降，步长为0.01.
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

##  开始训练

for i in range(1000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

## 评估模型

##　得到标签的序号：  tf.argmax(y,1)
##　返回ｂｏｏｌ数组　[True, False, True, True]
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

## 我们将布尔值转换为浮点数来代表对、错，然后取平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

##  打印准确率
print accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})