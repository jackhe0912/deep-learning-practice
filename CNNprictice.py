
import tensorflow as tf

#卷积层
filter_weight=tf.get_variable('weights',[5,5,3,16],initializer=tf.truncated_normal_initializer(stddev=0.1))
biases=tf.get_variable('biases',[16],initializer=tf.truncated_normal_initializer(0.1))
#input为四维矩阵，第一维对应于一个输入batch,后三维对应于一个节点矩阵，filter_weight为卷积层权重，第三个参数为卷积
#步长，其中第一维和第四维为固定为1，因为卷积步长只对矩阵的长和宽有效，padding='SAME'则添加全0填充，padding='VALID'则不添加

conv=tf.nn.conv2d(input,filter_weight,strides=[1,1,1,1],padding='SAME')
bias=tf.nn.bias_add(conv,biases)

#将计算结果通过relu激活函数完成去线性化
activated_conv=tf.nn.relu(bias)

#池化层 tf.nn.max_pool实现最大池化层的前向传播过程，它的参数和tf.nn.conv2d函数类似
#ksize提供了过滤器的尺寸，strides提供了步长的信息，padding提供了是否使用全0填充
pool=tf.nn.max_pool(activated_conv,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
