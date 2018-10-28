
import tensorflow as tf

lstm=tf.nn.rnn_cell.BasicLSTMCell(lstm_size)

#使用DropoutWrapper类来实现dropout功能。该类通过两个参数来 控制dropout的概率，一个参数为input_keep_prob,
# 可以用来控制输入的dropout概率，另一个为output_keep_prob,它控制输出的dropout概率

dropout_lstm=tf.nn.rnn_cell.DropoutWrapper(lstm,output_keep_prob=0.5)

#在使用了dropout的基础上定义
stacked_lstm=tf.nn.rnn_cell.MultiRNNCell([dropout_lstm]*number_of_layers)

