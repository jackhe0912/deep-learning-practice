import tensorflow as tf

#定义一个基本的LSTM结构作为循环体的基础结构，当然深层循环神经网络也支持使用其他的循环体
lstm_size=20
lstm=tf.nn.rnn_cell.BasicLSTMCell(lstm_size)

#通过MultiRNNCell类实现深层神经网络中的每一个时刻前向传播过程。其中number_of_layers表示了有多少层，
# 即输入xt到输出ht需要经过多少个LSTM结构
stacked_lstm=tf.nn.rnn_cell.MultiRNNCell([lstm]*number_of_layers)

#通过zero_state函数来获取初始状态
state=stacked_lstm.zero_state(batch_size=20,dtypes=tf.float32)

num_steps=100
for i in range(len(num_steps)):
    if i >0:
        tf.get_variable_scope().reuse_variables()
    stacked_lstm_output,state=stacked_lstm(current_input,state)
    final_output=fully_connected(stacked_lstm_output)
    loss+=calc_loss(final_output,expected_output)
