
import tensorflow as tf
#定义一个 LSTM结构

lstm_hidden_size=100
lstm=tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size)

#将 LSTM中的状态初始化为全0数组，每次使用一个batch的训练样本
state=lstm.zero_state(batch_size=80,dtype=tf.float32)

#定义损失函数
loss=0.0

#规定最大序列长度，用num_step表示
num_step=40   #最大序列长度

for i in range(num_step):
    # 在第一个时刻声明LSTM结构中使用的变量，在之后的时刻都需要复用之前定义好的变量
    if i  >0:
        tf.get_variable_scope().reuse_variables()

    #每一步处理时间序列中的一个时刻。将当前输入(current_input)和前一时刻状态 （state)传入定义的LSTM结构可以得到
    # 当前LSTM结构的输出lstm_output和更新后的状态state
    lstm_output,state=lstm(current_input,state)

    #将当前时刻的输出传入一个全连接层得到最后的输出
    final_output=fully_connected(lstm_output)

    #计算当前时刻输出的损失
    loss +=calc_loss(final_output,expected_output)



