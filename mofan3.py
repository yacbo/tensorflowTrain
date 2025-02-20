#import tensorflow as tf
#我的是因为在tf2下使用了tf1的API。
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt

#添加层
def add_layer(inputs,in_size,out_size,activation_function = None):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size])+0.1)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs,Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs
#定义数据
x_data = np.linspace(-1,1,500)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

#define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,1],name='x_input')
    ys = tf.placeholder(tf.float32,[None,1],name='y_input')
#add hidden layer
l1 = add_layer(xs,1,10,activation_function = tf.nn.relu)
#add output layer
prediction = add_layer(l1,10,1,activation_function = None)
#the error between prediction and real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean( tf.reduce_sum(tf.square(ys - prediction),
                        reduction_indices=[1]))
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init = tf.initialize_all_variables()
sess = tf.Session()
#writer = tf.train.write_graph(sess.graph,"logs/","aa.Morvan")
#writer = tf.summary.FileWriter("log_1",sess.graph)
#writer.close()
#important step
sess.run(init)

#画原始数据图
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()

#开始训练
for i in range(10000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50==0:
        #to see the improvement
        #print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction,feed_dict={xs:x_data})
        lines = ax.plot(x_data,prediction_value,'r-',lw=3)
        # ax.lines.remove(lines[0])
        plt.pause(0.1)





