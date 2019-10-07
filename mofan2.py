import tensorflow as tf

#定义变量
state = tf.Variable(0,name='count')# print(state.name)
one = tf.constant(1)
new_value = tf.add(state , one)
update = tf.assign(state,new_value)
init = tf.initialize_all_variables() #must have if define variable
with tf.Session() as sess:
    sess.run(init)
    for _ in range(8):
        sess.run(update)
        print(sess.run(state))
#赋值
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1,input2)
with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[7.],input2:[2.]}))

#添加层
def add_layer(inputs,in_size,out_size,activation_function = None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b = tf.matmul(inputs,Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs
