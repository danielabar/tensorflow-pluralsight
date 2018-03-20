import tensorflow as tf

# Represent Two Equations
# y = Ax^2 + Bx + C
# y = Ax^2 + Bx^2

# Setup constants
A = tf.constant([4], tf.int32, name='A')
B = tf.constant([5], tf.int32, name='B')
C = tf.constant([6], tf.int32, name='C')

# x is a placeholder whose value will be filled in when equations are evaluated
x = tf.placeholder(tf.int32, name='x')

# Setup computations for first equation: y = Ax^2 + Bx + C
with tf.name_scope('Equation_1'):
    Ax2 = tf.multiply(A, tf.pow(x, 2), name='Ax2')
    Bx = tf.multiply(B, x, name='Bx')
    y1 = tf.add_n([Ax2, Bx, C], name='y1')

# Setup computations for second equation: y = Ax^2 + Bx^2
with tf.name_scope('Equation_2'):
    Ax2 = tf.multiply(A, tf.pow(x, 2), name='Ax2')
    Bx2 = tf.multiply(B, tf.pow(x, 2), name='Bx2')
    y2 = tf.add_n([Ax2, Bx2], name='y2')

# Final computation
with tf.name_scope('Final_Sum'):
    y = y1 + y2

with tf.Session() as sess:
    print(sess.run(y, feed_dict={x: [10]}))
    writer = tf.summary.FileWriter('./logs/m3_example5', sess.graph)
    writer.close()