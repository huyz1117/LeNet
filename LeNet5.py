import tensorflow as tf
from tensorflow.contrib.layers import flatten
from tensorflow.examples.tutorials.mnist import input_data

num_epochs = 1
batch_size = 128
lr = 0.001
mean = 0.0
stddev = 0.1

mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)

print("Image size: {}".format(mnist.train.images[0].shape))
print("Training set: {}".format(mnist.train.images.shape))
print("Training set: {}".format(mnist.validation.images.shape))
print("Training set: {}".format(mnist.test.images.shape))

X = tf.placeholder(tf.float32, shape=[None, 784])
X_img = tf.reshape(X, shape=[-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, shape=[None, 10])

X_img = tf.pad(X_img, [[0, 0], [2, 2], [2, 2], [0, 0]])
print(X_img.shape)


# 第一层卷积： 28*28*1 --> 28*28*6
conv1_W = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 6], mean=mean, stddev=stddev))
conv1_b = tf.Variable(tf.zeros([6]))
conv1 = tf.nn.conv2d(X_img, conv1_W, strides=[1, 1, 1, 1], padding="VALID")
conv1 = tf.nn.bias_add(conv1, conv1_b)
conv1 = tf.nn.relu(conv1)
print(conv1.shape)

# 池化：28*28*6 --> 14*14*6
pool_1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

# 第二层卷积: 14*14*6 --> 10*10*16
conv2_W = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], mean=mean, stddev=stddev))
conv2_b = tf.Variable(tf.zeros([16]))
conv2 = tf.nn.conv2d(pool_1, conv2_W, strides=[1, 1, 1, 1], padding="VALID")
conv2 = tf.nn.bias_add(conv2, conv2_b)
conv2 = tf.nn.relu(conv2)

# 池化：10*10*16 --> 5*5*16
pool_2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

# 全连接：5*5*16 --> 120
flatten = flatten(pool_2)
fc1_W = tf.Variable(tf.truncated_normal(shape=[5*5*16, 120], mean=mean, stddev=stddev))
fc1_b = tf.Variable(tf.zeros([120]))
fc1 = tf.matmul(flatten, fc1_W) + fc1_b
fc1 = tf.nn.relu(fc1)

# 全连接：120 --> 86
fc2_W = tf.Variable(tf.truncated_normal(shape=[120, 80], mean=mean, stddev=stddev))
fc2_b = tf.Variable(tf.zeros([80]))
fc2 = tf.matmul(fc1, fc2_W) + fc2_b
fc2 = tf.nn.relu(fc2)

# 全连接：86 --> 10
fc3_W = tf.Variable(tf.truncated_normal(shape=[80, 10], mean=mean, stddev=stddev))
fc3_b = tf.Variable(tf.zeros([10]))
logits = tf.matmul(fc2, fc3_W) + fc3_b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

is_correction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correction, tf.float32))

optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

saver = tf.train.Saver()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    print("Learning started...")

    for epoch in range(num_epochs):
        avg_accuracy = 0
        avg_cost = 0
        num_batches = int(mnist.train.num_examples / batch_size)
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_x, Y: batch_y}
        for i in range(num_batches):
            _, a, c = sess.run([optimizer, accuracy, cost], feed_dict=feed_dict)
            avg_accuracy += a / num_batches
            avg_cost += c / num_batches
        print("Epoch: {}\tLoss: {:.9f}\tAccuracy: {:.3%}".format(epoch+1, avg_cost, avg_accuracy))
    print("Training finished!")

    saver.save(sess, "ckpt_examples/LeNet/lenet.ckpt")
    print("Model saved!")

with tf.Session() as sess:
    model_file = tf.train.latest_checkpoint("ckpt_examples/LeNet")
    saver.restore(sess, model_file)
    accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})
    print("Accuracy on test set: {:.3%}".format(accuracy))
