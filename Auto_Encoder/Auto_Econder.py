import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# xavier初始化器　用于权重初始化
def xavier_init(fan_in, fan_out, constant=1):
    low = -constant*np.sqrt(6.0 / (fan_in + fan_out))
    high = constant*np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


# 去噪自编码器类
class AdditiveGaussianNoiseAutoEncoder(object):
    # 输入参数为n_input：输入变量数　n_hidden：隐含层节点数 transfer_function：隐含层激活函数,默认softplus
    #         optimizer：优化器默认Adam　scale:高斯噪声系数　默认0.1　

    def __init__(self, n_input, n_hidden, optimizer=tf.train.AdamOptimizer(),
                 transfer_function=tf.nn.softplus, scale=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale*tf.random_normal((n_input,)),
                                    self.weights['w1']),
                                    self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']),
                                     self.weights['b2'])
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, x):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: x, self.scale: self.training_scale})
        return cost

    def calc_total_cost(self, x):
        return self.sess.run(self.cost, feed_dict={self.x: x, self.scale: self.training_scale})

    def transform(self, x: object) -> object:
        return self.sess.run(self.hidden, feed_dict={self.x: x, self.scale: self.training_scale})

    def generate(self, hidden = None):
        if hidden is None:
            hidden = np.random.normal(size = self.weights["b1"])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, x):
        return self.sess.run(self.reconstruction, feed_dict={self.x: x, self.scale: self.training_scale})

    def get_weights(self):
        return self.sess.run(self.weights['w1'])

    def get_biases(self):
        return self.sess.run(self.weights['b1'])


def stand_scale(x_train, x_test):
    preprocessor = prep.StandardScaler().fit(x_train)
    x_train = preprocessor.transform(x_train)
    x_test = preprocessor.transform(x_test)
    return x_train, x_test


# get block data from data sets randomly
def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data)-batch_size)
    return data[start_index:(start_index+batch_size)]

m = input_data.read_data_sets('MNIST_data', one_hot=True)
X_train, X_test = stand_scale(m.train.images, m.test.images)
n_samples = int(m.train.num_examples)
training_epochs = 20
batch_size = 128
display_step = 1
ae = AdditiveGaussianNoiseAutoEncoder(n_input=784, n_hidden=200, transfer_function=tf.nn.softplus,
                                      optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                      scale=0.01)
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)
        Cost = ae.partial_fit(batch_xs)
        avg_cost += Cost / n_samples * batch_size
    if epoch % display_step == 0:
        print("Epoch", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
print("Total cost:" + str(ae.calc_total_cost(X_test)))
