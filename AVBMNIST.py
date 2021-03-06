import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 32
z_dim = 10
eps_dim = 4
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
#number of nodes in hidden layer
h_dim = 128
lr = 1e-3

#change add 1e-10 to avoid log 0
def log(x):
    return tf.log(x + 1e-10)


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

#similar xavier initialization that uses normal dist
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


#P(X|z)
P_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
P_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
P_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
P_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_P = [P_W1, P_W2, P_b1, P_b2]


def P(z):
    h = tf.nn.relu(tf.matmul(z, P_W1) + P_b1)
    #logits is output of final layer before activation function, used for
    #sigmoid_cross_entropy_with_logits
    logits = tf.matmul(h, P_W2) + P_b2
    prob = tf.nn.sigmoid(logits)
    return prob, logits


#Q(z|X,eps)
X = tf.placeholder(tf.float32, shape=[None, X_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])
eps = tf.placeholder(tf.float32, shape=[None, eps_dim])

Q_W1 = tf.Variable(xavier_init([X_dim + eps_dim, h_dim]))
Q_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
Q_W2 = tf.Variable(xavier_init([h_dim, z_dim]))
Q_b2 = tf.Variable(tf.zeros(shape=[z_dim]))

theta_Q = [Q_W1, Q_W2, Q_b1, Q_b2]


def Q(X, eps):
    inputs = tf.concat(axis=1, values=[X, eps])
    h = tf.nn.relu(tf.matmul(inputs, Q_W1) + Q_b1)
    z = tf.matmul(h, Q_W2) + Q_b2
    return z

#Discriminator
#P_D(z)
D_W1 = tf.Variable(xavier_init([X_dim + z_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
D_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]


def D(X, z):
    inputs = tf.concat([X, z], axis=1)
    h = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    return tf.matmul(h, D_W2) + D_b2


#Training
z_sample = Q(X, eps)
_, X_logits = P(z_sample)
D_sample = D(X, z_sample)
#T(x,z(eps))
D_q = tf.nn.sigmoid(D(X,z_sample))
#T(x,z)
D_prior = tf.nn.sigmoid(D(X, z))

# Sample from random z
X_samples, _ = P(z)
#-T(x,z)
disc = tf.reduce_mean(-D_sample)
nll = tf.reduce_sum(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=X_logits, labels=X),
    axis=1
)
#P(X|z)=log(p(x|z))
loglike = -tf.reduce_mean(nll)
#-T(x,z)+p(x|z)
elbo = disc + loglike
#P_D(z) = log(sigmoid(T(x,z(eps))))+log(1-sigmoid(T(x,z)))
D_loss = tf.reduce_mean(log(D_q) + log(1. - D_prior))

VAE_solver = tf.train.AdamOptimizer().minimize(-elbo, var_list=theta_P+theta_Q)
D_solver = tf.train.AdamOptimizer().minimize(-D_loss, var_list=theta_D)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

for it in range(100000):
    X_mb, _ = mnist.train.next_batch(mb_size)
    eps_mb = np.random.randn(mb_size, eps_dim)
    z_mb = np.random.randn(mb_size, z_dim)

    _, elbo_curr = sess.run([VAE_solver, elbo],
                            feed_dict={X: X_mb, eps: eps_mb, z: z_mb})

    _, D_loss_curr = sess.run([D_solver, D_loss],
                              feed_dict={X: X_mb, eps: eps_mb, z: z_mb})

    if it % 1000 == 0:
        print('Iter: {}; ELBO: {:.4}; D_Loss: {:.4}'
              .format(it, elbo_curr, D_loss_curr))

        samples = sess.run(X_samples, feed_dict={z: np.random.randn(16, z_dim)})

        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)
