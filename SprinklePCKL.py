import numpy as np
import tensorflow as tf
from matplotlib import pylab as plt
import seaborn as sns
tf.reset_default_graph()

epsilon=0.0000000001
batch_size=32
learning_rate=0.001
z_dim = 2
noise_dim=3
gen_hidden_dim=64
x_dim=1
ratio_hidden_dim=64

#Stuff for making true posterior graph (copied from Huszar)
xmin = -5
xmax = 5
xrange = np.linspace(xmin,xmax,300)
x = np.repeat(xrange[:,None],300,axis=1)
x = np.concatenate([[x.flatten()],[x.T.flatten()]])
prior_variance = 1
logprior = -(x**2).sum(axis=0)/2/prior_variance
def likelihoodd(x, y, beta_0=3, beta_1=1):
    beta = beta_0 + (beta_1*(x**3).clip(0,np.Inf).sum(axis=0))
    return -np.log(beta) - y/beta
y = [None]*5
y[0] = 0
y[1] = 5
y[2] = 8
y[3] = 12
y[4] = 50
llh = [None]*5
llh[0] = likelihoodd(x, y[0])
llh[1] = likelihoodd(x, y[1])
llh[2] = likelihoodd(x, y[2])
llh[3] = likelihoodd(x, y[3])
llh[4] = likelihoodd(x, y[4])
#The values of x which we input into posterior later
xgen=np.array(y, dtype=np.float32)
#Xavier Initiaizlier
def xavier_init(fan_in, fan_out, constant=1):
    low=-constant*np.sqrt(6.0/(fan_in+fan_out))
    high=constant*np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform((fan_in, fan_out),minval=low,maxval=high, dtype=tf.float32)
def likelihood(z):
    return tf.random_gamma(shape=(x_dim, batch_size), alpha=1, beta=3+tf.pow(tf.maximum(0.0,z[:,0]),3)+tf.pow(tf.maximum(0.0,z[:,1]),3))
weights = {
    'post_hidden1': tf.Variable(xavier_init(noise_dim+x_dim, gen_hidden_dim)),
    'post_hidden2': tf.Variable(xavier_init(gen_hidden_dim, gen_hidden_dim)),
    'post_out': tf.Variable(xavier_init(gen_hidden_dim, z_dim)),
    'ratio_hidden1': tf.Variable(xavier_init(z_dim+x_dim, ratio_hidden_dim)),
    'ratio_hidden2': tf.Variable(xavier_init(ratio_hidden_dim, ratio_hidden_dim)),
    'ratio_out': tf.Variable(xavier_init(ratio_hidden_dim, 1))
}
biases = {
    'post_hidden1': tf.Variable(tf.zeros([gen_hidden_dim])),
    'post_hidden2': tf.Variable(tf.zeros([gen_hidden_dim])),
    'post_out': tf.Variable(tf.zeros([z_dim])),
    'ratio_hidden1': tf.Variable(tf.zeros([ratio_hidden_dim])),
    'ratio_hidden2': tf.Variable(tf.zeros([ratio_hidden_dim])),
    'ratio_out': tf.Variable(tf.zeros([1]))
}
#Posterior = q_phi(z|x,eps)
def posterior(x, noise):
    input = tf.concat([x, noise], axis=1)
    hidden_layer1 = tf.matmul(input, weights['post_hidden1'])
    hidden_layer1 = tf.add(hidden_layer1, biases['post_hidden1'])
    hidden_layer1 = tf.nn.relu(hidden_layer1)
    hidden_layer2 = tf.matmul(hidden_layer1, weights['post_hidden2'])
    hidden_layer2 = tf.add(hidden_layer2, biases['post_hidden2'])
    hidden_layer2 = tf.nn.relu(hidden_layer2)
    out_layer = tf.matmul(hidden_layer2, weights['post_out'])
    out_layer = tf.add(out_layer, biases['post_out'])
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer

#Ratio Estimator = r_alpha(z;x)
def ratiomator(z, x):
    input = tf.concat([z, x], axis=1)
    hidden_layer1 = tf.matmul(input, weights['ratio_hidden1'])
    hidden_layer1 = tf.add(hidden_layer1, biases['ratio_hidden1'])
    hidden_layer1 = tf.nn.relu(hidden_layer1)
    hidden_layer2 = tf.matmul(hidden_layer1, weights['ratio_hidden2'])
    hidden_layer2 = tf.add(hidden_layer2, biases['ratio_hidden2'])
    hidden_layer2 = tf.nn.relu(hidden_layer2)
    out_layer = tf.matmul(hidden_layer2, weights['ratio_out'])
    out_layer = tf.add(out_layer, biases['ratio_out'])
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer

x_input = tf.placeholder(tf.float32, shape=[None, x_dim], name='x_input')
noise_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='noise_input')
prior_input = tf.placeholder(tf.float32, shape=[None, z_dim], name='prior_input')

post_sample=posterior(x_input, noise_input)
#E_q*(x)q(z|x)[r_alpha(z;x)]
qratio=ratiomator(post_sample, x_input)
#E_q*(x)p(z)[r_alpha(z;x)]
pratio=ratiomator(prior_input, x_input)
#p(x|q_phi(z|x,eps))
X_like=likelihood(post_sample)
neglogLL=-tf.reduce_mean(tf.log(X_like+epsilon))

ratioloss=tf.reduce_mean(pratio-1-tf.log(qratio+epsilon))
postloss=neglogLL+tf.reduce_mean(tf.log(qratio+epsilon))

post_vars = [weights['post_hidden1'], weights['post_hidden2'], weights['post_out'], biases['post_hidden1'], biases['post_hidden2'], biases['post_out']]
ratio_vars = [weights['ratio_hidden1'], weights['ratio_hidden2'], weights['ratio_out'], biases['ratio_hidden1'], biases['ratio_hidden2'], biases['ratio_out']]

train_post = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(postloss, var_list=post_vars)
train_ratio = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(ratioloss, var_list=ratio_vars)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #pre-train ratio Estimator
    for i in range(2000):
        #Prior sample N(0,I_2x2)
        prior=np.random.randn(batch_size, z_dim)
        noise=np.random.randn(batch_size, noise_dim)
        if i % 5 == 0:
            xin=np.repeat(0,batch_size)
        if i % 5 == 1:
            xin=np.repeat(5,batch_size)
        if i % 5 == 2:
            xin=np.repeat(8,batch_size)
        if i % 5 == 3:
            xin=np.repeat(12,batch_size)
        if i % 5 == 4:
            xin=np.repeat(50,batch_size)
        xin=xin.reshape(batch_size, 1)
        feed_dict = {x_input: xin, noise_input: noise, prior_input: prior}
        _, rl = sess.run([train_ratio, ratioloss], feed_dict=feed_dict)
        if i % 100 == 0 or i == 1:
            print('Step %i: Ratio Loss: %f' % (i, rl))
    for i in range(20000):
        prior=np.random.randn(batch_size, z_dim)
        noise=np.random.randn(batch_size, noise_dim)
        if i % 5 == 0:
            xin=np.repeat(0,batch_size)
        if i % 5 == 1:
            xin=np.repeat(5,batch_size)
        if i % 5 == 2:
            xin=np.repeat(8,batch_size)
        if i % 5 == 3:
            xin=np.repeat(12,batch_size)
        if i % 5 == 4:
            xin=np.repeat(50,batch_size)
        xin=xin.reshape(batch_size, 1)
        feed_dict = {x_input: xin, noise_input: noise, prior_input: prior}
        _, _, rl, pl = sess.run([train_ratio, train_post, ratioloss, postloss], feed_dict=feed_dict)
        if i % 100 == 0 or i ==1:
            print('Step %i: Ratio Loss: %f, Post Loss: %f' % (i, rl, pl))

    sns.set_style('whitegrid')
    sns.set_context('poster')

    plt.subplots(figsize=(20,8))
    #make 5000 noise and 1000 of each x sample
    N_samples=1000
    noise=np.random.randn(5*N_samples, noise_dim).astype('float32')
    x_gen=np.repeat(xgen,1000)
    x_gen=x_gen.reshape(5000,1)
    #plug into posterior
    z_samples=posterior(x_gen,noise)
    z_samples=tf.reshape(z_samples,[xgen.shape[0], N_samples, 2]).eval()
    #print(z_samples)
    #Plots
    for i in range(5):
        plt.subplot(2,5,i+1)
        sns.kdeplot(z_samples[i,:,0], z_samples[i,:,1], cmap='Greens')
        #plt.scatter(z_samples[i,:,0],z_samples[i,:,1])
        plt.axis('square');
        plt.title('q(z|x={})'.format(y[i]))
        plt.xlim([xmin,xmax])
        plt.ylim([xmin,xmax])
        plt.xticks([])
        plt.yticks([]);
        plt.subplot(2,5,5+i+1)
        plt.contour(xrange, xrange, np.exp(logprior+llh[i]).reshape(300,300).T, cmap='Greens')
        plt.axis('square');
        plt.title('p(z|x={})'.format(y[i]))
        plt.xlim([xmin,xmax])
        plt.ylim([xmin,xmax])
        plt.xticks([])
        plt.yticks([]);
    plt.show()
