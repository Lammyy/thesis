import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from matplotlib import pylab as plt
import seaborn as sns
tf.reset_default_graph()
#params
epsilon=0.0000000001
batch_size=100
learning_rate_p=0.0003
learning_rate_d=0.0003
z_dim = 2
noise_dim=3
gen_hidden_dim1=30
gen_hidden_dim2=60
data_dim=1
disc_hidden_dim1=30
disc_hidden_dim2=60
#Stuff for making true posterior graph (copied from Huszar)
xmin = -5
xmax = 5
xrange = np.linspace(xmin,xmax,300)
x = np.repeat(xrange[:,None],300,axis=1)
x = np.concatenate([[x.flatten()],[x.T.flatten()]])
prior_variance = 2
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

weights = {
    #post_hidden11 and post_hidden12 work on the x input
    'post_hidden11': tf.Variable(xavier_init(data_dim, gen_hidden_dim1)),
    'post_hidden12': tf.Variable(xavier_init(gen_hidden_dim1, gen_hidden_dim2)),
    #post_hidden2 works on noise input
    'post_hidden2': tf.Variable(xavier_init(noise_dim, gen_hidden_dim2)),
    #post_hidden31, post_hidden32 and post_out work on concatenated x and noise
    'post_hidden31': tf.Variable(xavier_init(gen_hidden_dim2+gen_hidden_dim2, gen_hidden_dim1)),
    'post_hidden32': tf.Variable(xavier_init(gen_hidden_dim1, gen_hidden_dim2)),
    'post_out': tf.Variable(xavier_init(gen_hidden_dim2, z_dim)),
    #disc_hidden11 and disc_hidden12 work on z input
    'disc_hidden11': tf.Variable(xavier_init(z_dim, disc_hidden_dim1)),
    'disc_hidden12': tf.Variable(xavier_init(disc_hidden_dim1, disc_hidden_dim2)),
    #disc_hidden21 and disc_hidden22 work on x input
    'disc_hidden21': tf.Variable(xavier_init(data_dim, disc_hidden_dim1)),
    'disc_hidden22': tf.Variable(xavier_init(disc_hidden_dim1, disc_hidden_dim2)),
    #disc_hidden31, disc_hidden32 and disc_out work on concatenated z and x
    'disc_hidden31': tf.Variable(xavier_init(disc_hidden_dim2+disc_hidden_dim2, disc_hidden_dim1)),
    'disc_hidden32': tf.Variable(xavier_init(disc_hidden_dim1, disc_hidden_dim2)),
    'disc_out': tf.Variable(xavier_init(disc_hidden_dim2, 1))
}
biases = {
    'post_hidden11': tf.Variable(tf.zeros([gen_hidden_dim1])),
    'post_hidden12': tf.Variable(tf.zeros([gen_hidden_dim2])),
    'post_hidden2': tf.Variable(tf.zeros([gen_hidden_dim2])),
    'post_hidden31': tf.Variable(tf.zeros([gen_hidden_dim1])),
    'post_hidden32': tf.Variable(tf.zeros([gen_hidden_dim2])),
    'post_out': tf.Variable(tf.zeros([z_dim])),
    'disc_hidden11': tf.Variable(tf.zeros([disc_hidden_dim1])),
    'disc_hidden12': tf.Variable(tf.zeros([disc_hidden_dim2])),
    'disc_hidden21': tf.Variable(tf.zeros([disc_hidden_dim1])),
    'disc_hidden22': tf.Variable(tf.zeros([disc_hidden_dim2])),
    'disc_hidden31': tf.Variable(tf.zeros([disc_hidden_dim1])),
    'disc_hidden32': tf.Variable(tf.zeros([disc_hidden_dim2])),
    'disc_out': tf.Variable(tf.zeros([1]))
}
#likeli = p(x|z)
def likelihood(z, x, beta_0=3., beta_1=1.):
    beta = beta_0 + tf.reduce_sum(beta_1*tf.maximum(0.0, z**3), 1)
    return -tf.log(beta) - x/beta

#def likelihood(z):
#    return tf.random_gamma(shape=(data_dim, batch_size), alpha=1, beta=3+tf.pow(tf.maximum(0.0,z[:,0]),3)+tf.pow(tf.maximum(0.0,z[:,1]),3))
#post = q(z|x,eps)
def posterior(x, noise):
    hidden_layer11 = tf.nn.relu(tf.matmul(x, weights['post_hidden11'])+biases['post_hidden11'])
    hidden_layer12 = tf.nn.relu(tf.matmul(hidden_layer11, weights['post_hidden12'])+biases['post_hidden12'])
    hidden_layer2 = tf.nn.relu(tf.matmul(noise, weights['post_hidden2'])+biases['post_hidden2'])
    hidden_layer = tf.concat([hidden_layer12, hidden_layer2],axis=1)
    hidden_layer31 = tf.nn.relu(tf.matmul(hidden_layer, weights['post_hidden31'])+biases['post_hidden31'])
    hidden_layer32 = tf.nn.relu(tf.matmul(hidden_layer31, weights['post_hidden32'])+biases['post_hidden32'])
    out_layer = tf.matmul(hidden_layer32, weights['post_out'])+biases['post_out']
    return out_layer

def discriminator(z, x):
    hidden_layer11 = tf.nn.relu(tf.matmul(z, weights['disc_hidden11'])+biases['disc_hidden11'])
    hidden_layer12 = tf.nn.relu(tf.matmul(hidden_layer11, weights['disc_hidden12'])+biases['disc_hidden12'])
    hidden_layer21 = tf.nn.relu(tf.matmul(x, weights['disc_hidden21'])+biases['disc_hidden21'])
    hidden_layer22 = tf.nn.relu(tf.matmul(hidden_layer21, weights['disc_hidden22'])+biases['disc_hidden22'])
    hidden_layer = tf.concat([hidden_layer12, hidden_layer22], axis=1)
    hidden_layer31 = tf.nn.relu(tf.matmul(hidden_layer, weights['post_hidden31'])+biases['post_hidden31'])
    hidden_layer32 = tf.nn.relu(tf.matmul(hidden_layer31, weights['post_hidden32'])+biases['post_hidden32'])
    out_layer = tf.matmul(hidden_layer32, weights['disc_out'])+biases['disc_out']
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer
#Build Networks
#if no NVIDIA CUDA remove this line and unindent following lines
with tf.device('/gpu:0'):

    x_input = tf.placeholder(tf.float32, shape=[None, data_dim], name='x_input')
    noise_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='noise_input')
    prior_input = tf.placeholder(tf.float32, shape=[None, z_dim], name='disc_input')
    post_sample = posterior(x_input, noise_input)

    disc_prior = discriminator(prior_input, x_input)
    disc_post = discriminator(post_sample, x_input)

    disc_loss = -tf.reduce_mean(tf.log(disc_post+epsilon))-tf.reduce_mean(tf.log(1.0-disc_prior+epsilon))

    negLL=-tf.reduce_mean(likelihood(post_sample, x_input))
    ratio=tf.reduce_mean(tf.log(tf.divide(disc_post+epsilon,1-disc_post+epsilon)))
    nelbo=ratio+negLL

    post_vars = [weights['post_hidden11'],weights['post_hidden12'], weights['post_hidden2'], weights['post_hidden31'], weights['post_hidden32'], weights['post_out'],
    biases['post_hidden11'], biases['post_hidden12'], biases['post_hidden2'], biases['post_hidden31'], biases['post_hidden32'], biases['post_out']]

    disc_vars = [weights['disc_hidden11'], weights['disc_hidden12'], weights['disc_hidden21'], weights['disc_hidden22'], weights['disc_hidden31'], weights['disc_hidden32'], weights['disc_out'],
    biases['disc_hidden11'], biases['disc_hidden12'], biases['disc_hidden21'], biases['disc_hidden22'], biases['disc_hidden31'], biases['disc_hidden32'], biases['disc_out']]

    train_elbo = tf.train.AdamOptimizer(learning_rate=learning_rate_p).minimize(nelbo, var_list=post_vars)
    train_disc = tf.train.AdamOptimizer(learning_rate=learning_rate_d).minimize(disc_loss, var_list=disc_vars)


#if no NVIDIA CUDA take out config=...
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    #Pre-Train Discriminator
    for i in range(10000):
        z=np.sqrt(2)*np.random.randn(5*batch_size, z_dim)
        xin=np.repeat(xgen,batch_size)
        xin=xin.reshape(5*batch_size, 1)
        noise=np.random.randn(5*batch_size, noise_dim)
        feed_dict = {prior_input: z, x_input: xin, noise_input: noise}
        _, dl = sess.run([train_disc, disc_loss], feed_dict=feed_dict)
        if i % 1000 == 0 or i == 1:
            print('Step %i: Discriminator Loss: %f' % (i, dl))
    for j in range(1000):
        #Train Discriminator
        for i in range(2001):
            #Prior sample N(0,I_2x2)
            z=np.sqrt(2)*np.random.randn(5*batch_size, z_dim)
            xin=np.repeat(xgen,batch_size)
            xin=xin.reshape(5*batch_size, 1)
            noise=np.random.randn(5*batch_size, noise_dim)
            feed_dict = {prior_input: z, x_input: xin, noise_input: noise}
            _, dl = sess.run([train_disc, disc_loss], feed_dict=feed_dict)
            if i % 1000 == 0 or i == 1:
                print('Step %i: Discriminator Loss: %f' % (i, dl))
        #Train Posterior on the 5 values of x specified at the start
        for k in range(1):
            xin=np.repeat(xgen,batch_size)
            xin=xin.reshape(5*batch_size, 1)
            noise=np.random.randn(5*batch_size, noise_dim)
            feed_dict = {x_input: xin, noise_input: noise}
            _, nelboo = sess.run([train_elbo, nelbo], feed_dict=feed_dict)
            if k % 1000 == 0 or k ==1:
                print('Step %i: NELBO: %f' % (k, nelboo))

    sns.set_style('whitegrid')
    sns.set_context('poster')

    plt.subplots(figsize=(20,8))
    #make 5000 noise and 1000 of each x sample
    N_samples=2000
    noise=np.random.randn(5*N_samples, noise_dim).astype('float32')
    x_gen=np.repeat(xgen,2000)
    x_gen=x_gen.reshape(10000,1)
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
