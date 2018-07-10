import numpy as np
import tensorflow as tf
from matplotlib import pylab as plt
import seaborn as sns
tf.reset_default_graph()
#params
epsilon=0.0000000001
batch_size=32
learning_rate=0.001
z_dim = 2
noise_dim=3
gen_hidden_dim=32
data_dim=1
disc_hidden_dim=32
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

weights = {
    'post_hidden1': tf.Variable(xavier_init(noise_dim+data_dim, gen_hidden_dim)),
    'post_hidden2': tf.Variable(xavier_init(gen_hidden_dim, gen_hidden_dim)),
    'post_out': tf.Variable(xavier_init(gen_hidden_dim, z_dim)),
    'disc_hidden1': tf.Variable(xavier_init(z_dim, disc_hidden_dim)),
    'disc_hidden2': tf.Variable(xavier_init(disc_hidden_dim, disc_hidden_dim)),
    'disc_out': tf.Variable(xavier_init(disc_hidden_dim, 1))
}
biases = {
    'post_hidden1': tf.Variable(tf.zeros([gen_hidden_dim])),
    'post_hidden2': tf.Variable(tf.zeros([gen_hidden_dim])),
    'post_out': tf.Variable(tf.zeros([z_dim])),
    'disc_hidden1': tf.Variable(tf.zeros([disc_hidden_dim])),
    'disc_hidden2': tf.Variable(tf.zeros([disc_hidden_dim])),
    'disc_out': tf.Variable(tf.zeros([1]))
}
#likeli = p(x|z)

def likelihood(z):
    return tf.random_gamma(shape=(data_dim, batch_size), alpha=1, beta=3+tf.pow(tf.maximum(0.0,z[:,0]),3)+tf.pow(tf.maximum(0.0,z[:,1]),3))
#post = q(z|x,eps)
def posterior(x, noise):
    input = tf.concat([x, noise], axis=1)
    hidden_layer1 = tf.nn.relu(tf.matmul(input, weights['post_hidden1'])+biases['post_hidden1'])
    hidden_layer2 = tf.matmul(hidden_layer1, weights['post_hidden2'])
    hidden_layer2 = tf.add(hidden_layer2, biases['post_hidden2'])
    hidden_layer2 = tf.nn.relu(hidden_layer2)
    out_layer = tf.matmul(hidden_layer2, weights['post_out'])
    out_layer = tf.add(out_layer, biases['post_out'])
    #out_layer = tf.nn.sigmoid(out_layer)
    return out_layer

def discriminator(z):
    hidden_layer1 = tf.matmul(z, weights['disc_hidden1'])
    hidden_layer1 = tf.add(hidden_layer1, biases['disc_hidden1'])
    hidden_layer1 = tf.nn.relu(hidden_layer1)
    hidden_layer2 = tf.matmul(hidden_layer1, weights['disc_hidden2'])
    hidden_layer2 = tf.add(hidden_layer2, biases['disc_hidden2'])
    hidden_layer2 = tf.nn.relu(hidden_layer2)
    out_layer = tf.matmul(hidden_layer2, weights['disc_out'])
    out_layer = tf.add(out_layer, biases['disc_out'])
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer
#Build Networks
x_input = tf.placeholder(tf.float32, shape=[None, data_dim], name='x_input')
noise_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='noise_input')
disc_input = tf.placeholder(tf.float32, shape=[None, z_dim], name='disc_input')
post_sample = posterior(x_input, noise_input)

#beta_expr=3+tf.pow(tf.maximum(0.0,post_sample[:,0]),3)+tf.pow(tf.maximum(0.0,post_sample[:,1]),3)
#X_like=tf.log(beta_expr+epsilon)+xgen/beta_expr
X_like = likelihood(post_sample)
disc_prior = discriminator(disc_input)
disc_post = discriminator(post_sample)

disc_loss = -tf.reduce_mean(tf.log(disc_post+epsilon)+tf.log(1.0-disc_prior+epsilon))

negLL=-tf.reduce_mean(X_like)
ratio=tf.reduce_mean(tf.log(tf.divide(disc_post+epsilon,(1-disc_post+epsilon))))
nelbo=ratio+negLL

post_vars = [weights['post_hidden1'], weights['post_hidden2'], weights['post_out'], biases['post_hidden1'], biases['post_hidden2'], biases['post_out']]
disc_vars = [weights['disc_hidden1'], weights['disc_hidden2'], weights['disc_out'], biases['disc_hidden1'], biases['disc_hidden2'], biases['disc_out']]

train_elbo = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(nelbo, var_list=post_vars)
train_disc = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(disc_loss, var_list=disc_vars)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for j in range(5):
        #Train Discriminator
        for i in range(10000):
            #Prior sample N(0,I_2x2)
            z=np.random.randn(batch_size, z_dim)
            #Use prior samples to make an x sample
            x=np.random.exponential(size=(data_dim, batch_size), scale=3+np.power(np.maximum(0.0,z[:,0]),3)+np.power(np.maximum(0.0,z[:,1]),3))
            x=np.transpose(x)
            noise=np.random.randn(batch_size, noise_dim)
            feed_dict = {disc_input: z, x_input: x, noise_input: noise}
            _, dl = sess.run([train_disc, disc_loss], feed_dict=feed_dict)
            if i % 1000 == 0 or i == 1:
                print('Step %i: Discriminator Loss: %f' % (i, dl))
        #Train Posterior on the 5 values of x specified at the start
        for k in range(1):
        #    if i % 5 == 0:
        #        xin=np.repeat(0,batch_size)
        #    if i % 5 == 1:
        #        xin=np.repeat(5,batch_size)
        #    if i % 5 == 2:
        #        xin=np.repeat(8,batch_size)
        #    if i % 5 == 3:
        #        xin=np.repeat(12,batch_size)
        #    if i % 5 == 4:
        #        xin=np.repeat(50,batch_size)
        #    xin=xin.reshape(batch_size, 1)
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
