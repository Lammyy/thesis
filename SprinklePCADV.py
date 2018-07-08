import numpy as np
import tensorflow as tf
from matplotlib import pylab as plt
import seaborn as sns
tf.reset_default_graph()
#z=tf.Variable(tf.random_normal([2],mean=0.0,stddev=1.0,dtype=tf.float32))
#x=tf.Variable(tf.random_gamma([1],alpha=1,beta=3+np.power(tf.maximum(0.0,z[0]),3)+np.power(tf.maximum(0.0,z[1]),3)),dtype=tf.float32)
#noise=tf.Variable(tf.random_normal([1],mean=0.0,stddev=1.0,dtype=tf.float32))
#input=tf.concat([x,noise],0)
#params
epsilon=0.00000001
batch_size=32
learning_rate=0.001
z_dim = 2
noise_dim=3
like_hidden_dim=64
gen_hidden_dim=64
data_dim=1
disc_hidden_dim=64
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
xgen=np.array(y, dtype=np.float32)
llh = [None]*5
llh[0] = likelihoodd(x, y[0])
llh[1] = likelihoodd(x, y[1])
llh[2] = likelihoodd(x, y[2])
llh[3] = likelihoodd(x, y[3])
llh[4] = likelihoodd(x, y[4])
#Xavier Initiaizlier
def xavier_init(fan_in, fan_out, constant=1):
    low=-constant*np.sqrt(6.0/(fan_in+fan_out))
    high=constant*np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform((fan_in, fan_out),minval=low,maxval=high, dtype=tf.float32)

weights = {
    'like_hidden1': tf.Variable(xavier_init(z_dim, like_hidden_dim)),
    'like_out': tf.Variable(xavier_init(like_hidden_dim, data_dim)),
    'post_hidden1': tf.Variable(xavier_init(noise_dim+data_dim, gen_hidden_dim)),
    'post_out': tf.Variable(xavier_init(gen_hidden_dim, z_dim)),
    'disc_hidden1': tf.Variable(xavier_init(z_dim, disc_hidden_dim)),
    'disc_out': tf.Variable(xavier_init(disc_hidden_dim, 1))
}
biases = {
    'like_hidden1': tf.Variable(tf.zeros([like_hidden_dim])),
    'like_out': tf.Variable(tf.zeros([data_dim])),
    'post_hidden1': tf.Variable(tf.zeros([gen_hidden_dim])),
    'post_out': tf.Variable(tf.zeros([z_dim])),
    'disc_hidden1': tf.Variable(tf.zeros([disc_hidden_dim])),
    'disc_out': tf.Variable(tf.zeros([1]))
}
#likeli = p(x|z)
def likelihood(z):
    return tf.random_gamma(shape=(data_dim, batch_size), alpha=1, beta=3+tf.pow(tf.maximum(0.0,z[:,0]),3)+tf.pow(tf.maximum(0.0,z[:,1]),3))
#def likelihood(z):
#    hidden_layer = tf.matmul(z, weights['like_hidden1'])
#    hidden_layer = tf.add(hidden_layer, biases['like_hidden1'])
#    hidden_layer = tf.nn.sigmoid(hidden_layer)
#    logits=tf.matmul(hidden_layer, weights['like_out'])
#    logits=tf.add(logits, biases['like_out'])
#    out_layer=tf.nn.sigmoid(logits)
#    return out_layer, logits
#post = q(z|x,eps)
def posterior(x, noise):
    #noise=tf.Variable(tf.random_normal([noise_dim],mean=0.0,stddev=1.0,dtype=tf.float32))
    #input=tf.concat([x,noise],0)
    input = tf.concat([x, noise], axis=1)
    hidden_layer = tf.matmul(input, weights['post_hidden1'])
    hidden_layer = tf.add(hidden_layer, biases['post_hidden1'])
    hidden_layer = tf.nn.sigmoid(hidden_layer)
    out_layer = tf.matmul(hidden_layer, weights['post_out'])
    out_layer = tf.add(out_layer, biases['post_out'])
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer

def discriminator(z):
    hidden_layer = tf.matmul(z, weights['disc_hidden1'])
    hidden_layer = tf.add(hidden_layer, biases['disc_hidden1'])
    hidden_layer = tf.nn.sigmoid(hidden_layer)
    out_layer = tf.matmul(hidden_layer, weights['disc_out'])
    out_layer = tf.add(out_layer, biases['disc_out'])
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer
#Build Networks
x_input = tf.placeholder(tf.float32, shape=[None, data_dim], name='data_input')
noise_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='data_input')
disc_input = tf.placeholder(tf.float32, shape=[None, z_dim], name='disc_input')
post_sample = posterior(x_input, noise_input)
X_like = likelihood(post_sample)
disc_real = discriminator(disc_input)
disc_fake = discriminator(post_sample)

disc_loss = -tf.reduce_mean(tf.log(disc_real+epsilon)+tf.log(1.0-disc_fake+epsilon))
negLL=-tf.reduce_mean(X_like)
ratio=-tf.reduce_mean(tf.log(tf.divide(disc_fake+epsilon,(1-disc_fake+epsilon))))
nelbo=ratio+negLL


#like_vars = [weights['like_hidden1'], weights['like_out'], biases['like_hidden1'], biases['like_out']]
post_vars = [weights['post_hidden1'], weights['post_out'], biases['post_hidden1'], biases['post_out']]
disc_vars = [weights['disc_hidden1'], weights['disc_out'], biases['disc_hidden1'], biases['disc_out']]

train_elbo = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(nelbo, var_list=post_vars)
train_disc = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(disc_loss, var_list=disc_vars)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for j in range(4):
        for i in range(15000):
            #z=tf.random_normal([batch_size,2],mean=0.0,stddev=1.0,dtype=tf.float32)
            z=np.random.randn(batch_size, z_dim)
            #x=tf.random_gamma([batch_size,1],alpha=1,beta=3+np.power(tf.maximum(0.0,z[0]),3)+np.power(tf.maximum(0.0,z[1]),3),dtype=tf.float32)
            #x=likelihood(z)
            x=np.random.exponential(size=(data_dim, batch_size), scale=3+np.power(np.maximum(0.0,z[:,0]),3)+np.power(np.maximum(0.0,z[:,1]),3))
            x=np.transpose(x)
            #noise=tf.random_normal([noise_dim],mean=0.0,stddev=1.0,dtype=tf.float32)
            noise=np.random.randn(batch_size, noise_dim)
            feed_dict = {disc_input: z, x_input: x, noise_input: noise}
            _, dl = sess.run([train_disc, disc_loss], feed_dict=feed_dict)
            if i % 1000 == 0 or i == 1:
                print('Step %i: Discriminator Loss: %f' % (i, dl))
        for k in range(4000):
            feed_dict = {x_input: x, noise_input: noise}
            _, nelboo = sess.run([train_elbo, nelbo], feed_dict=feed_dict)
            if k % 1000 == 0 or k ==1:
                print('Step %i: NELBO: %f' % (k, nelboo))
sns.set_style('whitegrid')
sns.set_context('poster')

plt.subplots(figsize=(20,8))
N_samples=1000
noise=np.random.randn(5*N_samples, noise_dim).astype('float32')
#noise=tf.random_normal([N_samples,z],mean=0.0,stddev=1.0,dtype=tf.float32)
x_gen=np.repeat(xgen,1000)
x_gen=x_gen.reshape(5000,1)
z_samples=posterior(x_gen,noise)
z_samples=tf.reshape(z_samples,[xgen.shape[0], N_samples, 2])
for i in range(5):
    plt.subplot(2,5,i+1)
    #sns.kdeplot(z_samples[i,:,:], cmap='Greens')
    plt.scatter(z_samples[i,:,0],z_samples[i,:,1])
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
