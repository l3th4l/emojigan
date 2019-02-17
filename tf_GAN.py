import os, time, itertools, pickle#, imageio
import prep
import os.path as path
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def l_relu(x, alpha = 0.2):
    return tf.maximum(x, alpha * x)

def tconv(x, out_depth, filter_size = [4,4], strides = [2,2], padding = 'same'):
    return tf.layers.conv2d_transpose(x, out_depth, filter_size, strides = strides, padding = padding)

def conv(x, out_depth, filter_size = [4,4], strides = [2,2], padding = 'same'):
    return tf.layers.conv2d(x, out_depth, filter_size, strides = strides, padding = padding)

def bnorm(x, train = False):
    return tf.layers.batch_normalization(x, training = train)

#G(x): The generator network 
def gen(x, train = True, reuse = False):
    with tf.variable_scope('gen', reuse = reuse):
        #1st upsampling layer 
        #output dims = 4 x 4 x 1024
        tcn1 = tconv(x, 1024, strides = [1,1], padding = 'valid')
        tcn1 = bnorm(tcn1, train)
        lrlu1 = l_relu(tcn1)

        #2nd upsampling layer
        #output dims = 8 x 8 x 512 
        tcn2 = tconv(lrlu1, 512, filter_size = [5,5])
        tcn2 = bnorm(tcn2, train)
        lrlu2 = l_relu(tcn2)

        #3rd upsampling layer 
        #output dims = 16 x 16 x 256
        tcn3 = tconv(lrlu2, 256, filter_size = [5,5])
        tcn3 = bnorm(tcn3, train)
        lrlu3 = l_relu(tcn3)

        #4th upsampling layer 
        #output dims = 32 x 32 x 128
        tcn4 = tconv(lrlu3, 128, filter_size = [5,5])
        tcn4 = bnorm(tcn4, train)
        lrlu4 = l_relu(tcn4)

        #5th upsampling layer
        #output dims = 64 x 64 x 32
        tcn5 = tconv(lrlu4, 32)

        #output layer 
        #output dims = 64 x 64 x 4        
        tcn6 = conv(tcn5, 4, filter_size = [5,5], strides = [1,1])

        out = tf.nn.sigmoid(tcn6) 

        return out

#D(x): The discriminator network 
def disc(x, train = True, reuse = False):
    with tf.variable_scope('disc', reuse = reuse):
        #1st downsampling layer 
        #output size = 32 x 32 x 128
        cn1 = conv(x, 128, filter_size = [5,5])
        cn1 = bnorm(cn1, train)
        lrlu1 = l_relu(cn1)

        #2nd downsampling layer 
        #output size = 16 x 16 x 256
        cn2 = conv(lrlu1, 256, filter_size = [5,5])
        cn2 = bnorm(cn2, train)
        lrlu2 = l_relu(cn2)

        #3rd downsampling layer
        #output size = 8 x 8 x 512
        cn3 = conv(lrlu2, 512, filter_size = [5,5])
        cn3 = bnorm(cn3, train)
        lrlu3 = l_relu(cn3)

        #4th downsampling layer 
        #output size = 4 x 4 x 1024
        cn4 = conv(lrlu3, 1024, filter_size = [5,5])
        cn4 = bnorm(cn4, train)
        lrlu4 = l_relu(cn4)

        #output layer
        #output size = 1 x 1 x 1
        logits = conv(lrlu4, 1, strides = [1,1], padding = 'valid')
        out = tf.nn.sigmoid(logits)

        return logits, out

#A fixed synthetic dataset consisting
#of 25 random 100 dimentional vectors
fixed_z = np.random.normal(0, 1, [25, 1, 1, 100])

#Hyperparams 
lrate = 0.003
batch_size = 135
epochs = 3000

#load data
dat = prep.load_chunk(path.abspath('dat_bin/dat_0.npy'))
#shuffle
np.random.shuffle(dat)

#variables placeholder 
y = tf.placeholder(tf.float32, shape = [None, 64, 64, 4])
z = tf.placeholder(tf.float32, shape = [None, 1, 1, 100])
_train = tf.placeholder(tf.bool)

#generator network 
G_z = gen(z, _train)

#discriminator network
D_logits_real, D_real = disc(y, _train)
D_logits_fake, D_fake = disc(G_z, _train, reuse = True)

#Losses: 
# |__Discriminator loss
# |   |__Real:
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_logits_real, labels = tf.ones([batch_size, 1, 1, 1])))
# |   |__Fake:
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_logits_fake, labels = tf.zeros([batch_size, 1, 1, 1])))
# |  :
D_loss = D_loss_real + D_loss_fake
# |__Generator loss:
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_logits_fake, labels = tf.ones([batch_size, 1, 1, 1])))

#trainable variables
D_vars = tf.trainable_variables(scope = 'disc')
G_vars = tf.trainable_variables(scope = 'gen')

#optimizers 
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    D_opt = tf.train.AdamOptimizer(lrate, beta1 = 0.7).minimize(D_loss, var_list = D_vars)
    G_opt = tf.train.AdamOptimizer(lrate * 0.85, beta1 = 0.7).minimize(G_loss, var_list = G_vars)

#make tf session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

#resize images
dat = tf.image.resize_images(dat, [64, 64]).eval() 

def show_result(num_epoch, show = False, save = False, path = 'result.png'):
    test_images = sess.run(G_z, {z: fixed_z, _train: False})

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(size_figure_grid*size_figure_grid):
        i = k // size_figure_grid
        j = k % size_figure_grid
        ax[i, j].cla()
        ax[i, j].imshow(np.reshape(test_images[k], (64, 64, 4)), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

# results save folder
root = 'DCGAN_results/'
model = 'DCGAN_'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results'):
    os.mkdir(root + 'Fixed_results')

tvars = tf.trainable_variables()
saver = tf.train.Saver(var_list = tvars)
filedir, filename = 'saved_state', 'test_2.ckpt'
path = os.path.abspath('%s/%s' % (filedir, filename))

try:
    saver.restore(sess, path)
    print('model restored')
except:
    try:
        os.mkdir(os.path.abspath(filedir))
    except:
        pass
    saver.save(sess, path)
    print('model saved at %s' % (path))

#training iterations
gen_losses = []
disc_losses = []
print(dat.shape[0] // batch_size)
for epoch in range(epochs):
    saver.save(sess, path)
    gen_loss = []
    disc_loss = []
    ep_time = time.time()
    for _iter in range(dat.shape[0] // batch_size):
        print('iteration :' + str(_iter))
        #optimize discriminator 
        _y = dat[_iter * batch_size : (_iter + 1) * batch_size]
        _z = np.random.normal(0, 1, [batch_size, 1, 1, 100])
        _d_loss, _ = sess.run([D_loss, D_opt], {y : _y, z : _z, _train : True})
        disc_loss.append(_d_loss)

        #optimize generator
        _z = np.random.normal(0, 1, [batch_size, 1, 1, 100])
        _g_loss, _ = sess.run([G_loss, G_opt], {z : _z, y : _y, _train : True})
        gen_loss.append(_g_loss)

    ep_time = time.time() - ep_time
    disc_losses.append(np.mean(disc_loss))
    gen_losses.append(np.mean(gen_loss))
    print('ep[%d/%d]: d_loss = %.3f ; g_loss = %.3f ; ep_time = %.2f' % (epoch + 1, epochs, ep_time, np.mean(disc_loss), np.mean(gen_loss)))

    fixed_path = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
    show_result((epoch + 1), save = True, path = fixed_path)

try:
    os.listdir('losses')
except:
    os.mkdir('losses')

d_l_file = os.path.abspath('losses/d_losses.npy')
g_l_file = os.path.abspath('losses/g_losses.npy')

np.save(d_l_file, np.array(disc_losses))
np.save(g_l_file, np.array(gen_losses))

sess.close() 