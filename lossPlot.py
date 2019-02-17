from matplotlib import pyplot as plt 
import numpy as np 
import os

try:
    d_loss = np.load(os.path.abspath('losses/d_losses.npy'))
    g_loss = np.load(os.path.abspath('losses/g_losses.npy'))
except:
    print('loss data not found')
    quit()

plt.plot(d_loss, label = 'discriminator loss', color = 'red')
plt.plot(g_loss, label = 'genrator loss', color = 'blue')
plt.xlabel('epoch')
plt.ylabel('loss')

plt.legend()

plt.savefig(os.path.abspath('losses/plot.png'))