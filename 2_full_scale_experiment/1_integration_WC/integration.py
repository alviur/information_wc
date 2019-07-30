import numpy as np
import tensorflow as tf
import utils
import scipy.io as sio
import time

path = ''
savePath = '/home/agomez/7_WC_integration/1_data/results_2/'

# Parameters
g = 0.5
epsilon = 0.1
sizeT = 10025
maxSteps = 650;
k = 1;
deltat = 1e-5;
mat_w = sio.loadmat('alfa_m.mat')
Da = (mat_w['alfa_m'])

# Load and prepare interaction kernel W
W = utils.loadKernelW_3wscales(path+'W.mat','W')



xt = tf.placeholder(tf.float32, shape=(sizeT,1))
e = tf.placeholder(tf.float32, shape=(sizeT,1))
xm = Da

#f = tf.cast(tf.squeeze(utils.saturation_f(tf.abs(tf.expand_dims(xt,axis=1)),g,tf.expand_dims(xt,axis=1),epsilon,sizeT)),dtype =tf.float64)
#xtm1 = xt + e*deltat - k*Da*xt*deltat - k*WDxm*(sign(xt).*abs(xt).^ga)*deltat;

interactions = (k * utils.apply_H4_3scales(tf.sign(xt)*(tf.pow(tf.abs(xt), tf.scalar_mul(g, tf.ones([sizeT,1])))),W))* deltat
saturation = k * Da * xt * deltat
sample = e * deltat
inStep = xt + e * deltat - saturation - tf.expand_dims(interactions,axis=1)
#inStep = xt + e * deltat - k * Da * xt * deltat - (k * utils.apply_H4(f,W))* deltat
#inStep = utils.apply_H4(f,W)





# Launch the default graph.
with tf.Session() as sess:


    for file in range(21):


        if(file+1 != 3):

            # Load data
            mainPath = '/home/agomez/7_WC_integration/1_data/TF/'
            mat_data = sio.loadmat(mainPath +str(file+1)+'.mat')
            data = (mat_data['batchTest'])

            integrated = np.zeros((10025, data.shape[1]))

            for img in range(data.shape[1]):

                xtm1 = data[:, img]
                imgTemp = data[:,img]

                print(str(file+1)+'.mat '+str(img), ' of ',data.shape[1])

                for t in range(maxSteps):


                    #xtm1 = xt + e * deltat - k * Da * xt * deltat - (k * W * f). * deltat;
                    xtm1,intera, sat,s = sess.run([inStep,interactions,saturation,sample],feed_dict={xt: np.expand_dims(np.squeeze(xtm1),axis=1), e: np.expand_dims(imgTemp,axis=1)})

                    xt_ant = xtm1; # save previous step
                    #print(xt_ant.shape,s.shape,intera.shape,sat.shape,Da.shape)

                #elapsed = time.clock() - time1
                #print('Computing time:',elapsed)

                integrated[:,img] = np.squeeze(xtm1)

            print('Saving '+str(file+1))
            sio.savemat(savePath+str(file+1)+'.mat', {'integrated': integrated})
