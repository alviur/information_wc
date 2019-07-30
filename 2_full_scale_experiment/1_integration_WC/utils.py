import numpy as np
import tensorflow as tf
import scipy.io as sio


def apply_H4_3scales(f_flat,W):


    sizes = [40, 40, 40, 40, 40, 20, 20, 20, 20, 10, 10, 10, 10, 5]

    w1 = np.zeros((1, 40, 40, 14))
    w2 = np.zeros((1, 20, 20, 14))
    w3 = np.zeros((1, 10, 10, 14))
    w4 = np.zeros((1, 5, 5, 14))

    print(f_flat)

    # Separate in list of bands
    f = [tf.reshape(f_flat[0:40 * 40], [40, 40]),
         tf.reshape(f_flat[40 * 40:40 * 40 * 2], [40, 40]),
         tf.reshape(f_flat[40 * 40 * 2:40 * 40 * 3], [40, 40]),
         tf.reshape(f_flat[40 * 40 * 3:40 * 40 * 4], [40, 40]),
         tf.reshape(f_flat[40 * 40 * 4:40 * 40 * 5], [40, 40]),
         tf.reshape(f_flat[40 * 40 * 5:8400], [20, 20]),
         tf.reshape(f_flat[8400:8800], [20, 20]),
         tf.reshape(f_flat[8800:9200], [20, 20]),
         tf.reshape(f_flat[9200:9600], [20, 20]),
         tf.reshape(f_flat[9600:9700], [10, 10]),
         tf.reshape(f_flat[9700:9800], [10, 10]),
         tf.reshape(f_flat[9800:9900], [10, 10]),
         tf.reshape(f_flat[9900:10000], [10, 10]),
         tf.reshape(f_flat[10000:10025], [5, 5])]

    for subband in range(14):

        if (subband < 5):

            w1 += (tf.nn.depthwise_conv2d(
                input=tf.expand_dims(tf.expand_dims(f[subband], axis=0), axis=3),
                # filter=tf.constant((kernelW_t['w1'][:, :, :, 0 * 18:0 * 18 + 18])),
                filter=tf.constant((W['w1'][:, :, :, subband * 14:subband * 14 + 14]),dtype = tf.float32),
                strides=[1, 1, 1, 1],
                padding="SAME"))


        elif (subband < 9):

            w2 += (tf.nn.depthwise_conv2d(
                input=tf.expand_dims(tf.expand_dims(f[subband], axis=0), axis=3),
                filter=tf.constant(W['w2'][:, :, :, (subband - 5) * 14:(subband - 5) * 14 + 14],dtype = tf.float32),
                strides=[1, 1, 1, 1],
                padding="SAME"))

        elif (subband < 13):

            w3 += tf.nn.depthwise_conv2d(
                input=tf.expand_dims(tf.expand_dims(f[subband], axis=0), axis=3),
                filter=tf.constant(W['w3'][:, :, :, (subband - 9) * 14:(subband - 9) * 14 + 14],dtype = tf.float32),
                strides=[1, 1, 1, 1], padding="SAME")

        else:

            w4 += (tf.nn.depthwise_conv2d(
                input=tf.expand_dims(tf.expand_dims(f[subband], axis=0), axis=3),
                filter=tf.constant(W['w4'][:, :, :, (subband - 13) * 14:(subband - 13) * 14 + 14],dtype = tf.float32),
                strides=[1, 1, 1, 1],
                padding="SAME"))

    out = []

    for i in range(14):
        w1s = tf.cast(tf.image.resize_images(tf.expand_dims(w1[:, :, :, i], axis=3), [sizes[i], sizes[i]],
                                             method=tf.image.ResizeMethod.BICUBIC, align_corners=True), tf.float32)
        w2s = tf.cast(tf.image.resize_images(tf.expand_dims(w2[:, :, :, i], axis=3), [sizes[i], sizes[i]],
                                             method=tf.image.ResizeMethod.BICUBIC, align_corners=True), tf.float32)
        w3s = tf.cast(tf.image.resize_images(tf.expand_dims(w3[:, :, :, i], axis=3), [sizes[i], sizes[i]],
                                             method=tf.image.ResizeMethod.BICUBIC, align_corners=True), tf.float32)
        w4s = tf.cast(tf.image.resize_images(tf.expand_dims(w4[:, :, :, i], axis=3), [sizes[i], sizes[i]],
                                             method=tf.image.ResizeMethod.BICUBIC, align_corners=True), tf.float32)

        out.append(w1s + w2s + w3s + w4s)

    flatOut2 = tf.concat(
        [tf.reshape(out[0], [-1]), tf.reshape(out[1], [-1]), tf.reshape(out[2], [-1]), tf.reshape(out[3], [-1]),
         tf.reshape(out[4], [-1]), tf.reshape(out[5], [-1]), tf.reshape(out[6], [-1]), tf.reshape(out[7], [-1]),
         tf.reshape(out[8], [-1]), tf.reshape(out[9], [-1]), tf.reshape(out[10], [-1]), tf.reshape(out[11], [-1]),
         tf.reshape(out[12], [-1]), tf.reshape(out[13], [-1])], 0)

    return flatOut2

def apply_H4(f_flat,W):

    sizes = [64, 64, 64, 64, 64, 32, 32, 32, 32, 16, 16, 16, 16, 8, 8, 8, 8, 4]

    w1 = np.zeros((1, 64, 64, 18))
    w2 = np.zeros((1, 32, 32, 18))
    w3 = np.zeros((1, 16, 16, 18))
    w4 = np.zeros((1, 8, 8, 18))
    w5 = np.zeros((1, 4, 4, 18))

    # Separate in list of bands
    f = [tf.reshape(f_flat[0:64 * 64], [64, 64]), tf.reshape(f_flat[64 * 64:64 * 64 * 2], [64, 64]),
         tf.reshape(f_flat[64 * 64 * 2:64 * 64 * 3], [64, 64]),
         tf.reshape(f_flat[64 * 64 * 3:64 * 64 * 4], [64, 64]),
         tf.reshape(f_flat[64 * 64 * 4:64 * 64 * 5], [64, 64]), tf.reshape(f_flat[64 * 64 * 5:21504], [32, 32]),
         tf.reshape(f_flat[21504:22528], [32, 32]), tf.reshape(f_flat[22528:23552], [32, 32]),
         tf.reshape(f_flat[23552:24576], [32, 32]),
         tf.reshape(f_flat[24576:24832], [16, 16]), tf.reshape(f_flat[24832:25088], [16, 16]),
         tf.reshape(f_flat[25088:25344], [16, 16]),
         tf.reshape(f_flat[25344:25600], [16, 16]), tf.reshape(f_flat[25600:25664], [8, 8]),
         tf.reshape(f_flat[25664:25728], [8, 8]),
         tf.reshape(f_flat[25728:25792], [8, 8]), tf.reshape(f_flat[25792:25856], [8, 8]),
         tf.reshape(f_flat[25856:25872], [4, 4])]

    for subband in range(18):

        if (subband < 5):

            w1 += (tf.nn.conv2d(
                input=tf.expand_dims(tf.expand_dims(f[subband], axis=0), axis=3),
                filter=tf.constant((W['w1'][:, :, :, subband * 18:subband * 18 + 18])),
                strides=[1, 1, 1, 1],
                padding="SAME"))

        elif (subband < 9):

            w2 += (tf.nn.conv2d(
                input=tf.expand_dims(tf.expand_dims(f[subband], axis=0), axis=3),
                filter=tf.constant(W['w2'][:, :, :, (subband - 5) * 18:(subband - 5) * 18 + 18]),
                strides=[1, 1, 1, 1],
                padding="SAME"))

        elif (subband < 13):

            w3 += (tf.nn.conv2d(
                input=tf.expand_dims(tf.expand_dims(f[subband], axis=0), axis=3),
                filter=tf.constant(W['w3'][:, :, :, (subband - 9) * 18:(subband - 9) * 18 + 18]),
                strides=[1, 1, 1, 1],
                padding="SAME"))

        elif (subband < 17):

            w4 += (tf.nn.conv2d(
                input=tf.expand_dims(tf.expand_dims(f[subband], axis=0), axis=3),
                filter=tf.constant(W['w4'][:, :, :, (subband - 13) * 18:(subband - 13) * 18 + 18]),
                strides=[1, 1, 1, 1],
                padding="SAME"))

        else:
            w5 += (tf.nn.conv2d(
                input=tf.expand_dims(tf.expand_dims(f[subband], axis=0), axis=3),
                filter=tf.constant(W['w5'][:, :, :, (subband - 17) * 18:(subband - 17) * 18 + 18]),
                strides=[1, 1, 1, 1],
                padding="SAME"))


    out = []

    for i in range(18):
        out.append(tf.cast(tf.image.resize_images(tf.expand_dims(w1[:, :, :, i], axis=3), [sizes[i], sizes[i]],
                                                  method=tf.image.ResizeMethod.BICUBIC, align_corners=False),
                           tf.float32) + tf.cast(tf.image.resize_images(tf.expand_dims(w2[:, :, :, i], axis=3), [sizes[i], sizes[i]],
                                   method=tf.image.ResizeMethod.BICUBIC, align_corners=False), tf.float32) +
                   tf.cast(tf.image.resize_images(tf.expand_dims(w3[:, :, :, i], axis=3), [sizes[i], sizes[i]],
                                                  method=tf.image.ResizeMethod.BICUBIC, align_corners=False),
                           tf.float32) + tf.cast(tf.image.resize_images(tf.expand_dims(w4[:, :, :, i], axis=3), [sizes[i], sizes[i]],
                                   method=tf.image.ResizeMethod.BICUBIC, align_corners=False), tf.float32) +
                   tf.cast(tf.image.resize_images(tf.expand_dims(w5[:, :, :, i], axis=3), [sizes[i], sizes[i]],
                                                  method=tf.image.ResizeMethod.BICUBIC, align_corners=False),tf.float32))

    flatOut = tf.concat(
        [tf.reshape(out[0], [-1]), tf.reshape(out[1], [-1]), tf.reshape(out[2], [-1]), tf.reshape(out[3], [-1]),
         tf.reshape(out[4], [-1]), tf.reshape(out[5], [-1]), tf.reshape(out[6], [-1]), tf.reshape(out[7], [-1]),
         tf.reshape(out[8], [-1]), tf.reshape(out[9], [-1]), tf.reshape(out[10], [-1]), tf.reshape(out[11], [-1]),
         tf.reshape(out[12], [-1]), tf.reshape(out[13], [-1]), tf.reshape(out[14], [-1]), tf.reshape(out[15], [-1]),
         tf.reshape(out[16], [-1]), tf.reshape(out[17], [-1])], 0)





    return flatOut


def saturation_f(x,g,xm,epsilon,sizeT):

    K = tf.pow(xm, tf.scalar_mul(1 - g, tf.ones([sizeT,1])))
    K = tf.where(tf.is_nan(K), tf.zeros_like(K), K)

    a = (2 - g) * K*(epsilon**(g - 1))
    a = tf.where(tf.is_nan(a), tf.zeros_like(a), a)
    b = (g-1) * K*(epsilon**(g - 2))
    b = tf.where(tf.is_nan(b), tf.zeros_like(b), b)

    pG = tf.math.greater(x, tf.ones([sizeT,1]) * epsilon)
    pG_zeros = tf.count_nonzero(pG)

    # pp = find((x <= epsilon) & (x >= 0));
    pp1 = tf.math.less_equal(x, tf.ones([sizeT,1]) * epsilon)
    pp2 = tf.math.greater_equal(x, tf.zeros([sizeT,1]))
    pp1_zeros = tf.count_nonzero(pp1)
    pp2_zeros = tf.count_nonzero(pp2)

    #
    nG = tf.math.less(x, -tf.ones([sizeT,1]) * epsilon)
    np1 = tf.math.greater(x, -tf.ones([sizeT,1]) * epsilon)
    np2 = tf.math.less_equal(x, tf.zeros([sizeT,1]))
    nG_zeros = tf.count_nonzero(nG)
    np1_zeros = tf.count_nonzero(np1)
    np2_zeros = tf.count_nonzero(np2)

    f = x;

    def f1(): return tf.where(pG, K*tf.pow(x, (g * tf.ones([sizeT,1]))), f)
    def f2(): return f
    f1 = tf.cond(tf.math.greater(pG_zeros, 0), f1, f2)


    def f3(): return tf.where(nG, -K*tf.pow(tf.abs(x), (g * tf.ones([sizeT,1]))), f1)
    def f4(): return f1
    f2 = tf.cond(tf.math.greater(nG_zeros, 0), f3, f4)


    def f5(): return tf.where(tf.math.logical_and(pp1,pp2), a * tf.abs(x) + b *tf.pow(x, 2 * tf.ones([sizeT,1])), f2)
    def f6(): return f2
    f3 = tf.cond(tf.math.greater(pp1_zeros + pp2_zeros, 1), f5, f6)


    def f7(): return tf.where(tf.math.logical_and(np1,np2), -(a * tf.abs(x) + b * tf.pow(x, 2 * tf.ones([sizeT,1]))), f3)
    def f8(): return f3
    f4 = tf.cond(tf.math.greater(np1_zeros + np2_zeros, 1), f7, f8)

    return f4


def loadKernelW_3wscales(path,name):

    mat_w = sio.loadmat(path)
    W = mat_w[name]

    kernelW_t = {
        # 40x40
        'w1': (np.zeros([40, 40, 1, 70])),
        # 20x201
        'w2': (np.zeros([20, 20, 1, 56])),
        # 10x10
        'w3': (np.zeros([10, 10, 1, 56])),
        # 5x5
        'w4': (np.zeros([5, 5, 1, 14]))
    }

    # Save kernel in list
    contW1 = contW2 = contW3 = contW4  = 0;

    for j in range(14):
        for i in range(14):

            if (W[i, j].shape[1] == 40):
                kernelW_t['w1'][:, :, :, contW1] = np.expand_dims(W[i, j], axis=2)
                contW1 += 1

            elif (W[i, j].shape[1] == 20):
                kernelW_t['w2'][:, :, :, contW2] = np.expand_dims(W[i, j], axis=2)
                contW2 += 1
            elif (W[i, j].shape[1] == 10):
                kernelW_t['w3'][:, :, :, contW3] = np.expand_dims(W[i, j], axis=2)
                contW3 += 1
            else:
                kernelW_t['w4'][:, :, :, contW4] = np.expand_dims(W[i, j], axis=2)
                contW4 += 1

    return kernelW_t


def loadKernelW(path,name):



    mat_w = sio.loadmat(path )
    W = mat_w[name]

    kernelW_t = {
        # 64x64
        'w1': (np.zeros([64, 64, 1, 90])),
        # 32x32
        'w2': (np.zeros([32, 32, 1, 72])),
        # 16x16
        'w3': (np.zeros([16, 16, 1, 72])),
        # 8x8
        'w4': (np.zeros([8, 8, 1, 72])),
        # 4x4
        'w5': (np.zeros([4, 4, 1, 18]))
    }

    # Save kernel in list
    contW1 = contW2 = contW3 = contW4 = contW5 = 0;

    for j in range(18):
        for i in range(18):
            if (W[i, j].shape[1] == 64):
                kernelW_t['w1'][:, :, :, contW1] = np.expand_dims(W[i, j], axis=2)
                contW1 += 1

            elif (W[i, j].shape[1] == 32):
                kernelW_t['w2'][:, :, :, contW2] = np.expand_dims(W[i, j], axis=2)
                contW2 += 1
            elif (W[i, j].shape[1] == 16):
                kernelW_t['w3'][:, :, :, contW3] = np.expand_dims(W[i, j], axis=2)
                contW3 += 1
            elif (W[i, j].shape[1] == 8):
                kernelW_t['w4'][:, :, :, contW4] = np.expand_dims(W[i, j], axis=2)
                contW4 += 1
            else:
                kernelW_t['w5'][:, :, :, contW5] = np.expand_dims(W[i, j], axis=2)
                contW5 += 1

    return kernelW_t
