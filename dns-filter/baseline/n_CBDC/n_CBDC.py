import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import os

# C_1 size: 38
C_1 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
     'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '-', '.']

# C_2 size: 1444 = 38*38
C_2 = []
for i in range(len(C_1)):
    for j in range(len(C_1)):
        C_2.append(C_1[i]+C_1[j])

Model_Save_Path = './Model/'


# 将各组域名用n-gram表示作为神经网络的输入(n = 2)
def n_gram_representation(dns_data, c=C_1, c_2=C_2, N_MAX=64):
    data_input = np.zeros(
        [dns_data.shape[0], len(c)*len(c), N_MAX], dtype='float32')
    for i in range(dns_data.shape[0]):
        dns = dns_data[i][0]
        dns_C = []
        for j in range(len(dns)-1):
            dns_C.append(dns[j]+dns[j+1])
        for j in range(len(dns_C)):
            index_ = c_2.index(dns_C[j])
            data_input[i][index_][j] = 1
    return data_input


# # 将dns_normal 用n-gram表示
# train_data_normal = n_gram_representation(dns_normal)
# train_data_normal_inputs = train_data_normal[:, :, :, np.newaxis]

# train_data_dga = n_gram_representation(dns_dga)
# train_data_dga_inputs = train_data_dga[:, :, :, np.newaxis]


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def Conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def Max_Pool(x):
    return tf.nn.max_pool(x, ksize=[1, 38, 2, 1], strides=[1, 38, 2, 1], padding='SAME')


def FC(inputs, n_in, n_out):
    scale = np.sqrt(6./n_in+n_out)
    weights = weight_variable([n_in, n_out])
    biases = bias_variable([n_out])
    return tf.matmul(inputs, weights) + biases


def Dropout(inputs, keep_prob):
    return tf.nn.dropout(inputs, keep_prob)


def feature_extraction(inputs, in_channels=1, out_channels=64):
    W_conv2 = weight_variable([38, 2, in_channels, out_channels])
    W_conv3 = weight_variable([38, 3, in_channels, out_channels])
    W_conv5 = weight_variable([38, 5, in_channels, out_channels])
    W_conv7 = weight_variable([38, 7, in_channels, out_channels])

    b_conv2 = bias_variable([out_channels])
    b_conv3 = bias_variable([out_channels])
    b_conv5 = bias_variable([out_channels])
    b_conv7 = bias_variable([out_channels])

    h_conv2 = tf.nn.elu(Conv2d(inputs, W_conv2)+b_conv2)
    h_conv3 = tf.nn.elu(Conv2d(inputs, W_conv3)+b_conv3)
    h_conv5 = tf.nn.elu(Conv2d(inputs, W_conv5)+b_conv5)
    h_conv7 = tf.nn.elu(Conv2d(inputs, W_conv7)+b_conv7)

    h_max_pool2 = Max_Pool(h_conv2)
    h_max_pool3 = Max_Pool(h_conv3)
    h_max_pool5 = Max_Pool(h_conv5)
    h_max_pool7 = Max_Pool(h_conv7)

    concatenation = tf.concat(
        [h_max_pool2, h_max_pool3, h_max_pool5, h_max_pool7], axis=3)

    outputs = tf.reshape(concatenation, [-1, 38*32*64*4])

    return outputs


def CBDC_net(inputs):
    classification_inputs = feature_extraction(
        inputs, in_channels=1, out_channels=64)

    fc1 = FC(classification_inputs, 38*32*64*4, 64)
    drop1 = Dropout(fc1, 0.8)

    fc2 = FC(drop1, 64, 64)
    drop2 = Dropout(fc2, 0.8)

    fc3 = FC(drop2, 64, 64)
    drop3 = Dropout(fc3, 0.8)

    outputs = tf.nn.softmax(drop3)
    return outputs


def run_training():
    dns_normal = pd.read_csv('data/dga-agg-220907.csv')  # bazardoor,efhhilbfpejt.bazar
    dns_normal = dns_normal.values
    dns_dga = pd.read_csv('data/dns_dga_train_label.csv')
    dns_dga = dns_dga.values

    train_data = np.concatenate([dns_normal, dns_dga])
    # np.random.shuffle(train_data)

    train_x = train_data[:, 1]
    train_x = np.reshape(train_x, [train_x.shape[0], 1])
    train_y = train_data[:, 2]
    train_num = train_x.shape[0]

    # construct the computation graph
    inputs = tf.placeholder(
        tf.float32, shape=[None, 1444, 64, 1], name='inputs')
    labels = tf.placeholder(tf.int32, shape=[None], name='labels')
    learning_rate = 0.001

    logits = CBDC_net(inputs)

    # softmax_loss
    labels = tf.to_int32(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

    # evaluation
    prediction = tf.argmax(logits, axis=1, name='predcition')
    correct = tf.equal(tf.cast(prediction, tf.int32), labels, name='correct')
    eval_correct = tf.reduce_sum(
        tf.cast(correct, tf.float32), name='valuation')

    # training_operation
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    epoch = 10
    batch_size = 5000
    batch_num = train_num // batch_size if train_num % batch_size == 0 else train_num // batch_size+1

    saver = tf.train.Saver(max_to_keep=10)

    index = list(range(train_num))

    for i in range(epoch):
        loss = 0
        score = 0
        np.random.shuffle(index)

        for j in range(batch_num):
            batch_x = train_x[index[(j * batch_size) %
                                    train_num:((j+1)*batch_size) % train_num]]
            batch_y = train_y[index[(j * batch_size) %
                                    train_num:((j+1)*batch_size) % train_num]]
            batch_y = batch_y.astype('int8')

            batch_x_ngram_temp = n_gram_representation(batch_x)

            batch_x_ngram = batch_x_ngram_temp[:, :, :, np.newaxis]

            result = sess.run([train_op, loss, eval_correct], feed_dict={
                              inputs: batch_x_ngram, labels: batch_y})

            loss += result[1]
            score += result[2]
            print(i)

            # release memory
            del batch_x_ngram_temp
            del batch_x_ngram

        score /= train_num
        if i % batch_num == 0:
            print(
                'epoch {}: traing: loss - -> {:.3f}, acc --> {:.3f}%'.format(i, loss, score * 100))

        saver.save(sess, Model_Save_Path + 'ocr.model', global_step=i)

    sess.close()


if __name__ == "__main__":
    run_training()
