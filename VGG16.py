import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential
import os
import read_CSV_data
import read_CSV_label
import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)

conv_layers = [

    layers.UpSampling2D(size=3),

    # unit 1
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit 2
    layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit 3
    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit 4
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit 5
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')

]


def preprocess(x, y):

    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.float32)
    return x, y


def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return pdf


x_filename = 
y_filename = 
fig_filename = 

x = read_CSV_data.CSV_data(x_filename)
y = read_CSV_label.CSV_label(y_filename)

x_test = read_CSV_data.CSV_data_test(x_filename)
y_test = read_CSV_label.CSV_label_test(y_filename)

y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test, axis=1)

print(x.shape, y.shape)
print(x_test.shape, y_test.shape)

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(1000).batch(100)

test_db = tf.data.Dataset.from_tensor_slices((x, y))
test_db = test_db.shuffle(1000).batch(1)

sample = next(iter(train_db))
sample1 = next(iter(test_db))



def main():

    global correct_num, var, mean
    conv_net = Sequential(conv_layers)

    fc_net = Sequential([
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(500, activation=tf.nn.relu),
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dense(1, activation=None),
    ])

    conv_net.build(input_shape=[None, 8, 8, 3])
    fc_net.build(input_shape=[None, 512])
    optimizer = optimizers.Adam(lr=1e-2)


    variables = conv_net.trainable_variables + fc_net.trainable_variables

    for epoch in range(100):

        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out = conv_net(x)

                out = tf.reshape(out, [-1, 512])
 
                logits = fc_net(out)

                loss = tf.losses.MSE(y, logits)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))

            print(epoch, step, 'loss:', float(loss))

        total_num = 0
        acc_list = np.zeros(100, dtype=float)
        loss_list = np.zeros(100, dtype=float)

        i = 0

        for x1, y1 in test_db:
            out1 = conv_net(x1)
            out1 = tf.reshape(out1, [-1, 512])
            prediction = fc_net(out1)

            loss = prediction/y1
            loss_list[i] = loss

            acc = 1 - abs(prediction - y1) / y1
            acc_list[i] = acc

            i += 1

        mean = np.mean(acc_list)
        std = np.std(acc_list)
        mean1 = np.mean(loss_list)
        std1 = np.std(loss_list)


        print("epoch:", epoch)
        print("mean:", mean)
        print("var:", std)


        x_1 = np.arange(-1, 6, 0.001)
        y_1 = normfun(x_1, mean1, std1)

        plt.hist(loss_list, bins=20, rwidth=0.9, normed=True)
        plt.plot(x_1, y_1)

        plt.savefig(fig_filename + str(epoch))
        plt.close()


if __name__ == '__main__':
    main()
