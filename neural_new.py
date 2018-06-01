import cv2

import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
from math import ceil, sqrt

TRAIN_DIR = 'C:\\Users\\leonardo.zanella\\Documents\\machine_learning_sentdex\\train'
TEST_DIR = 'C:\\Users\\leonardo.zanella\\Documents\\machine_learning_sentdex\\test'
IMG_SIZE = 128
LR = 1e-3

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.1
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100000, 0.96, staircase=True)

MODEL_NAME = 'dogvscats-{}-{}.model'.format(LR, '2conv-128-64-32')


def label_imb(img):
    world_label = img.split('.')[-3]
    if world_label == 'cat':
        return [1, 0]
    elif world_label == 'dog':
        return [0, 1]


def create_train_data():
    train_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_imb(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        train_data.append([np.array(img), np.array(label)])
    shuffle(train_data)
    np.save('train_data.npy', train_data)
    return train_data


def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])

    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data


def get_train_data(train):
    if train == 'train':
        train_data = create_train_data()
    else:
        train_data = np.load('train_data.npy')
    return train_data


def get_test_data(train):
    if train == 'train':
        test_data = process_test_data()
    else:
        test_data = np.load('test_data.npy')
    return test_data


def CNN(load):
    tf.reset_default_graph()

    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

    # RELU function e sigmoid softmax
    qtd_filters = 64
    window_size = 10
    convnet = conv_2d(convnet, qtd_filters, window_size, activation='relu')
    # escolhe o maior numa janela de 9

    convnet = max_pool_2d(convnet, 12)

    convnet = conv_2d(convnet, 32, 10, activation='relu')
    convnet = max_pool_2d(convnet, 10)

    convnet = conv_2d(convnet, 16, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    # convnet = conv_2d(convnet, 8, 2, activation='relu')
    # convnet = max_pool_2d(convnet, 2)

    # convnet = conv_2d(convnet, 32, 5, activation='relu')
    # convnet = max_pool_2d(convnet, 5)

    # convnet = conv_2d(convnet, 64, 5, activation='relu')
    # convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 16, 5, activation='relu')

    number_neuron = 512

    convnet = fully_connected(convnet, number_neuron, activation='relu')
    # numero de neuronios que vai sair(aleatoriamente)
    convnet = dropout(convnet, 0.74)

    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')
    if load:
        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('model Loaded')

    return model


def fit_model(model, train_data):
    train = train_data[:-5000]
    test = train_data[-5000:]

    X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    Y = [i[1] for i in train]

    test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    test_y = [i[1] for i in test]
    model.fit({'input': X}, {'targets': Y}, n_epoch=25, validation_set=({'input': test_x}, {'targets': test_y}),
              snapshot_step=1000, show_metric=True, run_id=MODEL_NAME)

    model.save(MODEL_NAME)


def predict_model(model, test_data, dimension, num_fig):
    for _ in range(num_fig):
        fig = plt.figure()
        total_elements = dimension ** 2
        from random import randint

        # images = test_data[:total_elements]
        # for num, data in enumerate(test_data[:total_elements]):
        # img_num = data[1]
        # img_data = data[0]
        # dimension = int(ceil(sqrt(total_elements)))

        for i in range(dimension):
            for j in range(dimension):
                coord = (i * dimension + j) + 1
                y = fig.add_subplot(dimension, dimension, (i * dimension + j) + 1)
                orig = test_data[randint(0, len(test_data) - 1)][0]
                data = test_data[randint(0, len(test_data) - 1)][0].reshape(IMG_SIZE, IMG_SIZE, 1)
                model_out = model.predict([data])[0]
                if np.argmax(model_out) == 1:
                    str_label = 'Dog'
                else:
                    str_label = 'Cat'

                y.imshow(orig, cmap='gray')
                plt.title(str_label)
                y.axes.get_xaxis().set_visible(False)
                y.axes.get_yaxis().set_visible(False)
        plt.show()


def predict_my_image(model, path):
    # fig = plt.figure()
    img_orig = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.array(img)
    data = img.reshape(IMG_SIZE, IMG_SIZE, 1)
    model_out = model.predict([data])[0]
    if np.argmax(model_out) == 1:
        str_label = 'Dog'
    else:
        str_label = 'Cat'
    # y = fig.add_subplot(3, 4, 1)
    plt.imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB), cmap='hsv')
    plt.title(str_label)
    # y.axes.get_xaxis().set_visible(False)
    # y.axes.get_yaxis().set_visible(False)
    plt.show()


def csv_to_submit():
    with open('submission.csv', 'w') as f:
        f.write('id,label\n')
    with open('submission.csv', 'a') as f:
        for data in tqdm(test_data):
            img_num = data[1]
            img_data = data[0]
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
            model_out = model.predict([data])[0]
            f.write('{},{}\n'.format(img_num, model_out[1]))


##################################################

train_data = get_train_data('load')

test_data = get_test_data('load')

model = CNN(True)

# fit_model(model, train_data)

a = input('tamanho:')
b = input('times:')
# for _ in b:
predict_model(model, test_data, int(a), int(b))

# predict_my_image(model,'D:\\joao.jpg')
# csv_to_submit()







