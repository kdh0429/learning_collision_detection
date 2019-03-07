import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import preprocessing

# parameters
num_input_dyna = 28
num_input_cont = 14
num_output = 2


for i in range(10):
    path = 'test_data_set/20190226_collision_10_represent/' + str(i+1) + '/raw_data_.csv'
    # raw data
    f = open(path, 'r', encoding='utf-8')
    rdr = csv.reader(f)
    t = []
    x_dyna_data_raw = []
    x_cont_data_raw = []
    y_data_raw = []

    for line in rdr:
        line = [float(i) for i in line]
        x_dyna_data_raw.append(line[1:num_input_dyna+1])
        x_cont_data_raw.append(line[num_input_dyna+1:num_input_dyna+num_input_cont+1])
        y_data_raw.append(line[-num_output:])
    t = range(len(y_data_raw))
    t = np.reshape(t,(-1,1))
    x_dyna_data_raw = np.reshape(x_dyna_data_raw, (-1, num_input_dyna))
    x_cont_data_raw = np.reshape(x_cont_data_raw, (-1, num_input_cont))
    y_data_raw = np.reshape(y_data_raw, (-1, num_output))

    tf.reset_default_graph()
    sess = tf.Session()

    new_saver = tf.train.import_meta_graph('model/model.ckpt.meta')
    new_saver.restore(sess, 'model/model.ckpt')

    graph = tf.get_default_graph()
    x_dyna = graph.get_tensor_by_name("m1/input_dyna:0")
    x_cont = graph.get_tensor_by_name("m1/input_cont:0")
    y = graph.get_tensor_by_name("m1/output:0")
    keep_prob = graph.get_tensor_by_name("m1/keep_prob:0")
    hypothesis = graph.get_tensor_by_name("m1/hypothesis:0")

    hypo = sess.run(hypothesis, feed_dict={x_dyna: x_dyna_data_raw, x_cont : x_cont_data_raw, keep_prob: 1.0})

    prediction = np.argmax(hypo, 1)
    correct_prediction = np.equal(prediction, np.argmax(y_data_raw, 1))
    accuracy = np.mean(correct_prediction)

    print("Accuracy : %f" % accuracy)


    plt.plot(t,y_data_raw[:,0], color='r', marker="o", label='real')
    plt.plot(t,hypo[:,0], color='b',marker="x", label='prediction')
    plt.xlabel('time')
    plt.ylabel('Collision Probability')
    plt.legend()
    plt.savefig('Figure_' + str(i)+'.png')
    plt.clf()