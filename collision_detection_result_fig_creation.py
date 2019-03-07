import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import preprocessing

# parameters
num_input = 21
num_output = 7


for i in range(10):
    path = 'test_data_set/20190226_collision_10_represent/' + str(i+1) + '/raw_data_.csv'
    # raw data
    f = open(path, 'r', encoding='utf-8')
    rdr = csv.reader(f)
    t = []
    x_data_raw = []
    y_data_raw = []

    for line in rdr:
        line = [float(i) for i in line]
        #if(line[84] == 0):
        x_data_raw.append(line[1:num_input+1])
        y_data_raw.append(line[num_input+1+7:num_input+1+7+num_output])
    t = range(len(x_data_raw))
    t = np.reshape(t,(-1,1))
    x_data_raw = np.reshape(x_data_raw, (-1, num_input))
    y_data_raw = np.reshape(y_data_raw, (-1, num_output))

    tf.reset_default_graph()
    sess = tf.Session()

    new_saver = tf.train.import_meta_graph('model/model.ckpt.meta')
    new_saver.restore(sess, 'model/model.ckpt')

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("m1/input:0")
    y = graph.get_tensor_by_name("m1/output:0")
    keep_prob = graph.get_tensor_by_name("m1/keep_prob:0")
    hypothesis = graph.get_tensor_by_name("m1/hypothesis:0")

    hypo = sess.run(hypothesis, feed_dict={x: x_data_raw, keep_prob: 1.0})

    mean_error_1 = np.mean(np.abs(y_data_raw[:,0]-hypo[:,0]))
    mean_error_2 = np.mean(np.abs(y_data_raw[:,1]-hypo[:,1]))
    mean_error_3 = np.mean(np.abs(y_data_raw[:,2]-hypo[:,2]))
    mean_error_4 = np.mean(np.abs(y_data_raw[:,3]-hypo[:,3]))
    mean_error_5 = np.mean(np.abs(y_data_raw[:,4]-hypo[:,4]))
    mean_error_6 = np.mean(np.abs(y_data_raw[:,5]-hypo[:,5]))
    mean_error_7 = np.mean(np.abs(y_data_raw[:,6]-hypo[:,6]))
    print("Accuracy : %f" % mean_error_1)

    for j in range(7):
        plt.subplot(7,1,j+1)
        plt.plot(t,y_data_raw[:,j], color='r', label='real')
        plt.plot(t,hypo[:,0], color='b', label='prediction')
        plt.xlabel('time')
        plt.ylabel('qdot')
        plt.legend()

    plt.savefig('Figure_' + str(i)+'.png')
    plt.clf()
    #plt.show()