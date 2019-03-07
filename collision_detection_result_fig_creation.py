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
    not_collision_num = 0
    for line in rdr:
        line = [float(i) for i in line]
        if line[84] == 0:
            not_collision_num += 1
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

    mean_error_not_collision = np.zeros(7)
    mean_error_not_collision[0] = np.mean(np.abs(y_data_raw[0:not_collision_num,0]-hypo[0:not_collision_num,0]))
    mean_error_not_collision[1] = np.mean(np.abs(y_data_raw[0:not_collision_num,1]-hypo[0:not_collision_num,1]))
    mean_error_not_collision[2] = np.mean(np.abs(y_data_raw[0:not_collision_num,2]-hypo[0:not_collision_num,2]))
    mean_error_not_collision[3] = np.mean(np.abs(y_data_raw[0:not_collision_num,3]-hypo[0:not_collision_num,3]))
    mean_error_not_collision[4] = np.mean(np.abs(y_data_raw[0:not_collision_num,4]-hypo[0:not_collision_num,4]))
    mean_error_not_collision[5] = np.mean(np.abs(y_data_raw[0:not_collision_num,5]-hypo[0:not_collision_num,5]))
    mean_error_not_collision[6] = np.mean(np.abs(y_data_raw[0:not_collision_num,6]-hypo[0:not_collision_num,6]))

    mean_error_collision = np.zeros(7)
    mean_error_collision[0] = np.mean(np.abs(y_data_raw[not_collision_num:,0]-hypo[not_collision_num:,0]))
    mean_error_collision[1] = np.mean(np.abs(y_data_raw[not_collision_num:,1]-hypo[not_collision_num:,1]))
    mean_error_collision[2] = np.mean(np.abs(y_data_raw[not_collision_num:,2]-hypo[not_collision_num:,2]))
    mean_error_collision[3] = np.mean(np.abs(y_data_raw[not_collision_num:,3]-hypo[not_collision_num:,3]))
    mean_error_collision[4] = np.mean(np.abs(y_data_raw[not_collision_num:,4]-hypo[not_collision_num:,4]))
    mean_error_collision[5] = np.mean(np.abs(y_data_raw[not_collision_num:,5]-hypo[not_collision_num:,5]))
    mean_error_collision[6] = np.mean(np.abs(y_data_raw[not_collision_num:,6]-hypo[not_collision_num:,6]))
    print("Not Collision Error: %f" % np.mean(mean_error_not_collision))
    print("Collision Error: %f" % np.mean(mean_error_collision))

    for j in range(7):
        plt.subplot(7,1,j+1)
        plt.plot(t,y_data_raw[:,j], color='r', label='real')
        plt.plot(t,hypo[:,j], color='b', label='prediction')
        plt.xlabel('time')
        plt.ylabel('qdot')
        plt.legend()

    plt.savefig('Figure_' + str(i)+'.png')
    plt.clf()
    #plt.show()