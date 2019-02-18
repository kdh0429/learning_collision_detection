import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import preprocessing

# parameters
num_input = 28
num_output = 6
output_idx = 6


for i in range(10):
    path = 'test_data_set/20190211_10_represent/' + str(i+1) + '/raw_data_.csv'
    # raw data
    f = open(path, 'r', encoding='utf-8')
    rdr = csv.reader(f)
    t = []
    x_data_raw = []
    y_data_raw = []

    for line in rdr:
        line = [float(i) for i in line]
        t.append(line[0])
        x_data_raw.append(line[1:num_input+1])
        y_data_raw.append(line[-num_output:])
        #y_data_raw.append(line[-output_idx])

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

    # mean_error = graph.get_tensor_by_name("mean_error:0")
    # for op in graph.get_operations():
    #     print(op.name)

    # mean_error_test = sess.run(mean_error, feed_dict={x: x_data_raw, y: y_data_raw, keep_prob: 1.0})
    # print(mean_error_test)
    # print(np.mean(np.abs(y_data_raw - hypo)))

    res_idx = 0
    mean_error_x = np.mean(np.abs(y_data_raw[:,0]-hypo[:,0]))
    mean_error_y = np.mean(np.abs(y_data_raw[:,1]-hypo[:,1]))
    mean_error_z = np.mean(np.abs(y_data_raw[:,2]-hypo[:,2]))

    print("X Mean error : %f" % mean_error_x)
    print("Y Mean error : %f" % mean_error_y)
    print("Z Mean error : %f" % mean_error_z)

    max_error_x = np.max(np.abs(y_data_raw[:,0]-hypo[:,0]))
    max_error_y = np.max(np.abs(y_data_raw[:,1]-hypo[:,1]))
    max_error_z = np.max(np.abs(y_data_raw[:,2]-hypo[:,2]))

    print("X Max error : %f" % max_error_x)
    print("Y Max error : %f" % max_error_y)
    print("Z Max error : %f" % max_error_z)

    max_error_x_idx = np.argmax(np.abs(y_data_raw[:,0]-hypo[:,0]))
    max_error_y_idx = np.argmax(np.abs(y_data_raw[:,1]-hypo[:,1]))
    max_error_z_idx = np.argmax(np.abs(y_data_raw[:,2]-hypo[:,2]))

    print("X Max error time : %f" % t[max_error_x_idx])
    print("Y Max error time : %f" % t[max_error_y_idx])
    print("Z Max error time : %f" % t[max_error_z_idx])
    
    

    # plt.plot(t,y_data_raw[:,res_idx], 'r', label='real')
    # plt.plot(t,hypo[:,res_idx], 'b', label='prediction')
    # plt.xlabel('time')
    # plt.ylabel('Fx')
    # plt.legend()
    # plt.show()


    plt.subplot(3,1,1)
    plt.plot(t,y_data_raw[:,res_idx], 'r', label='real')
    plt.plot(t,hypo[:,res_idx], 'b', label='prediction')
    plt.xlabel('time')
    plt.ylabel('Fx')
    plt.legend()
    plt.subplot(3,1,2)
    plt.plot(t,y_data_raw[:,1], 'r', label='real')
    plt.plot(t,hypo[:,1], 'b', label='prediction')
    plt.xlabel('time')
    plt.ylabel('Fy')
    plt.legend()
    plt.subplot(3,1,3)
    plt.plot(t,y_data_raw[:,2], 'r', label='real')
    plt.plot(t,hypo[:,2], 'b', label='prediction')
    plt.xlabel('time')
    plt.ylabel('Fz')
    plt.legend()
    plt.savefig('Figure_' + str(i)+'.png')
    plt.clf()
    #plt.show()