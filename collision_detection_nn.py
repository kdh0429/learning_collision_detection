import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import preprocessing
import time
import wandb
import os

wandb_use = True
start_time = time.time()
if wandb_use == True:
    wandb.init(project="collision_detection", tensorboard=False)

class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.X = tf.placeholder(tf.float32, shape=[None, num_input], name = "input")
            self.Y = tf.placeholder(tf.int64, shape=[None, num_output], name= "output")

            # dropout (keep_prob) rate  0.7 on training, but should be 1 for testing
            self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

            # weights & bias for nn layers
            # http://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
            W1 = tf.get_variable("W1", shape=[num_input, 40], initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.random_normal([40]))
            L1 = tf.matmul(self.X, W1) +b1
            #L1 = tf.nn.relu(L1)
            L1 = tf.nn.sigmoid(tf.matmul(self.X, W1) + b1)
            L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob)

            W2 = tf.get_variable("W2", shape=[40, 40], initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.random_normal([40]))
            L2 = tf.matmul(L1, W2) +b2
            #L2 = tf.nn.relu(L2)
            L2 = tf.nn.sigmoid(tf.matmul(L1, W2) + b2)
            L2 = tf.nn.dropout(L2, keep_prob=self.keep_prob)

            # W3 = tf.get_variable("W3", shape=[80, 80], initializer=tf.contrib.layers.xavier_initializer())
            # b3 = tf.Variable(tf.random_normal([80]))
            # L3 = tf.matmul(L2, W3) +b3
            # L3 = tf.nn.relu(L3)
            # #L3 = tf.nn.relu(tf.sigmoid(L2, W3) + b3)
            # L3 = tf.nn.dropout(L3, keep_prob=self.keep_prob)

            # W4 = tf.get_variable("W4", shape=[80, 80], initializer=tf.contrib.layers.xavier_initializer())
            # b4 = tf.Variable(tf.random_normal([80]))
            # L4 = tf.matmul(L3, W4) +b4
            # L4 = tf.nn.relu(L4)
            # #L4 = tf.nn.relu(tf.sigmoid(L3, W4) + b4)
            # L4 = tf.nn.dropout(L4, keep_prob=self.keep_prob)

            # W5 = tf.get_variable("W5", shape=[80, 80], initializer=tf.contrib.layers.xavier_initializer())
            # b5 = tf.Variable(tf.random_normal([80]))
            # L5 = tf.matmul(L4, W5) +b5
            # L5 = tf.nn.relu(L5)
            # #L3 = tf.nn.relu(tf.sigmoid(L2, W3) + b3)
            # L5 = tf.nn.dropout(L5, keep_prob=self.keep_prob)

            # W6 = tf.get_variable("W6", shape=[80, 80], initializer=tf.contrib.layers.xavier_initializer())
            # b6 = tf.Variable(tf.random_normal([80]))
            # L6 = tf.matmul(L5, W6) +b6
            # L6 = tf.nn.relu(L6)
            # #L4 = tf.nn.relu(tf.sigmoid(L3, W4) + b4)
            # L6 = tf.nn.dropout(L6, keep_prob=self.keep_prob)

            # W7 = tf.get_variable("W7", shape=[80, 80], initializer=tf.contrib.layers.xavier_initializer())
            # b7 = tf.Variable(tf.random_normal([80]))
            # L7 = tf.matmul(L6, W7) +b7
            # L7 = tf.nn.relu(L7)
            # #L4 = tf.nn.relu(tf.sigmoid(L3, W4) + b4)
            # L7 = tf.nn.dropout(L7, keep_prob=self.keep_prob)

            # W8 = tf.get_variable("W8", shape=[80, 80], initializer=tf.contrib.layers.xavier_initializer())
            # b8 = tf.Variable(tf.random_normal([80]))
            # L8 = tf.matmul(L7, W8) +b8
            # L8 = tf.nn.relu(L8)
            # #L4 = tf.nn.relu(tf.sigmoid(L3, W4) + b4)
            # L8 = tf.nn.dropout(L8, keep_prob=self.keep_prob)

            W9 = tf.get_variable("W9", shape=[40, num_output], initializer=tf.contrib.layers.xavier_initializer())
            b9 = tf.Variable(tf.random_normal([num_output]))
            self.logits = tf.matmul(L2, W9) + b9
            self.hypothesis = tf.nn.softmax(self.logits)
            self.hypothesis = tf.identity(self.hypothesis, "hypothesis")

            # define cost/loss & optimizer
            self.l2_reg = tf.nn.l2_loss(W1)# + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(W4) + tf.nn.l2_loss(W5) + tf.nn.l2_loss(W6) + tf.nn.l2_loss(W7)
            self.l2_reg = regul_factor* self.l2_reg
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.Y))
            #self.cost = tf.reduce_mean(tf.reduce_mean(tf.square(self.hypothesis - self.Y)))
            self.optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate).minimize(self.cost + self.l2_reg)
        
        self.prediction = tf.argmax(self.hypothesis, 1)
        self.correct_prediction = tf.equal(self.prediction, tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def get_mean_error_hypothesis(self, x_test, y_test, keep_prop=1.0):
        return self.sess.run([self.accuracy, self.hypothesis, self.X, self.Y, self.l2_reg], feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: keep_prop})

    def train(self, x_data, y_data, keep_prop=1.0):
        return self.sess.run([self.accuracy, self.l2_reg, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.keep_prob: keep_prop})

    def next_batch(self, num, data):
        x_batch = []
        y_batch = []
        i = 0
        for line in data:
            line = [float(i) for i in line]
            #x_batch.append(line[1:num_input+1])
            x_batch.append(line[36:43])
            y_batch.append(line[-num_output:])
            #y_batch.append(line[-output_idx])
            i = i+1

            if i == num:
                break
        return [np.asarray(np.reshape(x_batch, (-1, num_input))), np.asarray(np.reshape(y_batch,(-1,num_output)))]
# input/output number
num_input = 7
num_output = 2
output_idx = 6
# loading testing data
f_test = open('testing_data_.csv', 'r', encoding='utf-8')
rdr_test = csv.reader(f_test)
t = []
x_data_test = []
y_data_test = []

for line in rdr_test:
    line = [float(i) for i in line]
    t.append(line[0])
    #x_data_test.append(line[1:num_input+1])
    x_data_test.append(line[36:43])
    y_data_test.append(line[-num_output:])
    #y_data_test.append(line[-output_idx])

t = np.reshape(t,(-1,1))
x_data_test = np.reshape(x_data_test, (-1, num_input))
#x_data_test = preprocessing.scale(x_data_test)
y_data_test = np.reshape(y_data_test, (-1, num_output))

# load validation data
f_val = open('validation_data_.csv', 'r', encoding='utf-8')
rdr_val = csv.reader(f_val)
x_data_val = []
y_data_val = []
for line in rdr_val:
    line = [float(i) for i in line]
    #x_data_val.append(line[1:num_input+1])
    x_data_val.append(line[36:43])
    y_data_val.append(line[-num_output:])
    #y_data_val.append(line[-output_idx])
x_data_val = np.reshape(x_data_val, (-1, num_input))
#x_data_val = preprocessing.scale(x_data_val)
y_data_val = np.reshape(y_data_val, (-1, num_output))

# parameters
learning_rate = 0.000001 #0.000001
training_epochs = 100
batch_size = 100
total_batch = 1800#int(np.shape(x_data_test)[0]/batch_size*4)
drop_out = 1.0
regul_factor = 0.00000

if wandb_use == True:
    wandb.config.epoch = training_epochs
    wandb.config.batch_size = batch_size
    wandb.config.learning_rate = learning_rate
    wandb.config.drop_out = drop_out
    wandb.config.num_input = num_input
    wandb.config.num_output = num_output
    wandb.config.total_batch = total_batch
    wandb.config.activation_function = "ReLU"
    wandb.config.training_episode = 1200
    wandb.config.hidden_layers = 5
    wandb.config.hidden_neurons = 40
    wandb.config.L2_regularization = regul_factor
    

# initialize
sess = tf.Session()
m1 = Model(sess, "m1")
sess.run(tf.global_variables_initializer())

train_mse = np.zeros(training_epochs)
validation_mse = np.zeros(training_epochs)

# train my model
for epoch in range(training_epochs):
    accu_train = 0
    avg_reg_cost = 0
    f = open('training_data_.csv', 'r', encoding='utf-8')
    rdr = csv.reader(f)

    for i in range(total_batch):
        batch_xs, batch_ys = m1.next_batch(batch_size, rdr)
        #batch_xs = preprocessing.scale(batch_xs)
        c, reg_c,_ = m1.train(batch_xs, batch_ys, drop_out)
        accu_train += c / total_batch
        avg_reg_cost += reg_c / total_batch

    print('Epoch:', '%04d' % (epoch + 1))
    print('Train Accuracy =', '{:.9f}'.format(accu_train), 'Train l2 reg cost =', '{:.9f}'.format(avg_reg_cost))

    [accu_val, hypo, x_val, y_val, l2_reg_val] = m1.get_mean_error_hypothesis(x_data_val, y_data_val)
    print('Validation Accuracy:', '{:.9f}'.format(accu_val), 'Validation l2 regularization:', '{:.9f}'.format(l2_reg_val))
    

    train_mse[epoch] = accu_train
    validation_mse[epoch] = accu_val

    if wandb_use == True:
        wandb.log({'training Accuracy': accu_train, 'validation Accuracy': accu_val, 'validation l2_reg': l2_reg_val})

        if epoch % 20 ==0:
            for var in tf.trainable_variables():
                name = var.name
                wandb.log({name: sess.run(var)})


print('Learning Finished!')


[accu_test, hypo, x_test, y_test, l2_reg_test] = m1.get_mean_error_hypothesis(x_data_test, y_data_test)
# print('Error: ', error,"\n x_data: ", x_test,"\nHypothesis: ", hypo, "\n y_data: ", y_test)
print('Test Accuracy: ', accu_test)
print('Test l2 regularization:', l2_reg_test)

elapsed_time = time.time() - start_time
print(elapsed_time)

saver = tf.train.Saver()
saver.save(sess,'model/model.ckpt')

if wandb_use == True:
    wandb.save(os.path.join(wandb.run.dir, 'model/model.ckpt'))
    wandb.config.elapsed_time = elapsed_time

epoch = np.arange(training_epochs)
plt.plot(epoch, train_mse, 'r', label='train')
plt.plot(epoch, validation_mse, 'b', label='validation')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('abs error')
plt.show()
