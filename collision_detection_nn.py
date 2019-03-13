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
    wandb.init(project="collision_detection_CNN", tensorboard=False)

class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.X = tf.placeholder(tf.float32, shape=[None, num_time_step*num_input], name = "input")
            self.X_input = tf.reshape(self.X, [-1, num_time_step, num_input, 1])
            self.Y = tf.placeholder(tf.int64, shape=[None, num_output], name= "output")
            self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
            self.is_train = tf.placeholder(tf.bool, name="is_train")
            self.hidden_layers = 0
            self.hidden_neurons = 20

            L1 = tf.layers.conv2d(inputs= self.X_input, filters= 64, kernel_size= [3,3], padding="SAME", activation=tf.nn.relu)
            L1 = tf.nn.relu(L1)
            L1 = tf.layers.batch_normalization(L1, training=self.is_train)
            L1 = tf.layers.dropout(L1, rate=self.keep_prob, training=self.is_train)

            L2 = tf.layers.conv2d(inputs= L1, filters= 128, kernel_size= [3,3], padding="SAME", activation=tf.nn.relu)
            L2 = tf.nn.relu(L2)
            L2 = tf.layers.batch_normalization(L2, training=self.is_train)
            L2 = tf.layers.dropout(L2, rate=self.keep_prob, training=self.is_train)
            self.hidden_layers += 1

            L3 = tf.layers.conv2d(inputs= L2, filters= 256, kernel_size= [3,3], padding="SAME", activation=tf.nn.relu)
            L3 = tf.nn.relu(L3)
            L3 = tf.layers.batch_normalization(L3, training=self.is_train)
            L3 = tf.layers.dropout(L3, rate=self.keep_prob, training=self.is_train)
            self.hidden_layers += 1

            Flat = tf.reshape(L3, [-1, 256*num_time_step*num_input])
            Dense1 = tf.layers.dense(inputs=Flat, units=self.hidden_neurons, activation=tf.nn.relu)
            Dense1 = tf.layers.batch_normalization(Dense1, training=self.is_train)
            self.hidden_layers += 1

            Dense2 = tf.layers.dense(inputs=Dense1, units=self.hidden_neurons, activation=tf.nn.relu)
            Dense2 = tf.layers.batch_normalization(Dense2, training=self.is_train)
            self.hidden_layers += 1
            
            self.logits = tf.layers.dense(inputs=Dense2, units=num_output)
            self.hypothesis = tf.nn.softmax(self.logits)
            self.hypothesis = tf.identity(self.hypothesis, "hypothesis")

            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.Y))
            self.optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate).minimize(self.cost)
        
        self.prediction = tf.argmax(self.hypothesis, 1)
        self.correct_prediction = tf.equal(self.prediction, tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def get_mean_error_hypothesis(self, x_test, y_test, keep_prop=1.0, is_train=False):
        return self.sess.run([self.accuracy, self.hypothesis, self.X, self.Y], feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: keep_prop, self.is_train: is_train})

    def train(self, x_data, y_data, keep_prop=1.0, is_train=True):
        return self.sess.run([self.accuracy, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.keep_prob: keep_prop, self.is_train: is_train})

    def next_batch(self, num, data):
        stack = []
        x_batch = []
        y_batch = []
        i = 0
        for line in data:
            line = [float(i) for i in line]
            stack.append(line[1:num_input+1])
            if i >= num_time_step-1:
                y_batch.append(line[-num_output:])
            i = i+1

            if i == num:
                break

        for j in range(num - num_time_step +1):
            for k in range(num_time_step):
                x_batch.append(stack[j+k:j+k+1])
        
        return [np.asarray(np.reshape(x_batch, (-1, num_time_step*num_input))), np.asarray(np.reshape(y_batch,(-1,num_output)))]

    def get_hidden_number(self):
        return [self.hidden_layers, self.hidden_neurons]

# input/output number
num_input = 28
num_output = 2
output_idx = 6
num_time_step = 5

# parameters
learning_rate = 0.000010 #0.000001
training_epochs = 1000
batch_size = 100
total_batch = 1800
drop_out = 1.0
regul_factor = 0.00000

# loading testing data
f_test = open('testing_data_.csv', 'r', encoding='utf-8')
rdr_test = csv.reader(f_test)

stack_test = []
x_data_test = []
y_data_test = []
i_test = 0
for line in rdr_test:
    line = [float(i) for i in line]
    stack_test.append(line[1:num_input+1])
    if i_test >= num_time_step-1:
        y_data_test.append(line[-num_output:])
    i_test = i_test+1

for j in range(i_test - num_time_step +1):
    for k in range(num_time_step):
        x_data_test.append(stack_test[j+k:j+k+1])

x_data_test = np.reshape(x_data_test, (-1, num_time_step*num_input))
y_data_test = np.reshape(y_data_test, (-1, num_output))


# load validation data
f_val = open('validation_data_.csv', 'r', encoding='utf-8')
rdr_val = csv.reader(f_val)

stack_val = []
x_data_val = []
y_data_val = []
i_val = 0
for line in rdr_val:
    line = [float(i) for i in line]
    stack_val.append(line[1:num_input+1])
    if i_val >= num_time_step-1:
        y_data_val.append(line[-num_output:])
    i_val = i_val+1

for j in range(i_val - num_time_step +1):
    for k in range(num_time_step):
        x_data_val.append(stack_val[j+k:j+k+1])

x_data_val = np.reshape(x_data_val, (-1, num_time_step*num_input))
y_data_val = np.reshape(y_data_val, (-1, num_output))


# initialize
sess = tf.Session()
m1 = Model(sess, "m1")
sess.run(tf.global_variables_initializer())


if wandb_use == True:
    wandb.config.epoch = training_epochs
    wandb.config.batch_size = batch_size
    wandb.config.learning_rate = learning_rate
    wandb.config.drop_out = drop_out
    wandb.config.num_input = num_input
    wandb.config.num_output = num_output
    wandb.config.total_batch = total_batch
    wandb.config.activation_function = "Sigmoid"
    wandb.config.training_episode = 1200
    wandb.config.hidden_layers, wandb.config.hidden_neurons = m1.get_hidden_number()
    wandb.config.L2_regularization = regul_factor 

# train my model
train_mse = np.zeros(training_epochs)
validation_mse = np.zeros(training_epochs)

for epoch in range(training_epochs):
    accu_train = 0
    f = open('training_data_.csv', 'r', encoding='utf-8')
    rdr = csv.reader(f)

    for i in range(total_batch):
        batch_xs, batch_ys = m1.next_batch(batch_size, rdr)
        c, _ = m1.train(batch_xs, batch_ys, drop_out)
        accu_train += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1))
    print('Train Accuracy =', '{:.9f}'.format(accu_train))

    [accu_val, hypo, x_val, y_val] = m1.get_mean_error_hypothesis(x_data_val, y_data_val)
    print('Validation Accuracy:', '{:.9f}'.format(accu_val))

    train_mse[epoch] = accu_train
    validation_mse[epoch] = accu_val

    if wandb_use == True:
        wandb.log({'training Accuracy': accu_train, 'validation Accuracy': accu_val})

        if epoch % 20 ==0:
            for var in tf.trainable_variables():
                name = var.name
                wandb.log({name: sess.run(var)})


print('Learning Finished!')
[accu_test, hypo, x_test, y_test] = m1.get_mean_error_hypothesis(x_data_test, y_data_test)
# print('Error: ', error,"\n x_data: ", x_test,"\nHypothesis: ", hypo, "\n y_data: ", y_test)
print('Test Accuracy: ', accu_test)

elapsed_time = time.time() - start_time
print(elapsed_time)

saver = tf.train.Saver()
saver.save(sess,'model/model.ckpt')

if wandb_use == True:
    saver.save(sess, os.path.join(wandb.run.dir, 'model/model.ckpt'))
    wandb.config.elapsed_time = elapsed_time

epoch = np.arange(training_epochs)
plt.plot(epoch, train_mse, 'r', label='train')
plt.plot(epoch, validation_mse, 'b', label='validation')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('abs error')
plt.show()
