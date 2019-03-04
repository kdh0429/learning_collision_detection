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
    wandb.init(project="collision_detection_dyna_cont_detached", tensorboard=False)

class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.X_dyna = tf.placeholder(tf.float32, shape=[None, num_input_dyna], name = "input_dyna")
            self.X_cont = tf.placeholder(tf.float32, shape=[None, num_input_cont], name = "input_cont")
            self.Y = tf.placeholder(tf.int64, shape=[None, num_output], name= "output")
            self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

            self.hidden_layers_dyna = 0
            self.hidden_layers_cont = 0
            self.hidden_neurons = 40

            # weights & bias for nn layers
            # http://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
            W1_dyna = tf.get_variable("W1_dyna", shape=[num_input_dyna, self.hidden_neurons], initializer=tf.contrib.layers.xavier_initializer())
            b1_dyna = tf.Variable(tf.random_normal([self.hidden_neurons]))
            L1_dyna = tf.matmul(self.X_dyna, W1_dyna) + b1_dyna
            L1_dyna = tf.nn.sigmoid(L1_dyna)
            L1_dyna = tf.nn.dropout(L1_dyna, keep_prob=self.keep_prob)

            W2_dyna = tf.get_variable("W2_dyna", shape=[self.hidden_neurons, self.hidden_neurons], initializer=tf.contrib.layers.xavier_initializer())
            b2_dyna = tf.Variable(tf.random_normal([self.hidden_neurons]))
            L2_dyna = tf.matmul(L1_dyna, W2_dyna) + b2_dyna
            L2_dyna = tf.nn.sigmoid(L2_dyna)
            L2_dyna = tf.nn.dropout(L2_dyna, keep_prob=self.keep_prob)
            self.hidden_layers_dyna += 1

            W3_dyna = tf.get_variable("W3_dyna", shape=[self.hidden_neurons, self.hidden_neurons], initializer=tf.contrib.layers.xavier_initializer())
            b3_dyna = tf.Variable(tf.random_normal([self.hidden_neurons]))
            L3_dyna = tf.matmul(L2_dyna, W3_dyna) + b3_dyna
            L3_dyna = tf.nn.sigmoid(L3_dyna)
            L3_dyna = tf.nn.dropout(L3_dyna, keep_prob=self.keep_prob)
            self.hidden_layers_dyna += 1

            W4_dyna = tf.get_variable("W4_dyna", shape=[self.hidden_neurons, self.hidden_neurons], initializer=tf.contrib.layers.xavier_initializer())
            b4_dyna = tf.Variable(tf.random_normal([self.hidden_neurons]))
            L4_dyna = tf.matmul(L3_dyna, W4_dyna) +b4_dyna
            L4_dyna = tf.nn.sigmoid(L4_dyna)
            L4_dyna = tf.nn.dropout(L4_dyna, keep_prob=self.keep_prob)
            self.hidden_layers_dyna += 1

            W5_dyna = tf.get_variable("W5_dyna", shape=[self.hidden_neurons, 1], initializer=tf.contrib.layers.xavier_initializer())
            b5_dyna = tf.Variable(tf.random_normal([self.hidden_neurons]))
            L5_dyna = tf.matmul(L4_dyna, W5_dyna) +b5_dyna
            L5_dyna = tf.nn.sigmoid(L5_dyna)
            L5_dyna = tf.nn.dropout(L5_dyna, keep_prob=self.keep_prob)
            self.hidden_layers_dyna += 1

            W1_cont = tf.get_variable("W1_cont", shape=[num_input_cont, self.hidden_neurons], initializer=tf.contrib.layers.xavier_initializer())
            b1_cont = tf.Variable(tf.random_normal([self.hidden_neurons]))
            L1_cont = tf.matmul(self.X_cont, W1_cont) +b1_cont
            L1_cont = tf.nn.sigmoid(L1_cont)
            L1_cont = tf.nn.dropout(L1_cont, keep_prob=self.keep_prob)

            W2_cont = tf.get_variable("W2_cont", shape=[self.hidden_neurons, self.hidden_neurons], initializer=tf.contrib.layers.xavier_initializer())
            b2_cont = tf.Variable(tf.random_normal([self.hidden_neurons]))
            L2_cont = tf.matmul(L1_cont, W2_cont) +b2_cont
            L2_cont = tf.nn.sigmoid(L2_cont)
            L2_cont = tf.nn.dropout(L2_cont, keep_prob=self.keep_prob)
            self.hidden_layers_cont += 1

            W3_cont = tf.get_variable("W3_cont", shape=[self.hidden_neurons, 1], initializer=tf.contrib.layers.xavier_initializer())
            b3_cont = tf.Variable(tf.random_normal([self.hidden_neurons]))
            L3_cont = tf.matmul(L2_cont, W3_cont) +b3_cont
            L3_cont = tf.nn.sigmoid(L3_cont)
            L3_cont = tf.nn.dropout(L3_cont, keep_prob=self.keep_prob)
            self.hidden_layers_cont += 1

            L_last = tf.stack([L5_dyna, L3_cont])
            W_last = tf.get_variable("W_last", shape=[2, num_output], initializer=tf.contrib.layers.xavier_initializer())
            b_last = tf.Variable(tf.random_normal([num_output]))
            self.logits = tf.matmul(L_last, W_last) + b_last
            self.hypothesis = tf.nn.softmax(self.logits)
            self.hypothesis = tf.identity(self.hypothesis, "hypothesis")

            # define cost/loss & optimizer
            self.l2_reg = 0 #tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(W4) + tf.nn.l2_loss(W5) + tf.nn.l2_loss(W6) + tf.nn.l2_loss(W7)
            self.l2_reg = regul_factor* self.l2_reg
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.Y))
            #self.cost = tf.reduce_mean(tf.reduce_mean(tf.square(self.hypothesis - self.Y)))
            self.optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate).minimize(self.cost + self.l2_reg)
        
        self.prediction = tf.argmax(self.hypothesis, 1)
        self.correct_prediction = tf.equal(self.prediction, tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def get_mean_error_hypothesis(self, x_dyna_test, x_cont_test, y_test, keep_prop=1.0):
        return self.sess.run([self.accuracy, self.hypothesis, self.X_dyna, self.X_cont, self.Y, self.l2_reg], feed_dict={self.X_dyna: x_dyna_test, self.X_cont: x_cont_test, self.Y: y_test, self.keep_prob: keep_prop})

    def train(self, x_dyna_data, x_cont_data, y_data, keep_prop=1.0):
        return self.sess.run([self.accuracy, self.l2_reg, self.optimizer], feed_dict={
            self.X_dyna: x_dyna_data, self.X_cont: x_cont_data, self.Y: y_data, self.keep_prob: keep_prop})

    def next_batch(self, num, data):
        x_dyna_batch = []
        x_cont_batch = []
        y_batch = []
        i = 0
        for line in data:
            line = [float(i) for i in line]
            x_dyna_batch.append(line[1:num_input_dyna+1])
            x_cont_batch.append(line[num_input_dyna+1:num_input_dyna+num_input_cont+1])
            y_batch.append(line[-num_output:])
            i = i+1

            if i == num:
                break
        return [np.asarray(np.reshape(x_dyna_batch, (-1, x_dyna_batch))), np.asarray(np.reshape(x_cont_batch, (-1, num_input_cont))), np.asarray(np.reshape(y_batch,(-1,num_output)))]
    def get_hidden_number(self):
        return [self.hidden_layers_dyna, self.hidden_layers_cont, self.hidden_neurons]

# input/output number
num_input_dyna = 28
num_input_cont = 14
num_output = 2

# parameters
learning_rate = 0.000010 #0.000001
training_epochs = 3000
batch_size = 100
total_batch = 1800
drop_out = 1.0
regul_factor = 0.00000

# loading testing data
f_test = open('testing_data_.csv', 'r', encoding='utf-8')
rdr_test = csv.reader(f_test)
x_dyna_data_test = []
x_cont_data_test = []
y_data_test = []

for line in rdr_test:
    line = [float(i) for i in line]
    x_dyna_data_test.append(line[1:num_input_dyna+1])
    x_cont_data_test.append(line[num_input_dyna+1:num_input_dyna+num_input_cont+1])
    y_data_test.append(line[-num_output:])

x_dyna_data_test = np.reshape(x_dyna_data_test, (-1, num_input_dyna))
x_cont_data_test = np.reshape(x_cont_data_test, (-1, num_input_cont))
y_data_test = np.reshape(y_data_test, (-1, num_output))

# load validation data
f_val = open('validation_data_.csv', 'r', encoding='utf-8')
rdr_val = csv.reader(f_val)
x_dyna_data_val = []
x_cont_data_val = []
y_data_val = []
for line in rdr_val:
    line = [float(i) for i in line]
    x_dyna_data_val.append(line[1:num_input_dyna+1])
    x_cont_data_val.append(line[num_input_dyna+1:num_input_dyna+num_input_cont+1])
    y_data_val.append(line[-num_output:])
x_dyna_data_val = np.reshape(x_dyna_data_val, (-1, num_input_dyna))
x_cont_data_val = np.reshape(x_cont_data_val, (-1, num_input_cont))
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
    wandb.config.num_input_dyna = num_input_dyna
    wandb.config.num_input_dyna = num_input_cont
    wandb.config.num_output = num_output
    wandb.config.total_batch = total_batch
    wandb.config.activation_function = "Sigmoid"
    wandb.config.training_episode = 1200
    wandb.config.hidden_layers_dyna, wandb.config.hidden_layers_cont, wandb.config.hidden_neurons = m1.get_hidden_number()
    wandb.config.L2_regularization = regul_factor 

# train my model
train_mse = np.zeros(training_epochs)
validation_mse = np.zeros(training_epochs)

for epoch in range(training_epochs):
    accu_train = 0
    avg_reg_cost = 0
    f = open('training_data_.csv', 'r', encoding='utf-8')
    rdr = csv.reader(f)

    for i in range(total_batch):
        batch_xs_dyna, batch_xs_cont, batch_ys = m1.next_batch(batch_size, rdr)
        c, reg_c,_ = m1.train(batch_xs_dyna, batch_xs_cont, batch_ys, drop_out)
        accu_train += c / total_batch
        avg_reg_cost += reg_c / total_batch

    print('Epoch:', '%04d' % (epoch + 1))
    print('Train Accuracy =', '{:.9f}'.format(accu_train), 'Train l2 reg cost =', '{:.9f}'.format(avg_reg_cost))

    [accu_val, hypo, x_dyna_val, x_cont_val, y_val, l2_reg_val] = m1.get_mean_error_hypothesis(x_dyna_data_val,x_cont_data_val, y_data_val)
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
[accu_test, hypo, x_dyna_test, x_cont_test, y_test, l2_reg_test] = m1.get_mean_error_hypothesis(x_dyna_data_test, x_cont_data_test, y_data_test)
# print('Error: ', error,"\n x_data: ", x_test,"\nHypothesis: ", hypo, "\n y_data: ", y_test)
print('Test Accuracy: ', accu_test)
print('Test l2 regularization:', l2_reg_test)

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
