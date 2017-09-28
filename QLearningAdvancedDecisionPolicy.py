from DecisionPolicy import DecisionPolicy
import numpy as np
import tensorflow as tf
import random
import tensorflow.contrib.slim as slim
import tensorflow.contrib.rnn
import array
import json
import pprint

class QLearningAdvancedDecisionPolicy(DecisionPolicy):
    def __init__(self, actions, input_dim, trace_length, tensorboardLog):
        '''
        :param actions: ACTIONS ARE BUY, HOLD, SELL (Want to be able to extend these actions doe.)
        :param input_dim: 
        '''

        self.epsilon = 0.9  # Choose a random action about 10% of the time and decrease the probability of choose a random action as time goes on
        self.gamma = 0.001
        self.actions = actions
        self.tensorboardLog = tensorboardLog
        self.length_of_state = input_dim
        self.h_size_0 = 8
        self.h_size = 16
        self.num_hidden_rnn = 16
        self.amount_of_data_in_each_state = 6
        self.convolution_layer_1_length = 5
        self.convolution_layer_2_length = 5
        output_dim = len(actions) + 2

        #if trace length is 30 and batch size is 10 -> that is 300 values

        # Setting the training parameters
        self.trace_length = trace_length
        self.rnn_batch_size_val = self.h_size*self.length_of_state*self.amount_of_data_in_each_state/(4*self.trace_length*self.h_size)  # How many experience traces to use for each training step.
        print("THE RNN BATCH SIZE IS")
        print(self.rnn_batch_size_val)

         # How long each experience trace will be when training

        '''
        Input looks like this:
        price[x][0] -> low
        price[x][1] -> close
        price[x][2] -> high
        price[x][3] -> volume
        price[x][4] -> budget
        price[x][5] -> num_stocks
        '''
        self.x = tf.placeholder(tf.float32, [None, self.amount_of_data_in_each_state*self.length_of_state], name="x")  # PUT THE BATCH OF VALUES IN THE PLACEHOLDER X, size is 202.
        self.y = tf.placeholder(tf.float32, [output_dim], name="y")  # outputs an action but should also output a value tooooo!!!!!!!
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.trainLength = tf.placeholder(dtype=tf.int32, name="trainLength")

        self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.h_size, state_is_tuple=True)
        self.rnn_batch_size = tf.placeholder(dtype=tf.int32, shape=[], name="rnn_batch_size")

        #ALWAYS INITIALIZE YOUR WEIGHTS AND BIASES
        self.weights = {

            'wc1': tf.Variable(tf.random_normal([self.amount_of_data_in_each_state, self.convolution_layer_1_length, 1, self.h_size_0])),
            'wc2': tf.Variable(tf.random_normal([self.amount_of_data_in_each_state, self.convolution_layer_2_length, self.h_size_0, self.h_size])),
            'wd1': tf.Variable(tf.random_normal([int(self.amount_of_data_in_each_state/2 * self.length_of_state/2 * self.h_size), 1024])),
            'birnn_out' : tf.Variable(tf.random_normal([2 * self.num_hidden_rnn, output_dim]))
        }

        self.biases = {
            'bc1': tf.Variable(tf.random_normal([self.h_size_0])),
            'bc2': tf.Variable(tf.random_normal([self.h_size])),
        #    'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([output_dim]))
        }

        def conv2d(x, W, b, stride=1):
            x = tf.nn.conv2d(input=x, filter=W, strides=[1,stride,stride, 1], padding="SAME")
            x = tf.nn.bias_add(x, b)
            return tf.nn.relu(x)

        def bidirectionalLSTM(x, trainLength, num_hidden_rnn, weights, biases):
            # Current data input shape: (batch_size, timesteps, n_input)
            # Required shape: 'timesteps' tensors list of shape (batch_size, num_input)

            # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_input)
            x = tf.unstack(x, trainLength, 1)

            # Define lstm cells with tensorflow
            # Forward direction cell
            lstm_fw_cell = tensorflow.contrib.rnn.BasicLSTMCell(num_hidden_rnn, forget_bias=1.0)
            # Backward direction cell
            lstm_bw_cell = tensorflow.contrib.rnn.BasicLSTMCell(num_hidden_rnn, forget_bias=1.0)

            # Get lstm cell output
            try:
                outputs, _, _ = tensorflow.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
            except Exception:  # Old TensorFlow version only returns outputs not states
                outputs = tensorflow.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)

            # Linear activation, using rnn inner loop last output
            return tf.matmul(outputs[-1], weights['birnn_out']) + biases['out']

        def maxpool2d(x, k=2):
            return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k,k,1], padding="SAME")

        def deep_net(x, weights, biases, dropout):
            self.price_in = tf.reshape(x, shape=[-1, self.amount_of_data_in_each_state, self.length_of_state, 1])

            # Convolution Layer
            conv1 = conv2d(self.price_in, weights['wc1'], biases['bc1'])
            # Another Convolution Layer
            conv2 = conv2d(conv1, weights['wc2'], biases["bc2"])

            # Max Pooling (down-sampling)       #Cuts both dimensiosn in half
            conv2 = maxpool2d(conv2, k=2)

            # We take the output from the final convolutional layer and send it to a recurrent layer.
            # The input must be reshaped into [batch x trace x units] for rnn processing,
            # and then returned to [batch x units] when sent through the upper levles.

            self.convFlat = tf.reshape(slim.flatten(conv2), [self.rnn_batch_size, self.trainLength, self.h_size])
            QOut = bidirectionalLSTM(self.convFlat, self.trace_length, self.num_hidden_rnn, weights, biases)
            return QOut
            '''
            self.state_in = self.rnn_cell.zero_state(self.rnn_batch_size, tf.float32)

            self.rnn, self.rnn_state = tf.nn.dynamic_rnn( inputs=self.convFlat, cell=self.rnn_cell, dtype=tf.float32, initial_state=self.state_in)
            self.rnn = tf.reshape(self.rnn, shape=[-1, self.h_size])

            # The output from the recurrent layer is then split into separate Value and Advantage streams
            self.streamA, self.streamV = tf.split(self.rnn, 2, 1)
            self.AW = tf.Variable(tf.random_normal([self.h_size // 2, len(actions)]))
            self.VW = tf.Variable(tf.random_normal([self.h_size // 2, 1]))
            self.Advantage = tf.matmul(self.streamA, self.AW)       #-> Now you have only 3 sized array
            self.Value = tf.matmul(self.streamV, self.VW)          # -> Now you have 1 sized array

            self.salience = tf.gradients(self.Advantage, self.price_in)
            # Then combine them together to get our final Q-values.
            Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
            '''
            '''
            self.predict = tf.argmax(self.Qout, 1)

            # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
            self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions, 4, dtype=tf.float32)

            self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

            self.td_error = tf.square(self.targetQ - self.Q)

            # In order to only propogate accurate gradients through the network, we will mask the first
            # half of the losses for each trace as per Lample & Chatlot 2016
            self.maskA = tf.zeros([self.batch_size, self.trainLength // 2])
            self.maskB = tf.ones([self.batch_size, self.trainLength // 2])
            self.mask = tf.concat([self.maskA, self.maskB], 1)
            self.mask = tf.reshape(self.mask, [-1])
            self.loss = tf.reduce_mean(self.td_error * self.mask)

            self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
            self.updateModel = self.trainer.minimize(self.loss)
            
            return out
            '''

        #  W1 = tf.Variable(tf.random_normal([input_dim, h1_dim]))  # Go from 202 to 200
        #b1 = tf.Variable(tf.constant(0.1, shape=[h1_dim]))
        #h1 = tf.nn.relu(tf.matmul(self.x, W1) + b1)
        #W2 = tf.Variable(tf.random_normal([h1_dim, output_dim]))  # Go from 200 to number of actions
        #b2 = tf.Variable(tf.constant(0.1, shape=[output_dim]))
        #self.q = tf.nn.relu(tf.matmul(h1, W2) + b2)  # DETERMINES THE SCORE OF EACH ACTION
        self.q = deep_net(self.x, self.weights, self.biases, self.keep_prob)

        if self.tensorboardLog:
            tf.summary.histogram('q', self.q)
            tf.summary.histogam('W1', W1)
            tf.summary.histogram('W2', W2)
            tf.summary.histogram('histogram', W2)

        loss = tf.square(self.y - self.q)

        if self.tensorboardLog:
            tf.summary.scalar("loss", tf.reduce_sum(loss))
            tf.summary.scalar("loss histogram", loss)

        self.train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)
        with tf.device('/gpu:0'):
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
            self.sess.run(tf.global_variables_initializer())

    def createAgentState(self, price_data, budget, num_stocks):
        state = []
        for i in price_data:
            state.append(i[0])
            state.append(i[1])
            state.append(i[2])
            state.append(i[3])
            state.append(budget)
            state.append(num_stocks)
        return np.array(state)

    def select_action(self, current_state, step):
        # -> SIZE OF 202 WHICH IS the size of the input Dim for an input x. which is also hist + 2, the +2 is the price and other data
        # print("THE CURRENT STATE IS: " + str(current_state))
        '''
        THE CURRENT STATE IS: 
        [[ 292.89  285.81  288.14  276.18  272.2   265.8   262.3   251.21  267.29
        257.86  258.26  262.96  265.7   252.    235.    242.    232.11  219.94
        222.53  234.44  243.    247.9   252.99  249.7   251.23  256.55  256.29
        252.64  250.22  239.06  234.98  238.22  240.5   238.46  233.05  228.1
        219.1   218.46  228.62  224.55  234.32  239.67  252.15  265.71  260.72
        279.    287.23  271.21  260.87  263.04  278.23  270.54  270.01  273.5
        280.6   283.    284.    283.97  286.79  292.19  304.32  296.18  296.
        295.24  301.86  300.3   309.54  316.98  321.75  316.5   315.32  314.96
        316.22  314.5   316.98  319.45  318.26  318.63  316.06  309.82  301.66
        297.75  304.98  302.96  304.98  298.4   301.92  305.24  309.51  306.19
        301.34  304.    300.01  298.03  292.95  296.77  291.92  286.76  291.26
        290.44  290.56  292.03  291.92  296.3   297.68  304.82  306.02  295.88
        288.61  294.3   293.56  292.98  294.19  291.01  287.5   284.87  273.
        270.51  280.04  278.    273.27  274.69  272.09  274.52  267.51  266.99
        272.69  273.18  281.82  279.98  275.01  274.83  268.79  265.74  272.57
        268.    263.51  259.5   258.49  258.17  257.13  263.84  263.12  259.9
        262.84  257.02  256.81  257.84  263.2   260.88  265.    274.    269.56
        263.37  267.98  265.89  262.01  263.53  267.39  271.92  281.    278.3
        280.5   282.63  292.54  286.    285.    282.36  278.13  281.92  279.2
        280.43  281.2   278.29  276.    276.03  274.89  277.33  276.28  276.38
        281.61  280.35  281.49  281.07  281.    278.    278.62  279.5   279.69
        277.    276.37  277.69  278.69  281.26  279.16  279.65  281.49  281.5
        281.5   282.58   19.28   61.  ]]
        '''
        with tf.device('/gpu:0'):
            writer = None
            merged = None
            amout = 0
            if self.tensorboardLog:
                writer = tf.summary.FileWriter(whatToLearn + "/")
                merged = tf.summary.merge_all()
            threshold = min(self.epsilon, step / 1000.)
            if random.random() < threshold:
                # Exploit best option with probability epsilon
                summary = None
                action_q_vals = None
                current_state = current_state.reshape((1,self.length_of_state*self.amount_of_data_in_each_state))
                if self.tensorboardLog:
                    summary, action_q_vals = self.sess.run([self.merged, self.q], feed_dict={self.x: current_state, self.keep_prob : 0.90, self.rnn_batch_size: self.rnn_batch_size_val, self.trainLength: self.trace_length})
                    print("ACTION Q VALUES ARE")
                    print(action_q_vals)
                else:
                    action_q_vals = self.sess.run(self.q, feed_dict={self.x: current_state, self.keep_prob: 0.90, self.rnn_batch_size: self.rnn_batch_size_val, self.trainLength: self.trace_length})
                action_idx = np.argmax(action_q_vals[0:2])  # PICK THE MOST VALUABLE ACTION OUT OF THEM ALL
                print("ACTION IDX")
                print(action_q_vals)
                print(action_idx)
                action = self.actions[action_idx]  # Get the action
                if(action == "Buy"):
                    amout = action_q_vals[3]
                elif(action =="Sell"):
                    amout = action_q_vals[4]

                if self.tensorboardLog:
                    writer.add_summary(summary, step)

            else:
                # Explore random option with probability 1 - epsilon
                action = self.actions[random.randint(0, len(self.actions) - 1)]  # Chhoose a random action
            if self.tensorboardLog:
                writer.close()
            return (action, amout)

    def update_q(self, state, action, reward, next_state):  # TO UPDATE THE STATE MATRIX FOR BETTER PREDICTIONS
        '''

        :param state:  The current State
        :param action: BUY, HOLD, SELL, The chosen action at the time 
        :param reward: The reward you got for choosing that action
        :param next_state: The next state you are in after choosing the action
        :return: 
        '''
        #  print("THE STATE SHAPE IS ")
        # print(state)
        with tf.device('/gpu:0'):
            state = state.reshape((1,self.length_of_state*self.amount_of_data_in_each_state))     #THERE IS ONLY 1 THING BATCHED
            next_state = next_state.reshape((1,self.length_of_state*self.amount_of_data_in_each_state))   #THERE IS ONLY 1 THING BATCHED

            action_q_vals = self.sess.run(self.q, feed_dict={self.x: state, self.keep_prob: 0.90, self.rnn_batch_size: self.rnn_batch_size_val, self.trainLength: self.trace_length})  # WHAT ARE THE ACTION_Q VALUES FOR THE CURRENT STATE
            next_action_q_vals = self.sess.run(self.q, feed_dict={ self.x: next_state, self.keep_prob: 0.90, self.rnn_batch_size: self.rnn_batch_size_val, self.trainLength: self.trace_length})  # WHAT ARE THE ACTION_Q VALUES FOR THE NEXT STATE
            next_action_idx = np.argmax(
                next_action_q_vals[0:2]
            )  # WHATS THE BEST NEXT ACTION TO TAKE FROM THE NEW STATE, GET THAT ACTIONS ID
            #print("NEXT ACTION Q-VALS")
            #print(next_action_q_vals)
            '''
            TODO: Document what these 2 do soon.
            '''
            action_q_vals[0, next_action_idx] = reward + self.gamma * next_action_q_vals[0, next_action_idx]
            action_q_vals = np.squeeze(np.asarray(action_q_vals))

            self.sess.run(self.train_op,
                          feed_dict={self.x: state, self.y: action_q_vals, self.keep_prob : 0.90, self.rnn_batch_size: self.rnn_batch_size_val, self.trainLength: self.trace_length})  # RUN THE TRAIN OP TO IMPROVE THE PREDICTIONS

