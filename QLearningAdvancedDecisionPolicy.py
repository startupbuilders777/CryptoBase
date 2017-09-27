from DecisionPolicy import DecisionPolicy
import numpy as np
import tensorflow as tf
import random
import json
import pprint

class QLearningAdvancedDecisionPolicy(DecisionPolicy):
    def __init__(self, actions, input_dim, tensorboardLog):
        '''
        :param actions: ACTIONS ARE BUY, HOLD, SELL (Want to be able to extend these actions doe.)
        :param input_dim: 
        '''

        self.epsilon = 0.9  # Choose a random action about 10% of the time and decrease the probability of choose a random action as time goes on
        self.gamma = 0.001
        self.actions = actions
        self.tensorboardLog = tensorboardLog

        output_dim = len(actions)
        h1_dim = 200

        self.x = tf.placeholder(tf.float32,
                                [None, input_dim])  # PUT THE BATCH OF VALUES IN THE PLACEHOLDER X, size is 202.
        self.y = tf.placeholder(tf.float32,
                                [output_dim])  # outputs an action but should also output a value tooooo!!!!!!!

        W1 = tf.Variable(tf.random_normal([input_dim, h1_dim]))  # Go from 202 to 200
        b1 = tf.Variable(tf.constant(0.1, shape=[h1_dim]))
        h1 = tf.nn.relu(tf.matmul(self.x, W1) + b1)
        W2 = tf.Variable(tf.random_normal([h1_dim, output_dim]))  # Go from 200 to number of actions
        b2 = tf.Variable(tf.constant(0.1, shape=[output_dim]))
        self.q = tf.nn.relu(tf.matmul(h1, W2) + b2)  # DETERMINES THE SCORE OF EACH ACTION

        if tensorboardLog:
            tf.summary.histogram('q', self.q)
            tf.summary.histogam('W1', W1)
            tf.summary.histogram('W2', W2)
            tf.summary.histogram('histogram', W2)

        loss = tf.square(self.y - self.q)

        if tensorboardLog:
            tf.summary.scalar("loss", tf.reduce_sum(loss))
            tf.summary.scalar("loss histogram", loss)

        self.train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def createAgentState(self, price_data, budget, num_stocks):
        return np.asmatrix(np.hstack(price_data, budget, num_stocks))

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

        writer = None
        merged = None
        if tensorboardLog:
            writer = tf.summary.FileWriter(whatToLearn + "/")
            merged = tf.summary.merge_all()
        threshold = min(self.epsilon, step / 1000.)
        if random.random() < threshold:
            # Exploit best option with probability epsilon
            summary = None
            action_q_vals = None
            if tensorboardLog:
                summary, action_q_vals = self.sess.run([self.merged, self.q], feed_dict={self.x: current_state})
            else:
                action_q_vals = self.sess.run(self.q, feed_dict={self.x: current_state})
            action_idx = np.argmax(action_q_vals)  # PICK THE MOST VALUABLE ACTION OUT OF THEM ALL
            action = self.actions[action_idx]  # Get the action
            if tensorboardLog:
                writer.add_summary(summary, step)

        else:
            # Explore random option with probability 1 - epsilon
            action = self.actions[random.randint(0, len(self.actions) - 1)]  # Chhoose a random action
        if tensorboardLog:
            writer.close()
        return action

    def update_q(self, state, action, reward, next_state):  # TO UPDATE THE STATE MATRIX FOR BETTER PREDICTIONS
        '''

        :param state:  The current State
        :param action: BUY, HOLD, SELL, The chosen action at the time 
        :param reward: The reward you got for choosing that action
        :param next_state: The next state you are in after choosing the action
        :return: 
        '''
        action_q_vals = self.sess.run(self.q,
                                      feed_dict={self.x: state})  # WHAT ARE THE ACTION_Q VALUES FOR THE CURRENT STATE
        next_action_q_vals = self.sess.run(self.q, feed_dict={
            self.x: next_state})  # WHAT ARE THE ACTION_Q VALUES FOR THE NEXT STATE
        next_action_idx = np.argmax(
            next_action_q_vals)  # WHATS THE BEST NEXT ACTION TO TAKE FROM THE NEW STATE, GET THAT ACTIONS ID

        '''
        TODO: Document what these 2 do soon.
        '''
        action_q_vals[0, next_action_idx] = reward + self.gamma * next_action_q_vals[0, next_action_idx]
        action_q_vals = np.squeeze(np.asarray(action_q_vals))

        self.sess.run(self.train_op,
                      feed_dict={self.x: state, self.y: action_q_vals})  # RUN THE TRAIN OP TO IMPROVE THE PREDICTIONS

