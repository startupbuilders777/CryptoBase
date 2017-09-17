from matplotlib import pyplot as plt
def reinforcementAgent():
    import numpy as np
    import tensorflow as tf
    import random

    import json
    import pprint

    '''
    REMOVE THE TRAINING AND TESTING DATA SPLIT, DOESNT MAKE SENSE FOR REINFOORCEMENT AGENTS
    
    ACTIONS ARE BUY, SELL AND HOLD
    '''


    # Define an abstract class called DecisionPolicy

    class DecisionPolicy:
        def select_action(self, current_state, step):
            pass

        def update_q(self, state, action, reward, next_state):
            pass


    #Thats a good baseline. Now lets use a smarter approach using a neural network

    class QLearningDecisionPolicy(DecisionPolicy):
        def __init__(self, actions, input_dim):
            self.epsilon = 0.9
            self.gamma = 0.001
            self.actions = actions
            output_dim = len(actions)
            h1_dim = 200

            self.x = tf.placeholder(tf.float32, [None, input_dim])
            self.y = tf.placeholder(tf.float32, [output_dim])
            W1 = tf.Variable(tf.random_normal([input_dim, h1_dim]))
            b1 = tf.Variable(tf.constant(0.1, shape=[h1_dim]))
            h1 = tf.nn.relu(tf.matmul(self.x, W1) + b1)
            W2 = tf.Variable(tf.random_normal([h1_dim, output_dim]))
            b2 = tf.Variable(tf.constant(0.1, shape=[output_dim]))
            self.q = tf.nn.relu(tf.matmul(h1, W2) + b2)

            loss = tf.square(self.y - self.q)
            self.train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

        def select_action(self, current_state, step):
            threshold = min(self.epsilon, step / 1000.)
            if random.random() < threshold:
                # Exploit best option with probability epsilon
                action_q_vals = self.sess.run(self.q, feed_dict={self.x: current_state})
                action_idx = np.argmax(action_q_vals)  # TODO: replace w/ tensorflow's argmax
                action = self.actions[action_idx]
            else:
                # Explore random option with probability 1 - epsilon
                action = self.actions[random.randint(0, len(self.actions) - 1)]
            return action

        def update_q(self, state, action, reward, next_state):
            action_q_vals = self.sess.run(self.q, feed_dict={self.x: state})
            next_action_q_vals = self.sess.run(self.q, feed_dict={self.x: next_state})
            next_action_idx = np.argmax(next_action_q_vals)
            action_q_vals[0, next_action_idx] = reward + self.gamma * next_action_q_vals[0, next_action_idx]
            action_q_vals = np.squeeze(np.asarray(action_q_vals))
            self.sess.run(self.train_op, feed_dict={self.x: state, self.y: action_q_vals})

    def run_simulation(policy, initial_budget, initial_num_stocks, prices, hist, reinforcementAgentDecisions, debug=False):
        budget = initial_budget
        num_stocks = initial_num_stocks
        share_value = 0
        transitions = list()
        for i in range(len(prices) - hist - 1):
            if i % 100 == 0:
                print('progress {:.2f}%'.format(float(100*i) / (len(prices) - hist - 1)))
            current_state = np.asmatrix(np.hstack((prices[i:i+hist], budget, num_stocks)))
            current_portfolio = budget + num_stocks * share_value
            action = policy.select_action(current_state, i)
            share_value = float(prices[i + hist + 1])
            if action == 'Buy' and budget >= share_value:
                budget -= share_value
                num_stocks += 1
                reinforcementAgentDecisions.append("buy one")
            elif action == 'Sell' and num_stocks > 0:
                budget += share_value
                num_stocks -= 1
                reinforcementAgentDecisions.append("sell one")
            else:
                action = 'Hold'
                reinforcementAgentDecisions.append("hold")
            new_portfolio = budget + num_stocks * share_value
            reward = new_portfolio - current_portfolio
            next_state = np.asmatrix(np.hstack((prices[i+1:i+hist+1], budget, num_stocks)))
            transitions.append((current_state, action, reward, next_state))
            policy.update_q(current_state, action, reward, next_state)

        portfolio = budget + num_stocks * share_value
        if debug:
            print('${}\t{} shares'.format(budget, num_stocks))
        return portfolio

    #We want to run simulations multiple times and average out the performances:

    def run_simulations(policy, budget, num_stocks, prices, hist, reinforcementAgentDecisions):
        num_tries = 10
        final_portfolios = list()
        for i in range(num_tries):
            final_portfolio = run_simulation(policy, budget, num_stocks, prices, hist, reinforcementAgentDecisions)
            final_portfolios.append(final_portfolio)
        avg, std = np.mean(final_portfolios), np.std(final_portfolios)
        return avg, std

    #Call the following function to use the Yahoo Finance library and obtain useful stockmarket data.

    def get_prices(share_symbol, start_date, end_date, cache_filename='stock_prices.npy'):
        try:
            stock_prices = np.load(cache_filename)
        except IOError:
            share = Share(share_symbol)
            stock_hist = share.get_historical(start_date, end_date)
            stock_prices = [stock_price['Open'] for stock_price in stock_hist]
            np.save(cache_filename, stock_prices)

        return stock_prices

    #Who wants to deal with stock market data without looking a pretty plots? No one. So we need this out of law:

    def plot_prices_all(prices):
        plt.title('Closing stock prices for all data')
        plt.xlabel('hour')
        plt.ylabel('price ($)')
        plt.plot(prices)
        plt.savefig('all_prices.png')
        plt.show()

    def plot_prices_test(prices):
        plt.title('Closing stock prices for test data')
        plt.xlabel('hour')
        plt.ylabel('price ($)')
        plt.plot(prices)
        plt.savefig('test_prices.png')
        plt.show()

    def plot_prices_train(prices):
        plt.title('Closing stock prices for train data')
        plt.xlabel('hour')
        plt.ylabel('price ($)')
        plt.plot(prices)
        plt.savefig('train_prices.png')
        plt.show()

    def standardizeData(trading_info):
        prices = []
        for item in trading_info:
            prices.append(item["close"])
        return prices

    '''
    Just use the closing prices of stocks right now but will add more later
    
    Further analysis of the candlestick required?
    Maybe a convolutional Analysis on the candlesticj 
    Followed by LSTM Analysis
    '''
    def splitDataToTestAndTrain(trading_info):
        #The first 90% of the data is for training the neural net and is old data
        #Assuming the data is ordererd from oldest to newest

        length = len(trading_info)
        train_len = length * 0.9
        train = trading_info[:int(train_len)]

        #The next 10% is recent data the neural net has not seen and will be tested on this data
        test_len = length * 0.1
        test = trading_info[int(train_len):]

        return (train, test)

    '''
    Candlestick analysis requires viewing several candlesticks at once and performing convolutions on these candle batches
    MUST BE FOLLOWED BY LSTM
    '''

    if __name__ == '__main__':
        with open('etherdata.json') as data_file:
            data = json.load(data_file)
        print("THE DATA IS")
        print(data)

        data.sort(key=lambda x: x["date"])

        trainData, testData = splitDataToTestAndTrain(data)
        print("AMOUT OF DATA IS: ")
        print(len(data))
        print("Amount of training data is: ")
        print(len(trainData))
        print("Amount of test data is: ")
        print(len(testData))

        all_prices = standardizeData(data)
        plot_prices_all(all_prices)
        print(all_prices)

        train_prices = standardizeData(trainData)
        plot_prices_train(train_prices)
        print(train_prices)

        test_prices = standardizeData(testData)
        plot_prices_test(test_prices)
        print(test_prices)

        actions = ['Buy', 'Sell', 'Hold']
        reinforcementAgentDecisions = []

        hist = 200
        policy = QLearningDecisionPolicy(actions, hist + 2)
        budget = 1000
        num_stocks = 0
        #avg, std = run_simulations(policy, budget, num_stocks, train_prices, hist)
        print("The trained amout earned was")
        #print(avg)
        print("The standard deviation for this is: ")
        #print(std)

        budget = 1000
        num_stocks = 0
        avg, std = run_simulations(policy, budget, num_stocks, all_prices, hist, reinforcementAgentDecisions)
        print("The end capital earned by the agent is: ")
        print(avg)
        print("The standard deviation for this capital amount in the test batch is: ")
        print(std)
        print("The agenets decisions were:")
        print(reinforcementAgentDecisions)

reinforcementAgent()



