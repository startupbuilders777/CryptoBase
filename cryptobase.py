import csv
import urllib.request
from datetime import datetime
from flask import Flask, request, flash, url_for, redirect, render_template, abort, jsonify, make_response
from matplotlib import pyplot as plt
from flask_sqlalchemy import SQLAlchemy
import sqlalchemy.orm
import gdax #GDAX WRAPPER FOR GDAX EXCHANGE API CALLS
from cockroachdb.sqlalchemy import run_transaction
import datetime
import tensorflow as tf
from datetime import timedelta
from flask_cors import CORS
from matplotlib import pyplot as plt
import os

app = Flask(__name__)
app.config.from_pyfile('hello.cfg')
db = SQLAlchemy(app)
sessionmaker = sqlalchemy.orm.sessionmaker(db.engine)
CORS(app)

class Todo(db.Model):
    __tablename__ = 'todos'
    id = db.Column('todo_id', db.Integer, primary_key=True)
    title = db.Column(db.String(60))
    text = db.Column(db.String)
    done = db.Column(db.Boolean)
    pub_date = db.Column(db.DateTime)

    def __init__(self, title, text):
        self.title = title
        self.text = text
        self.done = False
        self.pub_date = datetime.utcnow()

class Currency(db.Model):
    __tablename__ = "currency"
    id = db.Column("currency_id", db.Integer, primary_key=True)
    name = db.Column(db.String, unique=True)
    priceHistory = db.relationship("Price", backref="currencyAyy", lazy="dynamic")
    tradingHistory = db.relationship("Hist", backref="currency", lazy="dynamic")

    def __init__(self, name):
        self.name = name

    @property
    def serialize(self):
        '''Returns object data in easily serializable format'''
        return {
            "name" : self.name,
            "id": self.id
        }
    @property
    def serializeWithPrices(self):
        return{
            "name" : self.name,
            "prices": [price.serialize() for price in self.priceHistory],
            "hist": [tradinginfo.serialize() for tradinginfo in self.tradingHistory],
            "id": self.id
        }

class Price(db.Model):
    __tablename__ = "price"
    id = db.Column("price_id", db.Integer, primary_key=True)
    value = db.Column(db.Float)
    currency_id = db.Column(db.Integer, db.ForeignKey("currency.currency_id"))
    month = db.Column(db.Integer)
    day = db.Column(db.Integer)
    year = db.Column(db.Integer)
    hour = db.Column(db.Integer)
    minute = db.Column(db.Integer)
    second = db.Column(db.Integer)


    def __init__(self, value, currency_id, month, day, year, hour, minute, second):
        self.value = value
        self.currency_id = currency_id
        self.month = month
        self.day = day
        self.year = year
        self.hour = hour
        self.minute = minute
        self.second = second


    def serialize(self):
        return {
            "price": self.value,
            "month": self.month,
            "day": self.day,
            "year": self.year,
            "hour": self.hour,
            "minute": self.minute,
            "second": self.second
        }


class Agent(db.Model):
    __tablename__ = "agent"
    id = db.Column("agent_id", db.Integer, primary_key=True)
    action = db.Column(db.Integer)

    def __init__(self, action):
            self.action = action

    def serialize(self):
        return {
            "action": self.action
            }

from coinbase.wallet.client import Client

client = Client(
    "3ec6fdcab466768f96c39dd8ac2b2b900b12b051b7eda60c149276cf4786b90f",
    "be25c7a84a548717aea087d1fa412ca58e137ec0289b2c5f1795e24c0b5dbfd9",
    api_version='2017-09-17')

###### RESTFUL API TUTORIAL ENDPOINTS#########################################################################################
'''
@app.route('/todo/api/v1.0/tasks/fook', methods=['GET'])
def get_tasks_fook():
    return jsonify({'tasks': [make_public_task(task) for task in tasks]})

@app.route('/todo/api/v1.0/tasks', methods=['GET'])
def get_tasks():
    return jsonify({'tasks': tasks})

@app.route('/todo/api/v1.0/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    task = [task for task in tasks if task['id'] == task_id]
    if len(task) == 0:
        abort(404)
    return jsonify({'task': task[0]})

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)



$ curl -i -H "Content-Type: application/json" -X POST -d '{"title":"Read a book"}' 
http://localhost:5000/todo/api/v1.0/tasks
HTTP/1.0 201 Created
Content-Type: application/json
Content-Length: 104
Server: Werkzeug/0.8.3 Python/2.7.3
Date: Mon, 20 May 2013 05:56:21 GMT

{
  "task": {
    "description": "",
    "done": false,
    "id": 3,
    "title": "Read a book"
  }
}
'''
@app.route('/currency/new', methods=['POST'])
def create_currency():
    if not request.json or not 'name' in request.json:
        abort(400)

    def callback(session):
        currency = Currency(request.json['name'])
        session.add(currency)
    run_transaction(sessionmaker, callback)
    flash("Currency Created")
    return jsonify({"Created" : True})



@app.route('/currency/<int:currency_id>', methods=['POST'])
def add_prices(currency_id):
    if not request.json or not 'prices' in request.json:
        abort(400)
    def callback(session):
        if(session.query(Currency).filter_by(name=currency_id) is None):
            jsonify({"USER NOT FOUND": True})
            abort(404)

        prices = request.json["prices"]
        dates = request.json["dates"]

        for price, date in zip(prices, dates):
            newPrice = Price(value=float(price), currency_id=currency_id, month=int(date["month"]),
                             day=int(date["day"]), year=int(date["year"]), hour=int(date["hour"]),
                             minute=int(date["minute"]), second=int(date["second"]))
            session.add(newPrice)
    run_transaction(sessionmaker, callback)
    flash("Prices added")
    return jsonify({"Created" : True})

@app.route('/currency/<int:currency_id>', methods=['GET'])
def get_currency(currency_id):
    def callback(session):
        if(session.query(Currency).filter_by(id=currency_id) is None):
            jsonify({"USER NOT FOUND": True})
            abort(404)
        currency = session.query(Currency).filter_by(id=currency_id).one()
        return jsonify(currency.serializeWithPrices)
    return run_transaction(sessionmaker, callback)

@app.route("/currency", methods=["GET"])
def get_all_currencies():
    def callback(session):
        currencies = session.query(Currency)
        return jsonify(json_list=[currency.serialize for currency in currencies.all()])

    return run_transaction(sessionmaker, callback)



    '''
    task = {
        'id': tasks[-1]['id'] + 1,
        'title': request.json['title'],
        'description': request.json.get('description', ""),
        'done': False
    }
    tasks.append(task)
    return jsonify({'task': task}), 201
    '''

@app.route('/currency/<int:currency_id>/<string:new_name>', methods=['PUT'])
def update_task(currency_id, new_name):
    def callback(session):
        for currency in session.query(Currency).all():
            if(currency.id == currency_id):
                currency.name = new_name
    run_transaction(sessionmaker, callback)
    return jsonify({"Completed !:": True})

#YOU CAN ALSO UPDATE BY JUST USING THE HEADERS COOL WAY

@app.route('/currency/headers', methods=['PUT'])
def update_task_with_headers():
    if(request.headers.get("name") is None or request.headers.get("id") is None):
        jsonify({"Headers Is Bad": True})
        abort(400)
    def callback(session):
        for currency in session.query(Currency).all():
            if (currency.id == int(request.headers.get("id"))):
                currency.name = request.headers.get("name")
    run_transaction(sessionmaker, callback)
    return jsonify({"Completed !:": True})

    '''
    task = [task for task in tasks if task['id'] == task_id]
    if len(task) == 0:
        abort(404)
    if not request.json:
        abort(400)
    if 'title' in request.json and type(request.json['title']) != unicode:
        abort(400)
    if 'description' in request.json and type(request.json['description']) is not unicode:
        abort(400)
    if 'done' in request.json and type(request.json['done']) is not bool:
        abort(400)
    task[0]['title'] = request.json.get('title', task[0]['title'])
    task[0]['description'] = request.json.get('description', task[0]['description'])
    task[0]['done'] = request.json.get('done', task[0]['done'])
    return jsonify({'task': task[0]})
    '''
@app.route('/currency/tasks/<int:task_id>', methods=['DELETE'])
def delete_task(task_id):
    task = [task for task in tasks if task['id'] == task_id]
    if len(task) == 0:
        abort(404)
    #tasks.remove(task[0])
    return jsonify({'result': True})

######TO DO TUTORIAL ENDPOINTS ##############################################################################
@app.route('/')
def show_all():
    def callback(session):
        return render_template(
            'show_all.html',
            todos=session.query(Todo).order_by(Todo.pub_date.desc()).all())
    return run_transaction(sessionmaker, callback)

@app.route('/new', methods=['GET', 'POST'])
def new():
    if request.method == 'POST':
        if not request.form['title']:
            flash('Title is required', 'error')
        elif not request.form['text']:
            flash('Text is required', 'error')
        else:
            def callback(session):
                todo = Todo(request.form['title'], request.form['text'])
                session.add(todo)
            run_transaction(sessionmaker, callback)
            flash(u'Todo item was successfully created')
            return redirect(url_for('show_all'))
    return render_template('new.html')


@app.route('/update', methods=['POST'])
def update_done():
    def callback(session):
        for todo in session.query(Todo).all():
            todo.done = ('done.%d' % todo.id) in request.form
    run_transaction(sessionmaker, callback)
    flash('Updated status')
    return redirect(url_for('show_all'))



class TradingInfo(db.Model):
    __tablename__ = "trading_info"
    id = db.Column("trading_info_id", db.Integer, primary_key=True)
    name = db.Column(db.String, unique=True)
    open = db.Column(db.Float)
    close = db.Column(db.Float)
    high = db.Column(db.Float)
    low = db.Column(db.Float)
    day = db.Column(db.Integer)
    year = db.Column(db.Integer)
    hour = db.Column(db.Integer)
    minute = db.Column(db.Integer)
    second = db.Column(db.Integer)
    volume = db.Column(db.Float)

    def __init__(self, name, open, close, high, low, month, day, year, hour, minute, second, volume):
        self.name = name
        self.open = open
        self.close = close
        self.high = high
        self.low = low
        self.month = month
        self.day = day
        self.year = year
        self.hour = hour
        self.minute = minute
        self.second = second
        self.volume = volume


    @property
    def serialize(self):
        return {
            "name": self.name,
            "open": self.open,
            "close": self.close,
            "high": self.high,
            "low": self.low,
            "month": self.month,
            "day": self.day,
            "year": self.year,
            "hour": self.hour,
            "minute": self.minute,
            "second": self.second,
            "volume": self.volume,
            "id": self.id
        }

        



def serializeHistoricData(data):
    if(data is not None):
        data.sort()
        serializedData = []
        for i in data:
            time = datetime.datetime.utcfromtimestamp(i[0])
            serializedData.append(
                {"unix epoch time": i[0],
                 "nicetime": datetime.datetime.utcfromtimestamp(i[0]),
                 "time": {"year": time.year,
                 "month": time.month,
                 "day": time.day,
                 "hour": time.hour},
                 "low": i[1],
                 "high": i[2],
                  "open": i[3],
                  "close": i[4],
                  "volume": i[5],
                 })
    return serializedData









#########################################################################################################
######################################################################################################
####MACHINE LEARNING ALGO

#https://api-public.sandbox.gdax.com/products/ETH-USD/ticker
#https://api-public.sandbox.gdax.com/products/ETH-USD/candles
public_client = gdax.PublicClient()

'''
THIS ENDPOINT SHOULD ALLOW FOR GRANULARITY IN THE DATA
AND PICK WHATEVER YOU WANT


'''

@app.route('/send', methods=['GET'])
def send():
    if request.method == 'GET':
        data = public_client.get_product_historic_rates("ETH-USD", granularity=3600)  # GRANULAIRTY IS PER HOUR DATA
        print(len(data))

       # data2 = getData()
       #print(jsonify(serializeHistoricData(data2)))
        return jsonify(serializeHistoricData(data))

'''
FOR PRODUCT HISTORIC RATES
[
    [ time, low, high, open, close, volume ],
    [ 1415398768, 0.32, 4.2, 0.35, 4.2, 12.3 ],
    
HTTP REQUEST

GET /products/<product-id>/candles

PARAMETERS

Param	Description
start	Start time in ISO 8601
end	End time in ISO 8601
granularity	Desired timeslice in seconds
    ...
    
Each bucket is an array of the following information:

time bucket start time
low lowest price during the bucket interval
high highest price during the bucket interval
open opening price (first trade) in the bucket interval
close closing price (last trade) in the bucket interval
volume volume of trading activity during the bucket interval    
]

 The maximum number of data points for a single request is 200 candles. If your selection of start/end time and granularity 
 will result in more than 200 data points, your request will be rejected. If you wish to retrieve fine granularity data over a 
 larger time range, you will need to make multiple requests with new start/end ranges.






'''

def getData():
    '''Start from 2016 and collect to 2017, and test with latest 30% of data, data is hourly'''
    dateStart = datetime.datetime(2016, 9, 1, 0, 0, 0)
    dateStartISO = dateStart.isoformat()
    dateEnd = datetime.datetime(2017, 9, 14, 0, 0, 0)
    dateEndISO = dateEnd.isoformat()
    dateNext = dateStart

    data = []
    for i in range(1,300):
        dateStart = dateNext
        dateNext  = dateNext + timedelta(days=1)
        dateStartISO = dateStart.isoformat()
        dateNextISO = dateNext.isoformat()
       # print("date start")
       # print(dateStartISO)
       # print("date next is again")
       # print(dateNextISO)
       # print("CURRENT DATA IS")
        someData = public_client.get_product_historic_rates("LTC-USD", start=dateStartISO,  end=dateNextISO, granularity=3600)
        #someData = public_client.get_product_historic_rates('ETH-USD', granularity=3000)
       # print(someData)

        data += someData


    print(dateStart)
    #  print(dateEnd)

    #data = public_client.get_product_historic_rates("ETH-USD", start=dateStartISO, granularity=3600)  # GRANULAIRTY IS PER HOUR DATA
    print(data)

    print("LENGTH IS")
    print(len(data))

    with open("ltcUSD.csv", "w") as f:
        wr = csv.writer(f)
        wr.writerows(data)

#last_24_hours_data = public_client.get_product_24hr_stats("ETH-USD")
#print(last_24_hours_data)

#Convert to ISOps

#whatToLearn = BTC-USD
#whatToLearn = ETH-USD
#whatToLearn = ETH-USD
#whatToLearn = LITE-USD

@app.route('/executeNet/<string:whatToLearn>/<int:startingCapital>', methods=['POST'])
def reinforcementAgent(whatToLearn, startingCapital):

    '''
    IF YOU HAVE 
    log: True 
    in the header of the request, tesorboard will be loggin the weights 
    
    :param whatToLearn: 
    :param startingCapital: 
    :return: 
    '''
    import numpy as np
    import tensorflow as tf
    import random

    import json
    import pprint

    log = False
    tensorboardLog = False
    reloadOldResults = False

    if request.headers.get("log") is not None and request.headers.get("log") ==  "TRUE":
        log = True
    elif request.headers.get("tensorboard-log") is not None and request.headers.get("tensorboard-log").upper() == "TRUE":
        tensorboardLog = True
    elif request.headers.get("reload") is not None and  request.headers.get("reload").upper()== "TRUE": #RELOAD WORKS, ITS BEEN TESTED
        reloadOldResults = True

    if(os.path.isfile(whatToLearn + "X" + ".json") and reloadOldResults):
        with open(whatToLearn + "X" + ".json") as data_file:
            data = json.load(data_file)
            return jsonify(data)

    '''
    REMOVE THE TRAINING AND TESTING DATA SPLIT, DOESNT MAKE SENSE FOR REINFOORCEMENT AGENTS
    ADDDD TENSORBOARD
    ADDDD RNN
    ADD MULTIHTREADING
    ACTIONS ARE BUY, SELL AND HOLD
    '''

    # Define an abstract class called DecisionPolicy

    class DecisionPolicy:
        def select_action(self, current_state, step):
            pass

        def update_q(self, state, action, reward, next_state):
            pass

    # Thats a good baseline. Now lets use a smarter approach using a neural network



    class QLearningDecisionPolicy(DecisionPolicy):
        def __init__(self, actions, input_dim):
            '''
            
            
            :param actions: ACTIONS ARE BUY, HOLD, SELL (Want to be able to extend these actions doe.)
            :param input_dim: 
            '''

            self.epsilon = 0.9      #Choose a random action about 10% of the time and decrease the probability of choose a random action as time goes on
            self.gamma = 0.001
            self.actions = actions
            output_dim = len(actions)
            h1_dim = 200

            self.x = tf.placeholder(tf.float32, [None, input_dim])      #PUT THE BATCH OF VALUES IN THE PLACEHOLDER X, size is 202.
            self.y = tf.placeholder(tf.float32, [output_dim])           #outputs an action but should also output a value tooooo!!!!!!!

            W1 = tf.Variable(tf.random_normal([input_dim, h1_dim]))     #Go from 202 to 200
            b1 = tf.Variable(tf.constant(0.1, shape=[h1_dim]))
            h1 = tf.nn.relu(tf.matmul(self.x, W1) + b1)
            W2 = tf.Variable(tf.random_normal([h1_dim, output_dim]))    #Go from 200 to number of actions
            b2 = tf.Variable(tf.constant(0.1, shape=[output_dim]))
            self.q = tf.nn.relu(tf.matmul(h1, W2) + b2)     #DETERMINES THE SCORE OF EACH ACTION

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

        def select_action(self, current_state, step):
            # -> SIZE OF 202 WHICH IS the size of the input Dim for an input x. which is also hist + 2, the +2 is the price and other data
            #print("THE CURRENT STATE IS: " + str(current_state))
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
                action_idx = np.argmax(action_q_vals)           #PICK THE MOST VALUABLE ACTION OUT OF THEM ALL
                action = self.actions[action_idx]               #Get the action
                if tensorboardLog:
                    writer.add_summary(summary, step)

            else:
                # Explore random option with probability 1 - epsilon
                action = self.actions[random.randint(0, len(self.actions) - 1)] #Chhoose a random action
            if tensorboardLog:
                writer.close()
            return action

        def update_q(self, state, action, reward, next_state):      #TO UPDATE THE STATE MATRIX FOR BETTER PREDICTIONS
            '''
            
            :param state:  The current State
            :param action: BUY, HOLD, SELL, The chosen action at the time 
            :param reward: The reward you got for choosing that action
            :param next_state: The next state you are in after choosing the action
            :return: 
            '''
            action_q_vals = self.sess.run(self.q, feed_dict={self.x: state})    #WHAT ARE THE ACTION_Q VALUES FOR THE CURRENT STATE
            next_action_q_vals = self.sess.run(self.q, feed_dict={self.x: next_state})  #WHAT ARE THE ACTION_Q VALUES FOR THE NEXT STATE
            next_action_idx = np.argmax(next_action_q_vals) #WHATS THE BEST NEXT ACTION TO TAKE FROM THE NEW STATE, GET THAT ACTIONS ID

            '''
            TODO: Document what these 2 do soon.
            '''
            action_q_vals[0, next_action_idx] = reward + self.gamma * next_action_q_vals[0, next_action_idx]
            action_q_vals = np.squeeze(np.asarray(action_q_vals))

            self.sess.run(self.train_op, feed_dict={self.x: state, self.y: action_q_vals})  #RUN THE TRAIN OP TO IMPROVE THE PREDICTIONS


    def run_simulation(policy, initial_budget, initial_num_stocks, prices, hist, reinforcementAgentDecisions, current_portfolio_value_list,
                       debug=False):
        '''
        Runs the simiulation for the Q-LEARNING ALGORITHM
        
        :param policy: Class that encapsulates the learning agent
        :param initial_budget: Its a number like 1000$ that the agent can play with
        :param initial_num_stocks: How many coins does the agent start off with?
        :param prices: 
        :param hist: Size by default is 200. THE STATE FOR THE QLEARNING ALGO IS BASICALLY an array of 
        :param reinforcementAgentDecisions: NOT USED IN ALGORITHM, just there to accumulate data for graphs
        :param current_portfolio_value_list:NOT USED IN ALGORITHM, just there to accumulate data for graphs 
        :param debug: 
        :return: 
        '''
        budget = initial_budget
        num_stocks = initial_num_stocks
        share_value = 0
        transitions = list()

        for i in range(len(prices) - hist - 1):
            if i % 100 == 0:
                print('progress {:.2f}%'.format(float(100 * i) / (len(prices) - hist - 1)))

            current_state = np.asmatrix(np.hstack((prices[i:i + hist], budget, num_stocks)))
            '''THE STATE OF THE Q-ALGORITHM:

                THE STATE is a definite place.
                One sort of finite definite place is a list of prices from i to i + 200 along with the budget and num_stocks
                there is a finite number of these types of states. Also the prices are a contiguous block.
                
                To determine the best possible action to take given this 202 dimensional state (if hist == 200),
                 convert the blcok into a 200 weight -> then 200 to 3 states, BUY, SELL, HOLD
                

            '''
            current_portfolio = budget + num_stocks * share_value
            action = policy.select_action(current_state, i)
            share_value = float(prices[i + hist + 1])

            #RESTRICTION TO BUY, SELL ONLY ONE AT A TIME. change this so it gets better.
            if action == 'Buy' and budget >= share_value:           #BUY A STOCK, ONLY BUYS 1 AT A TIME RESTRICTION
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
            new_portfolio = budget + num_stocks * share_value      #Recalculate new portfolio value
            reward = new_portfolio - current_portfolio              #The REWARD IS THE INCREASE IN PORTFOLIO VALUE

            next_state = np.asmatrix(np.hstack((prices[i + 1:i + hist + 1], budget, num_stocks)))



            transitions.append((current_state, action, reward, next_state))
            policy.update_q(current_state, action, reward, next_state)

        portfolio = budget + num_stocks * share_value
        current_portfolio_value_list.append(portfolio)
        if debug:
            print('${}\t{} shares'.format(budget, num_stocks))
        return portfolio

    # We want to run simulations multiple times and average out the performances:

    def run_simulations(policy, budget, num_stocks, prices, hist, reinforcementAgentDecisions, current_portfolio_value_list):
        num_tries = 10
        final_portfolios = list()
        for i in range(num_tries):
            final_portfolio = run_simulation(policy, budget, num_stocks, prices, hist, reinforcementAgentDecisions, current_portfolio_value_list)
            final_portfolios.append(final_portfolio)
        avg, std = np.mean(final_portfolios), np.std(final_portfolios)
        return avg, std

    # Who wants to deal with stock market data without looking a pretty plots? No one. So we need this out of law:

    def plot_prices_all(prices):
        plt.title('Closing stock prices for all data')
        plt.xlabel('hour')
        plt.ylabel('price ($)')
        plt.plot(prices)
        plt.savefig(whatToLearn + " " + 'all_prices.png')
        #plt.show()

    def plot_prices_test(prices):
        plt.title('Closing stock prices for test data')
        plt.xlabel('hour')
        plt.ylabel('price ($)')
        plt.plot(prices)
        plt.savefig('test_prices.png')
        #plt.show()

    def plot_prices_train(prices):
        plt.title('Closing stock prices for train data')
        plt.xlabel('hour')
        plt.ylabel('price ($)')
        plt.plot(prices)
        plt.savefig('train_prices.png')
        #plt.show()

    def getCloseData(trading_info):
        prices = []
        for item in trading_info:
            prices.append(item["close"])
        return prices

    def getAllDataExceptTime(trading_info):
        '''
        DATA should look like this:
        "low": 14.89,
        "high": 14.9,
        "open": 14.89,
        "close": 14.9,
        "volume": 600.9523090700001        
        
        HERE A COOL RESEARCH QUESTION. Does it matter how we put the data in the array? 
        Like if we put close before low and high. Or put volume first. Hell put volume in the middle and close at the end?????
        '''
        prices = []
        for item in trading_info:
            prices.append([item["low"], item["close"], item["high"], item["volume"]])
        return prices

    '''
    Just use the closing prices of stocks right now but will add more later

    Further analysis of the candlestick required?
    Maybe a convolutional Analysis on the candlesticj 
    Followed by LSTM Analysis
    '''
    def splitDataToTestAndTrain(trading_info):
        # The first 90% of the data is for training the neural net and is old data
        # Assuming the data is ordererd from oldest to newest

        length = len(trading_info)
        train_len = length * 0.9
        train = trading_info[:int(train_len)]

        # The next 10% is recent data the neural net has not seen and will be tested on this data
        test_len = length * 0.1
        test = trading_info[int(train_len):]

        return (train, test)

    '''
    Candlestick analysis requires viewing several candlesticks at once and performing convolutions on these candle batches
    MUST BE FOLLOWED BY LSTM
    '''

    if __name__ == '__main__':
        data = []
        if(whatToLearn == "ETH-BTC"):
            with open('etherdata.json') as data_file:
                data = json.load(data_file)
                print("THE DATA IS")
                print(data)
        elif(whatToLearn == "BTC-USD"):
            with open('btcUSD.json') as data_file:
                data = json.load(data_file)
                print("THE DATA IS")
                print(data)
        elif(whatToLearn == "ETH-USD"):
            with open('ethUSD.json') as data_file:
                data = json.load(data_file)
                print("THE DATA IS")
                print(data)
        elif(whatToLearn == "LITE-USD"):
            with open('ltcUSD.json') as data_file:
                data = json.load(data_file)
                print("THE DATA IS")
                print(data)

        data.sort(key=lambda x: x["time"])
        all_prices = getCloseData(data)
        plot_prices_all(all_prices)
        print(all_prices)

        actions = ['Buy', 'Sell', 'Hold']
        reinforcementAgentDecisions = []
        current_portfolio_value_list = []

        hist = 200
        policy = QLearningDecisionPolicy(actions, hist + 2)         #-> INPUT DIM IS a matrix of size 202
        budget = startingCapital
        num_stocks = 0

        def serialize(data):
            serialized = []
            for i in data:
                time = datetime.datetime.utcfromtimestamp(i["time"])
                i["date"] = {"year": time.year, "month":time.month, "day" :time.day, "hour": time.hour}

            return data

        avg, std = run_simulations(policy, budget, num_stocks, all_prices, hist, reinforcementAgentDecisions, current_portfolio_value_list)
        print("The end capital earned by the agent is: ")
        print(avg)
        print("The standard deviation for this capital amount in the test batch is: ")
        print(std)
        print("The agenets decisions were:")
        print(reinforcementAgentDecisions)

        output = {"data": serialize(data),
                        "decisions": reinforcementAgentDecisions,
                        "current_portfolio_value_list": current_portfolio_value_list,
                        "average": avg,
                        "standard deviation": std
                        }

        with open(whatToLearn + "X" + ".json", 'w') as f:
            f.write(json.dumps(output, ensure_ascii = False))

        return jsonify(output)

if __name__ == '__main__':
    app.run()
