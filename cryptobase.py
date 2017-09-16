from datetime import datetime
from flask import Flask, request, flash, url_for, redirect, render_template, abort, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
import sqlalchemy.orm
import gdax #GDAX WRAPPER FOR GDAX EXCHANGE API CALLS
from cockroachdb.sqlalchemy import run_transaction
import datetime
from datetime import timedelta

app = Flask(__name__)
app.config.from_pyfile('hello.cfg')
db = SQLAlchemy(app)
sessionmaker = sqlalchemy.orm.sessionmaker(db.engine)


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

'''
class TradingInfo(db.Model):
    __tablename__ = "trading_info"
    id = db.Column("trading_info_id", db.Integer, primary_key=True)
    
    name = db.Column(db.String, unique=True)
    open = db.Column(db.Float)
    close = db.Column(db.Float)

    def __init__(self, name):
        self.name = name

    @property
    def serialize(self):
        return {
            "name": self.name,
            "id": self.id
        }

    @property
    def serializeWithPrices(self):
        return {
            "name": self.name,
            "prices": [price.serialize() for price in self.priceHistory],
            "id": self.id
        }
        
        
TUBA WILL MAKE THIS SHIT
'''

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

        data2 = getData()
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
    dateStart = datetime.datetime(2016, 1, 1, 0, 0, 0)
    dateStartISO = dateStart.isoformat()
    dateEnd = datetime.datetime(2016, 1, 2, 0, 0, 0)#datetime.datetime(2017,9,15,0,0,0)
    dateEndISO = dateEnd.isoformat()
    dateNext = dateStart

    data = []
    for i in range(1,10):
        dateStart = dateNext
        dateNext  = dateNext + timedelta(days=15)
        dateNextISO = dateNext.isoformat()
        print("date next is")
        print(dateStart)
        print("date next is again")
        print(dateNext)
        data +=  public_client.get_product_historic_rates("ETH-USD", start=dateStartISO, end=dateNextISO, granularity=3600)


    print(dateStart)
    #  print(dateEnd)

    #data = public_client.get_product_historic_rates("ETH-USD", start=dateStartISO, granularity=3600)  # GRANULAIRTY IS PER HOUR DATA
    print(data)

getData()

#last_24_hours_data = public_client.get_product_24hr_stats("ETH-USD")
#print(last_24_hours_data)

#Convert to ISO





if __name__ == '__main__':
    app.run()
