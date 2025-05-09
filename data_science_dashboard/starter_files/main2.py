# Stock Data Dashboard Application

# GUI imports
import taipy as tp
import taipy.gui.builder as tgb
from taipy.gui import Icon
from taipy import Config
# machine learning imports
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
# data imports
import datetime
import plotly.graph_objects as go
# Pandas on CPU
import pandas as pd

# Flag to determine if TensorFlow is available
try:
    from tensorflow.keras import models
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
    print("TensorFlow imported successfully")
except ImportError:
    print("TensorFlow import failed, running with limited functionality")
    TENSORFLOW_AVAILABLE = False

###################################
# GLOBAL VARIABLES
###################################
# datasets

stock_data = pd.read_csv("data/sp500_stocks.csv")
company_data = pd.read_csv("data/sp500_companies.csv")
# country names and icons [for slider]
country_names = company_data["Country"].unique().tolist()
country_names = [(name, Icon("images/flags/" + name + ".png", name)) for name in country_names]

# company names [for slider]
company_names = company_data[
                    ["Symbol","Shortname"]
                    ].sort_values("Shortname").values.tolist()
print(country_names)
print(company_names)

# start and finish dates
dates = [
    stock_data["Date"].min(),
    stock_data["Date"].max()
]

# initial country and company selection
country = "Canada"
company = ["LULU"]

# initial prediction values
lin_pred = 0
knn_pred = 0
rnn_pred = 0

# initial graph values
graph_data = None
figure = None

###################################
# GRAPHIC USER INTERFACE
###################################

# create page
with tgb.Page() as page:
     # create horizontal group of elements
    # aligned to the center
    with tgb.part("text-center"):
        tgb.image("images/icons/logo.png", width="10vw")
        tgb.text(
        "# S&P 500 Stock Value Over Time", mode="md"
        )
        # create date range selector
        tgb.date_range(
        "{dates}",
        label_start="Start Date",
        label_end="End Date"
        ) 
         # create vertical group of 2 elements
        # taking 20% and 80% of the view poer
    with tgb.layout("20 80"):
        tgb.selector(
            label="Country",
            class_name="fullwidth",
            value="{country}",
            lov="{country_names}",
            dropdown=True,
            value_by_id=True    
        ) 
        tgb.selector(
            label="Company",
            class_name="fullwidth",
            value="{company}",
            lov="{company_names}",
            dropdown=True,
            value_by_id=True,
            multiple=True    
        )
    # create chart    
    tgb.chart(figure="{figure}")
    # vertical group of 8 elements
    with tgb.part("text-center"):
       with tgb.layout("4 72 4 4 4 4 4 4"):
        # company name and symbol
        tgb.image(
            "images/icons/id-card.png",
            width="3vw"
            )
        tgb.text("{company[-1]} | {company_data['Shortname'][company_data['Symbol'] == company[-1]].values[0]}", mode="md")
        # linear regression prediction
        tgb.image(
            "images/icons/lin.png",
            width="3vw"
            )
        tgb.text("{lin_pred}", mode="md")
        tgb.image(
            "images/icons/knn.png",
            width="3vw"
            )
        # KNN prediction
        tgb.text("{knn_pred}", mode="md")
        # RNN prediction
        tgb.image(
            "images/icons/rnn.png",
            width="3vw"
            )
        tgb.text("{rnn_pred}", mode="md")

###################################
# FUNCTIONS
###################################

def build_company_names(country):
   company_names = company_data[["Symbol","Shortname"]][
      company_data["Country"] == country
   ].sort_values("Shortname").values.tolist()
   return company_names

def build_graph_data(dates, company):
    print("----------------------------")
    print(dates, company)

    temp_data = stock_data[["Date", "Adj Close", "Symbol"]][
       # filter by dates
        (stock_data["Date"] > str(dates[0])) &
        (stock_data["Date"] < str(dates[1]))      
    ]
    # reconstruct temp_data with empty data frame
    graph_data = pd.DataFrame()
    # fetch dates column
    graph_data["Date"] = temp_data["Date"].unique()
    # fetch company values into new columns
    for i in company:
       graph_data[i] = temp_data["Adj Close"][ 
          temp_data["Symbol"] == i
       ].values

    return graph_data

def display_graph(graph_data):
   """
    draw stock value graphs
    - input graph_data: numpy array of stock values to plot
    - output figure: plotly Figure with visuallized graph_data
    """
   figure = go.Figure()
    # fetch symbols from column names
   symbols = graph_data.columns[1:]
    # draw historic data for each symbol
   for i in symbols:
    # add titles
    figure.add_trace(go.Scatter(
      x=graph_data["Date"],
      y=graph_data[i],
      name=i,
      showlegend=True
      ))
   figure.update_layout(
      xaxis_title="Date",
      yaxis_title="Stock Value"
        )
   return figure

def split_data(stock_data, dates, symbol):
    """
    arrange data for training and prediction
    
    - input stock_data: Pandas dataframe from global variable
    - input dates: list with a start date and a finish date
    - input symbol: string that represents a company symbol
    
    - output features: numpy array
    - output targets: numpy array
    - output eval_features: numpy array
    """
    temp_data = stock_data[
    # filter dates and symbol   
    (stock_data["Symbol"] == symbol) &  
    (stock_data["Date"] > str(dates[0])) &
    (stock_data["Date"] < str(dates[1]))      
    ].drop(["Date", "Symbol"], axis=1)

    # fetch evaluation sample
    eval_features = temp_data.values[-1]
    eval_features = eval_features.reshape(1, -1)
    # unsqueeze dimensions

    # fetch features and targets
    features = temp_data.values[:-1]
    targets = temp_data["Adj Close"].shift(-1).values[:-1]

    return features, targets, eval_features   

def get_lin(dates, company):
   """
    obtain prediction with Linear Regression
    
    - input dates: list with a start date and a finish date
    - input company: list of company symbols
    
    - output: floating point prediciton
    """
   x, y, eval_x = split_data(stock_data, dates, company[-1])

   lin_model.fit(x, y)
   lin_pred = lin_model.predict(eval_x) 

   return lin_pred[0]

def get_knn(dates, company):
   """
    obtain prediction with K Nearest Neighbors
    
    - input dates: list with a start date and a finish date
    - input company: list of company symbols
    
    - output: floating point prediciton
    """
   x, y, eval_x = split_data(stock_data, dates, company[-1])
   
   knn_model.fit(x, y)
   knn_pred = knn_model.predict(eval_x)
   
   return knn_pred[0]

def get_rnn(dates, company):
   """
    obtain prediction with RNN
    
    - input dates: list with a start date and a finish date
    - input company: list of company symbols
    
    - output: floating point prediciton
    """
   if not TENSORFLOW_AVAILABLE:
       print("TensorFlow not available, returning placeholder value")
       return 0.0
       
   x, y, eval_x = split_data(stock_data, dates, company[-1])
   
   # Simple training - this is a placeholder that should be improved
   rnn_model.fit(x, y, epochs=5, verbose=0)
   rnn_pred = rnn_model.predict(eval_x)
   
   return rnn_pred[0][0]

###################################
# BACKEND
###################################

# configure data nodes
country_cfg = Config.configure_data_node(
    id="country")
company_names_cfg = Config.configure_data_node(
   id="company_names")
dates_cfg = Config.configure_data_node(
    id="dates")
company_cfg = Config.configure_data_node(
   id="company")
graph_data_cfg = Config.configure_data_node(
    id="graph_data"
)
lin_pred_cfg = Config.configure_data_node(
   id="lin_pred")
knn_pred_cfg = Config.configure_data_node(
   id="knn_pred")
rnn_pred_cfg = Config.configure_data_node(
   id="rnn_pred")

# configure tasks
get_lin_cfg = Config.configure_task(
    input = [dates_cfg, company_cfg],
    output = lin_pred_cfg,
    function = get_lin,
    id = "get_lin",
    skippable = True
    )

get_knn_cfg = Config.configure_task(
    input = [dates_cfg, company_cfg],
    output = knn_pred_cfg,
    function = get_knn,
    id = "get_knn",
    skippable = True
    )

get_rnn_cfg = Config.configure_task(
    input = [dates_cfg, company_cfg],
    output = rnn_pred_cfg,
    function = get_rnn,
    id = "get_rnn",
    skippable = True
    )

build_graph_data_cfg = Config.configure_task(
    input = [dates_cfg, company_cfg],
    output = graph_data_cfg,
    function = build_graph_data,
    id = "build_graph_data",
    skippable = True
    )


build_company_names_cfg = Config.configure_task(
   input=country_cfg,
   output=company_names_cfg,
   function=build_company_names,
   id="build_company_names",
   skippable=True
)

# configure scenario
scenario_cfg = Config.configure_scenario(
    task_configs = [
        build_company_names_cfg, 
        build_graph_data_cfg, 
        get_lin_cfg,
        get_knn_cfg,
        get_rnn_cfg
    ],
    id="scenario"
    )

def on_init(state, name, value):
    """
    built-in Taipy function that runs once
    when the application first loads
    """
    # write inputs to scenario
    state.scenario.country.write(state.country)
    state.scenario.dates.write(state.dates)
    state.scenario.company.write(state.company) 
    # update scenario     
    state.scenario.submit(wait=True)
    # fetch updated outputs
    state.graph_data = state.scenario.graph_data.read()
    state.company_names = state.scenario.company_names.read()
    state.lin_pred = state.scenario.lin_pred.read()
    state.knn_pred = state.scenario.knn_pred.read()
    state.rnn_pred = state.scenario.rnn_pred.read()  
    # Display graph initially
    state.figure = display_graph(state.graph_data)
      
def on_change(state, name, value):
   """
    built-in Taipy function that runs every time
    a GUI variable is changed by user
    """
   if name == "country":
      print(name, "was modified", value)
      # update scenario with new country selection
      state.scenario.country.write(state.country)
      state.scenario.submit(wait=True)
      # Get company names for the new country
      state.company_names = state.scenario.company_names.read()
      # If no companies available for this country, don't update graph
      if state.company_names and len(state.company_names) > 0:
         state.company = [state.company_names[0][0]]  # Select first company
         state.scenario.company.write(state.company)
         state.scenario.submit(wait=True)
         state.graph_data = state.scenario.graph_data.read()
         state.figure = display_graph(state.graph_data)
         # Get predictions
         state.lin_pred = state.scenario.lin_pred.read()
         state.knn_pred = state.scenario.knn_pred.read()
         state.rnn_pred = state.scenario.rnn_pred.read()


   if name == "company" or name == "dates":
      print(name, "was modified", value)
      # update scenario with new company or dates selection  
      state.scenario.dates.write(state.dates)
      state.scenario.company.write(state.company)      
      state.scenario.submit(wait=True)
      state.graph_data = state.scenario.graph_data.read()
      state.figure = display_graph(state.graph_data)
      # Get predictions
      state.lin_pred = state.scenario.lin_pred.read()
      state.knn_pred = state.scenario.knn_pred.read()
      state.rnn_pred = state.scenario.rnn_pred.read()

def build_RNN(n_features):
    """
    create a Recurrent Neural Network
    - input n_features: integer with the number of features within x and eval_x
    - output model: RNN Tensorflow model
    """
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available, returning None")
        return None
        
    model = models.Sequential()
    model.add(layers.Dense(64, activation = 'relu', input_shape=(n_features, )))
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dense(1, activation = 'linear'))
    model.compile(optimizer = 'rmsprop', loss = 'mse', metrics = ['mae'])
    return model      

if __name__ == "__main__":
    # create machine learning models
    lin_model = LinearRegression()