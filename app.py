import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import dateutil

import simpy
import numpy as np
#import matplotlib.pyplot as plt

#holding cost per item per time unit:
h = 5
item_price = 100
cost_to_order = 50
n_delivery_days = 2
PRT = False

messg = []

def warehouse_run(env, order_cutoff, order_target, m, ip, co, hc, d):
    global inventory, balance, num_ordered, h, item_price, cost_to_order, n_delivery_days
    
    item_price, cost_to_order, h, n_delivery_days = ip, co, hc, d
    #print(item_price, cost_to_order, h, n_delivery_days)
    inventory = order_target
    # current cash, revenue - cost
    balance = 0 
    num_ordered = 0
        
    while True:
        interarrival = generate_interarrival()
        #wait for next customer for 'interarrival' days
        yield env.timeout(interarrival)
        balance -= h * inventory * interarrival
        demand = generate_demand()
        
        if inventory > demand:
            balance += item_price * demand
            inventory -= demand
            if PRT:
                print('{:.2f} sold {}'.format(env.now, demand))
            if m != None:
                m.append('     {:.2f}: sold {}'.format(env.now, demand))
        else: 
            balance += item_price * inventory
            inventory = 0
            if PRT:
                print('{:.2f} sold {} (out of stock)'.format(env.now, inventory))
            if m != None:
                m.append('     {:.2f}: sold {} (out of stock)'.format(env.now, inventory))

        #Make new orders (replanish) if needed
        if inventory < order_cutoff and num_ordered == 0:
              env.process(handle_order(env, order_target, m))

def handle_order(env, order_target, m):
        global inventory, balance, num_ordered
    
        num_ordered = order_target - inventory
        if PRT:
            print('{:.2f} {} in inventory, placed order for {} items'.format(env.now, inventory, num_ordered))
        if m != None:
            m.append('     {:.2f} {} in inventory, placed order for {} items'.format(env.now, inventory, num_ordered))
        balance -= cost_to_order * num_ordered
        #wait for next order for n_delivery_days days
        yield env.timeout(n_delivery_days)
        
        #After that, update counters
        inventory += num_ordered
        num_ordered = 0
        if PRT:
            print('{:.2f} received order, {} items in inventory'.format(env.now, inventory))
        if m != None:
            m.append('     {:.2f}: received order, {} items in inventory'.format(env.now, inventory))

def generate_interarrival():
        #customer inter-arrival mean time
        cust_lambda = 5
        return np.random.exponential(1./cust_lambda)

def generate_demand():
        #each customer demands D = uniform(1,4)
        #prod_min, prod_max = 1,7
        #return np.random.randint(prod_min, prod_max)
        return np.random.poisson(lam=3)
        #return np.random.normal(loc=3,scale=2)

#control function
def observe(env, obs_time, obs_inventory, obs_balance):
    global inventory, balance
    while True:
        obs_time.append(env.now)
        obs_inventory.append(inventory)
        obs_balance.append(balance)
        yield env.timeout(0.5)

def one_sim(seed, s,S, ip, co,h,d):
    np.random.seed(seed)
    env = simpy.Environment()

    messg = []
    env.process(warehouse_run(env, s,S, messg, ip, co,h,d))
    obs_time = []
    obs_inventory = []
    obs_balance = []
    env.process(observe(env, obs_time, obs_inventory, obs_balance))

    #run warehouse for 5 days
    env.run(until=7.0)
    
    return obs_time, obs_inventory, obs_balance, messg

# Optimization
def observe2(env, obs_balance):
    global inventory, balance
    while True:
        obs_balance.append(balance)
        yield env.timeout(0.5)
        
def opt_fun(x, ip, co,h,d):
    replications = 50
    bal_tot = 0
    #print('2',ip,co,h,d)
    for i in range(replications):
        np.random.seed(i)
        env = simpy.Environment()
        env.process(warehouse_run(env, int(x[0]), int(x[1]), None, ip,co,h,d))
        obs_balance = []
        env.process(observe2(env, obs_balance))
        env.run(until=7.0)
        bal_tot += obs_balance[-1]   
    return bal_tot/replications

def grid_optimize(s1,s2, S1, S2, step, ip, co,h,d):
    #print('1',ip,co,h,d)
    small_s = range(s1,s2,step)
    big_S   = range(S1,S2,step)
    bal_best = 0
    x1_best = -1
    x2_best = -1
    bal_d = {}
    for x1 in small_s:
      for x2 in big_S:
        if x1 > x2:
            bal_d[(x1, x2)] = 0
            continue
        bal = opt_fun([x1,x2], ip, co,h,d) 
        bal_d[(x1,x2)] = bal
        if bal > bal_best:
            bal_best = bal
            x1_best = x1
            x2_best = x2

    print(bal_best)
    print(x1_best,x2_best)

    x1 = [x[0] for x in list(bal_d.keys())]
    x2 = [x[1] for x in list(bal_d.keys())]
    df = pd.DataFrame({'x1':x1,'x2':x2,'bal':list(bal_d.values())})

    return df.query('bal>0')



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(

    dcc.Tabs([
        dcc.Tab(label='Simulation', 
                style={'width': '50%','font-size': '130%','height': '30%', 'color':'blue'},
                children=[

                html.Br(),
                html.H1('Inventory Simulation',style={'textAlign': 'center'}),
                html.Br(),

                html.Div(className="row", children=[
                    html.Div(className="two columns", children=[
                        html.H6('Item Price'),
                        dcc.Dropdown(
                            id='item_price',
                            options=[{'label':c, 'value':c} for c in range(50,151,10)],
                            value=100
                        )
                    ]),
                    html.Div(className="two columns", children=[
                        html.H6('Cost to Order'),
                        dcc.Dropdown(
                            id='cost_to_order',
                            options=[{'label':c, 'value':c} for c in range(0,61,10)],
                            value=50
                        )
                    ]),
                    html.Div(className="two columns", children=[
                        html.H6('Holding cost / item'),
                        dcc.Dropdown(
                            id='holding_cost',
                            options=[{'label':c, 'value':c} for c in range(0,31,5)],
                            value=5
                        )
                    ]),
                    html.Div(className="two columns", children=[
                        html.H6('Delivery days'),
                        dcc.Dropdown(
                            id='delivery_days',
                            options=[{'label':c, 'value':c} for c in range(1,6,1)],
                            value=2
                        )
                    ]),
                ]),

                html.Div(className="row", children=[
                    html.Div(className="two columns", children=[
                        html.H6('Order Target',style={'color': 'red'}),
                        dcc.Dropdown(
                            id='order_target',
                            options=[{'label':c, 'value':c} for c in range(30,61,1)],
                            #placeholder=50
                            value=50
                        )
                    ]),
                    html.Div(className="two columns", children=[
                        html.H6('Order Cutoff',style={'color': 'red'}),
                        dcc.Dropdown(
                            id='order_cutoff',
                            options=[{'label':c, 'value':c} for c in range(10,31,1)],
                            value=20
                        )
                    ]),
                    html.Div(className="two columns", children=[
                        html.H6('Random Seed'),
                        dcc.Dropdown(
                            id='rseed',
                            options=[{'label':c, 'value':c} for c in range(10)],
                            value=0
                        )
                    ]),
                ]),

                html.Div(className="row", children=[
                    html.Div(className="four columns", children=[
                        dcc.Graph(
                            style={'width': '120%','font-size': '100%','height': '30%'},
                            id="plot_inventory",
                            config={ 'displayModeBar': False }
                        )
                    ]),
                    html.Div(className="four columns", children=[
                        dcc.Graph(
                            style={'width': '120%','font-size': '100%','height': '30%'},
                            id="plot_balance",
                            config={ 'displayModeBar': False }
                        )
                    ]),
                    html.Div(className="four columns", children=[
                        html.Br(),
                        html.Br(),
                        html.H5('Inventory sale (time: action)',style={'textAlign': 'center'}),
                        dcc.Textarea(
                            id='display_messg',
                            style={'width': '90%','height': '270px','font_size': '10px','textAlign': 'left'}
                        )
                    ])
                ]),

        ]),

        dcc.Tab(label='Optimization', 
        style={'width': '50%','font-size': '130%','height': '30%', 'color':'blue'},
        children=[
                html.Br(),
                html.H1('Inventory Optimization',style={'textAlign': 'center'}),
                html.Br(),

                html.Div(className="row", children=[
                    html.Div(className="two columns", children=[
                        html.H6('Item Price'),
                        dcc.Dropdown(
                            id='item_price2',
                            options=[{'label':c, 'value':c} for c in range(50,151,10)],
                            value=100
                        )
                    ]),
                    html.Div(className="two columns", children=[
                        html.H6('Cost to Order'),
                        dcc.Dropdown(
                            id='cost_to_order2',
                            options=[{'label':c, 'value':c} for c in range(0,61,10)],
                            value=10
                        )
                    ]),
                    html.Div(className="two columns", children=[
                        html.H6('Holding cost / item'),
                        dcc.Dropdown(
                            id='holding_cost2',
                            options=[{'label':c, 'value':c} for c in range(0,31,5)],
                            value=5
                        )
                    ]),
                    html.Div(className="two columns", children=[
                        html.H6('Delivery days'),
                        dcc.Dropdown(
                            id='delivery_days2',
                            options=[{'label':c, 'value':c} for c in range(1,6,1)],
                            value=2
                        )
                    ]),
                    html.Div(className="two columns", children=[
                        html.H6('Order Target',style={'color': 'red'}),
                        dcc.Dropdown(
                            id='order_target2',
                            options=[{'label':c, 'value':c} for c in range(30,101,5)],
                            value=90
                        )
                    ]),

                 ]),

                 html.Div(className="row", children=[
                    html.Div(className="four columns", children=[
                        html.Br(), html.Br(),
                        html.H5('Balance vs Order target and cutoff',style={'textAlign': 'center','color': 'blue'}),  
                        html.P(id = "opt_res", style={'textAlign': 'center'}),
                        dcc.Graph(
                            style={'width': '120%','font-size': '100%','height': '30%'},
                            id="plot_opt1",
                            config={ 'displayModeBar': False }
                        )
                      ]),

                    html.Div(className="four columns", children=[
                        html.Br(), html.Br(),
                        html.H5('Balance vs Order Cutoff projection',style={'textAlign': 'center','color': 'blue'}),  
                        html.P(id = "opt_choice", style={'textAlign': 'center'}),
                        dcc.Graph(
                            style={'width': '120%','font-size': '100%','height': '30%'},
                            id="plot_opt2",
                            config={ 'displayModeBar': False }
                        )
                      ]),
                   
                ])

            ])
        ])     
)

def Plot(x,y, title,xlabel,ylabel):
    fig = go.Figure(data=go.Scatter(x=x, y=y))
    fig.update_layout(
        # margin=dict(
        # l=0,
        # #r=20,
        # #b=20,
        # #t=20
        # ),
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel)
    return fig

def Plot_opt(x,y, z, title, xlabel,ylabel,zlabel):
    #https://www.programcreek.com/python/example/103209/plotly.graph_objs.Scatter3d
    fig = go.Figure(data=go.Scatter3d(x=x, y=y, z=z, 
            mode='markers',
            type='scatter3d',
            marker=dict(size=5,opacity=0.9),
            #colorscale='Jet', #'Viridis'
            ))

    fig.update_layout(
        title="Retina",
        margin=dict(
        l=1,
        r=10,
        b=0,
        t=0),
        scene = dict(
            xaxis = dict(
                title=xlabel,
                #range=[0, 1],
                ),
            yaxis = dict(
                title=ylabel),
            zaxis = dict(
                title=zlabel)
            )
    )

    return fig

@app.callback(
    [Output('plot_inventory', 'figure'), Output('plot_balance', 'figure'), \
     Output('display_messg', 'value')],
    [Input('order_cutoff', 'value'), Input('order_target', 'value'), Input('rseed', 'value'),
     Input('item_price', 'value'), Input('cost_to_order', 'value'), Input('holding_cost', 'value'),
     Input('delivery_days','value')]
)
def update_plot1(s, S, seed, ip, co,h,d):
    obs_time, obs_inventory, obs_balance, messg = one_sim(seed, s,S, ip, co,h,d)
    return Plot(obs_time, obs_inventory,'','Time (days)','Inventory'),\
           Plot(obs_time, obs_balance, '','Time (days)','Balance'),\
           '\n'.join(messg)

df = pd.DataFrame()

@app.callback(
    [Output('plot_opt1', 'figure'), Output('opt_res', 'children')],
    #Output('plot_opt2', 'figure'), Output('opt_choice', 'children')
    [Input('item_price2', 'value'), Input('cost_to_order2', 'value'), Input('holding_cost2', 'value'),
     Input('delivery_days2','value')#, Input('order_target2', 'value')
     ]
)
def update_plot21(ip, co,h,d):
    global df
    df = grid_optimize(20, 91, 30, 101, 5, ip,co,h,d)
    tmp = df.query('bal == {}'.format(max(df.bal)))
    results = 'Best balance = ${} at order (target, cutoff) = ({},{})'.format(
               int(max(df.bal)),int(tmp.x2),int(tmp.x1))
    #df_sel = df.query('x2 == {}'.format(int(order_target)))
    #title = '\t\t You have chosen Order Target = {} items'.format(order_target)
    return Plot_opt(df.x1, df.x2, df.bal,'Optimized balance','Order Cutoff','Order Target','Balance'),\
           results
           #Plot(df_sel.x1, df_sel.bal,'','Order Cutoff','Balance'), title

@app.callback(
    [#Output('plot_opt1', 'figure'), 
     Output('plot_opt2', 'figure'), Output('opt_choice', 'children')],
    [Input('item_price2', 'value'), Input('cost_to_order2', 'value'), Input('holding_cost2', 'value'),
     Input('delivery_days2','value'), Input('order_target2', 'value')]
)
def update_plot22(ip, co,h,d, order_target):
    global df
    #df = grid_optimize(20, 90, 30, 103, 3, ip,co,h,d)
    title = 'Please choose Order Target...'
    df_sel = pd.DataFrame({'x1':[],'x2':[],'bal':[]})
    if len(df)>0:
        title = '\t\t You have chosen Order Target = {} items'.format(order_target)
        df_sel = df.query('x2 == {}'.format(int(order_target)))
    return Plot(df_sel.x1, df_sel.bal,'','Order Cutoff','Balance'), title

server = app.server

if __name__ == '__main__':
    app.run_server(host="0.0.0.0", debug=False)
