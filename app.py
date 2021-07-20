from os import remove
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input, State
from dash_html_components.H4 import H4
from dash_html_components.P import P
from dash_html_components.Tr import Tr
import visdcc
import dash_bootstrap_components as dbc
import io
import base64
import pandas as pd
from pipeline import pipeline
import numpy as np
from sklearn.metrics import mean_absolute_error
import time
from copy import deepcopy
from datetime import datetime

FA = "https://use.fontawesome.com/releases/v5.12.1/css/all.css"

app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP, FA],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, height=device-height, initial-scale=1"}
    ]
)

edge_mode = False
remove_edge = False
remove_node = False
run1,run2,run3,run4 = False,False,False,False

nodes = [{'id': 'node-data', 'label': 'Data', 'size':15, 'shape': 'dot'},
        {'id': 'node-feature', 'label': 'Features', 'size':15, 'shape': 'dot'},
        {'id': 'node-model', 'label': 'Model', 'size':15, 'shape': 'dot'},
        {'id': 'node-output', 'label': 'Output', 'size':15, 'shape': 'dot'},
]

nodes_orig = deepcopy(nodes)

edges = [{'id': 'data-feature', 'from':'node-data', 'to':'node-feature', 'width':3},
        {'id': 'feature-model', 'from':'node-feature', 'to':'node-model', 'width':3},
        {'id': 'model-output', 'from':'node-model', 'to':'node-output', 'width':3}
]

edges_orig = deepcopy(edges)

app.layout = html.Div(
    children=[
        html.Div(children=[dbc.Row(
            [
                dbc.Col(html.Div(children=[
                    html.H5(
                    children="Nodes", className="panel-title"
                    ),
                    html.Hr(style={'color': 'white', 'background': 'white', 'maxWidth': '90%'}),
                    dbc.ButtonGroup([
                                    dbc.Button("Logger", color="primary", id='logger', n_clicks=0), 
                                    dbc.Button("Viz", color="primary", id='viz', n_clicks=0),
                                    dbc.Button("Transform", color="primary", id='transform', n_clicks=0)
                                    ], size='sm'),
                    dbc.Button("Remove Node", color="danger", className="mr-1",  block=True, id="remove-node", n_clicks_timestamp=0, style={'marginTop':'10px'}),
                    html.Div(children=[],style={'height':'5%'}),
                    html.H5(
                    children="Edges", className="panel-title"
                    ),
                    html.Hr(style={'color': 'white', 'background': 'white', 'maxWidth': '90%'}),
                    html.Div(id='callback-out', style={'display':'none'}),
                    html.Div(id='callback-out1', style={'display':'none'}),
                    dbc.Button("+ Add Edge", color="info", className="mr-1",  block=True, id="add_edge", n_clicks=0),
                    dbc.Button("Remove Edge", color="danger", className="mr-1",  block=True, id="remove-edge", n_clicks_timestamp=0, style={'marginTop':'10px'}),
                ], style={'background':'#423e3e', 'height':'1000px', 'padding':'5px'}), width=2),
                dbc.Col(visdcc.Network(id = 'net', data={'nodes': nodes, 'edges': edges}, 
                                        options={'height': '750px', 'width':'100%', 'edges':{
                                            'arrows':{
                                                'to': {'enabled': True, 'type': 'arrow'}
                                            }},
                                            'layout': {
                                                'randomSeed' : 6789,
                                                # 'hierarchical': {'direction': 'LR'}
                                            },
                                            'interaction': {
                                                'multiselect': True
                                            }
                                            }))
            ]
        )], 
                style={'height':'100vh', 'width':'100vw', 'overflow':'hidden'}),
        dbc.Modal(
            [
                dbc.ModalHeader(children=[
                    html.H4(["Data",
                            html.H6([dbc.Badge("Running", color="success", className="mr-1", style={'visibility':'hidden'}, id='data-badge')], 
                            style={'display': 'inline', 'marginLeft':'5px'})
                    ], 
                    style={'color':'#036bfc', 'display': 'inline', 'marginRight': '20px'}),
                    dbc.Button(children=["Run"], color="success", className="mr-1", size="sm", style={'marginRight': '10px'}, id='data-run', n_clicks_timestamp=0),
                    dbc.Button(children=["Stop"], color="danger", className="mr-1", size="sm", style={'marginRight': '10px'}, id='data-stop'),
                ]),
                dbc.ModalBody(
                    children=[
                    dbc.Checklist(
                        options=[
                            {"label": "Run Downstream", "value": "downstream_yes"},
                        ],
                        value=[],
                        id="switches-input",
                        switch=True,
                        inline=True
                    ),
                    html.P("Upload data for training model", style={'marginTop': '15px', 'fontWeight':'bold'}),
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files')
                        ]),
                        style={
                            'width': '95%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
                        },
                    ),
                    html.H4("Preview"),
                    html.Div(id='preview'),]
                ),
                dbc.Button(
                        "Close", id="close-data", className="ml-auto", n_clicks=0, style={'float':'right', 'margin': '10px'}
                )
            ],
            id="modal-data",
            is_open=False,
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(children=[
                    html.H4([
                        "Features", 
                        html.H6([dbc.Badge("Running", color="success", className="mr-1", style={'visibility':'hidden'}, id='feature-badge')], 
                        style={'display': 'inline', 'marginLeft':'5px'})], 
                    style={'color':'#036bfc', 'display': 'inline', 'marginRight': '20px'}),
                    dbc.Button(children=["Run"], color="success", className="mr-1", size="sm", style={'marginRight': '10px'}, id='feature-run', n_clicks_timestamp=0),
                    dbc.Button(children=["Stop"], color="danger", className="mr-1", size="sm", style={'marginRight': '10px'}, id='feature-stop'),
                ]),
                dbc.ModalBody(children=[
                    dbc.Checklist(
                        options=[
                            {"label": "Run Downstream", "value": "downstream_yes"},
                        ],
                        value=[],
                        id="switches-input-1",
                        switch=True,
                        inline=True
                    ),
                    html.P("Select the features you want", style={'marginTop': '15px', 'fontWeight':'bold'}),
                    html.Div(id='column-choice', children=[
                        dcc.Dropdown(
                            options=[
                            ],
                            multi=True,
                            id = 'x_columns'
                        )
                    ]),
                    html.P("Output Column", style={'marginTop': '8px', 'fontWeight':'bold'}),
                    html.Div(id='output-choice', children=[
                        dcc.Dropdown(
                            options=[
                            ],
                            id = 'y_column'
                        )
                    ]),
                    html.P("Train Test split", style={'marginTop': '8px'}),
                    dcc.Slider(
                        id='split-slider',
                        min=0,
                        max=100,
                        step=5,
                        value=70,
                    ),
                    html.P("Train  70% / Test  30%", style={'marginTop': '3px', 'fontWeight': 'bold'}, id='slider-label'),
                ]),
                dbc.Button(
                        "Close", id="close-feature", className="ml-auto", n_clicks=0, style={'float':'right', 'margin': '10px'}
                )
            ],
            id="modal-feature",
            is_open=False,
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(children=[
                    html.H4([
                        "Model", 
                        html.H6([dbc.Badge("Running", color="success", className="mr-1", style={'visibility':'hidden'}, id='model-badge')], 
                        style={'display': 'inline', 'marginLeft':'5px'})], 
                    style={'color':'#036bfc', 'display': 'inline', 'marginRight': '20px'}),
                    dbc.Button(children=["Run"], color="success", className="mr-1", size="sm", style={'marginRight': '10px'}, id='model-run', n_clicks_timestamp=0),
                    dbc.Button(children=["Stop"], color="danger", className="mr-1", size="sm", style={'marginRight': '10px'}, id='model-stop'),
                ]),
                dbc.ModalBody([
                    html.P("Select model to train"),
                    dcc.Dropdown(
                        options=[
                            {'label':'Decision Tree Regressor', 'value': 'tree_regr'}
                        ],
                        value='tree_regr',
                        id='model-selection'
                    )
                ]),
                dbc.Button(
                        "Close", id="close-model", className="ml-auto", n_clicks=0, style={'float':'right', 'margin': '10px'}
                )
            ],
            id="modal-model",
            is_open=False,
        ),
        dbc.Modal(
            [
                dbc.ModalHeader([
                    html.H4([
                        "Model"], 
                        style={'display': 'inline', 'color':'#036bfc'})
                ]),
                dbc.ModalBody([
                    html.P("Output from model"),
                    html.P(id='output-console'),
                    dbc.Button("Sample", color="primary", disabled=True, id='sample-btn', n_clicks=0),
                    html.Div(id='sample-output', style={'marginTop':'10px'})
                ]),
                dbc.Button(
                        "Close", id="close-output", className="ml-auto", n_clicks=0, style={'float':'right', 'margin': '10px'}
                )
            ],
            id="modal-output",
            is_open=False,
        ),
    ]
)

@app.callback(
    dash.dependencies.Output('slider-label', 'children'),
    [dash.dependencies.Input('split-slider', 'value')])
def update_output(value):
    return "Train  {}% / Test  {}%".format(value, 100-value)

def parseContents(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), index_col=0)
        elif 'xls' in filename:
            df = pd.read_excel(io.StringIO(decoded))
        
        cols = df.columns
        table_header = [html.Thead(html.Tr([html.Th(col) for col in cols]))]

        rows = []
        for i in range(0,3):
            rows.append(html.Tr([html.Td(r) for r in df.loc[i, cols]]))
        
        table_body = [html.Tbody(rows)]
        table = dbc.Table(table_header + table_body, bordered=True,dark=True,hover=True,responsive=True,striped=True)
        globals()['df'] = df
    except Exception as e:
        return html.Div(['There as an error'])
    
    return html.Div(children=[table])

@app.callback([Output('preview', 'children'),Output('x_columns', 'options'),Output('y_column', 'options')],
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))

def display_file(content, name, date):
    if content is not None:
        parsed_contents = parseContents(content, name, date)
        return [parsed_contents, [{'label':col, 'value': col} for col in globals()['df']], [{'label':col, 'value': col} for col in globals()['df']]]
    
    return ['', [], []]

@app.callback(
    [Output("modal-data", "is_open"), Output("modal-feature", "is_open"), Output("modal-model", "is_open"), Output("modal-output", "is_open")],
    [Input('net', 'selection')]
)

def open_modal(x):
    global edge_mode, remove_node
    if edge_mode or remove_node:
        print('In edge/node add mode, no modal will be opened')
        return [False, False, False, False]

    if type(x) == type(None):
        return [False, False, False, False]

    s = None
    if len(x['nodes']) > 0 : s = str(x['nodes'][0])

    if s == 'node-data':
        return [True, False, False, False]
    elif s == 'node-feature':
        return [False, True, False, False]
    elif s == 'node-model':
        return [False, False, True, False]
    elif s == 'node-output':
        return [False, False, False, True]
    
    return [False, False, False, False]

@app.callback(
    [Output('net', 'data'), Output('output-console', 'children'), Output('sample-btn', 'disabled')],
    [Input('net', 'selection'),Input('data-run', 'n_clicks_timestamp'),Input('feature-run', 'n_clicks_timestamp'),Input('model-run', 'n_clicks_timestamp'),
    Input('split-slider', 'value'),Input('logger', 'n_clicks'),Input('viz', 'n_clicks'),Input('transform', 'n_clicks')],
    [State('x_columns', 'value'), State('y_column', 'value'), State('switches-input', 'value'), State('switches-input-1', 'value')]
)
def run_pipeline(selection,btn1,btn2,btn3,val,x_cols,y_col,run_down1,run_down2,nclicks1,nclicks2,nclicks3):
    global nodes,edges,run1,run2,run3,run4,edge_mode,remove_node,remove_edge,nodes_orig,edges_orig

    ctx = dash.callback_context

    if (ctx.triggered):
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if button_id == 'logger':
            nodes.append({'id': 'node-logger', 'label': 'Logger', 'size':15, 'shape': 'dot'})
            nodes_orig.append({'id': 'node-logger', 'label': 'Logger', 'size':15, 'shape': 'dot'})
        elif button_id == 'viz':
            nodes.append({'id': 'node-viz', 'label': 'Visualization', 'size':15, 'shape': 'dot'})
            nodes_orig.append({'id': 'node-viz', 'label': 'Visualization', 'size':15, 'shape': 'dot'})
        elif button_id == 'transform':
            nodes.append({'id': 'node-transform', 'label': 'Transformation', 'size':15, 'shape': 'dot'})
            nodes_orig.append({'id': 'node-transform', 'label': 'Transformation', 'size':15, 'shape': 'dot'})
    
    if type(selection) != type(None):
        #remove node
        if len(selection['nodes']) > 0 and remove_node:
            for i in range(len(nodes)):
                if nodes[i]['id'] == selection['nodes'][0]:
                    nodes.pop(i)
                    nodes_orig.pop(i)
                    break
            print('removed node', selection['nodes'][0])
            remove_node = False
        
        #remove edge
        if len(selection['edges']) > 0 and remove_edge:
            for i in range(len(edges)):
                if edges[i]['id'] == selection['edges'][0]:
                    edges.pop(i)
                    edges_orig.pop(i)
                    break
            print('removed edge', selection['edges'][0])
            remove_edge = False


        if len(selection['nodes']) > 1 and edge_mode:
            edge_mode = False
            print('adding edge...')
            #make sure edge doesn't exist
            node1,node2 = selection['nodes'][0],selection['nodes'][1]
            for e in edges:
                if e['from'] == node1 and e['to'] == node2:
                    print('edge already exists..')
                    return [{'nodes':nodes, 'edges':edges}, [], True]
            #new edge
            edges.append({'id': node1+'-'+node2, 'from':node1, 'to':node2, 'width':3})
            edges_orig.append({'id': node1+'-'+node2, 'from':node1, 'to':node2, 'width':3})
            print('added edge b/w ', node1, node2)
         

    if (int(btn1) > int(btn2) and int(btn1) > int(btn3)):
        #run data-node
        nodes = deepcopy(nodes_orig)
        edges = deepcopy(edges_orig)
        nodes[0]['color'] = 'green'
        globals()['pipeline'] = pipeline(globals()['df'], (1-val/100))
        print('Initiated pipeline...')
        if run_down1 == ["downstream_yes"]:
            print('running downstream from data node')
            run2 = True
            run3 = True
        else:
            return [{'nodes':nodes, 'edges':edges}, [], True]
    if (int(btn2) > int(btn1) and int(btn2) > int(btn3)) or run2:
        #run feature-node

        nodes[1]['color'] = 'green'
        if 'color' in nodes[0].keys():
            del nodes[0]['color']
        globals()['pipeline'].build_pipeline()
        print('Built pipeline...')  

        if run_down2 == ["downstream_yes"]:
            print('running downstream from feature node')
            run3 = True
        elif run2:
            run2 = False
        else:
            return [{'nodes':nodes, 'edges':edges}, [], True]
    if (int(btn3) > int(btn1) and int(btn3) > int(btn2)) or run3:
        #run model-node

        nodes[2]['color'] = 'green'
        nodes[3]['color'] = 'green'
        if 'color' in nodes[1].keys():
            del nodes[1]['color']
        sc,mae = globals()['pipeline'].fit_test(x_cols, [y_col])
        globals()['score'] = sc
        globals()['mae'] = mae
        print('model ran with score', sc, 'error', mae)

        output_console = [
            html.H5([dbc.Badge("Accuracy", className="ml-1", color="success", style={'marginRight':'20px'}), "{:.3f}%".format(sc*100)]),
            html.H5([dbc.Badge("Mean Absolute Error", className="ml-1", color="warning", style={'marginRight':'20px'}), "{:.3f}".format(mae)]),
        ]
        if run3:
            run3 = False
            nodes[1]['color'] = 'green'
            nodes[0]['color'] = 'green'
        return [{'nodes':nodes, 'edges':edges}, output_console, False]
    return [{'nodes':nodes, 'edges':edges}, [], True]

@app.callback(Output('callback-out', 'children'),Input('add_edge', 'n_clicks'))
def add_edge(n_clicks):
    global edge_mode
    print(datetime.now(),n_clicks)
    if n_clicks > 0:
        print('entering edge mode...')
        edge_mode = True

@app.callback(Output('callback-out1', 'children'),[Input('remove-node', 'n_clicks_timestamp'), Input('remove-edge', 'n_clicks_timestamp')])
def remove_net(btn1, btn2):
    global remove_edge, remove_node

    if int(btn1) > int(btn2):
        print('entering node removal..')
        remove_node = True
        remove_edge = False
    elif int(btn2) > int(btn1):
        print('entering edge removal..')
        remove_node = False
        remove_edge = True

@app.callback(
    Output('sample-output', 'children'),
    Input('sample-btn', 'n_clicks')
)
def sample_run(n_clicks):
    if n_clicks>0:
        x_test = globals()['pipeline'].x_test
        y_test = globals()['pipeline'].y_test
        y_pred = globals()['pipeline'].pipe.predict(x_test)
        error = np.abs(np.subtract(y_pred.reshape(-1,1), y_test))
        print(x_test.shape, y_test.shape, error.shape)
        df = pd.DataFrame(np.concatenate((np.round(x_test,2),np.round(y_test,2),np.round(y_pred.reshape(-1,1),2),np.round(error,2)), axis=1))
        cols = list(globals()['df'].columns) + ['y_pred','error']
        table_header = [html.Thead(html.Tr([html.Th(col) for col in cols]))]
        rows = []
        for i in range(0,5):
            rows.append(html.Tr([html.Td(r) for r in df.loc[i, :]]))
        table_body = [html.Tbody(rows)]
        table = dbc.Table(table_header + table_body, bordered=True,dark=True,hover=True,responsive=True,striped=True)
        return table
    return ""

if __name__ == "__main__":
    app.run_server(debug=True)
