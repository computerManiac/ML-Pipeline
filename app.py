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

FA = "https://use.fontawesome.com/releases/v5.12.1/css/all.css"

app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP, FA],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, height=device-height, initial-scale=1"}
    ]
)

nodes = [{'id': 'node-data', 'label': 'Data', 'size':15, 'shape': 'dot'},
        {'id': 'node-feature', 'label': 'Features', 'size':15, 'shape': 'dot'},
        {'id': 'node-model', 'label': 'Model', 'size':15, 'shape': 'dot'},
        {'id': 'node-output', 'label': 'Output', 'size':15, 'shape': 'dot'},
]

edges = [{'id': 'data-feature', 'from':'node-data', 'to':'node-feature', 'width':3},
        {'id': 'feature-model', 'from':'node-feature', 'to':'node-model', 'width':3},
        {'id': 'model-output', 'from':'node-model', 'to':'node-output', 'width':3}
]


app.layout = html.Div(
    children=[
        html.Div(children=[visdcc.Network(id = 'net', data={'nodes': nodes, 'edges': edges}, 
                                        options={'height': '750px', 'width':'100%', 'edges':{
                                            'arrows':{
                                                'to': {'enabled': True, 'type': 'arrow'}
                                            }
                                        }}),], 
                style={'height':'100vh', 'width':'100vw', 'overflow':'hidden', 'zIndex':-1}),
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
                        value=['downstream_yes'],
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
                        value=['downstream_yes'],
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
    [Input('data-run', 'n_clicks_timestamp'),Input('feature-run', 'n_clicks_timestamp'),Input('model-run', 'n_clicks_timestamp'),
    Input('split-slider', 'value')],
    [State('x_columns', 'value'), State('y_column', 'value'),]
)
def run_pipeline(btn1,btn2,btn3,val,x_cols,y_col):
    global nodes,edges
    if int(btn1) > int(btn2) and int(btn1) > int(btn3):
        #run data-node
        nodes[0]['color'] = 'green'
        globals()['pipeline'] = pipeline(globals()['df'], (1-val/100))
        print('Initiated pipeline...')
        return [{'nodes':nodes, 'edges':edges}, [], True]
    elif int(btn2) > int(btn1) and int(btn2) > int(btn3):
        #run feature-node
        nodes[1]['color'] = 'green'
        if 'color' in nodes[0].keys():
            del nodes[0]['color']
        globals()['pipeline'].build_pipeline()
        print('Built pipeline...')  
        return [{'nodes':nodes, 'edges':edges}, [], True]
    elif int(btn3) > int(btn1) and int(btn3) > int(btn2):
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

        return [{'nodes':nodes, 'edges':edges}, output_console, False]
    return [{'nodes':nodes, 'edges':edges}, [], True]

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