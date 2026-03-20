from dash import Dash, dcc, html, Input, Output
import colorlover as cl
import datetime as dt
import flask
import os
import pandas as pd
import time

app = Dash(__name__)

server = app.server

app.scripts.config.serve_locally = False

colorscale = cl.scales['9']['qual']['Paired']

df = pd.read_csv('https://raw.githubusercontent.com/lihkir/Uninorte/main/AppliedStatisticMS/DataVisualizationRPython/Lectures/Python/PythonDataSets/dash-stock-ticker-demo.csv')

app.layout = html.Div([
    html.Div([
        html.H2('Finance Explorer',
                style={'display': 'inline',
                       'float': 'left',
                       'font-size': '2.65em',
                       'margin-left': '7px',
                       'font-weight': 'bolder',
                       'font-family': 'Product Sans',
                       'color': "#e2e8f0",
                       'margin-top': '20px',
                       'margin-bottom': '0'
                       }),
        html.Img(src="https://s3-us-west-1.amazonaws.com/plotly-tutorials/logo/new-branding/dash-logo-by-plotly-stripe.png",
                style={
                    'height': '100px',
                    'float': 'right'
                },
        ),
    ]),
    dcc.Dropdown(
        id='stock-ticker-input',
        options=[{'label': s[0], 'value': str(s[1])}
                 for s in zip(df.Stock.unique(), df.Stock.unique())],
        value=['YHOO', 'GOOGL'],
        multi=True
    ),

        html.Hr(),

    html.Div([
        html.H4("Technical indicators (select 4)", style={'color': 'white'}),

        dcc.Dropdown(
            id='indicators',
            options=[
                {'label': 'Bollinger Bands (BB)', 'value': 'BB'},
                {'label': 'RSI', 'value': 'RSI'},
                {'label': 'MACD', 'value': 'MACD'},
                {'label': 'OBV', 'value': 'OBV'},
                {'label': 'Aroon Oscillator', 'value': 'AROON'}
            ],
            value=['BB', 'RSI', 'MACD', 'OBV'],
            multi=True
        ),

        html.Div([
            html.Div([
                html.Label("BB period", style={'color': '#e2e8f0'}),
                dcc.Dropdown(id='bb_period', options=[{'label': i, 'value': i} for i in [10, 15, 20, 30]], value=20, clearable=False),
            ], style={'width': '24%', 'display': 'inline-block', 'paddingRight': '10px'}),

            html.Div([
                html.Label("BB std", style={'color': '#e2e8f0'}),
                dcc.Dropdown(id='bb_std', options=[{'label': i, 'value': i} for i in [1, 2, 3]], value=2, clearable=False),
            ], style={'width': '24%', 'display': 'inline-block', 'paddingRight': '10px'}),

            html.Div([
                html.Label("RSI period", style={'color': '#e2e8f0'}),
                dcc.Dropdown(id='rsi_period', options=[{'label': i, 'value': i} for i in [7, 14, 21]], value=14, clearable=False),
            ], style={'width': '24%', 'display': 'inline-block', 'paddingRight': '10px'}),

            html.Div([
                html.Label("Aroon period", style={'color': '#e2e8f0'}),
                dcc.Dropdown(id='aroon_period', options=[{'label': i, 'value': i} for i in [14, 25, 50]], value=25, clearable=False),
            ], style={'width': '24%', 'display': 'inline-block'}),
        ], style={'marginTop': '10px'}),

        html.Div([
            html.Div([
                html.Label("MACD fast", style={'color': '#e2e8f0'}),
                dcc.Dropdown(id='macd_fast', options=[{'label': i, 'value': i} for i in [8, 12, 16]], value=12, clearable=False),
            ], style={'width': '32%', 'display': 'inline-block', 'paddingRight': '10px'}),

            html.Div([
                html.Label("MACD slow", style={'color': '#e2e8f0'}),
                dcc.Dropdown(id='macd_slow', options=[{'label': i, 'value': i} for i in [20, 26, 30]], value=26, clearable=False),
            ], style={'width': '32%', 'display': 'inline-block', 'paddingRight': '10px'}),

            html.Div([
                html.Label("MACD signal", style={'color': '#e2e8f0'}),
                dcc.Dropdown(id='macd_signal', options=[{'label': i, 'value': i} for i in [5, 9, 12]], value=9, clearable=False),
            ], style={'width': '32%', 'display': 'inline-block'}),
        ], style={'marginTop': '10px'}),

        html.Div(id='ind_warning', style={'color': 'crimson', 'marginTop': '8px'})
    ]),


    html.Div(id='graphs')
], className="container",
   style={
       'backgroundColor': '#0f172a',  # azul oscuro elegante
       'padding': '20px',
       'minHeight': '100vh'
   })


def bbands(price, window_size=10, num_of_std=5):
    rolling_mean = price.rolling(window=window_size).mean()
    rolling_std  = price.rolling(window=window_size).std()
    upper_band = rolling_mean + (rolling_std*num_of_std)
    lower_band = rolling_mean - (rolling_std*num_of_std)
    return rolling_mean, upper_band, lower_band
    
def rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def obv(close, volume):
    direction = close.diff().fillna(0).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (direction * volume).cumsum()

def aroon_oscillator(high, low, n=25):
    def _aroon_up(x):
        return 100 * (n - (n - x.argmax())) / n
    def _aroon_down(x):
        return 100 * (n - (n - x.argmin())) / n

    aroon_up = high.rolling(n+1).apply(_aroon_up, raw=True)
    aroon_down = low.rolling(n+1).apply(_aroon_down, raw=True)
    return aroon_up - aroon_down

@app.callback(
    Output('graphs','children'),
    [
        Input('stock-ticker-input', 'value'),
        Input('indicators', 'value'),
        Input('bb_period', 'value'),
        Input('bb_std', 'value'),
        Input('rsi_period', 'value'),
        Input('aroon_period', 'value'),
        Input('macd_fast', 'value'),
        Input('macd_slow', 'value'),
        Input('macd_signal', 'value'),
    ]
)

def update_graph(tickers, indicators, bb_period, bb_std, rsi_period, aroon_period, macd_fast, macd_slow, macd_signal):
    graphs = []
    indicators = indicators or []
    if len(indicators) > 4:
        indicators = indicators[:4]
        graphs.append(html.Div(
            "⚠️ Seleccionaste más de 4 indicadores. Se mostrarán solo los primeros 4.",
            style={'color': 'crimson', 'marginTop': '8px'}
        ))



    if not tickers:
        graphs.append(html.H3(
            "Select a stock ticker.",
            style={'marginTop': 20, 'marginBottom': 20}
        ))
    else:
        for i, ticker in enumerate(tickers):

            dff = df[df['Stock'] == ticker].copy()
            dff['Date'] = pd.to_datetime(dff['Date'])
            dff = dff.sort_values('Date')


            candlestick = {
                'x': dff['Date'],
                'open': dff['Open'],
                'high': dff['High'],
                'low': dff['Low'],
                'close': dff['Close'],
                'type': 'candlestick',
                'name': ticker,
                'legendgroup': ticker,
                'increasing': {'line': {'color': colorscale[0]}},
                'decreasing': {'line': {'color': colorscale[1]}}
            }

            bollinger_traces = []
            if 'BB' in indicators:
                bb_bands = bbands(dff.Close, window_size=bb_period, num_of_std=bb_std)
                bollinger_traces = [{
                    'x': dff['Date'], 'y': y,
                    'type': 'scatter', 'mode': 'lines',
                    'line': {'width': 1, 'color': colorscale[(i*2) % len(colorscale)]},
                    'hoverinfo': 'none',
                    'legendgroup': ticker,
                    'showlegend': True if i == 0 else False,
                    'name': '{} - bollinger bands'.format(ticker)
                } for i, y in enumerate(bb_bands)]



            extra_traces = []

            # RSI (0-100) en eje derecho
            if 'RSI' in indicators:
                r = rsi(dff['Close'], n=rsi_period)
                extra_traces.append({
                    'x': dff['Date'], 'y': r,
                    'type': 'scatter', 'mode': 'lines',
                    'yaxis': 'y2',
                    'line': {'width': 1},
                    'legendgroup': ticker,
                    'name': f'{ticker} RSI({rsi_period})'
                })

            # MACD en eje derecho
            if 'MACD' in indicators:
                m_line, s_line, _ = macd(dff['Close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
                extra_traces.append({
                    'x': dff['Date'], 'y': m_line,
                    'type': 'scatter', 'mode': 'lines',
                    'yaxis': 'y2',
                    'line': {'width': 1},
                    'legendgroup': ticker,
                    'name': f'{ticker} MACD'
                })
                extra_traces.append({
                    'x': dff['Date'], 'y': s_line,
                    'type': 'scatter', 'mode': 'lines',
                    'yaxis': 'y2',
                    'line': {'width': 1, 'dash': 'dot'},
                    'legendgroup': ticker,
                    'name': f'{ticker} MACD signal'
                })
            
            # OBV (si existe Volume)
            if 'OBV' in indicators and 'Volume' in dff.columns:
                o = obv(dff['Close'], dff['Volume'])
                extra_traces.append({
                    'x': dff['Date'], 'y': o,
                    'type': 'scatter', 'mode': 'lines',
                    'yaxis': 'y2',
                    'line': {'width': 1},
                    'legendgroup': ticker,
                    'name': f'{ticker} OBV'
                })
            
            # Aroon oscillator (-100..100)
            if 'AROON' in indicators:
                a = aroon_oscillator(dff['High'], dff['Low'], n=aroon_period)
                extra_traces.append({
                    'x': dff['Date'], 'y': a,
                    'type': 'scatter', 'mode': 'lines',
                    'yaxis': 'y2',
                    'line': {'width': 1},
                    'legendgroup': ticker,
                    'name': f'{ticker} AroonOsc({aroon_period})'
                })

            graphs.append(dcc.Graph(
                id=ticker,
                figure={
                    'data': [candlestick] + bollinger_traces + extra_traces,
                    'layout': {
                        'margin': {'b': 0, 'r': 60, 'l': 60, 't': 0},
                        'legend': {'x': 0},
                        'xaxis': {'rangeslider': {'visible': False}},
                        'yaxis2': {'overlaying': 'y', 'side': 'right', 'showgrid': False}
                    }
                }
            ))

    return graphs

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9000)