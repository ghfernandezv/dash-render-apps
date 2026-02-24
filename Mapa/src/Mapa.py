import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

app = Dash(__name__)

# ========= DATA =========
df = pd.read_csv(
    "https://raw.githubusercontent.com/lihkir/Uninorte/main/AppliedStatisticMS/DataVisualizationRPython/Lectures/Python/PythonDataSets/intro_bees.csv"
)

df = (
    df.groupby(["State", "ANSI", "Affected by", "Year", "state_code"], as_index=False)[
        "Pct of Colonies Impacted"
    ]
    .mean()
)

app.layout = html.Div(
    [
        html.Div(
            [
                html.Img(
                    src="https://static.thenounproject.com/png/8180992-512.png",
                    style={'height': '60px', 'marginRight': '15px'}
                ),
                html.H1(
                    "Bee Colonies Impact Map",
                    style={'color': 'black', 'margin': '0'}
                ),
            ],
            style={
                'display': 'flex',
                'justifyContent': 'center',
                'alignItems': 'center',
                'gap': '15px',
                'marginBottom': '30px'
            }
        ),

        html.Div(
            [
                dcc.Dropdown(
                    id="slct_year",
                    options=[
                        {"label": "2015", "value": 2015},
                        {"label": "2016", "value": 2016},
                        {"label": "2017", "value": 2017},
                        {"label": "2018", "value": 2018},
                    ],
                    multi=False,
                    value=2015,
                    clearable=False,
                    style={'width': "40%"}
                ),
                dcc.Dropdown(
                    id="slct_affected",
                    options=[{"label": x, "value": x} for x in sorted(df["Affected by"].unique())],
                    value="Varroa_mites",
                    clearable=False,
                    style={'width': "40%"}
                ),
            ],
            style={
                'display': 'flex',
                'justifyContent': 'center',
                'gap': '20px',
                'marginBottom': '20px'
            }
        ),

        html.Div(
            id='output_container',
            children=[],
            style={
                'marginTop': '10px',
                'marginBottom': '10px',
                'textAlign': 'center',
                'fontFamily': 'Helvetica, Arial, sans-serif',
                'fontSize': '1.4rem',
                'fontWeight': '500',
                'color': 'black'
            }
        ),

        html.Br(),

        dcc.Graph(id='my_bee_map', figure={}),
        html.Hr(),

        html.Div(
            [
                dcc.Graph(id='bar_state', figure={}, style={'width': '33%', 'display': 'inline-block'}),
                dcc.Graph(id='bar_affected', figure={}, style={'width': '33%', 'display': 'inline-block'}),
                dcc.Graph(id='line_year', figure={}, style={'width': '33%', 'display': 'inline-block'}),
            ]
        ),
    ],
    style={
        'fontFamily': 'Helvetica, Arial, sans-serif',
        'color': 'purple'
    }
)

@app.callback(
    [
        Output('output_container', 'children'),
        Output('my_bee_map', 'figure'),
        Output('bar_state', 'figure'),
        Output('bar_affected', 'figure'),
        Output('line_year', 'figure'),
    ],
    [
        Input('slct_year', 'value'),
        Input('slct_affected', 'value'),
    ]
)
def update_graph(option_slctd, affected):

    container = f"The year chosen by user was: {option_slctd}"

    dff_year = df[(df["Year"] == option_slctd) & (df["Affected by"] == affected)]

    # --- Mapa ---
    fig = px.choropleth(
        data_frame=dff_year,
        locationmode='USA-states',
        locations='state_code',
        scope="usa",
        color='Pct of Colonies Impacted',
        hover_data=['State', 'Pct of Colonies Impacted'],
        color_continuous_scale=px.colors.sequential.Plasma,
        labels={'Pct of Colonies Impacted': '% of Bee Colonies'},
        template='plotly_white'
    )

    fig.update_layout(
        title_text=f"Bee Colonies Impacted (%): {affected} — {option_slctd}",
        title_x=0.5,
        font=dict(family="Helvetica, Arial, sans-serif", color="black"),
        paper_bgcolor="white",
        plot_bgcolor="white"
    )

    # --- Barras: Top States ---
    top_states = dff_year.sort_values('Pct of Colonies Impacted', ascending=False).head(10)

    fig_state = px.bar(
        top_states,
        x='State',
        y='Pct of Colonies Impacted',
        title='Top 10 States',
        template='plotly_white'
    )

    fig_state.update_traces(marker_color="#F4B400")

    fig_state.update_layout(
        title_x=0.5,
        title_font=dict(size=22, family="Helvetica, Arial, sans-serif", color="black"),
        font=dict(family="Helvetica, Arial, sans-serif", color="black"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(
            tickfont=dict(color="black"),
            title=dict(font=dict(color="black"))
        ),
        yaxis=dict(
            tickfont=dict(color="black"),
            title=dict(font=dict(color="black"))
        )
    )

    # --- Barras: resumen por Affected ---
    by_aff = df[df["Year"] == option_slctd].groupby("Affected by", as_index=False)["Pct of Colonies Impacted"].mean()

    fig_aff = px.bar(
        by_aff.sort_values("Pct of Colonies Impacted", ascending=False),
        x="Affected by",
        y="Pct of Colonies Impacted",
        title=f"Affected by (Year {option_slctd})",
        template='plotly_white'
    )

    plasma_colors = px.colors.sequential.Plasma
    fig_aff.update_traces(marker_color=plasma_colors)

    fig_aff.update_layout(
        title_x=0.5,
        title_font=dict(size=22, family="Helvetica, Arial, sans-serif", color="black"),
        font=dict(family="Helvetica, Arial, sans-serif", color="black"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(tickfont=dict(color="black")),
        yaxis=dict(tickfont=dict(color="black"))
    )

    # --- Línea: tendencia por año ---
    by_year = df[df["Affected by"] == affected].groupby("Year", as_index=False)["Pct of Colonies Impacted"].mean()

    fig_year = px.line(
        by_year,
        x="Year",
        y="Pct of Colonies Impacted",
        markers=True,
        title=f"Trend by Year: {affected}",
        template='plotly_white'
    )

    fig_year.update_traces(line=dict(color="#222222"))
    fig_year.update_layout(
        title_x=0.5,
        title_font=dict(size=22, family="Helvetica, Arial, sans-serif", color="black"),
        font=dict(family="Helvetica, Arial, sans-serif", color="black"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(tickfont=dict(color="black")),
        yaxis=dict(tickfont=dict(color="black"))
    )

    return container, fig, fig_state, fig_aff, fig_year


if __name__ == "__main__":
    ##app.run(debug=True)
    
    app.run(debug=True, host="0.0.0.0", port=9000)

