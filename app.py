import base64
import io
import pathlib

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

from dash.dependencies import Input, Output, State, ClientsideFunction
from plotly.express import bar
import plotly.express as px
import dash_table
import numpy as np


# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()

app = dash.Dash(__name__,
                meta_tags=[{"name": "viewport", "content": "width=device-width"}])
server = app.server


df_consumption = pd.read_csv('./data/consumption.csv')


FILENAME = "./data/raw.xlsx"
df = pd.read_excel(FILENAME)
df['Date'] = pd.to_datetime(df['Date'])


def get_bootstrap_rates(df, col_name, n_iterations=100):
    df = df[(df[col_name] <= np.quantile(df[col_name], 0.95)) & (df[col_name] >= np.quantile(df[col_name], 0.05))]
    n_samples = df.shape[0]
    means = []
    stdevs = []
    for i in range(n_iterations):
        samples = np.random.randint(0, n_samples, size=n_samples)
        temp = df.iloc[samples]
        means.append(temp[col_name].mean())
        stdevs.append(temp[col_name].std())
    means = np.asarray(means)
    stdevs = np.asarray(stdevs)
    result = {
        'Тип машины': df.index.unique()[0][0],
        'Тип работы': df.index.unique()[0][1],
        'Нижняя граница доверительного интервала среднего': np.quantile(means, 0.025),
        'Верхняя граница доверительного интервала среднего': np.quantile(means, 0.975),
        'Нижняя граница доверительного интервала стандартного отклонения': np.quantile(stdevs, 0.075),
        'Верхняя граница доверительного интервала стандартного отклонения': np.quantile(stdevs, 0.975)
    }
    return result


def generate_table(dataframe, max_rows=30):
    df_stats = dataframe.groupby(['Тип машины', 'Тип работы']).agg(
        {'Среднее потребление на работу': [np.mean, np.std, np.count_nonzero]})
    df_stats.columns = ['Среднее потребление топлива на работу', 'Стандартное отклонение потребления топлива на работу',
                        'Количество работ']
    df_index_bs = df_stats[df_stats['Количество работ'] >= 10].index
    dataframe = dataframe.set_index(['Тип машины', 'Тип работы'])
    df_bs = dataframe[dataframe.index.isin(df_index_bs)]
    stats_list = []
    for index in df_bs.index.unique():
        res = get_bootstrap_rates(df_bs.loc[index], col_name='Среднее потребление на работу')
        stats_list.append(res)

    add_stats = pd.DataFrame(stats_list)
    add_stats = add_stats.set_index(['Тип машины', 'Тип работы'])
    df_final_stats = pd.merge(df_stats, add_stats, how='left', left_index=True, right_index=True).iloc[:, :5]\
        .round(decimals=1)
    df_final_stats = df_final_stats.reset_index().fillna('Недостаточно данных')\
        .sort_values(by='Среднее потребление топлива на работу', ascending=False)
    return dash_table.DataTable(
        id='table',
        style_data={
            'whiteSpace': 'normal',
            'height': 'auto',
            'width': 'auto'
        },
        columns=[{'name': col, 'id': col} for col in df_final_stats.columns],
        data=df_final_stats.to_dict('records'),
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        column_selectable="single",
        row_selectable="multi",
        row_deletable=True,
        selected_columns=[],
        selected_rows=[],
        page_action="native",
        page_current=0,
        page_size=10
    )


dates = df['Date'].unique()
dates.sort()
dates_dict = {}
i = 1
for date in dates:
    dates_dict[i] = date
    i += 1


# LAYOUT
app.layout = html.Div(
    [html.Div(id="output-clientside"), # empty Div to trigger javascript file for graph resizing
    html.H2("Прогноз расхода для плана работ"),
    html.Div(
        [html.Div(
            [html.Div(className='pretty_container',
                      children=html.Div(
                               dcc.Upload(id='upload-data',
                                          children=html.Div(['Перетащите сюда или ',
                                                             html.A('загрузите файлы')])))),
            html.Div(
                    dcc.Dropdown(id='agg-dropdown',
                        options=[{'label': 'Вид работ', 'value': 'work'},
                                 {'label': 'Машина', 'value': 'machine'}],
                        value='work'),
                    className='pretty_container',),
            html.Div(
                dcc.Dropdown(id='precise-dropdown',
                             options=[{'label': 'все', 'value': 'all'}],
                             value='all'), className='pretty_container'),
            html.Center([
                html.H2('общий перерасход'),
                html.H2('-----', id='total-label', style={'font-weight': 'bold'}),
                html.H2('литров')],
                className='pretty_container')],
            className="three columns",
            id="cross-filter-options",
            ),
            html.Div(
                html.Div(
                    dcc.Graph(id="count_graph"),
                    id="countGraphContainer",
                    className="pretty_container",
                    style={'height': '600px'}),
                id="right-column",
                className="nine columns")],
        className="row flex-display",
    ),
    html.H2("Статистика расхода топлива"),
    html.H5(children='     '),
    html.Div(generate_table(df))
])

# CALLBACKS
app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="resize"),
    Output("output-clientside", "children"),
    [Input("count_graph", "figure")],
)


@app.callback([Output('total-label', 'children'),
              Output('count_graph', 'figure'),
               Output('precise-dropdown', 'options')],
              [Input('upload-data', 'contents'),
               Input('agg-dropdown', 'value'),
               Input('precise-dropdown', 'value')],
              [State('upload-data', 'filename'),
              State('upload-data', 'last_modified')])
def update_output(list_of_contents, agg_col, second_level, list_of_names, list_of_dates):

    if list_of_contents is None:
        return '-----', bar(), []

    content_type, content_string = list_of_contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    except Exception as e:
        print(e)
        return html.Div(['There was an error processing this file.'])

    df_calc = pd.merge(df_consumption, df, on=['machine', 'work'], how='inner').reset_index()

    second_level_values = [{'label': 'все', 'value': 'all'}, ] + [{'label': item, 'value': item} for item in df_calc.reset_index()[agg_col].unique()]

    if second_level == 'all':
        df_total = df_calc.groupby(agg_col).sum().reset_index().sort_values(agg_col)
        fig = bar(df_total, y=agg_col, x=["volume_planned", "volume_factual"],
                  labels={"volume_planned": "Расход топлива, план (л)",
                          "volume_factual": "Расход топлива,  факт (л)",
                          agg_col: "Машина/работа"},
                  title="Расход топлива", orientation='h')
                  # color_continuous_scale='reds',
                  # color="volume_planned")
        consumption = int(df_total['volume_factual'].sum() - df_calc['volume_planned'].sum())
        fig.update_layout(barmode='group', yaxis={'categoryorder': 'total ascending'})
    else:
        df_subtotal = df_calc[df_calc[agg_col] == second_level].sort_values(agg_col)
        fig = bar(df_subtotal, y="machine_number", x=["volume_planned", "volume_factual"],
                  labels={"volume_planned": "Расход топлива, план (л)",
                          "volume_factual": "Расход топлива,  факт (л)",
                          agg_col: "Машина/работа"},
                  title="Расход топлива", orientation='h')
                  # color_continuous_scale='reds',
                  # color="volume_planned")
        consumption = int(df_subtotal['volume_factual'].sum()) - int(df_subtotal['volume_planned'].sum())
        fig.update_layout(barmode='group', yaxis={'categoryorder': 'total ascending'})

    return str(consumption), fig, second_level_values


# Main
if __name__ == "__main__":
    app.run_server(port=8877, debug=True)
