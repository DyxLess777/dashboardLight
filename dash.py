from dash import Dash, dcc, html
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output
from plotly import graph_objects as go
import numpy as np

# Load data (CSV вместо Excel)
df = pd.read_csv("WEOApr2024all.xls", sep='\t', encoding='utf-16-le')

# Фильтрация строк по ключевым кодам WEO
keywords = [
    'NGDPD', 'NGDP_D', 'NGDPDPC', 'NID_NGDP', 'NGSD_NGDP', 'PCPI',
    'TM_RPCH', 'TX_RPCH', 'LUR', 'LE', 'LP', 'GGR', 'GGX',
    'GGXCNL', 'GGSB', 'GGXONLB', 'GGXWDN', 'GGXWDG', 'BCA'
]
filtered_df = df[df.apply(lambda row: any(keyword in str(row).upper() for keyword in keywords), axis=1)]
df = filtered_df.copy()

# Заданные индикаторы и годы
indicators_list = [
    'NGDPDPC', 'NID_NGDP', 'NGSD_NGDP', 'PCPI',
    'TM_RPCH', 'TX_RPCH', 'LUR', 'LE', 'LP',
    'GGR', 'GGX', 'GGXCNL', 'GGSB', 'GGXONLB',
    'GGXWDN', 'GGXWDG', 'BCA'
]
years = [str(year) for year in range(2010, 2023)]

# Сводим данные в длинный формат и фильтруем по показателям
df_melted1 = df.melt(id_vars=['Country', 'ISO', 'WEO Subject Code'],
                     value_vars=years,
                     var_name='Year', value_name='Value')
df_filtered = df_melted1[df_melted1['WEO Subject Code'].isin(indicators_list)]

# Преобразуем в широкий формат (pivot):contentReference[oaicite:7]{index=7}
df_wide = df_filtered.pivot_table(index=['Country', 'ISO', 'Year'],
                                  columns='WEO Subject Code',
                                  values='Value',
                                  aggfunc='first').reset_index()

# Конверсия колонок с данными в числовой тип
for col in indicators_list:
    if col in df_wide.columns:
        df_wide[col] = df_wide[col].astype(str).str.replace(',', '', regex=True)
        df_wide[col] = pd.to_numeric(df_wide[col], errors='coerce')

# Используем df_wide как базовые данные (Country, ISO, Year, ... индикаторы)
df_raw = df_wide.copy()

# Перевод в длинный формат для графиков
df_melted = df_raw.melt(id_vars=['Country', 'ISO', 'Year'],
                        var_name="Indicator", value_name="Value")

# Описания индикаторов
indicators = {
    "NGDPDPC": {"name": "ВВП на душу населения (текущие цены)", "comment": "Средний уровень дохода на человека."},
    "PCPI": {"name": "Индекс потребительских цен (CPI)", "comment": "Измеряет изменение стоимости потребкорзины."},
    "TM_RPCH": {"name": "Рост импорта (%)", "comment": "Темпы роста импорта товаров и услуг."},
    "TX_RPCH": {"name": "Рост экспорта (%)", "comment": "Изменение объема экспорта товаров и услуг."},
    "LP": {"name": "Численность населения (млн)", "comment": "Общее количество людей, проживающих в стране."},
    "LUR": {"name": "Безработица (%)", "comment": "Доля безработных в рабочей силе."},
    "LE": {"name": "Занятость (млн)", "comment": "Количество занятых в экономике."},
    "NID_NGDP": {"name": "Инвестиции (% от ВВП)", "comment": "Объем валовых инвестиций от ВВП."},
    "NGSD_NGDP": {"name": "Сбережения (% от ВВП)", "comment": "Доля национальных сбережений в ВВП."},
    "GGR": {"name": "Госдоходы (млрд)", "comment": "Доходы государственного сектора."},
    "GGX": {"name": "Госрасходы (млрд)", "comment": "Объем государственных расходов."},
    "GGXCNL": {"name": "Чистое кредитование (%)", "comment": "Разница госдоходов и расходов (% ВВП)."},
    "GGSB": {"name": "Структурный баланс (млрд)", "comment": "Бюджетный баланс с учетом цикла."},
    "BCA": {"name": "Баланс текущего счета (млрд USD)", "comment": "Баланс экспорта и импорта."}
}

groups = {
    "Макроэкономика и рост": ["NGDPDPC", "PCPI", "TM_RPCH", "TX_RPCH"],
    "Рынок труда и население": ["LP", "LE", "LUR"],
    "Финансовая устойчивость и гос. сектор": ["NID_NGDP", "NGSD_NGDP", "GGR", "GGX", "GGXCNL", "GGSB", "BCA"],
    "Аналитика": []
}

# Для выбора топ-10 стран
latest_year = df_melted['Year'].max()
def get_top10(ind):
    temp = df_melted[(df_melted['Indicator'] == ind) & (df_melted['Year'] == latest_year)]
    return temp.sort_values(by='Value', ascending=False).head(10)['Country'].unique().tolist()

countries = sorted(df_melted['Country'].dropna().unique())

# Выбросы: считаем Z-оценки и подсчитываем >3 (IQR/межквартильный размах)
numeric_cols = [c for c in df_raw.columns if c not in ['Country','ISO','Year'] and pd.api.types.is_numeric_dtype(df_raw[c])]
df_numeric = df_raw[numeric_cols]
z_scores = ((df_numeric - df_numeric.mean()) / df_numeric.std()).abs()
outliers = (z_scores > 3).sum().reset_index()
outliers.columns = ['Indicator', 'OutlierCount']

# Запускаем Dash приложение
app = Dash(__name__)
app.layout = html.Div([
    html.H1("Экономический дашборд МВФ (WEO 2024)", style={"textAlign": "center"}),
    html.H3("Links to Github – github.com/DyxLess777/dashboardLight", style={"textAlign": "center"}),
    dcc.Tabs([
        # Вкладки с показателями
        dcc.Tab(label=group_name, children=[
            html.Div([
                *[html.Div([
                    html.H1(f"{indicators[ind]['name']}", style={"margin-top": "20px"}),
                    html.P(indicators[ind]['comment'], style={"font-style": "italic"}),
                    html.Label(f"Выберите страны для {indicators[ind]['name']}:"),
                    dcc.Dropdown(
                        id=f"dropdown-{ind}",
                        options=[{"label": country, "value": country} for country in countries],
                        value=get_top10(ind),
                        multi=True
                    ),
                    dcc.Graph(id=f"graph-{ind}")
                ]) for ind in groups[group_name]]
            ])
        ]) if group_name != "Аналитика" else
        # Вкладка "Аналитика"
        dcc.Tab(label="Аналитика", children=[
            html.Div([
                html.H1("1. Корреляционная матрица по странам", style={"margin-top": "20px"}),
                html.Label("Выберите страну (или 'Все страны'):"),
                dcc.Dropdown(
                    id='corr-country-dropdown',
                    options=[{"label": c, "value": c} for c in countries] + [{"label": "Все страны", "value": "all"}],
                    value="all",
                    multi=False
                ),
                dcc.Graph(id='corr-matrix-graph'),
                html.Div(id='top-correlated')
            ]),
            html.Div([
                html.H1('2. Интерактивная карта экономических показателей', style={"margin-top": "20px"}),
                html.Div(
                    dcc.Graph(
                        figure=px.choropleth(
                            df_wide,
                            locations='ISO',
                            locationmode='ISO-3',
                            color='NGDPDPC',
                            hover_name='Country',
                            hover_data={key: True for key in numeric_cols},
                            color_continuous_scale=['white', 'green'],
                            animation_frame='Year',
                            projection='mercator',
                            width=1200,
                            height=700
                        ).update_layout(
                            geo=dict(showframe=False, showcoastlines=False),
                            showlegend=False
                        )
                    ),
                    style={'display': 'flex', 'justifyContent': 'center', 'width': '100%'}
                )
            ]),
            html.Div([
                html.H1("3. Коэффициент вариации (CV)", style={"margin-top": "20px"}),
                html.P("CV = std/mean для каждого показателя:contentReference[oaicite:8]{index=8}:"),
                html.Div(id="cv-values")
            ]),
            html.Div([
                html.H1("4. Кластеризация стран (PCA + KMeans)", style={"margin-top": "20px"}),
                dcc.Graph(id="pca-kmeans")
            ]),
            html.Div([
                dcc.Input(id='dummy', value='init', type='hidden'),
                html.H1("5. Тепловая карта пропущенных значений"),
                dcc.Graph(id='missing-heatmap')
            ]),
            html.Div([
                html.H1("6. Выбросы (IQR)", style={"margin-top": "20px"}),
                dcc.Graph(figure=px.bar(outliers, x='Indicator', y='OutlierCount',
                                        title='Количество выбросов (|Z|>3) по показателям'))
            ])
        ]) for group_name in groups
    ])
])

# Колбэки для каждого графика по показателю
for group_name in groups:
    for ind in groups[group_name]:
        @app.callback(
            Output(f"graph-{ind}", "figure"),
            Input(f"dropdown-{ind}", "value")
        )
        def update_graph(selected_countries, ind=ind):
            df_plot = df_melted[(df_melted['Country'].isin(selected_countries)) & (df_melted['Indicator'] == ind)]
            fig = px.line(df_plot, x="Year", y="Value", color="Country", title=indicators[ind]['name'])
            return fig

# Корреляционная матрица и наиболее скоррелированные показатели
@app.callback(
    [Output("corr-matrix-graph", "figure"),
     Output("top-correlated", "children")],
    Input("corr-country-dropdown", "value")
)
def update_corr_matrix(selected_country):
    # Фильтрация данных
    if selected_country == "all":
        df_sel = df_melted[df_melted['Indicator'].isin(indicators.keys())]
    else:
        df_sel = df_melted[(df_melted['Country'] == selected_country) &
                            (df_melted['Indicator'].isin(indicators.keys()))]
    # Сводим в таблицу (Year x Indicator)
    df_pivot = df_sel.pivot_table(index="Year", columns="Indicator", values="Value")
    df_pivot = df_pivot.dropna(axis=1, thresh=int(len(df_pivot) * 0.6))
    corr_matrix = df_pivot.corr()

    # Находим пару с максимальной корреляцией
    corr_copy = corr_matrix.copy()
    np.fill_diagonal(corr_copy.values, np.nan)
    corr_pairs = corr_copy.abs().unstack().dropna()
    corr_pairs = corr_pairs[corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)]
    corr_pairs = corr_pairs.sort_values(ascending=False)
    if not corr_pairs.empty:
        ind1, ind2 = corr_pairs.index[0]
        corr_value = corr_matrix.loc[ind1, ind2]
        name1 = indicators[ind1]['name']
        name2 = indicators[ind2]['name']
        comment1 = indicators[ind1]['comment']
        comment2 = indicators[ind2]['comment']
        top_text = html.Div([
            html.H4("Наиболее скоррелированные показатели:"),
            html.P(f"{name1} ↔ {name2}, корреляция: {corr_value:.2f}")
        ])
    else:
        top_text = html.Div([html.H4("Недостаточно данных для оценки корреляции.")])

    # Построение тепловой карты
    text_vals = np.round(corr_matrix.values, 2).astype(str)
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=[indicators[i]['name'] for i in corr_matrix.columns],
        y=[indicators[i]['name'] for i in corr_matrix.index],
        colorscale='RdYlGn', zmin=-1, zmax=1,
        colorbar=dict(title="Корреляция"),
        text=text_vals, hoverinfo='text', showscale=True,
        texttemplate="%{text}", textfont=dict(size=12, color="black")
    ))
    fig.update_layout(title="Матрица корреляции показателей", height=700)
    return fig, top_text

# Коэффициент вариации (CV) для каждого показателя
@app.callback(
    Output('cv-values', 'children'),
    Input('corr-country-dropdown', 'value')
)
def cv_analysis(_):
    cv_vals = {}
    for ind in indicators:
        values = df_melted[df_melted['Indicator'] == ind]['Value']
        if len(values) > 0:
            std = values.std()
            mean = values.mean()
            cv = (std / mean) if mean != 0 else 0
        else:
            cv = 0
        cv_vals[ind] = cv
    fig = px.bar(
        x=[indicators[i]['name'] for i in indicators],
        y=list(cv_vals.values()),
        labels={'x': 'Индикатор', 'y': 'CV'},
        title="Коэффициент вариации для каждого показателя"
    )
    return dcc.Graph(figure=fig)

# PCA (SVD) + KMeans (имплементация вручную)
@app.callback(
    Output('pca-kmeans', 'figure'),
    Input('corr-country-dropdown', 'value')
)
def pca_kmeans_analysis(_):
    df_pivot = df_melted.pivot_table(index='Country', columns='Indicator', values='Value')
    df_pivot = df_pivot.dropna(axis=1, thresh=int(len(df_pivot) * 0.6))
    df_imputed = df_pivot.fillna(df_pivot.mean())
    df_scaled = (df_imputed - df_imputed.mean()) / df_imputed.std()

    # PCA через SVD:contentReference[oaicite:9]{index=9}
    X = df_scaled.values
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    pca_components = U[:, :2] * S[:2]

    # KMeans вручную (K=3):contentReference[oaicite:10]{index=10}
    Xdata = df_scaled.values
    K = 3
    np.random.seed(0)
    # Инициализируем центры случайно
    initial_idx = np.random.choice(range(Xdata.shape[0]), K, replace=False)
    centroids = Xdata[initial_idx]
    for _ in range(100):
        # Присвоение кластеров по ближайшему центроиду
        distances = np.linalg.norm(Xdata[:, None] - centroids[None, :], axis=2)
        clusters = np.argmin(distances, axis=1)
        # Обновление центроидов
        new_centroids = np.array([Xdata[clusters == k].mean(axis=0) if np.any(clusters==k) else centroids[k] for k in range(K)])
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    fig = px.scatter(
        x=pca_components[:, 0], y=pca_components[:, 1],
        color=clusters.astype(str),
        text=df_scaled.index,
        labels={"x": "PCA Комп-1", "y": "PCA Комп-2"},
        title="Кластеризация стран (PCA+KMeans)"
    )
    return fig

# Тепловая карта пропусков (количество пустых по странам)
@app.callback(
    Output('missing-heatmap', 'figure'),
    Input('dummy', 'value')
)
def plot_missing_treemap(_):
    missing = df_melted[df_melted['Value'].isnull()]
    missing_count = missing.groupby('Country').size().reset_index(name='missing_count')
    fig = px.treemap(
        missing_count,
        path=['Country'],
        values='missing_count',
        color='missing_count',
        color_continuous_scale='Reds',
        title="Пропущенные значения по странам"
    )
    fig.update_layout(height=600)
    return fig

if __name__ == '__main__':
    app.run(debug=True)
