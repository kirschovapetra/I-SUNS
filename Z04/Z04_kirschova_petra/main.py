import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# zanre + tagy
genres_tags = [
    # tagy - zanre
    'action', 'rpg', 'software', 'adventure', 'indie', 'education', 'strategy',
    'simulation', 'sports', 'early_access', 'racing', 'multiplayer',
    # tagy - nove
    'singleplayer', 'arcade', 'classic', 'fantasy', 'historical', 'horror', 'puzzle',
    'sci_fi', 'shooter', 'survival', 'vr', '2d', '3d', 'addictive', 'based_on_a_novel',
    'difficult', 'great_soundtrack', 'masterpiece'
]

# zanre
genres_only = ['action', 'rpg', 'software', 'adventure', 'indie', 'education', 'strategy', 'simulation', 'sports',
               'racing', 'arcade', 'fantasy', 'historical', 'horror', 'puzzle', 'sci_fi', 'shooter', 'survival']


''' ************************ Uprava datasetu ************************ '''


def get_avg_from_interval(column, delim):
    """
    Ziskanie priemernych hodnot z intervalov v stlpci v datasete
    :param column: stlpec z datasetu
    :param delim: oddelovaci znak
    :return: novy stlpec s priemernymi honotami
    """
    out = []
    for item in column:
        # rozdelenie stringu
        values = item.split(delim)
        # priemer
        values_sum = int(values[0]) + int(values[1])
        out.append(int(values_sum / 2))
    return out


def get_years(column, delim):
    """
    Ziskanie rokov z datumov v stlpci v datasete
    :param column: stlpec v datasete
    :param delim: oddelovaci znak
    :return: novy stlpec s rokmi
    """
    out = []
    for item in column:
        # rozdelenie stringu
        values = item.split(delim)
        # rok
        out.append(int(values[0]))
    return out


def get_unique_genres(column):
    """
    Ziskanie unikatnych hodnot stlpca v datasete
    :param column: stlpec v datasete
    :return: mnozina s unikatnymi hodnotami
    """
    column = column.str.lower()

    unique = set()
    for entry in column:
        # oddelenie stringu - v 1 bunke moze byt viacero hodnot oddelenych ';'
        unique.update(entry.split(";"))
    return unique


def merge_genres(column, original_values, result_value):
    """
    Funkcia na spojenie viacerych zanrov do jedneho
    :param column: stlpec z datasetu
    :param original_values: povodne nazvy zanrov
    :param result_value: hodnota, ktorou sa vsetky original_values nahradia
    :return: upraveny stlpec
    """

    new_col = column.copy()
    for indx in column.index:
        # vsetky zaznamy z column, ktore nadobudaju niektoru hodnotu z original_values
        if [value for value in original_values if (value in column[indx])]:
            # odstrania sa z konkretnej bunky vsetky vyskyty niektorej hodnoty z original_values
            cell = [cell_value for cell_value in column[indx].split(";") if cell_value not in original_values]
            # do bunky sa appendne vysledna hodnota
            cell.append(result_value)
            # ulozi sa string s hodnotami oddelenymi ';'
            new_col[indx] = ";".join(cell)

    return new_col


def genre_binary_df(data):
    """
    Vytvorenie noveho dataframe z unikatnych zanrov
    :param data: dataset
    :return novy dataframe {appid, value_1,...,value_n}, stlpce nadobudaju hodnoty 0/1
    """

    print("\n\n************** Genres **************")

    # dataframe so stlpcom appid
    new_df = pd.DataFrame({"appid": data.appid})

    # unikatne zanre
    unique_genres = get_unique_genres(data["genres"])

    # prechadza sa cez vsetky unikatne hodnoty
    for genre in unique_genres:
        print(".", end="")

        # najskor su v stlpci iba 0
        col = [0] * data.shape[0]

        for indx in data.index:
            # zapise sa 1 ak sa hodnota nachadza v data['genres'][index]
            if genre in data["genres"].array[indx]:
                col[indx] = 1

        # prida sa novy stlpec pre zaner
        new_df[genre] = col

    return new_df


def get_genres_tags_df(main_df, tag_df):
    """
    Ziskanie dataframe zlozeneho zo zanrov a tagov
    :param main_df: zakladny dataset
    :param tag_df: dataset tagov
    :return: novy dataset so zanrami a tagmi
    """
    main_df["genres"] = main_df["genres"].str.lower()

    # mergnutie zanrov do 1 zanru - software
    to_merge = ['utilities', 'game development', 'web publishing', 'animation & modeling',
                'design & illustration', 'audio production', 'video production', 'photo editing']
    main_df["genres"] = merge_genres(main_df["genres"], to_merge, 'software')

    # ziskanie dataframe pre zanre s hodnotami 0/1
    genres_df = genre_binary_df(main_df)

    # premenovanie stlpcov, aby sedeli s tagmi
    genres_df = genres_df.rename(columns={"early access": "early_access", "massively multiplayer": "multiplayer"})

    # odstranenie npotrebnych stlpcov
    genres_df.drop(columns=["nudity", "free to play", "gore", "violent", "sexual content",
                            'tutorial', 'software training', "accounting", "casual", "documentary"],
                   inplace=True)

    # spojenie dataframe zanrov a tagov. stlpce pre tagy budu mat suffix "_tag", ak sa stlpce opakuju
    genres_tags_df = genres_df.join(tag_df[genres_tags], rsuffix="_tag")

    for genre in genres_df.columns.drop("appid"):
        # ak je pocet tagnuti s 1 tagom pre hru vacsi ako 0, tak sa zaznam v genres_tags_df[genre]
        # prepise na hodnotu z genres_tags_df[<genre>_tag]
        genres_tags_df[genre] = np.where(genres_tags_df[genre + "_tag"] > 0,        # podmienka
                                         genres_tags_df[genre + "_tag"],            # co sa zapise
                                         genres_tags_df[genre])                     # kam sa zapise

        # ak hra ma priradeny zaner zo zakladneho datasetu, ale v tagoch sa zaner nenachadza,
        # priradi sa do genres_tags_df[genre] priemer zo stlpca genres_tags_df[<genre>_tag]
        genres_tags_df[genre] = np.where((genres_tags_df[genre] == 1) &
                                         (genres_tags_df[genre + "_tag"] == 0),             # podmienka
                                         int(genres_tags_df[genre + "_tag"].mean()),        # co sa zapise
                                         genres_tags_df[genre])                             # kam sa zapise

    # vymazu sa vsetky stlpce, ktore maju v nazve '_tag'
    genre_tag_cols = [i for i in genres_tags_df.columns.values if "_tag" in i]
    genres_tags_df.drop(columns=genre_tag_cols, inplace=True)

    return genres_tags_df


def get_publisher_game_count(data):
    """
    Ziskanie poctu hier pre kazdeho publishera
    :param data: dataset
    :return: novy stlpec s poctom hier
    """
    print("\n\n************ publisher game count **************")

    # vytvorenie grup podla publishera, ulozenie velkosti grup
    groups_count_all = data.groupby("publisher").size()

    games = [0] * data.shape[0]
    for indx in data.index:
        # vypis
        if indx % 500 == 0:
            print('.', end='')
        # ulozenie poctu hier
        games[indx] = groups_count_all[data["publisher"][indx]]

    return games


def get_publisher_avg(data, colname):
    """
    Ziskanie priemeru data[colname] pre kazdeho publishera
    :param data: dataset
    :param colname: nazov stlpca, v ktorom sa bude robit priemer
    :return: novy stlpec s priemermi
    """
    print("\n\n************ publisher " + colname + " avg **************")

    # vytvorenie grup podla publishera, ulozenie priemeru grup
    groups_avg_all = data.groupby("publisher")[colname].mean()

    avg = [0] * data.shape[0]
    for indx in data.index:
        # vypis
        if indx % 500 == 0:
            print('.', end='')
        # ulozenie priemeru
        avg[indx] = groups_avg_all[data["publisher"][indx]]

    return avg


def get_genre_lists(data):
    """
    Vrati string vsetkych zanrov, ktore ma dana hra (po uprave datasetu)
    :param data: dataset
    """
    genre_lists = []
    for indx in data.index:  # prechadzaju sa riadky
        if indx % 500 == 0:
            print('.', end='')

        genre_list_row = []
        for colname in genres_tags:  # prechadzaju sa stlpce
            if data[colname][indx] > 0:
                genre_list_row.append(colname)  # ak ma hra dany zaner, zaner sa prida do genre_list_row

        # finalny string
        genre_lists.append(";".join(genre_list_row))

    return genre_lists


def delete_outliers(data):
    """
    Vymazanie outlierov
    :param data: dataset
    """
    data.dropna(inplace=True)  # null hodnoty

    data.drop(data[data.average_playtime > 20000].index, inplace=True)  # priemerny cas hrania
    data.drop(data[data.owners > 10000000].index, inplace=True)  # pocet vlastnikov
    data.drop(data[data.price > 90].index, inplace=True)  # cena
    data.drop(data[data.publisher_avg_owners > 8000000].index, inplace=True)  # priemer vlastnikov publisherovych hier

    # plot_scatter_plots(data)


def fix_dataset(data, tags):
    # priemer vlastnikov namiesto intervalu
    data["owners"] = get_avg_from_interval(data["owners"], "-")

    # rok namiesto datumu
    data["release_year"] = get_years(data["release_date"], "-")

    # hodnotenie - vzorec: https://steamdb.info/blog/steamdb-rating/
    sum_ratings = data["positive_ratings"] + data["negative_ratings"]
    avg_ratings = data["positive_ratings"] / sum_ratings
    data["ratings"] = avg_ratings - (avg_ratings - 0.5) * (2 ** -np.log10(sum_ratings + 1))

    # publisher: pocet hier, priemerny pocet vlastnikov, priemerne hodnotenie hier
    data["publisher_games_count"] = get_publisher_game_count(data)
    data["publisher_avg_owners"] = pd.Series(get_publisher_avg(data, "owners")).astype('int')
    data["publisher_avg_rating"] = get_publisher_avg(data, "ratings")

    # spojenie datasetov
    data = data.join(get_genres_tags_df(steam, tags), rsuffix='_genres')

    # stlpec genres bude obsahovat vsetky zanre hry oddelene ;
    data["genres"] = get_genre_lists(data)

    # vymazanie nepotrebnych stlpcov
    data.drop(columns=["appid_genres", "release_date", "developer", "platforms", "categories", "positive_ratings",
                       "negative_ratings", "median_playtime", "achievements", "english", "required_age"],
              inplace=True)

    return data


''' ****************************** Grafy **************************** '''


def plot_correlation_matrix(data, fontsize):
    """
    Vykreslenie korelacnej matice
    :param fontsize: velkost pisma
    :param data: dataset
    """

    # vypocet korelacnej matice
    correlation = data.corr().round(2)

    # heatmap
    lst = correlation.columns.values.tolist()
    fig = ff.create_annotated_heatmap(correlation.values, x=lst, y=lst)
    fig.layout.font.size = fontsize
    fig.show()


def plot_scatter_plot(data, col):
    """
    Vykreslenie scatter grafu pre 1 stlpec z datasetu
    :param data: dataset
    :param col: nazov stlpca
    """
    fig = px.scatter(data, x='appid', y=col)
    fig.show()


def plot_scatter_plots(data):
    """
    Vykreslenie scatter grafov pre stlpce v datasete
    :param data: dataset
    """
    plot_scatter_plot(data, "average_playtime")
    plot_scatter_plot(data, "owners")
    plot_scatter_plot(data, "price")
    plot_scatter_plot(data, "ratings")
    plot_scatter_plot(data, "publisher_avg_owners")
    plot_scatter_plot(data, "publisher_avg_rating")


def plot_clustering_bar_charts(data, rows, cols, title_text):
    """
    Vykreslenie clusterov do bar grafov
    :param data: cely dataset
    :param rows: pocet riadkov
    :param cols: pocet stlpcov
    :param title_text: nadpis grafu
    """

    # subploty do mriezky rows x cols
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=data.columns.drop("cluster"))
    row = 1
    col = 1
    for colname in data.columns.drop("cluster"):
        # priemerna hodnota v danom clusteri pre kazdy stlpec z data
        cluster_avg_all = data.groupby("cluster").mean()

        # stlpce, ktore sa budu konvertovat na integer
        int_columns = ["cluster_size", "owners", "release_year", "publisher_games_count", "publisher_avg_owners"]
        # float stlpce
        float_columns = cluster_avg_all.columns.drop(int_columns)

        # konvertovanie, zaokrúhlenie
        cluster_avg_all[int_columns] = cluster_avg_all[int_columns].astype(int)
        cluster_avg_all[float_columns] = cluster_avg_all[float_columns].round(2)

        # bar graf
        fig.add_trace(
            go.Bar(x=cluster_avg_all.index, y=cluster_avg_all[colname], name=colname, text=cluster_avg_all[colname]),
            row=row, col=col)

        # posun riadkov a stlpcov
        if col == cols:
            row = row + 1
            col = 1
        else:
            col = col + 1

    fig.update_xaxes(type='category')
    fig.update_traces(textposition='auto', textfont=dict(color="black"))
    fig.update_layout(height=2500, showlegend=False, title_text=title_text)
    fig.show()


''' ****************************** Grafy - EDA **************************** '''


def top_10_bar_graph(data, colname_x, colname_y, title_text):
    """
    Bar graf s top 10 zaznamov pre atribut colname_y
    :param data: dataset
    :param colname_x: hodnoty na osi x - nazov stlpca v datasete
    :param colname_y: hodnoty na osi y - nazov stlpca v datasete
    :param title_text: nadpis
    """

    # pri publisheroch sa odstrania duplikatne zaznamy
    if colname_y == "publisher":
        data = data[[colname_y, colname_x]].drop_duplicates()

    # zoberie sa top 20 zaznamov zo zoradeneho datasetu podla colname_x
    data_to_plot = data.sort_values(by=[colname_x], ascending=False).head(10)

    # bar plot
    fig = px.bar(data_to_plot, x=colname_x, y=colname_y, color=colname_y, text=data_to_plot[colname_x])
    fig.update_layout(showlegend=False, title_text=title_text,
                      title=dict(yanchor="top", y=0.99, xanchor="left", x=0.38,
                                 font=dict(size=25)))
    fig.update_traces(textposition='auto', textfont=dict(color="black"))
    fig.show()


def genres_pie_graph(data, genre_list, title_text):
    """
    Pie chart percentualneho zastupenia zanrov medzi hrami
    :param data: dataset
    :param genre_list: zoznam zanrov
    :param title_text: nadpis
    """
    df = pd.DataFrame(index=genre_list, columns=["percent"])

    # vypocitanie percent pre kazdy zaner
    for genre in genre_list:
        count = data[data[genre] > 0][genre].count()
        percent = count / data.shape[0]
        df["percent"][genre] = percent

    # pie chart
    fig = px.pie(df, values="percent", names=genre_list, title=title_text)
    fig.update_layout(legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.01),
                      title=dict(yanchor="top", y=0.99, xanchor="left", x=0.38, font=dict(size=25)))
    fig.show()


def genres_tags_bar_graph(data, colname_y, title_text):
    """
    Bar graf pre zavislost zanrov od atributu colname_y
    :param data: dataset
    :param colname_y: hodnoty na osi y - stlpec z datasetu
    :param title_text: nadpis
    """
    indx = genres_tags  # index noveho dataframe = zanre + tagy

    if colname_y in indx:
        indx.remove(colname_y)

    # index = zanre a tagy, 1 stlpec = parameter colname_y
    df = pd.DataFrame(index=indx, columns=[colname_y])

    # priemer pre kazdy zaner
    for genre in indx:
        df[colname_y][genre] = data[data[genre] > 0][colname_y].mean()

    # bar graf
    fig = px.bar(df, x=indx, y=colname_y, color=indx)
    fig.update_xaxes(type='category')
    fig.update_yaxes(title_text=colname_y)
    fig.update_layout(showlegend=False, title_text=title_text,
                      title=dict(yanchor="top", y=0.99, xanchor="left", x=0.38, font=dict(size=25)))
    fig.show()


def singleplayer_multiplayer_bar_graph(data, title_text):
    """
    Bar graf s viacerymi stlpcami pre kazdy zaner
    :param data: dataset
    :param title_text: nadpis
    """
    indx = genres_only  # index noveho dataframe = zanre

    colnames_y = ["singleplayer", "multiplayer"]  # nazvy stlpcov

    for genre in colnames_y:
        if genre in indx:
            indx.remove(genre)

    # index = zanre, viacero stlpcov
    df = pd.DataFrame(index=indx, columns=colnames_y)

    bars = []  # bar grafy
    for colname in colnames_y:
        for genre in indx:
            df[colname][genre] = data[data[genre] > 0][colname].mean()  # priemer
        bars.append(go.Bar(name=colname, x=indx, y=df[colname], text=df[colname]))  # prida sa novy bar graf

    # viacero bar grafov na 1 figure
    fig = go.Figure(data=bars)
    fig.update_xaxes(type='category')
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(barmode='group', title_text=title_text, uniformtext_minsize=8, uniformtext_mode='hide',
                      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                      title=dict(yanchor="top", y=0.95, xanchor="left", x=0.3, font=dict(size=25)))
    fig.show()


def genres_multiple_bar_graphs(data, rows, cols, colnames_y, title_text):

    indx = genres_only  # index noveho dataframe = zanre

    for genre in colnames_y:
        if genre in indx:
            indx.remove(genre)

    # index = zanre, viacero stlpcov
    df = pd.DataFrame(index=indx, columns=colnames_y)

    # subploty do mriezky rows x cols
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=colnames_y)

    row = 1
    col = 1
    for colname in colnames_y:
        for genre in indx:
            df[colname][genre] = data[data[genre] > 0][colname].mean()  # priemer

        # bar graf
        fig.add_trace(
            go.Bar(x=indx, y=df[colname], name=colname, text=df[colname]),
            row=row, col=col
        )

        # posun riadkov a stlpcov
        if col == cols:
            row = row + 1
            col = 1
        else:
            col = col + 1

    fig.update_xaxes(type='category')
    fig.update_traces(texttemplate='%{text:.2f}', textposition='auto')
    fig.update_layout(showlegend=False, title_text=title_text, uniformtext_minsize=8, uniformtext_mode='hide',
                      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                      title=dict(yanchor="top", y=0.95, xanchor="left", x=0.3, font=dict(size=25)))
    fig.show()


def max_line_graph(data, colname_x, colname_y, title_text):
    """
    Line graf pre maximalne hodnoty
    :param data: dataset
    :param colname_x: hodnoty na osi x - nazov stlpca v datasete
    :param colname_y:  hodnoty na osi y - nazov stlpca v datasete
    :param title_text: nadpis
    """
    data = data.sort_values(by=[colname_x])
    indx = data[colname_x].unique()  # index noveho dataframe

    # index = zoradene hodnoty z data[colname_x], stlpec = colname_y
    df = pd.DataFrame(index=indx, columns=[colname_y])

    # vypocitanie max hodnoty pre kazdu hodnotu z indexu df
    for value in indx:
        df[colname_y][value] = data[data[colname_x] == value][colname_y].max()

    # line graf
    fig = px.line(df, x=indx, y=colname_y)
    fig.update_xaxes(title_text=colname_x)
    fig.update_layout(title_text=title_text, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.94),
                      title=dict(yanchor="top", y=0.99, xanchor="left", x=0.3, font=dict(size=25)))
    fig.show()


def year_line_graph_all(data, colname_y, title_text):
    """
    Line graf zavislosti roku od colname_y
    :param data: dataset
    :param colname_y: hodnoty na osi y - nazov stlpca v datasete
    :param title_text: nadpis
    """
    data = data.sort_values(by=["release_year"])
    fig = go.Figure()

    for genre in genres_tags:
        # priemer zo zaznamov pre konkretny zaner v data[colname_y]
        fig.add_trace(go.Scatter(x=data[data[genre] > 0]["release_year"].unique(),
                                 y=data[data[genre] > 0].groupby("release_year")[colname_y].mean(),
                                 mode="lines", name=genre))

    # priemer zo vsetkych zaznamov v data[colname_y]
    fig.add_trace(go.Scatter(x=data["release_year"].unique(),
                             y=data.groupby("release_year")[colname_y].mean(),
                             mode="lines+markers", name="all"))

    fig.update_xaxes(title_text="release_year", type='category')
    fig.update_yaxes(title_text=colname_y)
    fig.update_layout(title_text=title_text,
                      title=dict(yanchor="top", y=0.99, xanchor="left", x=0.35, font=dict(size=25)))

    fig.show()


def eda(data):
    # top 10 hier
    top_10_bar_graph(data, "ratings", "name", "Top 10 najlepšie hodnotených hier")
    top_10_bar_graph(data, "price", "name", "Top 10 najdrahších hier")

    # top 10 publisherov
    top_10_bar_graph(data, "publisher_games_count", "publisher",
                     "Top 10 publisherov s najväčším počtom vydaných hier")
    top_10_bar_graph(data, "publisher_avg_rating", "publisher",
                     "Top 10 najlepšie hodnotených publisherov")

    # % zastupenie zanrov
    genres_pie_graph(data, genres_only, "Zastúpenie žánrov medzi hrami")

    # bar graf pre zanre
    genres_tags_bar_graph(data, "price", "Závislosť priemernej ceny od žánru")
    genres_tags_bar_graph(data, "owners", "Závislosť priemerného počtu hráčov od žánru")

    # multi-bar graf pre zanre
    singleplayer_multiplayer_bar_graph(data, "Koľko ľudí priemerne zaradilo hry k vlastnostiam:<br>singleplayer, "
                                             "multiplayer?")
    # viacero bar grafov na 1 stranku
    genres_multiple_bar_graphs(data, 2, 2, ["difficult", "great_soundtrack", "masterpiece", "classic"],
                               "Koľko ľudí priemerne zaradilo hry k vlastnostiam:<br>"
                               "difficult, great_soundtrack, masterpiece, classic?")

    # line graf - max
    max_line_graph(data, "owners", "price", "Závislosť počtu hráčov od ceny hier")
    max_line_graph(data, "publisher_games_count", "publisher_avg_rating",
                   "Závislosť počtu vydaných hier od ratingu publishera")

    # line graf pre roky
    year_line_graph_all(data, "average_playtime", "Závislosť času hrania od roku vydania")
    year_line_graph_all(data, "price", "Závislosť ceny od roku vydania")
    year_line_graph_all(data, "ratings", "Závislosť hodnotenia od roku vydania")


''' ****************************** Clustering, PCA **************************** '''


def kmeans_clustering(data):
    """
    K-means clustering
    :param data: dataset
    """
    # cely dataset
    data_all = data.copy()
    # dataset bez string stlpcov, neskalovany
    data_to_fit = data.drop(columns=["publisher", "genres", "name", "appid"])
    # dataset bez string stlpcov, skalovany
    data_to_fit_scaled = pd.DataFrame(
        data=StandardScaler().fit_transform(data_to_fit),
        columns=data_to_fit.columns)

    # K-means
    kmeans = KMeans(init="random", n_clusters=5, max_iter=300).fit(data_to_fit_scaled)

    # pridanie id clusterov do datasetov
    data_all["cluster"] = kmeans.labels_
    data_to_fit["cluster"] = kmeans.labels_
    data_to_fit_scaled["cluster"] = kmeans.labels_

    # pridanie velkosti clusterov do datasetu
    sizes = []
    for label in kmeans.labels_:
        sizes.append(kmeans.labels_[kmeans.labels_ == label].size)
    data_to_fit.insert(0, "cluster_size", sizes)

    # vykreslenie bar grafov
    plot_clustering_bar_charts(data_to_fit, 10, 4, "K-means clusters")

    # PCA
    make_pca_2D(data_all, data_to_fit_scaled, "K-means")
    make_pca_3D(data_all, data_to_fit_scaled, "K-means")


def dbscan_clustering(data):
    """
    DBSCAN clustering
    :param data: dataset
    """
    # cely dataset
    data_all = data.copy()
    # dataset bez string stlpcov, neskalovany
    data_to_fit = data.drop(columns=["publisher", "genres", "name", "appid"])
    # dataset bez string stlpcov, skalovany
    data_to_fit_scaled = pd.DataFrame(
        data=StandardScaler().fit_transform(data_to_fit),
        columns=data_to_fit.columns)

    # dbscan
    dbscan = DBSCAN(eps=2, min_samples=15, n_jobs=-1).fit(data_to_fit_scaled)

    # pridanie id clusterov do datasetov
    data_all["cluster"] = dbscan.labels_
    data_to_fit["cluster"] = dbscan.labels_
    data_to_fit_scaled["cluster"] = dbscan.labels_

    # pridanie velkosti clusterov do datasetu
    sizes = []
    for label in dbscan.labels_:
        sizes.append(dbscan.labels_[dbscan.labels_ == label].size)
    data_to_fit.insert(0, "cluster_size", sizes)

    # vykreslenie bar grafov
    plot_clustering_bar_charts(data_to_fit, 10, 4, "DBSCAN clusters")

    # PCA
    make_pca_2D(data_all, data_to_fit_scaled, "DBSCAN")
    make_pca_3D(data_all, data_to_fit_scaled, "DBSCAN")


def make_pca_3D(data, scaled_data, title_text):
    """
    PCA + vykreslenie do 3D grafu
    :param data: cely dataset
    :param scaled_data: skalovany dataset bez stringov
    :param title_text: nadpis grafu
    """
    pca = PCA(n_components=3)
    reduced_dim_array = pca.fit_transform(scaled_data)

    # pridaju sa nove stlpce do datasetu - dimenzie x, y, z
    data['x'] = reduced_dim_array[:, 0]
    data['y'] = reduced_dim_array[:, 1]
    data['z'] = reduced_dim_array[:, 2]

    data = data.sort_values(by=["cluster"])

    # aby boli v grafe farby clusterov na legende izolovane
    data["cluster"] = data["cluster"].astype("category")

    # 3d scatter plot
    fig = px.scatter_3d(data, x="x", y="y", z="z",
                        hover_data=["name", "publisher", "ratings", "price", "genres"],
                        color="cluster", title="PCA: " + title_text)
    fig.show()


def make_pca_2D(data, scaled_data, title_text):
    """
    PCA + vykreslenie do 2D grafu
    :param data: cely dataset
    :param scaled_data: skalovany dataset bez stringov
    :param title_text: nadpis grafu
    """
    pca = PCA(n_components=2)
    reduced_dim_array = pca.fit_transform(scaled_data)

    # pridaju sa nove stlpce do datasetu - dimenzie x, y
    data['x'] = reduced_dim_array[:, 0]
    data['y'] = reduced_dim_array[:, 1]

    data = data.sort_values(by=["cluster"])

    # aby boli v grafe farby clusterov na legende izolovane
    data["cluster"] = data["cluster"].astype('category')

    # scatter plot
    fig = px.scatter(data,
                     x="x", y="y",
                     color="cluster",
                     hover_data=["name", "publisher", "ratings", "price", "genres"],
                     title="PCA: " + title_text)
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.95))
    fig.show()


if __name__ == '__main__':
    '''***************************** nacitanie dat *****************************'''

    steam = pd.read_csv('data/steam.csv')

    steam_tag = pd.read_csv('data/steamspy_tag_data.csv')
    # upraveny dataset sa ulozi do csv
    fix_dataset(steam, steam_tag).to_csv(r'data/steam_joined.csv', index=False, sep=",")
    steam_joined = pd.read_csv('data/steam_joined.csv')

    '''********************************** EDA **********************************'''

    eda(steam_joined)

    # korelacna matica
    plot_correlation_matrix(steam_joined.drop(columns="appid"), 10)

    '''****************************** Clustering *******************************'''

    # vymazanie outlierov
    delete_outliers(steam_joined)

    kmeans_clustering(steam_joined)
    dbscan_clustering(steam_joined)
