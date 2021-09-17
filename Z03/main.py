import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import plotly.express as px
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.tree import export_graphviz
import pydot
import warnings

warnings.filterwarnings("ignore")

classes = ["STAR", "GALAXY", "QSO"]

''' ********************** Uprava datasetu ********************** '''


def convert(data, list_of_cols, data_type):
    """
    Funkcia na konvertovanie stlpcov v datasete
    :param data: dataset
    :param list_of_cols: zoznam stlpcov, ktore sa budu konvertovat
    :param data_type: datovy typ
    """

    data_cp = data.copy()

    if data_type == "category":
        for i in list_of_cols:
            data_cp.loc[:, i] = data_cp[i].astype(data_type).cat.codes  # typ category
    else:
        data_cp.loc[:, [list_of_cols]] = data_cp[list_of_cols].astype(data_type)  # ostatne typy

    return data_cp


def normalize_data(data, exclude_cols):
    """
    Funkcia na normalizaciu datasetu
    :param data: dataset
    :param exclude_cols: stlpce, ktore sa nebudu normalizovat
    src: https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame
    """

    data_out = data.copy()

    for i in data_out.columns:
        if is_numeric_dtype(data_out[i]) & ~(i in exclude_cols):
            max_value = data_out[i].max()
            min_value = data_out[i].min()
            data_out[i] = (data_out[i] - min_value) / (max_value - min_value)

    return data_out


''' ********************** Grafy, vypisy ********************** '''


def plot_correlation_matrix(data):
    """
    Funkcia na vykreslenie korelacnej matice
    :param data: dataset
    """
    data_cp = convert(data, ["class"], "category")
    correlation = data_cp.corr().round(2)  # vypocet korelacnej matice

    # heatmap
    sb.set(font_scale=2)
    plt.subplots(figsize=(25, 20))
    sb.heatmap(correlation, annot=True,
               xticklabels=correlation.columns.values,
               yticklabels=correlation.columns.values)
    plt.title("Korelačná matica", fontsize=50)
    plt.show()


def plot_confusion_matrix(tst_y, pred_y):
    """
    Funkcia na vykreslenie confusion matice
    :param tst_y: povodne triedy
    :param pred_y: predikovane triedy
    """
    conf_matrix = confusion_matrix(tst_y, pred_y)

    # heatmap
    sb.set(font_scale=2)
    sb.heatmap(conf_matrix, center=True, annot=True, fmt='g', xticklabels=classes, yticklabels=classes)
    plt.title("Confusion matrix", fontsize=20)
    plt.yticks(rotation=0)
    plt.show()


def plot_regression_results(test, predict, title_text):
    """
      Funkcia na vykreslenie 3D a 2D grafov pre predpovedane a povodne hodnoty
      :param test: povodne hodnoty
      :param predict: predikovane hodnoty
      :param title_text: nadpis grafu
    """

    # ************************** 3D graf **************************
    sb.set(font_scale=0.7)
    sb.set_style("ticks")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title_text, fontsize=15)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # predict data
    ax.scatter(
        predict['x_coord'], predict['y_coord'], predict['z_coord'],
        c="navy", marker=".", s=5, label='predict')

    # test data
    ax.scatter(
        test['x_coord'], test['y_coord'], test['z_coord'],
        c="purple", marker=".", s=2, label='test')

    plt.legend(['predict', 'test'], fontsize=10, loc="upper left")

    plt.show()

    # ********************* kazda suradnica samostatne *********************
    x_axis = range(len(test))  # indexy na x osi

    # x_coord
    sb.scatterplot(x=x_axis, y=test["x_coord"], label="x-test", color='navy', s=10)         # x_coord test
    sb.scatterplot(x=x_axis, y=predict["x_coord"], label="x-pred", color='limegreen', s=5)  # x_coord predict
    plt.title(title_text + ": x_coord", fontsize=20)
    plt.legend(fontsize=15)
    plt.show()

    # y_coord
    sb.scatterplot(x=x_axis, y=test["y_coord"], label="y-test", color='navy', s=10)         # y_coord test
    sb.scatterplot(x=x_axis, y=predict["y_coord"], label="y-pred", color='turquoise', s=5)  # y_coord predict
    plt.title(title_text + ": y_coord", fontsize=20)
    plt.legend(fontsize=15)
    plt.show()

    # z_coord
    sb.scatterplot(x=x_axis, y=test["z_coord"], label="z-test", color='navy', s=10)          # z_coord test
    sb.scatterplot(x=x_axis, y=predict["z_coord"], label="z-pred", color='blueviolet', s=5)  # z_coord predict
    plt.title(title_text + ": z_coord", fontsize=20)
    plt.legend(fontsize=15)
    plt.show()


def print_regression_results(test_y_out, predict_y_out, title_text):
    """
      Funkcia na vypis vysledkov regresie
      :param test_y_out: povodne hodnoty
      :param predict_y_out: predikovane hodnoty
      :param title_text: nadpis grafu
    """

    print("************************ ", title_text, " ************************")

    # mae a r2 pre cely vystup
    print("[x_coord,y_coord,z_coord]: MAE =", mean_absolute_error(test_y_out, predict_y_out),
          "R2 =", r2_score(test_y_out, predict_y_out))

    # mae a r2 pre kazdu suradnicu zvlast
    print("\nx_coord: MAE=", mean_absolute_error(test_y_out["x_coord"], predict_y_out[:, 0]),
          "R2=", r2_score(test_y_out["x_coord"], predict_y_out[:, 0]))

    print("y_coord: MAE=", mean_absolute_error(test_y_out["y_coord"], predict_y_out[:, 1]),
          "R2=", r2_score(test_y_out["y_coord"], predict_y_out[:, 1]))

    print("z_coord: MAE=", mean_absolute_error(test_y_out["z_coord"], predict_y_out[:, 2]),
          "R2=", r2_score(test_y_out["z_coord"], predict_y_out[:, 2]))


def plot_residual_plot(test, predict, title_text):
    """
    Funkcia na vykreslenie residual plotu pre regresiu
    :param title_text: Nadpis grafu
    :param test: Skutocne suradnice
    :param predict: Predpovedane suradnice
    """

    # x_coord
    fig = px.scatter(
        x=predict["x_coord"], y=test["x_coord"],
        marginal_y='violin', trendline_color_override="red", trendline='ols'
    )
    fig.update_layout(
        xaxis_title="predict-x",
        yaxis_title="test-x",
        title=dict(
            text="Residual plot - " + title_text + " x_coord",
            x=0.45,
            xanchor='center',
            yanchor='top',
            font=dict(size=20)
        )
    )
    fig.show()

    # y_coord
    fig = px.scatter(
        x=predict["y_coord"], y=test["y_coord"],
        marginal_y='violin', trendline_color_override="red", trendline='ols'
    )
    fig.update_layout(
        xaxis_title="predict-y",
        yaxis_title="test-y",
        title=dict(
            text="Residual plot - " + title_text + " y_coord",
            x=0.45,
            xanchor='center',
            yanchor='top',
            font=dict(size=20)
        )
    )
    fig.show()

    # z_coord
    fig = px.scatter(
        x=predict["z_coord"], y=test["z_coord"],
        marginal_y='violin', trendline_color_override="red", trendline='ols'
    )
    fig.update_layout(
        xaxis_title="predict-z",
        yaxis_title="test-z",
        title=dict(
            text="Residual plot - " + title_text + " z_coord",
            x=0.45,
            xanchor='center',
            yanchor='top',
            font=dict(size=20)
        )
    )
    fig.show()


''' ********************** Klasifikatory ********************** '''


def train_random_forest_classifier(train_x_in, train_y_out, test_x_in, test_y_out):
    """
    Funkcia na trenovanie random forest klasifikatora
    :param train_x_in: vstupna trenovacia mnozina
    :param train_y_out: vystupna trenovacia mnozina
    :param test_x_in: vstupna testovacia mnozina
    :param test_y_out: vystupna testovacia mnozina
    """

    # random forest klasifikator
    classifier = RandomForestClassifier(criterion='entropy', verbose=1, n_jobs=-1, max_leaf_nodes=40,
                                        min_samples_leaf=10)

    classifier.fit(train_x_in, train_y_out)         # trenovanie
    predict_y_out = classifier.predict(test_x_in)   # predpovedane triedy

    # vysledky
    print("\n*************** RANDOM FOREST CLASSIFIER *****************")
    print(classification_report(test_y_out, predict_y_out, digits=3))

    # confusion matica
    plot_confusion_matrix(test_y_out, predict_y_out)

    # strom
    tree = classifier.estimators_[0]
    export_graphviz(tree, out_file='tree-classifier.dot', class_names=classes, feature_names=train_x_in.columns,
                    filled=True, rounded=True)
    (graph,) = pydot.graph_from_dot_file('tree-classifier.dot')
    graph.write_png('tree-classifier.png')


def train_mlp_classifier(train_x_in, train_y_out, test_x_in, test_y_out):
    """
    Funkcia na trenovanie MLP klasifikatora
    :param train_x_in: vstupne trenovacie data
    :param train_y_out: vystupne trenovacie data
    :param test_x_in: vstupne testovacie data
    :param test_y_out: vystupne testovacie data
    """

    # normalizacia
    train_x_in = normalize_data(train_x_in, [])
    test_x_in = normalize_data(test_x_in, [])

    # mlp klasifikator
    classifier = MLPClassifier(verbose=True, max_iter=600, alpha=0.01,
                               hidden_layer_sizes=(20,))

    classifier.fit(train_x_in, train_y_out)         # trenovanie
    predict_y_out = classifier.predict(test_x_in)   # predikcia vystupnych hodnot

    # vysledky
    print("\n***************** MLP CLASSIFIER *****************")
    print(classification_report(test_y_out, predict_y_out, digits=3))

    # confusion matica
    plot_confusion_matrix(test_y_out, predict_y_out)


''' ********************** Regresory ********************** '''


def train_random_forest_regressor(train_x_in, train_y_out, test_x_in, test_y_out):
    """
    Funkcia na trenovanie random forest regressora
    :param train_x_in: vstupna trenovacia mnozina
    :param train_y_out: vystupna trenovacia mnozina
    :param test_x_in: vstupna testovacia mnozina
    :param test_y_out: vystupna testovacia mnozina
    """

    # random forest regresor
    regressor = RandomForestRegressor(verbose=1, n_jobs=-1, criterion="mse", max_depth=15)

    regressor.fit(train_x_in, train_y_out)          # trenovanie
    predict_y_out = regressor.predict(test_x_in)    # predpovedane hodnoty

    # vystup konvertovany na dataframe
    predict_df = pd.DataFrame(data=predict_y_out, columns=["x_coord", "y_coord", "z_coord"])

    # grafy
    plot_residual_plot(test_y_out, predict_df, "Random forest regressor")
    plot_regression_results(test_y_out, predict_df, "Random forest regressor")

    # vypis vysledkov
    print_regression_results(test_y_out, predict_y_out, "RANDOM FOREST REGRESSOR")


def train_k_neighbors_regressor(train_x_in, train_y_out, test_x_in, test_y_out):
    """
    Funkcia na trenovanie k-neighbors regresora
    :param train_x_in: vstupne trenovacie data
    :param train_y_out: vystupne trenovacie data
    :param test_x_in: vstupne testovacie data
    :param test_y_out: vystupne testovacie data
    """

    # K neighbors regresor
    regressor = KNeighborsRegressor(n_jobs=-1, n_neighbors=8, weights='distance')
    #
    regressor.fit(train_x_in, train_y_out)          # trenovanie
    predict_y_out = regressor.predict(test_x_in)    # predikcia vystupnych hodnot

    # vystup konvertovany na dataframe
    predict_df = pd.DataFrame(data=predict_y_out, columns=["x_coord", "y_coord", "z_coord"])

    # grafy
    plot_residual_plot(test_y_out, predict_df, "K-neighbors regressor")
    plot_regression_results(test_y_out, predict_df, "K-neighbors regressor")

    # vypis vysledkov
    print_regression_results(test_y_out, predict_y_out, "K-NEIGHBORS REGRESSOR")


if __name__ == '__main__':
    # nacitanie dat
    train_dataset = pd.read_csv('data_z3/train.csv')
    test_dataset = pd.read_csv('data_z3/test.csv')

    # drop nepotrebnych stlpcov
    train_dataset.drop(columns=["objid", "fiberid", "specobjid", "run", "rerun", "camcol", "field"], inplace=True)
    test_dataset.drop(columns=["objid", "fiberid", "specobjid", "run", "rerun", "camcol", "field"], inplace=True)

    # korelacna matica po odstraneni stlpcov
    plot_correlation_matrix(train_dataset)

    # ************************** KLASIFIKATORY **********************************

    # vstupne stlpce - vsetky okrem class
    cols_x = train_dataset.columns.drop(["class"])

    # rozdelenie dat na vstupne a vystupne mnoziny
    train_x, train_y = train_dataset[cols_x], train_dataset["class"]
    test_x, test_y = test_dataset[cols_x], test_dataset["class"]

    # klasifikatory
    train_random_forest_classifier(train_x, train_y, test_x, test_y)
    train_mlp_classifier(train_x, train_y, test_x, test_y)

    # ************************** REGRESORY **********************************

    # konvertovanie stlpca class na ciselne hodnoty kategorii
    train_dataset = convert(train_dataset, ["class"], "category")
    test_dataset = convert(test_dataset, ["class"], "category")

    # vstupne stlpce - vsetky okrem x_coord, y_coord a z_coord
    cols_x = train_dataset.columns.drop(["x_coord", "y_coord", "z_coord"])

    # rozdelenie dat na vstupne a vystupne mnoziny
    train_x, train_y = train_dataset[cols_x], train_dataset[["x_coord", "y_coord", "z_coord"]]
    test_x, test_y = test_dataset[cols_x], test_dataset[["x_coord", "y_coord", "z_coord"]]

    # regresory
    train_random_forest_regressor(train_x, train_y, test_x, test_y)
    train_k_neighbors_regressor(train_x, train_y, test_x, test_y)
