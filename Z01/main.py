import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, r2_score, mean_squared_error, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.utils._testing import ignore_warnings

''' ********************** Uprava datasetu ********************** '''


def remove_values(data):
    """
    Funkcia na odstranenie nepotrebnych hodnot v datasete
    :param data: dataset
    """
    # odstranenie stlpca id
    data.drop(columns='id', inplace=True)

    # vymazanie riadkov, kde je null hodnota
    data.dropna(inplace=True)

    # active: vymaze sa active == 2 (musi byt binary 0/1)
    data.drop(data[data.active == 2].index, inplace=True)

    # ap_hi: nechaju sa hodnoty <60-250>
    data.drop(data[data.ap_hi < 60].index, inplace=True)
    data.drop(data[data.ap_hi > 250].index, inplace=True)

    # ap_lo: nechaju sa hodnoty <30-200>
    data.drop(data[data.ap_lo < 30].index, inplace=True)
    data.drop(data[data.ap_lo > 200].index, inplace=True)

    # age: vymaze sa vek > 40 000 dni (cca 110 rokov)
    data.drop(data[data.age > 40000].index, inplace=True)

    # weight: vymaze sa vaha < 30 kg
    data.drop(data[data.weight <= 30].index, inplace=True)

    # height: zostanu hodnoty (130-250)
    data.drop(data[data.height <= 130].index, inplace=True)
    data.drop(data[data.height >= 250].index, inplace=True)


def replace_values(data):
    """
    Funkcia na nahradenie hodnot v datasete
    :param data: dataset
    """

    # gender: (woman,man) -> (0,1)
    data.loc[data.gender == "woman", "gender"] = 1
    data.loc[data.gender == "man", "gender"] = 0

    # cholesterol: (normal, above normal,  well above normal) -> (0,0.5,1)
    data.loc[data.cholesterol == 'normal', "cholesterol"] = 0
    data.loc[data.cholesterol == 'above normal', "cholesterol"] = 0.5
    data.loc[data.cholesterol == 'well above normal', "cholesterol"] = 1

    # glucose: (normal, above normal, well above normal) -> (0,0.5,1)
    data.loc[data.glucose == 'normal', "glucose"] = 0
    data.loc[data.glucose == 'above normal', "glucose"] = 0.5
    data.loc[data.glucose == 'well above normal', "glucose"] = 1

    # konverzia datovych typov stlpcov
    convert(data, ["cholesterol", "glucose"], "float64")
    convert(data, ["gender"], "int64")


def calculate_bmi(data):
    """
    Funkcia na pocitanie bmi a pridanie noveho stlpca
    :param data: dataset
    :rtype: stlpec bmi
    """
    bmis = []
    for i in data.index:
        height = data["height"][i] / 100  # vyska v metroch
        weight = data["weight"][i]

        # bmi
        bmi = weight / (height * height)
        bmis.append(bmi)

    return bmis  # vrati sa novy stlpec bmi


def normalize_data(data):
    """
    Funkcia na normalizaciu datasetu
    :param data: dataset
    src: https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame
    """

    for i in data.columns:
        max_value = data[i].max()
        min_value = data[i].min()
        data[i] = (data[i] - min_value) / (max_value - min_value)


def convert(data, list_of_cols, data_type):
    """
    Funkcia na konvertovanie hodnot v datasete
    :param data_type: datovy typ
    :param data: dataset
    :param list_of_cols: zoznam stlpcov, ktore sa budu konvertovat
    """
    data[list_of_cols] = data[list_of_cols].astype(data_type)


def fix_data(data):
    """
    Funkcia na upravu dat v datasete
    :param data: dataset
    :return: upraveny dataset
    """
    # spravi sa kopia dat
    data_out = data.copy()

    # vymazanie hodnot
    remove_values(data_out)

    # nahradenie hodnot
    replace_values(data_out)

    # vytvorenie stlpca bmi
    bmi_col = calculate_bmi(data_out)

    # normalizacia
    normalize_data(data_out)

    # konverzia datovych typov stlpcov
    convert(data_out, ["gender", "smoke", "alco", "active", "cardio"], "int64")

    # pridanie stlpca bmi
    data_out["bmi"] = bmi_col

    # odstranenie riadkov pre hranicne outliers bmi
    data_out.drop(data_out[data_out.bmi > 60].index, inplace=True)
    data_out.drop(data_out[data_out.bmi < 12].index, inplace=True)

    return data_out


''' ********************** Grafy ********************** '''


def plot_correlation_matrix(data):
    """
    Funkcia na vykreslenie korelacnej matice
    :param data: dataset
    """
    correlation = data.corr().round(2)  # vypocet korelacnej matice

    # heatmap
    sb.set(font_scale=0.8)
    sb.heatmap(correlation, annot=True,
               xticklabels=correlation.columns.values,
               yticklabels=correlation.columns.values)
    plt.title("Korelačná matica", fontsize=20)
    plt.show()


def plot_residual_plot(test_y, predict_y, title_text):
    """
    Funkcia na vykreslenie residual plotu pre regresiu
    :param title_text: Popis grafu
    :param test_y: Skutocne bmi
    :param predict_y: Predpovedane bmi
    """

    # vytvorenie dataframe z predict_y a test_y
    df = pd.DataFrame({'predicted bmi': predict_y, 'original bmi': test_y.values.ravel()})

    # residual plot
    fig = px.scatter(
        df, x='predicted bmi', y='original bmi', title=title_text,
        marginal_y='violin', trendline_color_override="red", trendline='ols'
    )
    fig.show()


def plot_loss_graph(data, title_text):
    """
    Funkcia na vykreslenie loss grafu pri trenovani
    :param data: hodnoty loss
    :param title_text: popis grafu
    """
    plt.title(title_text, fontsize=20)
    plt.plot(data)
    plt.show()


''' ********************** Klasifikator, Regresor ********************** '''


def split_data(data, cols_x, col_y):
    """
    Funkcia na rozdelenie datasetu na trenovacie a testovacie data
    :param col_y: stlpec pre vystupne data
    :param cols_x: stlpce pre vstupne data
    :param data: dataset
    :return: train_x_in, test_x_in, train_y_out, test_y_out - testovacie a trenovacie data
    """

    x = data[cols_x]
    y = data[col_y]

    # train 70%, test 30%
    train_x_in, test_x_in, train_y_out, test_y_out = train_test_split(x, y, test_size=0.3)

    return train_x_in, test_x_in, train_y_out, test_y_out


@ignore_warnings(category=ConvergenceWarning)
def train_classifier(train_x_in, test_x_in, train_y_out, test_y_out):
    """
    Funkcia na trenovanie binarneho klasifikatora
    :param train_x_in: vstupne trenovacie data
    :param test_x_in: vstupne testovacie data
    :param train_y_out: vystupne trenovacie data
    :param test_y_out: vystupne testovacie data
    """

    # mlp klasifikator
    classifier = MLPClassifier(verbose=True, tol=0.000001, max_iter=300, alpha=0.01,
                               hidden_layer_sizes=(20, ))

    classifier.fit(train_x_in, train_y_out)  # trenovanie
    predict_y_out = classifier.predict(test_x_in)  # predikcia vystupnych hodnot

    # vysledky
    print("\n*************** Binarny klasifikator ***************")
    report = classification_report(test_y_out, predict_y_out, digits=3)
    print(report)

    # loss graph
    plot_loss_graph(classifier.loss_curve_, "Binarny klasifikator - loss graph")

    # confusion matica
    conf_matrix = confusion_matrix(test_y_out, predict_y_out)
    sb.set(font_scale=2)
    sb.heatmap(conf_matrix, center=True, annot=True, fmt='g')
    plt.title("Confusion matrix", fontsize=20)

    plt.show()


@ignore_warnings(category=ConvergenceWarning)
def train_regressor(train_x_in, test_x_in, train_y_out, test_y_out):
    """
    Funkcia na trenovanie regresora
    :param train_x_in: vstupne trenovacie data
    :param test_x_in: vstupne testovacie data
    :param train_y_out: vystupne trenovacie data
    :param test_y_out: vystupne testovacie data
    """

    # mlp regresor
    regressor = MLPRegressor(verbose=True, tol=0.000001, max_iter=300, alpha=0.01,
                             hidden_layer_sizes=(20, ))
    regressor.fit(train_x_in, train_y_out.values.ravel())  # trenovanie
    predict_y_out = regressor.predict(test_x_in)  # predikcia vystupnych hodnot

    # vypocet mse a r2
    mse = mean_squared_error(test_y_out.values.ravel(), predict_y_out)
    score = r2_score(test_y_out, predict_y_out)

    print("\n*************** MLP Regressor ***************")
    print("mse = ", mse, "r2 = ", score)

    # loss graph
    plot_loss_graph(regressor.loss_curve_, "MLP Regressor - loss graph")

    # residual plot
    plot_residual_plot(test_y_out, predict_y_out, "MLP Regressor - residual plot")


def linear_regression(train_x_in, test_x_in, train_y_out, test_y_out):
    """
    Funkcia na pocitanie linearnej regresie
    :param train_x_in: vstupne trenovacie data
    :param test_x_in: vstupne testovacie data
    :param train_y_out: vystupne trenovacie data
    :param test_y_out: vystupne testovacie data
    """

    # linearna regresia
    lin_regression = LinearRegression()
    lin_regression.fit(train_x_in, train_y_out.values.ravel())  # trenovanie
    predict_y_out = lin_regression.predict(test_x_in)  # predikcia vystupnych hodnot

    # vypocet mse a r2
    mse = mean_squared_error(test_y_out, predict_y_out)
    score = r2_score(test_y_out, predict_y_out)

    print("\n*************** Linearna regresia ***************")
    print("mse = ", mse, "r2 = ", score)

    # residual plot
    plot_residual_plot(test_y_out, predict_y_out, "Linear regression - residual plot")



''' ******************** Povodne parametre a nastavenie trenovania ******************** '''


def main():
    # nacitanie dat
    dataset = pd.read_csv('data/srdcove_choroby.csv')

    # uprava dat
    fixed_dataset = fix_data(dataset)

    # korelacna matica
    plot_correlation_matrix(fixed_dataset)

    # odstranenie stlpcov so zanedbatelnymi korelaciami
    fixed_dataset.drop(columns=['smoke', 'alco', 'active'], inplace=True)

    # korelacna matica po odstraneni nepotrebnych stlpcov
    plot_correlation_matrix(fixed_dataset)

    # *********************** binarna klasifikacia ***********************

    # rozdelenie datasetu na trenovacie a testovacie data pre klasifikator
    train_x, test_x, train_y, test_y = split_data(fixed_dataset,
                                                  ["age", "weight", "ap_lo", "ap_hi", "cholesterol"],
                                                  "cardio")

    # trenovanie klasifikatora
    train_classifier(train_x, test_x, train_y, test_y)

    # **************************** regresia ****************************

    # # rozdelenie datasetu na trenovacie a testovacie data pre regresiu
    train_x, test_x, train_y, test_y = split_data(fixed_dataset,
                                                  ["age", "ap_lo", "ap_hi", "cholesterol", "glucose", "gender", "cardio"],
                                                  "bmi")

    # trenovanie regresora
    train_regressor(train_x, test_x, train_y, test_y)

    # linearny regresor
    linear_regression(train_x, test_x, train_y, test_y)


''' ********************** Testy na vstupne parametre ********************** '''


# cely dataset
def test1_params_all_cols():
    # nacitanie dat
    dataset = pd.read_csv('data/srdcove_choroby.csv')

    # uprava dat
    fixed_dataset = fix_data(dataset)

    # *********************** binarna klasifikacia ***********************

    # rozdelenie datasetu na trenovacie a testovacie data pre klasifikator
    train_x, test_x, train_y, test_y = \
        split_data(fixed_dataset, fixed_dataset.columns.drop("cardio"), "cardio")  # vsetky stlpce

    # trenovanie klasifikatora
    train_classifier(train_x, test_x, train_y, test_y)

    # **************************** regresia ****************************

    # rozdelenie datasetu na trenovacie a testovacie data pre regresor
    train_x, test_x, train_y, test_y = \
        split_data(fixed_dataset, fixed_dataset.columns.drop("bmi"), "bmi")  # vsetky stlpce

    # trenovanie regresora
    train_regressor(train_x, test_x, train_y, test_y)

    # linearny regresor
    linear_regression(train_x, test_x, train_y, test_y)


# parametre s malou korelaciou
def test2_params_small_correlation():
    # nacitanie dat
    dataset = pd.read_csv('data/srdcove_choroby.csv')

    # uprava dat
    fixed_dataset = fix_data(dataset)

    # *********************** binarna klasifikacia ***********************

    train_x, test_x, train_y, test_y = \
        split_data(fixed_dataset, ["smoke", "alco", "active", "height", "gender"], "cardio")  # najmensia korelacia

    # trenovanie klasifikatora
    train_classifier(train_x, test_x, train_y, test_y)

    # **************************** regresia ****************************

    # rozdelenie datasetu na trenovacie a testovacie data pre regresiu
    train_x, test_x, train_y, test_y = \
        split_data(fixed_dataset, ["smoke", "alco", "active"], "bmi")  # najmensia korelacia

    # trenovanie regresora
    train_regressor(train_x, test_x, train_y, test_y)

    # linearny regresor
    linear_regression(train_x, test_x, train_y, test_y)


# parametre s najvacsou korelaciou
def test3_params_big_correlation():
    # nacitanie dat
    dataset = pd.read_csv('data/srdcove_choroby.csv')

    # uprava dat
    fixed_dataset = fix_data(dataset)

    # *********************** binarna klasifikacia ***********************

    # rozdelenie datasetu na trenovacie a testovacie data pre klasifikator
    train_x, test_x, train_y, test_y = \
        split_data(fixed_dataset, ["ap_hi", "ap_lo"], "cardio")  # malo stlpcov ale vysoka korelacia

    # trenovanie klasifikatora
    train_classifier(train_x, test_x, train_y, test_y)

    # **************************** regresia ****************************

    # # rozdelenie datasetu na trenovacie a testovacie data pre regresiu
    train_x, test_x, train_y, test_y = \
        split_data(fixed_dataset, ["ap_hi", "ap_lo"], "bmi")  # malo stlpcov ale vysoka korelacia

    # trenovanie regresora
    train_regressor(train_x, test_x, train_y, test_y)

    # linearny regresor
    linear_regression(train_x, test_x, train_y, test_y)


''' ********************************* Koniec funkcii ********************************* '''

if __name__ == "__main__":

    main()

    # ******************** testy na vstupne parametre ********************

    # print("---------------------------------------- TEST1 - VSETKY STLPCE -----------------------------------------")
    # test1_params_all_cols()
    # print("---------------------------------------- TEST2 - MALA KORELACIA ----------------------------------------")
    # test2_params_small_correlation()
    # print("-------------------------------- TEST3 - VELKA KORELACIA, MALO STLPCOV ---------------------------------")
    # test3_params_big_correlation()

