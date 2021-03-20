import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import plotly.express as px
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from keras.layers import Dense, Dropout, Input, BatchNormalization
import dateutil.parser as parser
from tensorflow.python.keras.utils.np_utils import to_categorical

genres = ["edm", "latin", "pop", "r&b", "rap", "rock"]


''' ********************** Uprava datasetu ********************** '''


def remove_values(data, is_train):
    """
    Funkcia na odstranenie nepotrebnych hodnot v datasete
    :param is_train: ci ide o testovacie alebo trenovacie data
    :param data: dataset
    """

    # vymazanie riadkov, kde je null hodnota
    data.dropna(inplace=True)

    # odstranenie outliers - iba pre trenovacie data
    if is_train:
        data.drop(data[data.speechiness >= 0.45].index, inplace=True)

        data.drop(data[data.duration_ms <= 115000].index, inplace=True)
        data.drop(data[data.duration_ms >= 400000].index, inplace=True)

        data.drop(data[data.tempo <= 60].index, inplace=True)
        data.drop(data[data.tempo >= 200].index, inplace=True)

        data.drop(data[data.loudness <= -20].index, inplace=True)
        data.drop(data[data.loudness >= -1].index, inplace=True)

        data.drop(data[data.danceability < 0.2].index, inplace=True)

        data.drop(data[data["track.popularity"] >= 85].index, inplace=True)

        data.drop(data[data.energy <= 0.2].index, inplace=True)

        data.drop(data[data.liveness >= 0.8].index, inplace=True)


def normalize_data(data, exclude_cols):
    """
    Funkcia na normalizaciu datasetu
    :param data: dataset
    :param exclude_cols: stlpce, ktore sa nebudu normalizovat
    src: https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame
    """

    for i in data.columns:
        if is_numeric_dtype(data[i]) & ~(i in exclude_cols):
            max_value = data[i].max()
            min_value = data[i].min()
            data[i] = (data[i] - min_value) / (max_value - min_value)


def convert(data, list_of_cols, data_type):
    """
    Funkcia na konvertovanie hodnot v datasete
    :param data: dataset
    :param list_of_cols: zoznam stlpcov, ktore sa budu konvertovat
    :param data_type: datovy typ
    """

    if data_type == "category":
        for i in list_of_cols:
            data[i] = data[i].astype(data_type).cat.codes           # typ category
    else:
        data[list_of_cols] = data[list_of_cols].astype(data_type)   # ostatne typy


def fix_dates(dates_col):
    """
    :param dates_col: stlpec v datasete s datumami
    :return: stlpec, kde su datumy nahradene rokmi
    """
    years = dates_col.copy()
    for i in years.index:
        years[i] = parser.parse(years[i]).year      # parsovanie roku z datumu

    return years


def fix_data(data, is_train):
    """
    Funkcia na upravu dat v datasete
    :param data: dataset
    :param is_train: ci ide o testovacie alebo trenovacie data
    :return: upraveny dataset
    """
    # spravi sa kopia dat
    data_out = data.copy()

    # odstranenie dat
    remove_values(data_out, is_train)

    # iba pri trenovacich datach
    if is_train:
        # nahradenie datumov rokmi
        data_out["track.album.release_date"] = fix_dates(data_out["track.album.release_date"])

    # zanre som konvertovala manualne, aby som s istotou vedela, akemu zanru je priradene ake cislo
    for val in genres:
        data_out.loc[data_out["playlist_genre"] == val, "playlist_genre"] = genres.index(val)

    # konverzia stringov na typ 'category'
    convert(data_out, data_out.select_dtypes(exclude='number').columns, 'category')

    # normalizacia - vsetky stlpce okrem playlist_genre
    normalize_data(data_out, ["playlist_genre"])

    return data_out


''' ********************** Grafy ********************** '''


def plot_correlation_matrix(data):
    """
    Funkcia na vykreslenie korelacnej matice
    :param data: dataset
    """

    correlation = data.corr().round(2)  # vypocet korelacnej matice

    # heatmap
    sb.set(font_scale=1.7)
    plt.subplots(figsize=(25, 20))
    sb.heatmap(correlation, annot=True,
               xticklabels=correlation.columns.values,
               yticklabels=correlation.columns.values)
    plt.title("Korelačná matica", fontsize=50)
    plt.show()


def plot_scatter_plot(data, col):
    """
    Funkcia na vykreslenie scatter grafu pre 1 stlpec z datasetu
    :param data: dataset
    :param col: nazov stlpca
    """
    fig = px.scatter(data, x='track.id', y=col)
    fig.show()


def plot_scatter_plots(data):
    """
    Funkcia na vykreslenie scatter grafov pre vsetky ciselne data v datasete
    :param data: dataset
    """

    plot_scatter_plot(data, "track.popularity")
    plot_scatter_plot(data, "energy")
    plot_scatter_plot(data, "danceability")
    plot_scatter_plot(data, "key")
    plot_scatter_plot(data, "loudness")
    plot_scatter_plot(data, "speechiness")
    plot_scatter_plot(data, "acousticness")
    plot_scatter_plot(data, "instrumentalness")
    plot_scatter_plot(data, "liveness")
    plot_scatter_plot(data, "valence")
    plot_scatter_plot(data, "tempo")
    plot_scatter_plot(data, "duration_ms")


def plot_classifier_results(history):
    """
    Funkcia na vykreslenie loss a accuracy grafov pre vysledky keras klasifikatora
    :param history: historia loss a accuracy pri trenovani
    src: https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
    """

    sb.set(font_scale=1)

    # accurracy
    plt.plot(history['categorical_accuracy'])
    plt.plot(history['val_categorical_accuracy'])
    plt.title('Klasifikator - accuracy', fontsize=15)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Klasifikator - loss', fontsize=15)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def plot_confusion_matrix(tst_y, pred_y):
    """
    Funkcia na vykreslenie confusion matice
    :param tst_y: predpokladane zanre
    :param pred_y: predikovane zanre
    """
    conf_matrix = confusion_matrix(tst_y, pred_y)

    # heatmap
    sb.set(font_scale=2)
    sb.heatmap(conf_matrix, center=True, annot=True, fmt='g', xticklabels=genres, yticklabels=genres)
    plt.title("Confusion matrix", fontsize=20)
    plt.yticks(rotation=0)
    plt.show()


''' ********************** Klasifikatory ********************** '''


def get_model(in_size, out_size, mode=""):
    """
    Funkcia na vytvorenie modelu neuronovej siete
    :param in_size: pocet vstupnych neuronov
    :param out_size: pocet vystupnych neuronov
    :param mode: metoda regularizacie, default = bez regularizacie
    :return model siete
    """
    # inicializacia modelu
    model = tf.keras.Sequential()

    model.add(Input(shape=(in_size,)))                          # vstupna vrstva
    # model.add(BatchNormalization(input_shape=(in_size,)))     # vstupna batch normalization vrstva

    # pretrenovanie siete bez regularizacie
    if mode == "overfit":
        model.add(Dense(500, activation='relu'))                # skryta vrstva

    # pretrenovanie siete, l1l2 regularizacia
    elif mode == "l1l2":
        model.add(Dense(500, activation='relu',
                        kernel_regularizer="l1_l2"))            # skryta vrstva

    # pretrenovanie siete, dropout regularizacia
    elif mode == "dropout":
        model.add(Dense(500, activation='relu'))                # skryta vrstva
        model.add(Dropout(0.5))                                 # skryta dropout vrstva

    # pretrenovanie siete, batch normalization
    elif mode == "batch":
        model.add(BatchNormalization())                         # batch normalization vrstva
        model.add(Dense(500, activation='sigmoid'))             # skryta vrstva
        model.add(BatchNormalization())                         # batch normalization vrstva

    # bez pretrenovania
    else:
        model.add(Dense(100, activation='relu'))                # skryta vrstva

    model.add(Dense(out_size, activation='softmax'))            # vystupna vrstva

    return model


def train_keras_classifier(train_x_in, train_y_out, test_x_in, test_y_out, mode=""):
    """
    Funkcia na trenovanie keras klasifikatora
    :param train_x_in: vstupna trenovacia mnozina
    :param train_y_out: vystupna trenovacia mnozina
    :param test_x_in: vstupna testovacia mnozina
    :param test_y_out: vystupna testovacia mnozina
    :param mode: metoda regularizacie, default = bez regularizacie
    """

    # pocet neuronov vo vstupnej a vystupnej vrstve
    input_size = train_x_in.columns.size
    output_size = train_y_out.nunique()

    # uprava Y na categorical
    train_y_out = to_categorical(train_y_out.values.ravel(), num_classes=output_size)
    test_y_out = to_categorical(test_y_out.values.ravel(), num_classes=output_size)

    # klasifikator
    classifier = get_model(input_size, output_size, mode)
    classifier.compile(loss='categorical_crossentropy',
                       optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                       # optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # overfit
                       metrics=['categorical_accuracy'])

    # trenovanie klasifikatora
    fit = classifier.fit(train_x_in, train_y_out,
                         batch_size=100, epochs=500, validation_split=0.2, use_multiprocessing=True)

    # predpovedany vystup
    predict_y_out = classifier.predict(test_x_in)

    print("\n****************************** evaluation ******************************")
    classifier.evaluate(test_x_in, test_y_out) # vysledky

    # grafy pre loss a accuracy
    plot_classifier_results(fit.history)

    # confusion matica
    plot_confusion_matrix(test_y_out.argmax(axis=1), predict_y_out.argmax(axis=1))


def train_svm_classifier(train_x_in, train_y_out, test_x_in, test_y_out):
    """
    Funkcia na trenovanie SVM klasifikatora
    :param train_x_in: vstupna trenovacia mnozina
    :param train_y_out: vystupna trenovacia mnozina
    :param test_x_in: vstupna testovacia mnozina
    :param test_y_out: vystupna testovacia mnozina
    """

    # inicializacia klasifikatora
    classifier = svm.SVC(kernel="rbf", tol=0.00001, verbose=1)
    # trenovanie
    classifier.fit(train_x_in, train_y_out)
    # predikcia vystupu
    predict_y_out = classifier.predict(test_x_in)

    # vysledky
    print("\n*************** SVM klasifikator ***************")
    report = classification_report(test_y_out, predict_y_out, digits=3)
    print(report)

    # confusion matica
    plot_confusion_matrix(test_y_out, predict_y_out)


if __name__ == '__main__':

    # nacitanie dat
    train_dataset = pd.read_csv('data_z2/train.csv')
    test_dataset = pd.read_csv('data_z2/test.csv')

    # uprava dat - pri testovacich datach sa len konvertuje na category a normalizuje
    fixed_train_dataset = fix_data(train_dataset, True)
    fixed_test_dataset = fix_data(test_dataset, False)

    # drop id
    fixed_train_dataset.drop(
        columns=["track.id", "track.album.id", "playlist_id"], inplace=True)
    fixed_test_dataset.drop(
        columns=["track.id", "track.album.id", "playlist_id"], inplace=True)
    print(fixed_test_dataset.shape)

    # korelacna matica
    plot_correlation_matrix(fixed_train_dataset)

    # vstupne parametre na trenovanie: vsetko okrem playlist_genre a playlist_subgenre
    cols_x = fixed_train_dataset.columns.drop(["playlist_genre", "playlist_subgenre"])

    # rozdelenie datasetu na trenovacie a testovacie data pre klasifikator
    train_x, train_y = fixed_train_dataset[cols_x], fixed_train_dataset["playlist_genre"]
    test_x, test_y = fixed_test_dataset[cols_x], fixed_test_dataset["playlist_genre"]

    print("\n****************************** KERAS CLASSIFIER ************************************")
    train_keras_classifier(train_x, train_y, test_x, test_y)

    # print("\n********************************** OVERFIT *****************************************")
    # train_keras_classifier(train_x, train_y, test_x, test_y, "overfit")
    #
    # print("\n********************************** L1L2 ********************************************")
    # train_keras_classifier(train_x, train_y, test_x, test_y, "l1l2")
    #
    # print("\n********************************* DROPOUT *****************************************")
    # train_keras_classifier(train_x, train_y, test_x, test_y, "dropout")
    #
    # print("\n********************************** BATCH ******************************************")
    # train_keras_classifier(train_x, train_y, test_x, test_y, "batch")

    print("\n****************************** SVM CLASSIFIER **************************************")
    train_svm_classifier(train_x, train_y, test_x, test_y)
