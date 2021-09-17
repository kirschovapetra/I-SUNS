import json
import math
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras.layers import Dense, Flatten, MaxPooling2D, Conv2D
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.models import load_model
from tensorflow.python.ops.confusion_matrix import confusion_matrix

pd.options.mode.chained_assignment = None


#                   0             1           2             3
class_names = ["Accessories", "Apparel", "Footwear", "Personal Care"]

IMG_SIZE = (32, 32)
BATCH_SIZE = 400

''' ************************ Grafy, obrazky ************************* '''


def plot_confusion_matrix(tst_y, pred_y):
    """
    Vykreslenie confusion matice
    :param tst_y: spravne triedy
    :param pred_y: predpovedane triedy
    """
    conf_matrix = confusion_matrix(tst_y, pred_y)

    # heatmap
    sb.heatmap(conf_matrix, center=True, annot=True, fmt='g', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion matrix", fontsize=20)
    plt.yticks(rotation=0)
    plt.show()


def plot_classifier_results(history):
    """
    Vykreslenie loss a accuracy grafov pre vysledky keras klasifikatora
    :param history: historia loss a accuracy pri trenovani
    src: https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
    """

    sb.set(font_scale=1)

    # accurracy
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
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


def plot_images(images_generator, cols, rows):
    """
    Vykreslenie obrazkov do mriezky cols x rows
    :param images_generator: generator obrazkov
    :param cols: pocet stlpcov
    :param rows: pocet riadkov
    """

    images = images_generator.next()[0]       # zoznam obrazkov
    class_num = images_generator.classes      # triedy (num)

    # vykreslenie prvych rows * cols obrazkov z datasetu
    plt.figure(figsize=(cols + 5, rows + 5))
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        class_name = class_names[class_num[i]]
        filename = images_generator.filenames[i]
        plot_image(images[i], filename + ": " + class_name)
    plt.show()


def plot_predictions(images_generator, data_predicted):
    """
    Vykreslenie obrazkov z testovacej mnoziny + bar plot pre predikcie triedy
    :param images_generator: generator obrazkov
    :param data_predicted: predpovedane hodnoty - %
    """

    images = images_generator.next()[0]                         # zoznam obrazkov
    class_num_original = images_generator.classes               # spravne triedy (num)
    class_num_predicted = np.argmax(data_predicted, axis=1)     # predpovedane triedy (num)

    # vykreslenie prvych 5 obrazkov z testovacieho datasetu
    for i in range(0, 5):
        plt.figure(figsize=(10, 5))
        # obrazok
        plt.subplot(1, 2, 1)
        plot_image(images[i], "Original: " + class_names[class_num_original[i]])
        # bar graf %
        plt.subplot(1, 2, 2)
        plot_predictions_bar_plot(data_predicted[i], class_num_predicted[i])
        plt.show()


def plot_image(image, label):
    """
    Vykreslenie obrazku
    :param image: obrazok
    :param label: trieda, do ktorej patri
    """
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(label)


def plot_predictions_bar_plot(predicted_percent_all, predicted_class):
    """
    Vykreslenie bar grafu pre % pravdepodobnosti zatriedenia do tried
    :param predicted_percent_all: pravdepodobnosti obrazkov pre kazdu triedu
    :param predicted_class: predpovedana trieda - ciselna hodnota
    """

    # percento pre vyslednu triedu
    highest_percent = max((predicted_percent_all * 100).round(3))

    # bar plot
    bar_plot = plt.bar(range(len(predicted_percent_all)), predicted_percent_all, color="gray")
    bar_plot[predicted_class].set_color('green')
    plt.xticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted: " + class_names[predicted_class] + " " + str(highest_percent) + "%")


def plot_filters(model):
    """
    Vizualizacia filtrov v konvolucnej vrstve
    :param model: model neuronovej siete
    """
    # prva konvolucna vrstva
    conv_layer = model.layers[0]

    # filtre vo vrstve
    filters = conv_layer.get_weights()[0]

    # normalizacia
    filters = (filters - filters.min()) / (filters.max() - filters.min())

    rgb = ["red", "green", "blue"]

    plot_id = 1
    plt.figure(figsize=(10, 10))

    # zobrazenie prvych 5 filtrov
    for i in range(5):
        # rgb spolu
        filter_multicolor = filters[:, :, :, i]
        # plot
        plt.subplot(5, 4, plot_id)
        plot_image(filter_multicolor, "Filter " + str(i) + ": red + green + blue")
        plot_id += 1

        for color in rgb:
            # oddelene rgb
            filter_greyscale = filter_multicolor[:, :, rgb.index(color)]
            # plot
            plt.subplot(5, 4, plot_id)
            plot_image(filter_greyscale, color)
            plot_id += 1

    plt.show()


''' ********************** Classifier ********************** '''


def prepare_data(train_labels, test_labels):
    """
    Generovanie obrazkov, vytvorenie trenovacej, testovacej a validacnej mnoziny
    :param test_labels: testovacie data - dataframe
    :param train_labels: trenovacie data - dataframe
    :return: trenovacia, testovacia a validacna mnozina
    """
    # img generator pre trenovacie a validacne data
    train_val_gen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)
    # img generator pre testovacie data
    test_gen = ImageDataGenerator(rescale=1.0 / 255.0)

    # testovacie data
    test_img_generator = test_gen.flow_from_dataframe(
        dataframe=test_labels,
        directory="data/images",
        x_col="filename",
        y_col="masterCategory",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    # trenovacie data
    train_img_generator = train_val_gen.flow_from_dataframe(
        dataframe=train_labels,
        directory="data/images",
        x_col="filename",
        y_col="masterCategory",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        subset="training",
        class_mode="categorical",
        shuffle=False
    )

    # validacne data
    validation_img_generator = train_val_gen.flow_from_dataframe(
        dataframe=train_labels,
        directory="data/images",
        x_col="filename",
        y_col="masterCategory",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        subset="validation",
        class_mode="categorical",
        shuffle=False
    )

    return test_img_generator, train_img_generator, validation_img_generator


def get_model(in_shape, out_shape, kernel_size, regularizer=None):
    """
    Vytvorenie neuronovej siete
    :param kernel_size: velkost kernelu
    :param regularizer: None alebo l2
    :param in_shape: shape vstupu
    :param out_shape: shape vystupu (pocet neuronov)
    :return: model neuronovej siete
    """

    model = Sequential()

    model.add(Conv2D(32, kernel_size, activation="relu",
                     input_shape=in_shape))
    model.add(MaxPooling2D())

    model.add(Conv2D(64, kernel_size, activation="relu",
                     kernel_regularizer=regularizer))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(out_shape, activation="softmax"))

    return model


def train_classifier(saved_model_filename, train_labels, test_labels, learning_rate, kernel_size, regularizer=None):
    """
    Klasifikacia
    :param kernel_size: velkost kernelu
    :param saved_model_filename: nazov suboru, kam sa ulozi model
    :param regularizer: regularizacia - None alebo l2
    :param learning_rate: parameter ucenia
    :param test_labels: testovacie data - dataframe
    :param train_labels: trenovacie data - dataframe
    """
    # ziskanie trenovacich, testovacich, validacnych dat
    test_img_generator, train_img_generator, validation_img_generator = prepare_data(train_labels, test_labels)

    # vykreslenie nacitanych obrazkov
    plot_images(train_img_generator, 5, 5)

    # model klasifikatora
    classifier = get_model((32, 32, 3), len(class_names), kernel_size, regularizer)

    # vizualizacia filtrov
    plot_filters(classifier)

    # kompilacia a trenovanie klasifikatora
    classifier.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                       loss='categorical_crossentropy', metrics=['accuracy'])

    steps_per_epoch = math.ceil(train_img_generator.n / BATCH_SIZE)
    validation_steps = math.ceil(validation_img_generator.n / BATCH_SIZE)

    fit = classifier.fit(train_img_generator,
                         verbose=True, epochs=30,
                         steps_per_epoch=steps_per_epoch,
                         validation_data=validation_img_generator,
                         validation_steps=validation_steps)

    # ulozenie modelu do suboru
    classifier.save(saved_model_filename+".h5")

    # ulozenie history
    with open(saved_model_filename+'_history.json', 'w') as file:
        json.dump(fit.history, file)

    # nacitanie modelu zo suboru
    loaded_classifier = load_model(saved_model_filename+".h5")

    # nacitanie history
    with open(saved_model_filename+'_history.json') as file:
        loaded_history = json.load(file)

    # predpoved na testovacich datach
    predict_images(loaded_classifier, test_img_generator)

    # vykreslenie history
    plot_classifier_results(loaded_history)


def predict_images(classifier, test_img_generator):
    """
    Predpoved tried pre testovaciu mnozinu
    :param classifier: klasifikator
    :param test_img_generator: generator obrazkov pre testovaciu mnozinu
    """
    print("\n********************************** evaluation **********************************")

    # predpovedane triedy pre testovacie data
    test_img_generator.reset()
    predicted = classifier.predict(test_img_generator)

    # classification report
    print(classification_report(test_img_generator.classes, np.argmax(predicted, axis=1),
                                target_names=class_names, digits=4))

    # graf pre predpovedane obrazky + %
    plot_predictions(test_img_generator, predicted)

    # confusion matica
    plot_confusion_matrix(test_img_generator.classes, np.argmax(predicted, axis=1))


if __name__ == '__main__':
    # nacitanie csv
    styles = pd.read_csv('data/styles.csv', index_col="id", warn_bad_lines=False, error_bad_lines=False)

    # zmazanie zaznamov
    styles.drop(styles[styles.masterCategory == "Home"].index, inplace=True)
    styles.drop(styles[styles.masterCategory == "Sporting Goods"].index, inplace=True)
    styles.drop(styles[styles.masterCategory == "Free Items"].index, inplace=True)

    # nazvy obrazkov do dataframe
    fnames = []
    for indx in styles.index:
        fnames.append(str(indx) + ".jpg")
    styles["filename"] = fnames

    # rozdelenie dat na trenovacie a testovacie
    train, test = train_test_split(styles[["filename", "masterCategory"]], test_size=0.2, random_state=1)

    # klasifikator
    # train_classifier("models/overfit", train, test, 0.001, (3, 3))                         # bez regularizacie
    train_classifier("models/overfit_l2", train, test, 0.001, (3, 3), "l2")                  # s regularizaciou
    #
    # train_classifier("models/learning_rate_0_0001", train, test, 0.0001, (3, 3), "l2")     # mensi learning rate
    # train_classifier("models/learning_rate_0_01", train, test, 0.01,  (3, 3), "l2")        # vacsi learning rate
    #
    # train_classifier("models/kernel_size_5", train, test, 0.001, (5, 5), "l2")             # iny kernel size
