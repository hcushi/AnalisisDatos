import dataloader as dl
import datasaver as ds
import simplifier as sp
import tensorflow as tf
import tflearn
import os
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from tflearn.data_utils import to_categorical
import os


#modelo de red neuronal profunda
__POSITIVE = "positivo"
__NEGATIVE = "negativo"

__TRAINING_SET = "datasets" + os.path.sep + "train-depurada2.csv"
__VOCABULARY = "models" + os.path.sep + "vocabulary.csv"
__CLASSES = "models" + os.path.sep + "classes.csv"
__NEURAL_MODEL = "models" + os.path.sep + "model1.tflearn"

def __getVocabulary(document, min_df):
    cv = CountVectorizer(min_df=min_df)
    cv.fit(document)

    return cv.get_feature_names()



def __prepareData(document, labels, vocabulary):
    cv = CountVectorizer(vocabulary=vocabulary)
    le = LabelEncoder()

    x = cv.fit_transform(document).toarray()
    y_vector = le.fit_transform(labels)
    
    classes = le.classes_

    num_classes = len(classes)
    y = to_categorical(y_vector, nb_classes=num_classes)

    return x, y, classes

def __buildModel(train_x, train_y, vocab_size, num_classes, learning_rate):
    tf.reset_default_graph()

    # input layers
    net = tflearn.input_data([None, vocab_size])

    # hiden layers
    #net = tflearn.embedding(net, input_dim=10000, output_dim=128)
    #net = tflearn.lstm(net, 128, dropout=0.8)
    net = tflearn.fully_connected(net, 10000, activation='ReLU')
    net = tflearn.fully_connected(net, 1000, activation='ReLU')
    # output layer
    net = tflearn.fully_connected(net, num_classes, activation='softmax')
    
    net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')


    # net = tflearn.fully_connected(net, 40, activation='ReLU')
    # net = tflearn.fully_connected(net, 60, activation='ReLU')
    # net = tflearn.fully_connected(net, 10, activation='ReLU')
    
    # net = tflearn.fully_connected(net, num_classes, activation='softmax')
    # net = tflearn.regression(net, optimizer='sgd', learning_rate=learning_rate, loss='categorical_crossentropy')

    model = tflearn.DNN(net)
    model.fit(train_x, train_y, show_metric=True, batch_size=32)    
    return model


def initNeuralNet():
    num_pos = 80
    num_neg = 80
    

    if os.path.exists(__VOCABULARY) and os.path.exists(__CLASSES):
        print("-----------------------------------------------------")
        print("Cargando vocabulario del archivo:", __VOCABULARY)
        print("Cargando clases del archivo", __CLASSES)
        vocabulary = dl.readFile(__VOCABULARY)["vocabulary"].values
        classes = dl.readFile(__CLASSES)["classes"].values
        print("\nTamaño del Vocabulario:", len(vocabulary))
        print("Numero de clases:", len(classes))
        print("-----------------------------------------------------")
    else:
        print("-----------------------------------------------------")
        if os.path.exists(__TRAINING_SET):     
            print("Cargando set de entrenamiento del archivo:", __TRAINING_SET)
            trainingDataSet = dl.readFile(__TRAINING_SET)
        #else:
        #    print("Descargando set de entrenamiento de la base de datos")
        #    trainingDataSet = __createDatasetFromDtabase(num_pos, num_neg, __TRAINING_SET)

        vocabulary = __getVocabulary(trainingDataSet["tweet"], 2)
        train_x, train_y, classes = __prepareData(trainingDataSet["tweet"].values, trainingDataSet["polaridad"].values, vocabulary)
        
        print("\nCreando un nuevo vocabulario:", __VOCABULARY)
        ds.writeCSV(["vocabulary"], __VOCABULARY)
        for row in vocabulary:
            ds.writeCSV([row], __VOCABULARY)

        print("Creando nuevas clases:", __CLASSES)
        ds.writeCSV(["classes"], __CLASSES)
        for row in classes:
            ds.writeCSV([row], __CLASSES)

        print("\nTamaño del set de entrenamiento:", len(trainingDataSet))
        print("Tamaño del Vocabulario:", len(vocabulary))
        print("Numero de clases:", len(classes))
        print("-----------------------------------------------------")

    if os.path.exists(__NEURAL_MODEL + ".index"):
        print("-----------------------------------------------------")
        print("Cargando modelo de red neuronal:", __NEURAL_MODEL)
        model = __loadModel(__NEURAL_MODEL, len(vocabulary), len(classes), 0.01)
        print("-----------------------------------------------------")
    else:
        print("-----------------------------------------------------")
        print("Entrenando un nuevo modelo de red neuronal")
        train_x, train_y, classes = __prepareData(trainingDataSet["tweet"].values, trainingDataSet["polaridad"].values, vocabulary)
        model = __buildModel(train_x, train_y, len(vocabulary), len(classes), 0.01)
        print("Guardando el modelo de red neuronal:", __NEURAL_MODEL)
        model.save(__NEURAL_MODEL)
        print("-----------------------------------------------------")
    
    return model, vocabulary, classes

def __loadModel(path, vocab_size, num_classes, learning_rate):
    tf.reset_default_graph()

    # input layers
    net = tflearn.input_data([None, vocab_size])

    # hiden layers
    #net = tflearn.embedding(net, input_dim=10000, output_dim=128)
    #net = tflearn.lstm(net, 128, dropout=0.80)
    net = tflearn.fully_connected(net, 10000, activation='ReLU')
    net = tflearn.fully_connected(net, 1000, activation='ReLU')

    # output layer
    net = tflearn.fully_connected(net, num_classes, activation='softmax')
    
    #net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')

    model = tflearn.DNN(net)
    model.load(path)

    return model


def __predict(vectors, model):
    return model.predict(vectors)

def __decodeY(matrix):    
    decode = []
    for row in matrix:
        max_prob = max(row)
        pos = 0
        for value in row:
            if value == max_prob:
                decode.append(pos)
                break
            else:
                pos += 1
    return decode

def __validateModel(x_test, y_test, model, classes):
    y_predict = __predict(x_test, model)
    y_predict = __decodeY(y_predict)
    y_test = __decodeY(y_test)
    pr=precision_score(y_test, y_predict)
    cm = confusion_matrix(y_test, y_predict)
    print("\nMATRIZ DE CONFUSION")
    print("  ", classes)
    for x in range(0, len(classes)):
        print("{} \t".format(classes[x]), cm[x])
    print("Precision_score: ")
    print(pr)
def dictionary(text, processed=False):
    if not processed:
        text = sp.minimize(text)
    
    dictionary = []
    for [positive, negative] in dl.dictionary().values:
        dictionary.append([sp.getStem(positive), sp.getStem(negative)])
    
    pos_count = 0
    neg_count = 0
    for [positive, negative] in dictionary:
        pos_count += text.count(positive)
        neg_count += text.count(negative)

    result = pos_count - neg_count

    if result > 0: 
        return __POSITIVE
    else: 
        return __NEGATIVE
def neuralNet(text, model, vocabulary, classes, processed=False):
    if not processed:
        text = sp.minimize(text)

    vector = __vectorize([text], vocabulary)
    prediction = __predict(vector, model)
    prediction_decoded = __decodeY(prediction)
    polarity = __decodeLabel(prediction_decoded, classes)
    return polarity[0]


def __vectorize(document, vocabulary):
    cv = CountVectorizer(vocabulary=vocabulary)
    return cv.fit_transform(document).toarray()

def __decodeLabel(vector, classes):
    le = LabelEncoder()
    le.fit(classes)
    return le.inverse_transform(vector)
if __name__ == '__main__':
    model, vocabulary, classes = initNeuralNet()

    path = "datasets" + os.path.sep + "validation-depurada2.csv"
    if os.path.exists(path):
        testSet = dl.readFile(__TRAINING_SET)
    #else:
    #    testSet = __createDatasetFromDtabase(30, 30, 30, path)

    test_x, test_y, classes = __prepareData(testSet["tweet"].values, testSet["polaridad"].values, vocabulary)

    __validateModel(test_x, test_y, model, classes)