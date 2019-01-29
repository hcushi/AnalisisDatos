import pandas as pd
import csv
import os

__STOPWORDS_PATH  = "resources" + os.path.sep + "stopwords.csv"
__DICTIONARY_PATH = "resources" + os.path.sep + "dictionary.csv"
__LEMMATIZATION_PATH = "resources" + os.path.sep + "lemmatization.csv"

# Lee el archivo csv que tiene la lista de stopwords
#
# return: vector de stopwords
def stopwords():
    return pd.read_csv(__STOPWORDS_PATH)

# Lee el archivo csv que tiene la lista de palabras positivas (col 1) 
# y negrativas (col 2)
#
# return: matriz con la lista de palabras positivas y negativas
def dictionary():
    return pd.read_csv(__DICTIONARY_PATH, sep=";")


# Lee el archivo csv que tiene la lista de lematizacion
#
# return: dataframe
def lemmas():
    return pd.read_csv(__LEMMATIZATION_PATH, sep=";")


# Lee el archivo csv que tiene la lista de entrenamiento
#
# return: dataframe
def readFile(file_path):
    return pd.read_csv(file_path, sep=";")