import dataloader as dl
import sentiment as sm 
import simplifier as sp
import os
import pandas as pd
import csv

__DATA_COMPLETA = "datasets" + os.path.sep + "datacompleta" + os.path.sep + "datadepu.xlsx"
__DATA_POLARIDAD = "datasets" + os.path.sep + "datacompleta" + os.path.sep + "dataconpolaridad.csv"
if __name__ == '__main__':
    model, vocabulary, classes = sm.initNeuralNet()
    #Cargar datos de CSV
    print(__DATA_COMPLETA)
   
   
    excelFile = pd.read_excel(__DATA_COMPLETA)
    tweets=excelFile['text'].values
    print(tweets)
    #id_tweet=tweets["id_tweet"]
    #text=tweets["text"]
    #tweets = db.getAllUnprocessedTweets().values
    polaridad=[]
    num = 0
    data_conlaridad=open(__DATA_POLARIDAD,"w",encoding="utf8")
    for text in tweets:
        #print(text)
        text = sp.minimize(text)
        print(text)
        polarityDict  = sm.dictionary(text, True) 
        polarityModel = sm.neuralNet(text, model, vocabulary, classes, True)
        #escribir csv
        polaridad.append(polarityModel)
        data_conlaridad.write("{};{}\n".format(polaridad[num],text))
        #db.saveResult([id_tweet, polarityDict, polarityModel, text])
        num += 1
        if num%10 == 0:
            print("{} tweets procesados".format(num))

    data_conlaridad.close()
    print("{} tweets procesados en total".format(num))