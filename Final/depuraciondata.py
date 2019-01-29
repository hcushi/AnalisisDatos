import datacsv as db
import simplifier as sp
import re
import csv
import os
import pandas as pd
import numpy as np
import dataloader as dl 


CALIFICADO = "datasets" + os.path.sep  + "train-telefonia.csv"

# Obtiene los datos del csv
polaridades = dl.readFile(CALIFICADO)["polaridad"].values
tweets = dl.readFile(CALIFICADO)["tweet"].values

#reorganizamos la dimencion de los vectorees
polaridades = polaridades.reshape(len(polaridades),1)
tweets = tweets.reshape(len(tweets),1)

datos = np.concatenate((polaridades, tweets), axis=1)
print(datos)
processed = 0
num_saved = 0
with open('train-depurada.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer=csv.writer(csvfile)
    headers = ('polaridad', 'tweet')
    writer.writerow(headers)
    for [polaridad, texto_original] in datos:
        tweet_limpio = sp.minimize(texto_original)
        if len(tweet_limpio) > 4:
                #saved = db.saveTrainingData([polaridad.replace(" ", ""), tweet_limpio])
            writer.writerow([polaridad.replace(" ", ""), tweet_limpio])
                #if saved:
                #num_saved += 1

        processed += 1
        if processed%10 == 0:
            print("{} tweet procesados".format(processed))
            #print("{} tweet guardados en la base".format(num_saved))





 
    
