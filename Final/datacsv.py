import csv

def saveTrainingData(data):
    with open('datadepurada.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer=csv.writer(csvfile)
        writer.writerow(data)
    return True