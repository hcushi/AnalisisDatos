import csv
import os

# Path de salida del archivo csv
__output_csv_path = "out" + os.path.sep

# Escribe datos en un archivo CSV
#
# row: arreglo con los datos a guardar
# file_path: path del archivo
def writeCSV(row, file_path):
    with open(file_path, 'a', encoding="utf-8", newline="") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=";", quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(row)

    csv_file.close()