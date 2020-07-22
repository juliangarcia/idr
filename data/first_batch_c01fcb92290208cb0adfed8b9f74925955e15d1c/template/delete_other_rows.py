import os
import csv

directory = 'results/'

for filename in os.listdir(directory):
    with open('reduced_results/' + filename, 'w', newline='\n') as out_file:
        writer = csv.writer(out_file)
        with open('results/' + filename, 'r') as in_file:
            reader = csv.reader(in_file, quoting=csv.QUOTE_NONNUMERIC)
            for i, row in enumerate(reader):
                if i % 2 == 0:
                    writer.writerow(row)
