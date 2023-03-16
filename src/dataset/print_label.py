import csv
csvFile = open("Celeb_label.csv", "r")
reader = csv.reader(csvFile)
for item in reader:
    print(item[0], item[1])