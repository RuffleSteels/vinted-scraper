import csv
import statistics

def read_prices_and_stats(csv_path):
    values = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                try:
                    values.append(float(row[1]))
                except ValueError:
                    pass
    mean_val = statistics.mean(values) if values else None
    stdev_val = statistics.stdev(values) if len(values) > 1 else None
    return mean_val, stdev_val

if __name__ == "__main__":
    mean_val, stdev_val = read_prices_and_stats('prices.csv')
    print("Mean:", mean_val)
    print("Standard Deviation:", stdev_val)