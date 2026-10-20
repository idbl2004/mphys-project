import csv, matplotlib.pyplot as plt
from numpy import abs, array

# read in the CSV file
with open('results/type2ChunkedResults.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    data = list(reader)

# remove the data points where epeak is > 1000
data = [row for row in data if float(row['epeak']) <= 1000]
sliceIDs = [int(row['sliceId']) for row in data]
epeak = [float(row['epeak']) for row in data]
epeakLow = [float(row['epeakLow']) for row in data]
epeakHigh = [float(row['epeakHigh']) for row in data]

# remove the data points where epeak is <0
data = [row for row in data if float(row['epeak']) >= 0]
sliceIDs = [int(row['sliceId']) for row in data]
epeak = [float(row['epeak']) for row in data]
epeakLow = [float(row['epeakLow']) for row in data]
epeakHigh = [float(row['epeakHigh']) for row in data]

# find the error values for epeak
epeakError = [[(e - low) for e, low in zip(epeak, epeakLow)], [(high - e) for e, high in zip(epeak, epeakHigh)]]
epeakError = array(epeakError)  # convert to numpy array for easier manipulation
epeakError = abs(epeakError)  # ensure all error values are positive



# create the plot
plt.figure(figsize=(12, 6))

plt.scatter(
    sliceIDs,
    epeak,
#    yerr=epeakError,
#    fmt='o',
    linestyle='None',
    color='r',
#    ecolor='gray',
#    capsize=3
    )


plt.tight_layout()
plt.show()