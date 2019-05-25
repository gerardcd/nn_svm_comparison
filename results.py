import matplotlib.pyplot as plt
import pandas

results = pandas.read_csv('results.csv')

_, ax = plt.subplots(figsize=(8, 6))

results.plot(x='Sample size', y='NN', ax=ax, style='+-')
results.plot(x='Sample size', y='SVM', ax=ax, style='+-')

ax.hlines(y=0.8, xmin=0, xmax=1024, linewidth=2, color='r', linestyles='dashdot')

plt.show()