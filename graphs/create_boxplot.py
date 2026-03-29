import matplotlib.pyplot as plt
import numpy as np
import math as mth

filename = 'tagged-concepts-per-course'
infile = f'./{filename}.txt'
outfile = f'./{filename}.png'

with open(infile, 'r') as f:
    data = [int(line.rstrip("\n")) for line in f if line.strip()]

fig = plt.figure(figsize =(5, 2))
plt.xlabel('Words')
plt.title('Words Per Slide')
plt.gca().axes.get_yaxis().set_visible(False)

plt.boxplot(data, vert=False, notch=False)
plt.savefig(outfile)

print(sum(data)/len(data))
