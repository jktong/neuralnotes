import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import ujson as json

context = sys.argv[1]
mode = sys.argv[2]

#plt.xkcd()


def plot_hist(x, y, title, x_label, y_label, color):
    #plt.figure(title)
    fig, ax = plt.subplots()
    xx = range(len(x))
    bars = ax.bar(xx, y, color=color)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(xx)
    ax.set_xticklabels(x)

    
template = './performance/lstm3_{}'.format(mode) + '_k{}.json'
x = []
y = []

for k in xrange (1,7):
    path = template.format(k)
    #x.append('  '.format(k))
    x.append(k)
    with open(path, 'r') as f:
        y.append(json.load(f)[context])

xx = []
yy = []

xx = ['                  {} to {}'.format(x[i-1],x[i])for i in xrange(1, len(x))]
yy = [y[i]-y[i-1] for i in xrange(1, len(y))]
title = 'k-note accuracy gain with increasing k, for context {} on {} data'.format(context, mode)
plot_hist(xx,yy,title,'k', 'Accuracy', 'red')
plt.show()
