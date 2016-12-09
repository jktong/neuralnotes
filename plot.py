import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import ujson as json

data_path = sys.argv[1]
flag = sys.argv[2]

#plt.xkcd()

def plot_line(x, y, title, x_label, y_label, color):
    plt.figure(title)
    plt.title(title)
    plt.plot(x, y, ls='solid', lw=2.5, c=color)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

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

if flag == 'train':
    with open(data_path, 'r') as f:
        data = defaultdict(list)
        for line in f:
            epoch, loss, acc = line.strip().split()
            epoch = int(epoch)
            loss = float(loss)
            acc = float(acc)
            data['epochs'].append(epoch)
            data['losses'].append(loss)
            data['accs'].append(acc)
    plot_line(data['epochs'], data['losses'], 'Cross-entropy loss vs. epoch', 'Epoch', 'Cross-entropy loss', 'r')
    plot_line(data['epochs'], data['accs'], 'Accuracy vs. epoch', 'Epoch', 'Accuracy', 'b')
    
else:
    mode, k = flag.split('_')
    title = '{}-note accuracy per context size, on {} data'.format(k, mode)
    with open(data_path, 'r') as f:
        data = json.load(f).items()
        data.sort(key=lambda p: int(p[0]))
        data = data[:9]
    contexts = [int(datum[0]) for datum in data]
    accs = [datum[1] for datum in data]
    labels = ['{} to {}'.format(contexts[i-1],contexts[i])for i in xrange(1, len(contexts))]
    diffs = [accs[i]-accs[i-1] for i in xrange(1, len(accs))]
    print contexts
    plot_hist(labels, diffs, title, 'Context size', 'Accuracy', 'g')
    #plot_hist(contexts, accs, title, 'Context size', 'Accuracy', 'y')
    

plt.show()
