import argparse
import math
import ujson as json

from abdi import Model
from get_data import get_iterator_per_song_per_context


def main(args):
    perf = {}
    contexts = range(2,33)
    max_N = max(contexts)
    m = Model(args.name, max_N=max_N, k=args.top_k).load(args.load_path)
    for N in contexts:
        total = total_correct = 0
        for data in get_iterator_per_song_per_context([N], max_N, pad_end=False, mode=args.mode):
            S, _, _ = data.shape
            acc = m.accuracy(data)
            total += S
            total_correct += math.floor(S*acc + 0.5)
        perf[N] = total_correct / total
        print N, perf[N]
    with open(args.save_path, 'w') as f:
        json.dump(perf, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name',
                        help='Name for model.',
                        default='main')
    parser.add_argument('-m', '--mode',
                        help='Check performance on a dataset.',
                        choices=['train', 'validate', 'test'],
                        required=True)
    parser.add_argument('-k', '--top_k',
                        help='Check performance using top k predicted notes.',
                        type=int,
                        default=6)
    parser.add_argument('-l', '--load_path',
                        help='Path to model on disk. Will initialize model params randomly if not given. Required',
                        required=True)
    parser.add_argument('-s', '--save_path',
                        help='Save path for model performance. Required.',
                        required=True)
    args = parser.parse_args()
    main(args)
