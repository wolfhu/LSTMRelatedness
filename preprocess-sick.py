"""
Preprocessing script for SICK data.

"""

import os
import glob

def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

def build_vocab(filepaths, dst_path, lowercase=True):
    vocab = set()
    for filepath in filepaths:
        with open(filepath) as f:
            for line in f:
                if lowercase:
                    line = line.lower()
                vocab |= set(line.split())
    with open(dst_path, 'w') as f:
        for w in sorted(vocab):
            f.write(w + '\n')

def split(filepath, dst_dir):
    with open(filepath) as datafile, \
         open(os.path.join(dst_dir, 'a.txt'), 'w') as afile, \
         open(os.path.join(dst_dir, 'b.txt'), 'w') as bfile,  \
         open(os.path.join(dst_dir, 'id.txt'), 'w') as idfile, \
         open(os.path.join(dst_dir, 'sim.txt'), 'w') as simfile:
            datafile.readline()
            for line in datafile:
                i, a, b, sim, ent = line.strip().split('\t')
                idfile.write(i + '\n')
                afile.write(a + '\n')
                bfile.write(b + '\n')
                simfile.write(sim + '\n')

if __name__ == '__main__':
    print('=' * 80)
    print('Preprocessing SICK dataset')
    print('=' * 80)

    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    sick_dir = os.path.join(data_dir, 'sick')
    lib_dir = os.path.join(base_dir, 'lib')
    train_dir = os.path.join(sick_dir, 'train')
    dev_dir = os.path.join(sick_dir, 'dev')
    test_dir = os.path.join(sick_dir, 'test')
    make_dirs([train_dir, dev_dir, test_dir])

    # split into separate files
    split(os.path.join(sick_dir, 'SICK_train.txt'), train_dir)
    split(os.path.join(sick_dir, 'SICK_trial.txt'), dev_dir)
    split(os.path.join(sick_dir, 'SICK_test_annotated.txt'), test_dir)

    # get vocabulary
    build_vocab(
        glob.glob(os.path.join(sick_dir, '*/*.toks')),
        os.path.join(sick_dir, 'vocab.txt'))
    build_vocab(
        glob.glob(os.path.join(sick_dir, '*/*.toks')),
        os.path.join(sick_dir, 'vocab-cased.txt'),
        lowercase=False)
