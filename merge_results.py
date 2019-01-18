import os

import numpy as np
import pandas as pd


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', required=True)
    parser.add_argument('-p', '--predict-file', required=True)
    parser.add_argument('-o', '--output-file')
    args = parser.parse_args()
    name, ext = os.path.splitext(args.input_file)
    args.output_file = args.output_file or '{}_{}{}'.format(name, 'predict', ext)
    names = ['question', 'text', 'label']
    df = pd.read_csv(args.input_file, sep='\t', header=None, names=names)
    predictions = np.loadtxt(args.predict_file)
    df['predictions'] = np.argmax(predictions, axis=-1)
    df.to_csv(args.output_file, index=False, sep='\t', header=False, float_format='%.0f')


if __name__ == '__main__':
    main()
