# 參考 https://gist.github.com/Deepblue129/a51defb2dc9506945f58a165026d1a96#file-subword_text_tokenizer-py

import os
import sys
import argparse

pardir = os.path.realpath(os.path.join(os.path.abspath(__file__), '../../..'))
if pardir not in sys.path:
    sys.path.insert(0, pardir)

from ape.preprocess import tokenizer, subword_text_tokenizer
# from ape.dataset.billion import BillionWord


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-output_filename', type=str, default='./subword.vocab')
    parser.add_argument('-load_from', type=str, default=None)

    parser.add_argument('-corpus_filepattern', type=str, default=None)
    parser.add_argument('-vocab_filepattern', type=str, default=None)

    parser.add_argument('-min_count', type=int, default=5)
    parser.add_argument('-corpus_max_lines', type=int, default=None)
    parser.add_argument('-num_iterations', type=int, default=4)

    parser.add_argument('-split_on_newlines', type=bool, default=True)

    opt = parser.parse_args()

    # opt.corpus_filepattern = BillionWord.corpus_file_paths()

    opt.output_filename = './data/dual_gan/subword.vocab'
    # opt.corpus_filepattern = [BillionWord.path + '/*', ]
    opt.corpus_filepattern = '/media/data/datasets/1-billion-word-language-modeling-benchmark-r13output/' \
                             'training-monolingual.tokenized.shuffled/*'
    opt.corpus_max_lines = 1000000

    if opt.corpus_filepattern:
        token_counts = tokenizer.corpus_token_counts(opt.corpus_filepattern,
                                                     opt.corpus_max_lines,
                                                     split_on_newlines=opt.split_on_newlines)
    elif opt.vocab_filepattern:
        token_counts = tokenizer.vocab_token_counts(opt.vocab_filepattern,
                                                    opt.corpus_max_lines)
    elif opt.load_from is None:
        raise ValueError()

    encoder = subword_text_tokenizer.SubwordTextTokenizer()

    if opt.load_from is not None:
        encoder._load_from_file('./my.subword_text_encoder')
    else:
        encoder.build_from_token_counts(token_counts, opt.min_count, opt.num_iterations)
        encoder.store_to_file(opt.output_filename)

    # print(token_counts)

    # ['this_', 'is_', 'a_', 'test', '_', ' ...', '_']
    encoded = encoder.encode('this is a test ...')

    # print(encoded)


if __name__ == '__main__':
    main()
