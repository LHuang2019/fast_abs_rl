""" run decoding of rnn-ext + abs + RL (+ rerank)"""
import re
import argparse
import json
import os
import sys
import nltk
from os.path import join
from datetime import timedelta
from time import time
from collections import Counter, defaultdict
from itertools import product
from functools import reduce
import operator as op

from cytoolz import identity, concat, curry

import torch
from torch.utils.data import DataLoader
from torch import multiprocessing as mp

from .data.batcher import tokenize

from .decoding import Abstractor, RLExtractor, DecodeDataset, BeamAbstractor
from .decoding import make_html_safe

def preprocess_json(data_list):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    counter = 0
    json_list = []
    for article in data_list:
        article_lines = tokenizer.tokenize(article)

        js_example = {}
        js_example['id'] = " "
        js_example['article'] = article_lines
        js_example['abstract'] = " "
        json_list.append(js_example)

    return json_list

def summary_postprocessing(summary_list):

    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    processed_list = []

    for summary in summary_list:
        sentences = sent_tokenizer.tokenize(summary)

        # capitalize first words of every sentences
        sentences = [sent.capitalize() for sent in sentences]
        text = ' '.join(sentences)

        # remove spaces before punctuations
        text = re.sub(r'\s([,?.!"](?:\s|$))', r'\1', text)

        # remove redundant punctuations
        text = re.sub(r'[\?\.\!\,]+(?=[\?\.\!\,])', '', text)

        processed_list.append(text)

    return processed_list

def decode(data_list):
    
    data_list = preprocess_json(data_list)

    cuda = torch.cuda.is_available()
    model_dir = os.path.dirname(os.path.realpath(__file__)) + '/pretrained/'
    batch_size = 32
    beam_size = 5
    diverse = 1.0
    max_len = 30

    start = time()
    # setup model
    with open(join(model_dir, 'meta.json')) as f:
        meta = json.loads(f.read())
    if meta['net_args']['abstractor'] is None:
        # NOTE: if no abstractor is provided then
        #       the whole model would be extractive summarization
        assert beam_size == 1
        abstractor = identity
    else:
        if beam_size == 1:
            abstractor = Abstractor(join(model_dir, 'abstractor'),
                                    max_len, cuda)
        else:
            abstractor = BeamAbstractor(join(model_dir, 'abstractor'),
                                        max_len, cuda)
    extractor = RLExtractor(model_dir, cuda=cuda)

    # setup loader
    def coll(batch):
        articles = list(filter(bool, batch))
        return articles
    dataset = DecodeDataset(data_list)

    n_data = len(dataset)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=coll
    )

    summary_list = []
    
    # Decoding
    i = 0
    with torch.no_grad():
        for i_debug, raw_article_batch in enumerate(loader):
            tokenized_article_batch = map(tokenize(None), raw_article_batch)
            ext_arts = []
            ext_inds = []
            for raw_art_sents in tokenized_article_batch:
                ext = extractor(raw_art_sents)[:-1]  # exclude EOE
                if not ext:
                    # use top-5 if nothing is extracted
                    # in some rare cases rnn-ext does not extract at all
                    ext = list(range(5))[:len(raw_art_sents)]
                else:
                    ext = [i.item() for i in ext]
                ext_inds += [(len(ext_arts), len(ext))]
                ext_arts += [raw_art_sents[i] for i in ext]
            if beam_size > 1:
                all_beams = abstractor(ext_arts, beam_size, diverse)
                dec_outs = rerank_mp(all_beams, ext_inds)
            else:
                dec_outs = abstractor(ext_arts)
            assert i == batch_size*i_debug

            for j, n in ext_inds:
                decoded_sents = [' '.join(dec) for dec in dec_outs[j:j+n]]
                summary_list.append(make_html_safe(('\n'.join(decoded_sents).encode('ascii', 'ignore')).decode('ascii')))
                i += 1
                print('{}/{} ({:.2f}%) decoded in {} seconds\r'.format(
                    i, n_data, i/n_data*100,
                    timedelta(seconds=int(time()-start))
                ), end='')

    print()
    return summary_postprocessing(summary_list)

_PRUNE = defaultdict(
    lambda: 2,
    {1:5, 2:5, 3:5, 4:5, 5:5, 6:4, 7:3, 8:3}
)

def rerank(all_beams, ext_inds):
    beam_lists = (all_beams[i: i+n] for i, n in ext_inds if n > 0)
    return list(concat(map(rerank_one, beam_lists)))

def rerank_mp(all_beams, ext_inds):
    beam_lists = [all_beams[i: i+n] for i, n in ext_inds if n > 0]
    with mp.Pool(8) as pool:
        reranked = pool.map(rerank_one, beam_lists)
    return list(concat(reranked))

def rerank_one(beams):
    @curry
    def process_beam(beam, n):
        for b in beam[:n]:
            b.gram_cnt = Counter(_make_n_gram(b.sequence))
        return beam[:n]
    beams = map(process_beam(n=_PRUNE[len(beams)]), beams)
    best_hyps = max(product(*beams), key=_compute_score)
    dec_outs = [h.sequence for h in best_hyps]
    return dec_outs

def _make_n_gram(sequence, n=2):
    return (tuple(sequence[i:i+n]) for i in range(len(sequence)-(n-1)))

def _compute_score(hyps):
    all_cnt = reduce(op.iadd, (h.gram_cnt for h in hyps), Counter())
    repeat = sum(c-1 for g, c in all_cnt.items() if c > 1)
    lp = sum(h.logprob for h in hyps) / sum(len(h.sequence) for h in hyps)
    return (-repeat, lp)
