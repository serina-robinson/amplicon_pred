#!/usr/bin/env python

import subprocess
from sys import argv
import os

from Bio import SearchIO

from lib.data_preparation.run_hmmpfam import *
from lib.data_preparation.parse_hmmpfam import *
import argparse

def define_arguments():
    parser = argparse.ArgumentParser(description = "Extract 34 aa using hmmalign.")
    parser.add_argument("-i", type = str, required = True,
                        help = "Input fasta directory.")
    parser.add_argument("-o", type = str, required = True,
                        help = "Output fasta directory.")
    return parser

def extract_34_aa(fasta_dir, out_dir):
    run_hmmpfam2('AMP-binding.hmm', fasta_dir, 'temp_hmm.txt')

    out_file = open(out_dir, 'w')

    for result in SearchIO.parse('temp_hmm.txt', 'hmmer2-text'):
        for i, hit in enumerate(result.hits):
            if hit.id == 'AMP-binding':
                hsp = result.hsps[i]
                seq = hsp.query.seq
                seq = remove_insertions(seq)
                seq34 = extract_34(seq)
                header = make_header(result.id, result.description)
                out_file.write(">%s\n%s\n" % (header, seq34))


if __name__ == "__main__":
    parser = define_arguments()
    args = parser.parse_args()
    in_dir = args.i
    out_dir = args.o
    
    extract_34_aa(in_dir, out_dir)
    
