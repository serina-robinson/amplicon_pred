#!/usr/bin/env python

"""
Script to train, cross-validate and test a random forest classifier
"""

from sys import argv, path
from statistics import stdev
from typing import Any, Dict, List, Tuple, Optional

import random
import argparse
import os
import copy
import pickle

from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

current_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(current_folder)

path.append("%s/lib/data_preparation/" % parent_folder)
path.append("%s/lib/machine_learning/" % parent_folder)
path.append("%s/lib/testing/" % parent_folder)

from lib.machine_learning.train_RF import train_RF, store_model
from lib.data_preparation.get_seq_properties import *
from lib.data_preparation.make_test_set import *
from lib.testing.test_classifier import *

PROPERTIES_15 = "%s/data_preparation/15_aa_properties.txt" % parent_folder
PROPERTIES_4 = "%s/data_preparation/aa_properties.txt" % parent_folder

def define_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description = "Run random forest.")
    
    parser.add_argument("--fasta", required = True, type = str,
                        help = "Fasta directory.")
    parser.add_argument("--trees", required = True, nargs = '+', type = int,
                        help = "Forest sizes to try.")
    parser.add_argument("--depth", required = True, nargs = '+',
                        help = "Maximum tree depths to try.")
    parser.add_argument("--features", required = True, nargs = '+',
                        help = "Feature numbers to try.")
    parser.add_argument("--iterations", required = True, type = int,
                        help = "Number of times to run cross-validation.")
    parser.add_argument("--train_size", required = True, nargs = '+',
                        type = float, help = "Test sizes to try.")
    parser.add_argument("--groups", type = string_to_bool, default = False,
                        help = "Cross-validate group specificities only.")
    parser.add_argument("--cutoffs", nargs = '+', type = int,
                        help = "Minimum support of data points for a class.")
    parser.add_argument("--clean", required = True, type = string_to_bool,
                        default = True, help = "True if you want to start from\
scratch, False if you want to include previous results")
    parser.add_argument("--suffix", required = True, type = str,
                        help = "Suffix for output files.")

    return parser

def parse_depths(depths):
    depth_list = []
    for depth in depths:
        if depth == 'None':
            depth_list.append(None)
        else:
            depth_list.append(int(depth))

    return depth_list

def parse_max_features(max_features):
    feature_list = []

    for max_feature in max_features:
        if max_feature == 'log2' or max_feature == 'auto':
            feature_list.append(max_feature)
        else:
            feature_list.append(float(max_feature))

    return feature_list
        

def crossval(dataset, trees, depths, max_features,
             iterations, train_size, cutoff, suffix):

    results = []

    
    for forest_size in trees:
        for depth in depths:
            for max_feature in max_features:
                result = CrossvalResult(forest_size, depth, max_feature,
                                        iterations, train_size, cutoff,
                                        dataset, suffix)
                for i in range(iterations):
                    
                    forest = train_RF(dataset.training_features,
                                      dataset.training_response,
                                      forest_size, max_feature, depth)

                    accuracy = forest.oob_score_
                    result.add_accuracy(accuracy)

                result.calc_properties()
                results.append(result)

    results.sort(key = lambda result: result.average_accuracy, reverse = True)

    return results
                    
def write_results(results, out_dir):
    out_file = open(out_dir, 'a+')
    out_file.write("\tTraining size\tCutoff\tTrees\tDepth\tMax_features\tIterations\tOOB_score\tStd_dev")
    out_file.write("\n")
    
    for result in results:
        result.write_result(out_file)

    out_file.close()

def parse_results(results_dir):
    results_file = open(results_dir, 'r')
    results = []

    results_file.readline()
    
    for line in results_file:
        line = line.strip()
        features = line.split('\t')

        train_size = float(features[0])
        cutoff = int(features[1])
        trees = int(features[2])
        depth = features[3]
        max_features = features[4]
        iterations = int(features[5])
        accuracy = float(features[6])
        stddev = float(features[7])
        dataset = None

        result = CrossvalResult(trees, depth, max_features, iterations,
                                train_size, cutoff, dataset)
        result.average_accuracy = accuracy
        result.stddev = stddev

        results.append(result)

    results_file.close()

    return results


def test_best_result(all_results):
    best_result = all_results[0]
    trees = best_result.trees
    depth = best_result.depth
    max_features = best_result.max_features
    suffix = best_result.suffix
    
    dataset = best_result.dataset
    
    classifier = train_RF(dataset.training_features, dataset.training_response,
                          trees, max_features, depth)
    # filename = "models/best_rf_monomer_amplicon_model.sav" # save finalized model for monomers
    filename = "models/best_rf_groups_amplicon_model.sav"
    pickle.dump(classifier, open(filename, 'wb'))
    
    store_model(classifier, "best_RF_%s" % suffix)
    
    overall_accuracy = test(classifier, dataset.test_features,
                            dataset.test_response)
    class_accuracies = test_per_class(classifier, dataset)
    print(overall_accuracy)
    print_accuracy_dict(class_accuracies)
 
class CrossvalResult():
    def __init__(self, trees, depth, max_features, iterations, train_size,
                 cutoff, dataset, suffix):
        self.trees = trees
        self.depth = depth
        self.max_features = max_features
        self.iterations = iterations
        self.train_size = train_size
        self.cutoff = cutoff
        self.dataset = dataset
        self.suffix = suffix


        self.depth_ID = self.get_depth_ID()
        self.features_ID = self.get_features_ID()
        
        self.ID = "%d_%s_%s_%d_%.2f_%d_%s" % (self.trees, self.depth_ID,
                                              self.features_ID, self.iterations,
                                              self.train_size, self.cutoff,
                                              self.suffix)

        self.accuracies = []

    def __hash__(self):
        return self.ID

    def write_result(self, out_file):
        out_file.write("%.2f\t%d\t%d\t%s\t%s\t%d\t%.3f\t%.3f\n" % \
                       (self.train_size, self.cutoff, self.trees,
                        self.depth_ID, self.features_ID, self.iterations,
                        self.average_accuracy, self.stddev))

    def get_features_ID(self):
        features_ID = str(self.max_features)
        return features_ID

    def set_suffix(self, suffix):
        self.ID += "_%s" % suffix

    def get_depth_ID(self):
        if not self.depth:
            depth_ID = 'max'
        else:
            depth_ID = str(self.depth)

        return depth_ID

    def add_accuracy(self, accuracy):
        self.accuracies.append(accuracy)

    def set_stdev(self):
        try:
            self.stddev = stdev(self.accuracies)
        except Exception:
            self.stddev = 0.0

    def set_average_accuracy(self):
        assert self.iterations == len(self.accuracies)
        self.average_accuracy = sum(self.accuracies)/self.iterations

    def calc_properties(self):
        self.set_stdev()
        self.set_average_accuracy()

    def get_properties(self):
        return self.ID, self.average_accuracy, self.stddev



    

def main(fasta, trees, depths, max_features, iterations, train_sizes, groups,
         cutoffs, clean, suffix):
    depths = parse_depths(depths)
    max_features = parse_max_features(max_features)
    
    properties_15 = parse_15_properties(PROPERTIES_15)
    properties_4 = parse_4_properties(PROPERTIES_4)
    one_hot = one_hot_encoding()

    if groups:

        dir_15 = 'results_15_groups.txt'
        dir_4 = 'results_4_groups.txt'
        dir_hot = 'results_hot_groups.txt'

        if clean:
            
            all_results_15 = []
            all_results_4 = []
            all_results_hot = []
            
        else:
            assert os.path.isfile(dir_15)
            
            all_results_15 = parse_results(dir_15)
            all_results_4 = parse_results(dir_4)
            all_results_hot(dir_hot)

        if os.path.isfile(dir_15):
            os.remove(dir_15)

        if os.path.isfile(dir_4):
            os.remove(dir_4)
            
        
        for train_size in train_sizes:
            data_15 = DataSet(fasta, properties_15, True)
            data_4 = DataSet(fasta, properties_4, True)
            data_hot = DataSet(fasta, one_hot, True)

            data_15.stratify_data(train_size)
            data_4.stratify_data(train_size)
            data_hot.stratify_data(train_size)

            data_15.write_test_training(train_size, "15_groups_%s" % suffix)
            data_4.write_test_training(train_size, "4_groups_%s" % suffix)
            data_hot.write_test_training(train_size, "onehot_groups_%s" % suffix)

            results_15 = crossval(data_15, trees, depths, max_features,
                                  iterations, train_size, 0, suffix)

            results_4 = crossval(data_4, trees, depths, max_features,
                                 iterations, train_size, 0, suffix)
            
            results_hot = crossval(data_hot, trees, depths, max_features,
                                   iterations, train_size, 0, suffix)

            all_results_15 += results_15
            all_results_4 += results_4
            all_results_hot += results_hot

        all_results_15.sort(key = lambda result: result.average_accuracy, reverse = True)
        all_results_4.sort(key = lambda result: result.average_accuracy, reverse = True)
        all_results_hot.sort(key = lambda result: result.average_accuracy, reverse = True)

        write_results(all_results_15, dir_15)
        write_results(all_results_4, dir_4)
        write_results(all_results_hot, dir_hot)

        print("Testing 15 properties:")
        #test_best_result(all_results_15)
 #       print("Testing 4 properties:")
 #       test_best_result(all_results_4)
 #       print("Testing one-hot properties:")
 #       test_best_result(all_results_hot)
    else:

        dir_15 = 'results_15.txt'
        dir_4 = 'results_4.txt'
        dir_hot = 'results_hot.txt'

        if clean:
            
            all_results_15 = []
            all_results_4 = []
            all_results_hot = []
            
        else:
            assert os.path.isfile(dir_15)
            
            all_results_15 = parse_results(dir_15)
            all_results_4 = parse_results(dir_4)
            all_results_hot = parse_results(dir_hot)

        if os.path.isfile(dir_15):
            os.remove(dir_15)

        if os.path.isfile(dir_4):
            os.remove(dir_4)

        for train_size in train_sizes:
            for cutoff in cutoffs:
                data_15 = DataSet(fasta, properties_15, False, cutoff)
                data_4 = DataSet(fasta, properties_4, False, cutoff)
                data_hot = DataSet(fasta, one_hot, False, cutoff)

                data_15.stratify_data(train_size)
                data_4.stratify_data(train_size)
                data_hot.stratify_data(train_size)

                data_15.write_test_training(train_size, "15_cutoff%d_%s" % (cutoff, suffix))
                data_4.write_test_training(train_size, "4_cutoff%d_%s" % (cutoff, suffix))
                data_hot.write_test_training(train_size, "onehot_cutoff%d_%s" % (cutoff, suffix))

                results_15 = crossval(data_15,
                                      trees, depths, max_features,
                                      iterations, train_size, cutoff, suffix)

                results_4 = crossval(data_4,
                                     trees, depths, max_features,
                                     iterations, train_size, cutoff, suffix)

                results_hot = crossval(data_hot,
                                       trees, depths, max_features,
                                       iterations, train_size, cutoff, suffix)

                all_results_15 += results_15
                all_results_4 += results_4
                all_results_hot += results_hot


        all_results_15.sort(key = lambda result: result.average_accuracy, reverse = True)
        all_results_4.sort(key = lambda result: result.average_accuracy, reverse = True)
        all_results_hot.sort(key = lambda result: result.average_accuracy, reverse = True)

        write_results(all_results_15, dir_15)
        write_results(all_results_4, dir_4)
        write_results(all_results_hot, dir_hot)


        print("Testing 15 properties:")
        test_best_result(all_results_15)
 #       print("Testing 4 properties:")
 #       test_best_result(all_results_4) # Commented out to remove 4 property test
 #       print("Testing one-hot properties:")
 #       test_best_result(all_results_hot)

def string_to_bool(value):
    return value.lower() == 'true'

if __name__ == "__main__":
    parser = define_arguments()
    args = parser.parse_args()

    fasta = args.fasta
    trees = args.trees
    depths = args.depth
    max_features = args.features
    iterations = args.iterations
    train_sizes = args.train_size
    groups = args.groups
    cutoffs = args.cutoffs
    clean = args.clean
    suffix = args.suffix

    print(type(groups))
    print(groups)

    main(fasta, trees, depths, max_features, iterations, train_sizes, groups,
         cutoffs, clean, suffix)

    




        
