import numpy as np
import pandas as pd
import os
import h5py
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torchvision import transforms
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
from sys import getsizeof

class WGBSDataset():
    """WGBS dataset"""

    def __init__(self, fast5_dir, transform = None):
        """
        Args:
            fast5_dir (string): Path to the fast5 files
        """
        
        self.alignment_path = 'Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment'
        self.events_path = 'Analyses/RawGenomeCorrected_000/BaseCalled_template/Events'
        self.chroms = ['NC_000002.12', 'NC_000016.10', 'NC_000006.12', 'NC_000004.12', 'NC_000017.11', 'NC_000014.9', 'NC_000024.10', 'NC_000007.14', 'NC_000013.11', 'NC_000020.11', 'NC_000015.10', 'NC_000019.10', 'NC_000009.12', 'NC_000003.12', 'NC_000023.11', 'NC_000005.10', 'NC_000012.12', 'NC_000018.10', 'NC_000022.11', 'NC_000011.10', 'NC_000010.11', 'NC_000008.11', 'NC_000001.11', 'NC_000021.9']

        fast5_length = []
        
        for root, dirs, files in os.walk(fast5_dir):
            for file in files:
                #only fast5 files
                if file.endswith('.fast5'):
                    full_path = os.path.join(root, file)
                    with h5py.File(full_path, 'r+') as handle:
                        #only succesfully aligned reads
                        if 'BaseCalled_template' in handle['Analyses/RawGenomeCorrected_000'].keys():
                            if 'Alignment' in handle['Analyses/RawGenomeCorrected_000/BaseCalled_template'].keys():
                                chrom = handle[self.alignment_path].attrs['mapped_chrom']
                                if chrom in self.chroms:
                                    read_start = handle[self.alignment_path].attrs['mapped_start']
                                    read_end = handle[self.alignment_path].attrs['mapped_end']
                                    fast5_length.append(abs(read_end - read_start))
        
        self.overlap = max(fast5_length)
        
    def filter_WGBS(self, coverage_dir, WGBS_dir, min_coverage, upper_cutoff = 90.0, lower_cutoff = 0.0):
        """
        Filter WGBS coverage files

        Args:
            coverage_dir (string): Path to the coverage files
            WGBS_dir (string): Path to store the filtered coverage files
            min_coverage (int): Minimum amount of coverage for a position to be included
            upper_cutoff (float): Minimum percentage to be considered methylated
            lower_cutoff (float): Maximum percentage to be considered unmethylated
        """

        counter = 0
        """
        #what about these chroms????????????
        """

        with open(coverage_dir, 'r') as f:
            while True:
                #read coverage file line per line untill the end
                line = f.readline()
                if not line:
                    break

                #parse the line
                line_list = list(line.split('\t'))
                chrom = line_list[0]
                pos = line_list[1]
                meth_percentage = float(line_list[3])
                count_methylated = int(line_list[4])
                count_unmethylated = int(line_list[5])

                #evaluate the position and write to file if evaluation is passed
                if chrom in self.chroms:
                    if count_methylated + count_unmethylated >= min_coverage:
                        if (meth_percentage >= upper_cutoff) | (meth_percentage == lower_cutoff):
                            file_name = WGBS_dir + chrom + '_cov' + str(min_coverage) + '.txt'
                            with open(file_name, 'a') as l:
                                if meth_percentage >= upper_cutoff:
                                    meth_status = 1
                                elif meth_percentage <= lower_cutoff:
                                       meth_status = 0
                                new_line = '\t'.join([chrom, pos, str(meth_status)])
                                l.write(new_line + '\n')
                            l.close()

                #show progress
                counter += 1
                if counter%10000000 == 0:
                    print(counter)

        f.close()

    def merge_WGBS(self, WGBS_dir_1, WGBS_dir_2, WGBS_dir, min_coverage):
        """
        Merge (filtered) WGBS coverage files

        Args:
            WGBS_dir_1 (string): Path to the first set of coverage files
            WGBS_dir_2 (string): Path to the second set of coverage files
            WGBS_dir (string): Path to store the merged coverage files
            min_coverage (int): Minimum amount of coverage for a position to be included
        """

        for root, dirs, files in os.walk(WGBS_dir_1):
            for file in files:
                if file.endswith('_cov' + str(min_coverage) + '.txt'):
                    WGBS_df_1 = pd.read_csv(WGBS_dir_1 + file, sep='\t', names = ['chrom', 'pos', 'meth'])
                    WGBS_df_2 = pd.read_csv(WGBS_dir_2 + file, sep='\t', names = ['chrom', 'pos', 'meth'])
                    WGBS_df_merged = WGBS_df_1.merge(WGBS_df_2, how = 'inner', on = ['chrom', 'pos', 'meth'])
                    WGBS_df_merged.to_csv(WGBS_dir + file, sep="\t", header = False, index = False)
                    print('merged')

    def prepare_WGBS(self, WGBS_dir, label_dict_dir, min_coverage, interval_size = 1000000):
        """
        Turn the coverage files into a smaller arrays

        Args:
            WGBS_dir (string): Path to coverage files
            label_dict_dir (string): Path to store label dictionnaries
            min_coverage (int): Minimum amount of coverage for a position to be included
            interval_size (int): Size of the intervals used to store the WGBS data
        """

        #start from coverage files
        WGBS_positions = {}
        WGBS_methylation_status = {}
        chrom_interval_start = defaultdict(list)

        for root, dirs, files in os.walk(WGBS_dir):
            for file in files:
                if file.endswith('_cov' + str(min_coverage) + '.txt'):
                    full_path = os.path.join(root, file)
                    chrom = file[:-9]
                    WGBS_df = pd.read_csv(full_path, sep='\t', names = ['chrom', 'pos', 'meth'])
                    chrom_length = len(WGBS_df)
                    n = math.ceil(chrom_length/interval_size)

                    #first interval
                    temp_pos = np.array(WGBS_df['pos'][0:interval_size + self.overlap])
                    temp_meth = np.array(WGBS_df['meth'][0:interval_size + self.overlap])
                    loc = chrom + '_' + str(0)
                    chrom_interval_start[chrom].append(0)
                    WGBS_positions[loc] = temp_pos
                    WGBS_methylation_status[loc] = temp_meth

                    #last interval
                    temp_pos = np.array(WGBS_df['pos'][(n - 1)*interval_size - self.overlap:chrom_length - 1])
                    temp_meth = np.array(WGBS_df['meth'][(n - 1)*interval_size - self.overlap:chrom_length - 1])
                    loc = chrom + '_' + str(temp_pos[0])
                    chrom_interval_start[chrom].append(temp_pos[0])
                    WGBS_positions[loc] = temp_pos
                    WGBS_methylation_status[loc] = temp_meth

                    for i in range(1, n - 2):
                        temp_pos = np.array(WGBS_df['pos'][i*interval_size - self.overlap:(i + 1)*interval_size + self.overlap])
                        temp_meth = np.array(WGBS_df['meth'][i*interval_size - self.overlap:(i + 1)*interval_size + self.overlap])
                        loc = chrom + '_' + str(temp_pos[0])
                        chrom_interval_start[chrom].append(temp_pos[0])
                        WGBS_positions[loc] = temp_pos
                        WGBS_methylation_status[loc] = temp_meth
                        temp_pos = np


        print('saving')
        pi_dir = label_dict_dir + 'position_indexer.npz'
        mi_dir = label_dict_dir + 'meth_indexer.npz'
        cis_dir = label_dict_dir + 'chrom_interval_size.npz'
        np.savez_compressed(pi_dir, **WGBS_positions)
        np.savez_compressed(mi_dir, **WGBS_methylation_status)
        np.savez_compressed(cis_dir, **chrom_interval_start)
        print('saved')
