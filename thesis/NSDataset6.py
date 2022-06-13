#requires resquiggled fast5 data and WGBS coverage files
import numpy as np
import pandas as pd
import os
import h5py
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

class NSDataset(Dataset):

    def __init__(self, fast5_dir, label_dict_dir = None, CpG = True, transform = None):
        self.alignment_path = 'Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment'
        self.events_path = 'Analyses/RawGenomeCorrected_000/BaseCalled_template/Events'
        self.chroms = ['NC_000002.12', 'NC_000016.10', 'NC_000006.12', 'NC_000004.12', 'NC_000017.11', 'NC_000014.9', 'NC_000024.10', 'NC_000007.14', 'NC_000013.11', 'NC_000020.11', 'NC_000015.10', 'NC_000019.10', 'NC_000009.12', 'NC_000003.12', 'NC_000023.11', 'NC_000005.10', 'NC_000012.12', 'NC_000018.10', 'NC_000022.11', 'NC_000011.10', 'NC_000010.11', 'NC_000008.11', 'NC_000001.11', 'NC_000021.9']
        
        #determine if labels have to come from WGBS data
        if label_dict_dir == None:
            self.WGBS = False
        else:
            self.WGBS = True
            self.pi_dir = label_dict_dir + 'position_indexer.npz'
            self.mi_dir = label_dict_dir + 'meth_indexer.npz'
            self.cis_dir = label_dict_dir + 'chrom_interval_size.npz'
           
        self.CpG = CpG
        fast5_list = []
        fast5_length = []
        
        for root, dirs, files in os.walk(fast5_dir):
            for file in files:
                #only fast5 files
                if file.endswith('.fast5'):
                    full_path = os.path.join(root, file)
                    #print(full_path)
                    with h5py.File(full_path, 'r+') as handle:
                        
                        #only succesfully aligned reads
                        if 'RawGenomeCorrected_000' in handle['Analyses/'].keys():
                            if 'BaseCalled_template' in handle['Analyses/RawGenomeCorrected_000'].keys():
                                if 'Alignment' in handle['Analyses/RawGenomeCorrected_000/BaseCalled_template'].keys():  
                                    
                                    chrom = handle[self.alignment_path].attrs['mapped_chrom']
                                    if self.WGBS:
                                        if chrom in self.chroms:
                                        #only if there are any labels
                                            if any(self.get_tensors_WGBS(full_path)[1] != 2):
                                                fast5_list.append(full_path)
                                                read_start = handle[self.alignment_path].attrs['mapped_start']
                                                read_end = handle[self.alignment_path].attrs['mapped_end']
                                                fast5_length.append(abs(read_end - read_start))
                                    else:
                                        #only if there are any labels
                                        if any(self.get_tensors(full_path)[1] != 2):
                                            fast5_list.append(full_path)
                                            read_start = handle[self.alignment_path].attrs['mapped_start']
                                            read_end = handle[self.alignment_path].attrs['mapped_end']
                                            fast5_length.append(abs(read_end - read_start))

        
        indices = np.argsort(fast5_length)
        #self.lens = fast5_length
        fast5_list = [fast5_list[i] for i in indices]
        self.fast5s = fast5_list

    def __len__(self):
        return len(self.fast5s)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the desired read
        """
        if self.WGBS:
            return self.get_tensors_WGBS(self.fast5s[idx])
        else:        
            return self.get_tensors(self.fast5s[idx])
        
    def get_tensors_WGBS(self, path):
        """
        Args: 
            path: path to a fast5 file
        """
        
        position_indexer = np.load(self.pi_dir)
        meth_status_indexer = np.load(self.mi_dir)
        chrom_interval_start = np.load(self.cis_dir)
        
        with h5py.File(path, 'r+') as handle:
            
            #collect mapping info
            chrom = handle[self.alignment_path].attrs['mapped_chrom']
            read_start = handle[self.alignment_path].attrs['mapped_start']
            read_end = handle[self.alignment_path].attrs['mapped_end']
            orientation = handle[self.alignment_path].attrs['mapped_strand']

            #collect event data
            events = np.array(handle[self.events_path])
            norm_mean, norm_stdev, start, length, base = zip(*events)

            #get sequence
            tomboseq = ''
            for b in base:
                tomboseq += b.decode('UTF-8')
            read_length = len(tomboseq)
            Gs = np.where(np.array(list(tomboseq)) == 'G')[0]

            #1-hot encode sequence
            ordering = ["A", "T", "C", "G"]
            seq = np.expand_dims(np.array(ordering),-1) == np.array(list(tomboseq))

            read_data = torch.tensor(np.transpose(np.array([norm_mean, norm_stdev, length, seq[0], seq[1], seq[2], seq[3]], dtype = np.float32)), dtype = torch.float32)
                                
            #get interval start
            temp = chrom_interval_start[chrom] - read_start
            """
            maybe different number than 1B
            """
            temp[temp > 0] = 1000000000
            pos_start = chrom_interval_start[chrom][np.argmin(abs(temp))]
            pos = chrom + '_' + str(pos_start)

            #collect label data
            inds = np.where(abs(position_indexer[pos] - read_start - (read_length - 1)/2 - 1) < read_length/2)[0]
            meth_status = meth_status_indexer[pos][inds]
            positions = position_indexer[pos][inds]

            #genomic coordinates for negative and positive labels
            labs_neg_positions = positions[np.where(meth_status == 0)[0]]
            labs_pos_positions = positions[np.where(meth_status == 1)[0]]

            #turn genomic coordinates into read indexes
            if orientation == '-':
                labs_neg_ind = read_end - labs_neg_positions
                labs_pos_ind = read_end - labs_pos_positions
            elif orientation == '+':
                labs_neg_ind = labs_neg_positions - (read_start + 1)
                labs_pos_ind = labs_pos_positions - (read_start + 1)

            labs = np.full(read_length, 2)
            
            #only add labels to C's before G's
            if self.CpG:
                prev_ind = np.array([Gs - 1])
                prev_ind = prev_ind[prev_ind >= 0]
                CpG_inds = prev_ind[np.array(list(tomboseq))[prev_ind] == 'C']
                labs_neg_ind = np.intersect1d(CpG_inds, labs_neg_ind)
                labs_pos_ind = np.intersect1d(CpG_inds, labs_pos_ind)

            labs[labs_neg_ind] = 0
            labs[labs_pos_ind] = 1
            labs[Gs] = 2

            label_data = torch.tensor(np.array(labs), dtype = torch.float32)

        return read_data, label_data
    
    def get_tensors(self, path):
        """
        Args: 
            path: path to a fast5 file
        """
        
        with h5py.File(path, 'r+') as handle:
            
            #collect mapping info
            chrom = handle[self.alignment_path].attrs['mapped_chrom']
            read_start = handle[self.alignment_path].attrs['mapped_start']
            read_end = handle[self.alignment_path].attrs['mapped_end']
            orientation = handle[self.alignment_path].attrs['mapped_strand']

            #collect event data
            events = np.array(handle[self.events_path])
            norm_mean, norm_stdev, start, length, base = zip(*events)

            #get sequence
            tomboseq = ''
            for b in base:
                tomboseq += b.decode('UTF-8')
            read_length = len(tomboseq)
            Gs = np.where(np.array(list(tomboseq)) == 'G')[0]

            #1-hot encode sequence
            ordering = ["A", "T", "C", "G"]
            seq = np.expand_dims(np.array(ordering),-1) == np.array(list(tomboseq))

            read_data = torch.tensor(np.transpose(np.array([norm_mean, norm_stdev, length, seq[0], seq[1], seq[2], seq[3]], dtype = np.float32)), dtype = torch.float32)
                                
            #determine if read needs positive or negative labels
            if 'MSssI' in path:
                label = 1
            else:
                label = 0
                
            #add labels at CpG sites
            prev_ind = np.array([Gs - 1])
            prev_ind = prev_ind[prev_ind >= 0]
            CpG_inds = prev_ind[np.array(list(tomboseq))[prev_ind] == 'C']
            labs = np.full(read_length, 2)
            labs[CpG_inds] = label

            label_data = torch.tensor(np.array(labs), dtype = torch.float32)

        return read_data, label_data
    
    def get_tensors_WGBS_extra(self, idx):
        """
        Args: 
            path: path to a fast5 file
        """
        
        path = self.fast5s[idx]
        
        position_indexer = np.load(self.pi_dir)
        meth_status_indexer = np.load(self.mi_dir)
        chrom_interval_start = np.load(self.cis_dir)
        
        with h5py.File(path, 'r+') as handle:
            
            #collect mapping info
            chrom = handle[self.alignment_path].attrs['mapped_chrom']
            read_start = handle[self.alignment_path].attrs['mapped_start']
            read_end = handle[self.alignment_path].attrs['mapped_end']
            orientation = handle[self.alignment_path].attrs['mapped_strand']

            #collect event data
            events = np.array(handle[self.events_path])
            norm_mean, norm_stdev, start, length, base = zip(*events)

            #get sequence
            tomboseq = ''
            for b in base:
                tomboseq += b.decode('UTF-8')
            read_length = len(tomboseq)
            Gs = np.where(np.array(list(tomboseq)) == 'G')[0]

            #1-hot encode sequence
            ordering = ["A", "T", "C", "G"]
            seq = np.expand_dims(np.array(ordering),-1) == np.array(list(tomboseq))

            read_data = torch.tensor(np.transpose(np.array([norm_mean, norm_stdev, length, seq[0], seq[1], seq[2], seq[3]], dtype = np.float32)), dtype = torch.float32)
                                
            #get interval start
            temp = chrom_interval_start[chrom] - read_start
            """
            maybe different number than 1B
            """
            temp[temp > 0] = 1000000000
            pos_start = chrom_interval_start[chrom][np.argmin(abs(temp))]
            pos = chrom + '_' + str(pos_start)

            #collect label data
            inds = np.where(abs(position_indexer[pos] - read_start - (read_length - 1)/2 - 1) < read_length/2)[0]
            meth_status = meth_status_indexer[pos][inds]
            positions = position_indexer[pos][inds]

            #genomic coordinates for negative and positive labels
            labs_neg_positions = positions[np.where(meth_status == 0)[0]]
            labs_pos_positions = positions[np.where(meth_status == 1)[0]]

            #turn genomic coordinates into read indexes
            if orientation == '-':
                labs_neg_ind = read_end - labs_neg_positions
                labs_pos_ind = read_end - labs_pos_positions
            elif orientation == '+':
                labs_neg_ind = labs_neg_positions - (read_start + 1)
                labs_pos_ind = labs_pos_positions - (read_start + 1)

            labs = np.full(read_length, 2)
            
            #only add labels to C's before G's
            if self.CpG:
                prev_ind = np.array([Gs - 1])
                prev_ind = prev_ind[prev_ind >= 0]
                CpG_inds = prev_ind[np.array(list(tomboseq))[prev_ind] == 'C']
                labs_neg_ind = np.intersect1d(CpG_inds, labs_neg_ind)
                labs_pos_ind = np.intersect1d(CpG_inds, labs_pos_ind)

            labs[labs_neg_ind] = 0
            labs[labs_pos_ind] = 1
            labs[Gs] = 2

            label_data = torch.tensor(np.array(labs), dtype = torch.float32)

        return read_data, label_data, chrom, read_start, read_end, orientation
    
