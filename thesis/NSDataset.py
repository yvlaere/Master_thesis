#requires resquiggled fast5 data and WGBS coverage files
class NSDataset(Dataset):
    """NS dataset"""

    def __init__(self, fast5_dir, transform=None):
        """
        Args:
            fast5_dir (string): Path to the fast5 files
        """
        self.alignment_path = 'Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment'
        self.events_path = events_path = 'Analyses/RawGenomeCorrected_000/BaseCalled_template/Events'
        self.chroms = ['NC_000002.12', 'NC_000016.10', 'NC_000006.12', 'NC_000004.12', 'NC_000017.11', 'NC_000014.9', 'NC_000024.10', 'NC_000007.14', 'NC_000013.11', 'NC_000020.11', 'NC_000015.10', 'NC_000019.10', 'NC_000009.12', 'NC_000003.12', 'NC_000023.11', 'NC_000005.10', 'NC_000012.12', 'NC_000018.10', 'NC_000022.11', 'NC_000011.10', 'NC_000010.11', 'NC_000008.11', 'NC_000001.11', 'NC_000021.9']

        fast5_list = []
        fast5_length = []
        
        for root, dirs, files in os.walk(fast5_dir):
            for file in files:
                if file.endswith('.fast5'):
                    full_path = os.path.join(root, file)
                    with h5py.File(full_path, 'r+') as handle:
                        if 'BaseCalled_template' in handle['Analyses/RawGenomeCorrected_000'].keys():
                            if 'Alignment' in handle['Analyses/RawGenomeCorrected_000/BaseCalled_template'].keys():
                                chrom = handle[self.alignment_path].attrs['mapped_chrom']
                                if chrom in self.chroms:
                                    fast5_list.append(full_path)
                                    read_start = handle[self.alignment_path].attrs['mapped_start']
                                    read_end = handle[self.alignment_path].attrs['mapped_end']
                                    fast5_length.append(abs(read_end - read_start))
        
        indices = np.argsort(fast5_length)
        self.overlap = max(fast5_length)
        fast5_list = [fast5_list[i] for i in indices]
        self.fast5s = fast5_list

    def __len__(self):
        return len(self.fast5s)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the desired read
        """
        position_indexer = np.load('position_indexer.npz')
        meth_status_indexer = np.load('meth_indexer.npz')
        chrom_interval_start = np.load('c_i_s.npz')
        
        with h5py.File(self.fast5s[idx], 'r+') as handle:
            
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
            labs[labs_neg_ind] = 0
            labs[labs_pos_ind] = 1
            labs[Gs] = 2

            label_data = torch.tensor(np.array(labs), dtype = torch.float32)

        return read_data, label_data
    
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

    def prepare_WGBS(self, WGBS_dir, min_coverage, interval_size = 1000000):
        """
        Turn the coverage files into a smaller arrays

        Args:
            WGBS_dir (string): Path to coverage files
            interval_size (int): Size of the intervals used to store the WGBS data
            min_coverage (int): Minimum amount of coverage for a position to be included
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
        np.savez_compressed('position_indexer.npz', **WGBS_positions)
        np.savez_compressed('meth_indexer.npz', **WGBS_methylation_status)
        np.savez_compressed('c_i_s.npz', **chrom_interval_start)
        print('saved')
