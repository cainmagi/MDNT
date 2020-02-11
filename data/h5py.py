'''
################################################################
# Data - h5py
# @ Modern Deep Network Toolkits for Tensorflow-Keras
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6+
#   tensorflow r1.13+
# Use generator to wrap the h5py.
# Warning:
#   The standard tf dataset is proved to be incompatible with 
#   tf-K architecture. We need to wait until tf fix the bug.
# Version: 0.26 # 2020/02/10
# Comments:
#   1. Provide a converter to transform dataset formats.
# Version: 0.24 # 2019/10/23
# Comments:
#   1. Finsish `H5VGParser` for splitting valid set from a
#   train set.
#   2. Fix a bug for `H5GCombiner` when adding new parsers.
# Version: 0.23 # 2019/10/7
# Comments:
#   Modify `H5SupSaver` to enable it to add more data to an
#   existed file.
# Version: 0.22 # 2019/9/10
# Comments:
#   Modify `H5SupSaver` to enable it to resize dataset if the
#   data is dumped in series. Now dumping data into an exist-
#   ing dataset would expand the set.
# Version: 0.20 # 2019/3/31
# Comments:
#   Add a new class, `H5GCombiner`.
# Version: 0.18 # 2019/3/27
# Comments:
#   1. Upgrade the generators (parsers) to keras.util version.
#   2. Make the suffle optional for parsers.
# Version: 0.15 # 2019/3/26
# Comments:
#   1. Enable the config of H5SupSaver to be overriden.
#   2. Change the behavior of H5GParser, and let the pre-
#      processing function work with batches.
# Version: 0.10 # 2019/3/26
# Comments:
#   Create this submodule.
################################################################
'''

import h5py
import numpy as np
import tensorflow as tf
import os
import io

class H52TXT:
    '''An example of converter between HDF5 and TXT'''
    
    def read(self, fileName):
        '''read function, for converting TXT to HDF5.
        fileName is the name of the single input file'''
        with open(os.path.splitext(fileName)[0] + '.txt', 'r') as f:
            sizeText = io.StringIO(f.readline())
            sze = np.loadtxt(sizeText, dtype=np.int)
            data = np.loadtxt(f, dtype=np.float32)
            return np.reshape(data, sze)
    
    def write(self, h5data, fileName):
        '''write function, for converting HDF5 to TXT.
        fileName is the name of the single output file.'''
        with open(os.path.splitext(fileName)[0] + '.txt', 'w') as f:
            np.savetxt(f, np.reshape(h5data.shape, (1, h5data.ndim)), fmt='%d')
            if h5data.ndim > 1:
                for i in range(h5data.shape[0]):
                    np.savetxt(f, h5data[i, ...].ravel(), delimiter='\n')
            else:
                np.savetxt(f, h5data[:].ravel(), delimiter='\n')

class H52BIN:
    '''An example of converter between HDF5 and bin file'''
    
    def read(self, fileName):
        '''read function, for converting bin file to HDF5.
        fileName is the name of the single input file'''
        with open(os.path.splitext(fileName)[0] + '.bin', 'rb') as f:
            ndims = np.fromfile(f, dtype=np.int, count=1)[0]
            sze = np.fromfile(f, dtype=np.int, count=ndims)
            data = np.fromfile(f, dtype=np.float32, count=-1)
            return np.reshape(data, sze)
    
    def write(self, h5data, fileName):
        '''write function, for converting HDF5 to bin file.
        fileName is the name of the single output file.'''
        with open(os.path.splitext(fileName)[0] + '.bin', 'wb') as f:
            get_ndim = np.array(h5data.ndim)
            get_shape = np.array(h5data.shape)
            get_ndim.tofile(f)
            get_shape.tofile(f)
            if get_ndim > 1:
                for i in range(get_shape[0]):
                    h5data[i, ...].ravel().astype(np.float32).tofile(f)
            else:
                h5data[:].ravel().astype(np.float32).tofile(f)

class H5Converter:
    '''Conversion between HDF5 data and other formats.
    The "other formats" would be arranged in to form of several
    folders and files. Each data group would be mapped into a
    folder, and each dataset would be mapped into a file.
    '''
    def __init__(self, fileName, oformat, toOther=True):
        '''
        Initialization and set format.
        Arguments:
            fileName: a path where we find the dataset. If the
                      conversion is h52other, the path should
                      refer a folder containing several subfiles,
                      otherwise, it should refer an HD5 file.
            oformat:  the format function for a single dataset,
                      it could be provided by users, or use the
                      default configurations. (avaliable: 'txt',
                      'bin'.)
            toOther:  the flag for conversion mode. If set True,
                      the mode would be h52other, i.e. an HDF5
                      set would be converted into other formats.
                      If set False, the conversion would be
                      reversed.
        '''
        self.__read = (not toOther)
        self.folder = os.path.splitext(fileName)[0]
        if not self.__read:
            if not os.path.isfile(fileName):
                if (os.path.isfile(fileName+'.h5')):
                    fileName += '.h5'
                    self.folder = os.path.splitext(fileName)[0]
                else:
                    raise FileNotFoundError('Could not read the HDF5 dataset: {0}.'.format(fileName))
            if os.path.exists(self.folder):
                raise FileExistsError('Could not write to the folder: {0}, because it already exists.'.format(self.folder))
        else:
            if not os.path.isdir(fileName):
                raise FileNotFoundError('Could not open the folder {0}.'.format(fileName))
            if os.path.exists(fileName+'.h5'):
                raise FileExistsError('Could not write to the HDF5 dataset {0}.h5, because it already exists.'.format(fileName))
            self.folder = fileName
            fileName = fileName + '.h5'
        self.f = h5py.File(fileName, 'w' if self.__read else 'r')
        
        if oformat == 'txt':
            self.__func = H52TXT()
        elif oformat == 'bin':
            self.__func = H52BIN()
        else:
            if self.__read:
                if not hasattr(oformat, 'write'):
                    raise AttributeError('The "oformat" should contains the write method for applying the conversion.')
            else:
                if not hasattr(oformat, 'read'):
                    raise AttributeError('The "oformat" should contains the read method for applying the conversion.')
            self.__func = oformat

    @staticmethod
    def __h5iterate(g, func):
        if isinstance(g, h5py.Dataset):
            func(g)
        else:
            for item in g:
                H5Converter.__h5iterate(g[item], func)

    def __savefunc(self, g):
        path = os.path.join(self.folder, g.name.replace(':', '-')[1:])
        folder = os.path.split(path)[0]
        if not os.path.isdir(folder):
            os.makedirs(folder)
        self.__func.write(g, path)
        print('Have dumped {0}'.format(g.name))

    def __h52other(self):
        self.__h5iterate(self.f, self.__savefunc)
    
    def __other2h5(self):
        for root, _, files in os.walk(self.folder, topdown=False):
            for name in files:
                dsetName = '/'+ os.path.relpath(os.path.join(root, os.path.splitext(name)[0]), start=self.folder).replace('\\', '/')
                self.f.create_dataset(dsetName, data=self.__func.read(os.path.join(root, name)))
                print('Have dumped {0}'.format(dsetName))

    def convert(self):
        if self.__read:
            self.__other2h5()
        else:
            self.__h52other()

class H5SupSaver:
    '''Save supervised data set as .h5 file
    This class allows users to dump multiple datasets into one file
    handle, then it would save it as a .h5 file. The keywords of the
    sets should be assigned by users.
    '''
    def __init__(self, fileName, enableRead=False):
        '''
        Create the .h5 file while initialization.
        Arguments:
            fileName:   a path where we save the file.
            enableRead: when set True, enable the read/write mode.
                        This option is used when adding data to an
                        existed file.
        '''
        self.f = None
        self.logver = 0
        self.__kwargs = dict()
        self.open(fileName, enableRead)
        self.config(dtype='f')
        
    def config(self, **kwargs):
        '''
        Make configuration for the saver.
        Argumetns for this class:
            logver (int): the log level for dumping files.
        Arguments often used:
            chunks (tuple):         size of data blocks.
            compression (str):      compression method.
            compression_opts (int): compression parameter.
            shuffle (bool):         shuffle filter for data compression.
            fletcher32 (bool):      check sum for chunks.
        Learn more available arguments here:
            http://docs.h5py.org/en/latest/high/dataset.html
        '''
        logver = kwargs.pop('logver', None)
        if logver is not None:
            self.logver = logver
        self.__kwargs.update(kwargs)
        if self.logver > 0:
            print('Current configuration is:', self.__kwargs)
    
    def dump(self, keyword, data, **kwargs):
        '''
        Dump the dataset with a keyword into the file.
        Arguments:
            keyword: the keyword of the dumped dataset.
            data:    dataset, should be a numpy array.
        Providing more configurations for `create_dataset` would override
        the default configuration defined by self.config()
        If the provided `keyword` exists, the dataset would be resized for
        accepting more data.
        '''
        if self.f is None:
            raise OSError('Should not dump data before opening a file.')
        newkw = self.__kwargs.copy()
        newkw.update(kwargs)
        dshape = data.shape[1:]
        if keyword in self.f:
            ds = self.f[keyword]
            dsshape = ds.shape[1:]
            if np.all(np.array(dshape, dtype=np.int) == np.array(dsshape, dtype=np.int)):
                N = len(ds)
                newN = data.shape[0]
                ds.resize(N+newN, axis=0)
                ds[N:N+newN, ...] = data
                if self.logver > 0:
                    print('Dump {smp} data samples into the existed dataset {ds}. The data shape is {sze} now.'.format(smp=newN, ds=keyword, sze=ds.shape))
            else:
                raise ValueError('The data set shape {0} does not match the input shape {1}.'.format(dsshape, dshape))
        else:
            self.f.create_dataset(keyword, data=data, maxshape=(None, *dshape), **newkw)
            if self.logver > 0:
                print('Dump {0} into the file. The data shape is {1}.'.format(keyword, data.shape))
    
    def open(self, fileName, enableRead=False):
        '''
        The dumped file name (path), it will produce a .h5 file.
        Arguments:
            fileName: a path where we save the file.
            enableRead: when set True, enable the read/write mode.
                        This option is used when adding data to an
                        existed file.
        '''
        if fileName[-3:] != '.h5':
            fileName += '.h5'
        self.close()
        if enableRead:
            fmode = 'a'
        else:
            fmode = 'w'
        self.f = h5py.File(fileName, fmode)
        if self.logver > 0:
            print('Open a new file:', fileName)
        
    def close(self):
        if self.f is not None:
            self.f.close()
        self.f = None
        
class H5HGParser(tf.keras.utils.Sequence):
    '''Homogeneously parsing .h5 file by h5py module
    This class allows users to feed one .h5 file, and convert it to 
    tf.data.Dataset. The realization could be described as:
        (1) Create .h5 file handle.
        (2) Estimate the dataset size, and generate indices.
        (3) Define the randomly shuffling method for indices in each epoch.
        (4) Use Dataset.map() to address the data by the index from the
            index dataset.
    Note that in the file, there may be multiple datasets. This parser
    supports reading both single set and multiple sets.
    Note that all datasets in the same file should share the same shape.
    '''
    def __init__(self, fileName, batchSize=32, shuffle=True):
        '''
        Create the parser and its h5py file handle.
        Arguments:
            fileName: the data path of the file (could be without postfix).
            batchSize: number of samples in each batch.
            shuffle: if on, shuffle the data set at the end of each epoch.
        '''
        super(H5HGParser, self).__init__()
        self.f = None
        if (not os.path.isfile(fileName)) and (os.path.isfile(fileName+'.h5')):
            fileName += '.h5'
        self.f = h5py.File(fileName, 'r')
        self.size = self.__createSize()
        self.shuffle = shuffle
        if self.mutlipleMode:
            self.__indices, self.__secInd = self.__indexDataset()
            if shuffle:
                self.__shuffle()
            self.__mapfunc = self.__mapMultiple
        else:
            self.__indices = self.__indexDataset()
            if shuffle:
                self.__shuffle()
            self.__mapfunc = self.__mapSingle
        self.__batchSize = batchSize
    
    def __len__(self):
        '''
        Automatically calculate the steps for iterate the whole dataset.
        '''
        return np.ceil(np.sum(self.size)/self.__batchSize).astype(np.int)
        
    def __getitem__(self, idx):
        batchIndices = self.__indices[idx * self.__batchSize:(idx + 1) * self.__batchSize]
        res = []
        for ind in batchIndices:
            if self.mutlipleMode:
                res.append(self.__mapMultiple(self.__secInd[ind]))
            else:
                res.append(self.__mapSingle(ind))
        res = np.stack(res, axis=0)
        return res
            
    def on_epoch_end(self):
        '''
        Shuffle the data set according to the settings.
        '''
        if self.shuffle:
            self.__shuffle()
    
    def __createSize(self):
        '''
        Find the number of items in the dataset, only need to be run for once.
        '''
        if len(self.f) == 1:
            self.mutlipleMode = False
            self.__dnameIndex = list(self.f.keys())[0]
            return len(self.f[self.__dnameIndex])
        else:
            self.mutlipleMode = True
            self.__dnameIndex = list(self.f.keys())
            return tuple(len(self.f[fk]) for fk in self.__dnameIndex)
        
    def __indexDataset(self):
        '''
        Create a tensorflow index dataset, only need to be run for once.
        Should be run after __createSize
        '''
        if self.mutlipleMode:
            def genOneIndex(num):
                return np.stack((num*np.ones(self.size[num], dtype=np.int), np.arange(self.size[num], dtype=np.int)), axis=1)
            indices = np.concatenate(list(genOneIndex(n) for n in range(len(self.size))), axis=0)
            return np.arange(indices.shape[0], dtype=np.int), indices
        else:
            return np.arange(self.size, dtype=np.int)
            
    def __shuffle(self):
        '''
        Resort the indices randomly.
        '''
        np.random.shuffle(self.__indices)
    
    def __mapMultiple(self, index):
        '''
        Map function, for multiple datasets mode.
        '''
        dname = self.__dnameIndex[index[0]]
        numSample = index[1]
        return self.f[dname][numSample]
        
    def __mapSingle(self, index):
        '''
        Map function, for multiple datasets mode.
        '''
        dname = self.__dnameIndex
        return self.f[dname][index]

class H5GCombiner(tf.keras.utils.Sequence):
    '''Combiner designed for H5GParser
    In some applications, we may need to use multiple H5GParser
    simultaneously, and apply different preprocessing functions
    respectively. This class is used for such a case. It accepts
    multiple H5GParsers, and maintains a current index for each
    subsets (H5Parsers). If the shuffle flag is enabled in any subset, 
    the shuffle operation would be triggered respectively.
    Note that in each iteration, the H5GCombiner would returns a
    combined list for all returned results from each subset. Hence
    generally an external preprocessing function is needed for this
    class so that the returned data could be rearranged correctly.
    This class is used for merging unrelated datasets. If you need to
    combine multiple related datasets with the same number of samples,
    you should use H5GParser rather than this class, and store those
    related datasets in the same .h5 file with different keywords.
    '''
    def __init__(self, *args, preprocfunc=None):
        '''
        Merge multiple H5Parsers.
        Arguments:
            args: one or more H5Parsers (At least one H5Parsers).
            preprocfunc: the function applied for the combined outputs
                         from all subsets (H5Parsers).
        Note that the length of this combiner, or the end of an epoch would
        by tagged by the end of the first dataset.
        '''
        super(H5GCombiner, self).__init__()
        self.__setSize = len(args)
        if self.__setSize == 0:
            raise TypeError('Should combine at least one H5Parser.')
        for p in args:
            if not isinstance(p, H5GParser):
                raise TypeError('The type of one input argument is not H5Parser, need to check the inputs.')
        self.__parserList = list(args)
        self.__sizeList = [len(p) for p in self.__parserList]
        self.__indexList = [0] * self.__setSize
        self.__preprocfunc = preprocfunc
        
    def __len__(self):
        '''
        Use the size of first dataset as the epoch size.
        '''
        return self.__sizeList[0]
    
    def __getitem__(self, idx):
        """
        Gets batch at position `index`.
        In this class, the usage of `idx` (or `index`) parameter would be canceled,
        i.e. it only return samples in step by step.
        """
        collection = []
        for i in range(self.__setSize):
            ind = self.__indexList[i] # Get current index
            collection.append(self.__parserList[i][ind])
            ind = (ind + 1) % self.__sizeList[i] # Increse the current index
            self.__indexList[i] = ind
            if ind == 0: # If reach the end of a subset, call on_epoch_end
                self.__parserList[i].on_epoch_end()
        if self.__preprocfunc is not None:
            return self.__preprocfunc(*collection)
        else:
            return tuple(collection)
    
    def append(self, newparser):
        '''
        Add a new H5Parser into this combination.
        The added parser (subset) would be appended on the end of the parser list.
        '''
        if not isinstance(newparser, H5GParser):
            raise TypeError('The type of appended instance is not H5Parser, need to check the input.')
        self.__indexList.append(0)
        self.__sizeList.append(len(newparser))
        self.__parserList.append(newparser)
        self.__setSize += 1

class H5VGParser:
    '''Grouply parsing dataset for training/validating.
    This is a factory class. It accepts the same arguments of H5GParser,
    but split the dataset into a train set and a valid set.
    '''
    def __init__(self, fileName, keywords, batchSize=32, shuffle=True, preprocfunc=None):
        '''
        Initialize the H5VGParser. This parser could not be used directly, it requires users to call
        a split method and get two H5GParsers.
        '''
        self.trainSet = H5GParser(fileName, keywords, batchSize, shuffle, preprocfunc, _hasValidator=True)
        self.validSet = H5GParser(fileName, keywords, batchSize, shuffle, preprocfunc, _hasValidator=True)
        self.size = self.trainSet.size

    def numberSplit(self, validRate=0.1):
        '''
        Split the input set into a train set and a valid set by a specified index.
        This method would split the original set into two parts by a index. The first
        part is the train set while the other part is the valid set.
        Arguments:
            validRate: the ratio of the samples of the splitted valid set.
        '''
        validSize = int(validRate * self.size)
        if validSize <= 0 or validSize >= self.size:
            raise ValueError('The validation rate should be in (0.0, 1.0) to ensure that'
                             'both train set and validation set have more than one sample.')
        trainSize = self.size - validSize
        allInd = np.arange(self.size, dtype=np.int)
        self.trainSet.applyValidator(allInd[:trainSize])
        self.validSet.applyValidator(allInd[trainSize:])

    def randomSplit(self, validRate=0.1, seed=None):
        '''
        Split the input set into a train set and a valid set by random selection.
        Arguments:
            validRate: the ratio of the samples of the splitted valid set.
            seed:      random seed, recommend to specify a number.
        '''
        validSize = int(validRate * self.size)
        if validSize <= 0 or validSize >= self.size:
            raise ValueError('The validation rate should be in (0.0, 1.0) to ensure that'
                             'both train set and validation set have more than one sample.')
        if seed is not None:
            st = np.random.get_state()
            np.random.seed(seed)
        validChoice = np.random.choice(self.size, size=validSize, replace=False, p=None)
        if seed is not None:
            np.random.set_state(st)
        validChoice = np.sort(validChoice)
        validInd = []
        trainInd = []
        s = 0
        for i in range(self.size):
            if s < validSize and i == validChoice[s]:
                s += 1
                validInd.append(i)
            else:
                trainInd.append(i)
        validInd = np.asarray(validInd, dtype=np.int)
        trainInd = np.asarray(trainInd, dtype=np.int)
        self.trainSet.applyValidator(trainInd)
        self.validSet.applyValidator(validInd)

class H5GParser(tf.keras.utils.Sequence):
    '''Grouply parsing dataset
    This class allows users to feed one .h5 file, and convert it to 
    tf.keras.utils.Sequence. The realization could be described as:
        (1) Create .h5 file handle.
        (2) Using the user defined keywords to get a group of datasets.
        (3) Estimate the dataset sizes, and generate indices. Note each
            dataset should share the same size (but could be different 
            shapes).
        (4) Use the indices to create a tf.data.Dataset, and allows it
            to randomly shuffle the indices in each epoch.
        (5) Use Dataset.map() to address the data by the index from the
            index dataset.
    Certainly, you could use this parser to load a single dataset.
    '''
    def __init__(self, fileName, keywords, batchSize=32, shuffle=True, preprocfunc=None, _hasValidator=False):
        '''
        Create the parser and its h5py file handle.
        Arguments:
            fileName: the data path of the file (could be without postfix).
            keywords: should be a list of keywords (or a single keyword).
            batchSize: number of samples in each batch.
            shuffle: if on, shuffle the data set at the end of each epoch.
            preprocfunc: this function would be added to the produced data
                         so that it could serve as a pre-processing tool.
                         Note that this tool would process the batches
                         produced by the parser.
        Reserved arguments:
            _hasValidator: a flag for existence of a validator, which is
                           used to a train set and a valid set simultane-
                           ously. This argument should not be used by user.
        '''
        super(H5GParser, self).__init__()
        self.f = None
        if isinstance(keywords, str):
            self.keywords = (keywords,)
        else:
            self.keywords = keywords
        if (not os.path.isfile(fileName)) and (os.path.isfile(fileName+'.h5')):
            fileName += '.h5'
        self.f = h5py.File(fileName, 'r')
        self.__dsets = self.__creatDataSets()
        self.size = self.__createSize()
        if not _hasValidator:
            self.__indices = self.__indexDataset()
        self.shuffle = shuffle
        if shuffle and (not _hasValidator):
            self.__shuffle()
        self.__preprocfunc = preprocfunc
        self.__batchSize = batchSize
        self.__dsize = len(self.__dsets)
    
    def applyValidator(self, validIndices):
        '''
        Apply a validator. This method accept indices produced by a validator
        and apply them to self. This method should not be called by user.
        '''
        self.__indices = validIndices
        self.size = len(validIndices)
        if self.shuffle:
            self.__shuffle()
        
    def __len__(self):
        '''
        Automatically calculate the steps for iterate the whole dataset.
        '''
        return np.ceil(np.sum(self.size)/self.__batchSize).astype(np.int)
        
    def __getitem__(self, idx):
        batchIndices = self.__indices[idx * self.__batchSize:(idx + 1) * self.__batchSize]
        res = []
        for j in range(self.__dsize):
            res.append([])
        for ind in batchIndices:
            smp = self.__mapSingle(ind)
            for j in range(self.__dsize):
                res[j].append(smp[j])
        for j in range(self.__dsize):
            res[j] = np.stack(res[j], axis=0)
        if self.__preprocfunc is not None:
            return self.__preprocfunc(*res)
        else:
            return tuple(res)
            
    def on_epoch_end(self):
        '''
        Shuffle the data set according to the settings.
        '''
        if self.shuffle:
            self.__shuffle()
        
    def __creatDataSets(self):
        '''
        Find all desired dataset handles, and store them.
        '''
        dsets = []
        for key in self.keywords:
            dsets.append(self.f[key])
        if not dsets:
            raise KeyError('Keywords are not mapped to datasets in the file.')
        return dsets
        
    def __createSize(self):
        '''
        Find the number of items in the dataset, only need to be run for once.
        Should be run after __creatDataSets.
        '''
        sze = len(self.__dsets[0])
        for dset in self.__dsets:
            if sze != len(dset):
                raise TypeError('The assigned keywords do not correspond to each other.')
        return sze
        
    def __indexDataset(self):
        '''
        Create a tensorflow index dataset, only need to be run for once.
        Should be run after __createSize.
        '''
        return np.arange(self.size, dtype=np.int)
        
    def __shuffle(self):
        '''
        Resort the indices randomly.
        '''
        np.random.shuffle(self.__indices)

    def __mapSingle(self, index):
        '''
        Map function, for multiple datasets mode.
        '''
        return tuple(dset[index] for dset in self.__dsets)
