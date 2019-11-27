'''
################################################################
# Utilities - Extended tensorboard tools
# @ Modern Deep Network Toolkits for Tensorflow-Keras
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6+
#   tensorflow r1.13+
# Extended tools for parsing the logs in the tensorboard file.
# It enables users to extract records from tensorboard without
# launching the web interface. It also provides a python func-
# tion for launching the web interface.
# Version: 0.20 # 2019/11/27
# Comments:
#   Finish TensorLogHandle. It may be updated in future
#   versions.
# Version: 0.10 # 2019/11/26
# Comments:
#   Create this submodule and finish TensorBoardTool, launch.
################################################################
'''

import os, sys, logging
import numpy as np
import h5py
from tensorboard import default
from tensorboard import program
from tensorboard.backend.event_processing import event_accumulator

class TensorBoardTool:
    '''Tensorboard web interface launcher.
    Adapted from the original work here:
        https://stackoverflow.com/a/52295534
    This class is equivalent to call launch() in this module.
    Arguments:
        log_dir: the path where we store the logs.
        ip [optional]: the IP address for the web interface.
        port [optional]: the port number for the web interface.
    '''
    def __init__(self, log_dir, ip=None, port=None):
        '''Initialization
        see the docstring of this class.
        '''
        self.log_dir = log_dir
        self.ip = ip
        self.port = port

    def __collect_argvs(self):
        argvs = [None, '--logdir', str(self.log_dir)]
        if self.ip:
            argvs.extend(['--host', str(self.ip)])
        if self.port:
            argvs.extend(['--port', str(self.port)])
        return argvs

    def run(self):
        '''Launch the tensorboard.
        Note that this method would not block the main thread, we
        suggest to use launch() instead of this when you do not need
        to work with subthread.
        '''
        program.setup_environment()
        # Remove http messages
        log = logging.getLogger('werkzeug').setLevel(logging.ERROR)
        # Start tensorboard server
        _tb = program.TensorBoard(
            default.get_plugins(),
            program.get_default_assets_zip_provider())
        _tb.configure(argv=self.__collect_argvs())
        url = _tb.launch()
        print('TensorBoard at {0}, working on path: {1}.'.format(url, self.log_dir))

class TensorLogHandle:
    '''Read a tensorboard log file.
    This is a dictionary-lite interface for parsing a tensorboard
    file. It manages a EventAccumulator and wrap it with key-driven
    interfaces.
    Sometimes the handle may be slow, this is caused by the backend
    EventAccumulator. A possible way for solving this problem is
    passing a size guide during the initialization, but this sugge-
    stion could not guarantee the efficiency.
    Arguments:
        path: A file path to a directory containing tf events
            files, or a single tf events file. The accumulator
            will load events from this path.
        mode: The default working mode. Should be one of the
            avaliable list:
            (1) scalars (2) images (3) audio (4) histograms
            (5) distributions (6) tensors (7) metadata
        size_guidance: Information on how much data the
            EventAccumulator should store in memory. The 
            DEFAULT_SIZE_GUIDANCE tries not to store too much so as
            to avoid OOMing the client. The size_guidance should be
            a map from a `tagType` string to an integer representing
            the number of items to keep per tag for items of that
            `tagType`. If the size is 0, all events are stored.
        compression_bps: Information on how the `EventAccumulator`
            should compress histogram data for the
            `CompressedHistograms` tag (for details see
            `ProcessCompressedHistogram`).
        purge_orphaned_data: Whether to discard any events that
            were "orphaned" by a TensorFlow restart.
    '''
    MODE_LIST = {'scalars':event_accumulator.SCALARS,
                 'images': event_accumulator.IMAGES,
                 'audio': event_accumulator.AUDIO,
                 'histograms': event_accumulator.HISTOGRAMS,
                 'distributions': event_accumulator.COMPRESSED_HISTOGRAMS,
                 'tensors': event_accumulator.TENSORS,
                 'metadata': event_accumulator.RUN_METADATA}

    def __init__(self, path, mode='scalars', size_guidance=None,
                 compression_bps=event_accumulator.NORMAL_HISTOGRAM_BPS,
                 purge_orphaned_data=True):
        '''Initialization
        see the docstring of this class.
        '''
        self.__curMode = None
        self.setDefaultMode(mode)
        self.accumulator = event_accumulator.EventAccumulator(path=path, 
            size_guidance=size_guidance, compression_bps=compression_bps,
            purge_orphaned_data=purge_orphaned_data)
        self.accumulator.Reload()
        self.__keys = self.accumulator.Tags()

    def setDefaultMode(self, mode):
        '''Set the default working mode.
        Arguments:
            mode: The default mode, should be chosen from the avaliable
                list:
                (1) scalars (2) images (3) audio (4) histograms
                (5) tensors
        '''
        if self.__checkMode(mode):
            self.__curMode = self.MODE_LIST[mode]
        else:
            raise KeyError('Should choose mode from: {0}.'.format(self.MODE_LIST.keys()))
    
    @classmethod
    def __checkMode(cls, mode):
        return mode in cls.MODE_LIST

    def __contains__(self, key):
        return key in self.__keys[self.__curMode]

    def __getitem__(self, key):
        try:
            if isinstance(key, tuple) and len(key) == 2:
                if not (key[1] in self.__keys[self.MODE_LIST[key[0]]]):
                    raise KeyError
                return self.__getval(self.MODE_LIST[key[0]], key[1])
            else:
                if not (key in self.__keys[self.__curMode]):
                    raise KeyError
                return self.__getval(self.__curMode, key)
        except KeyError:
            raise KeyError('Could not find the item: {0}.'.format(key))

    def __len__(self):
        return len(self.__keys[self.__curMode])

    def __bool__(self):
        return bool(self.__keys[self.__curMode])

    def __iter__(self):
        return iter(self.__keys[self.__curMode])

    def keys(self, mode=None):
        '''Get all avaliable keys.
        Arguments:
            mode: The working mode, if not specified, would use
                default mode.
        '''
        if mode is not None:
            if not self.__checkMode(mode):
                raise KeyError('The specified mode is invalid, should choose from {0}.'.format(self.MODE_LIST.keys()))
            return iter(self.__keys[self.MODE_LIST[mode]])
        else:
            return iter(self.__keys[self.__curMode])

    def items(self, mode=None):
        '''Get all avaliable (k, v) pairs.
        Arguments:
            mode: The working mode, if not specified, would use
                default mode.
        '''
        if mode is not None:
            if not self.__checkMode(mode):
                raise KeyError('The specified mode is invalid, should choose from {0}.'.format(self.MODE_LIST.keys()))
            return map(lambda key: (key, self.__getval(self.MODE_LIST[mode], key)), self.__keys[self.MODE_LIST[mode]])
        else:
            return map(lambda key: (key, self.__getval(self.__curMode, key)), self.__keys[self.__curMode])
    
    def values(self, mode=None):
        '''Get all avaliable values.
        Arguments:
            mode: The working mode, if not specified, would use
                default mode.
        '''
        if mode is not None:
            if not self.__checkMode(mode):
                raise KeyError('The specified mode is invalid, should choose from {0}.'.format(self.MODE_LIST.keys()))
            return map(lambda key: self.__getval(mode, key), self.__keys[self.MODE_LIST[mode]])
        else:
            return map(lambda key: self.__getval(self.__curMode, key), self.__keys[self.__curMode])

    def __getval(self, mode, key):
        '''Protected function for getting item.
        Should not be called by users.
        '''
        if mode == event_accumulator.SCALARS:
            return self.__parserScalar(self.accumulator.Scalars(key))
        elif mode == event_accumulator.IMAGES:
            return self.accumulator.Images(key)
        elif mode == event_accumulator.AUDIO:
            return self.accumulator.Audio(key)
        elif mode == event_accumulator.HISTOGRAMS:
            return self.__parserHistogram(self.accumulator.Histograms(key))
        elif mode == event_accumulator.COMPRESSED_HISTOGRAMS:
            return self.__parserDistribution(self.accumulator.CompressedHistograms(key))
        elif mode == event_accumulator.RUN_METADATA:
            return self.accumulator.RunMetadata(key)
        elif mode == event_accumulator.TENSORS:
            return self.accumulator.Tensors(key)
        else:
            raise KeyError('The specified mode is invalid.')

    @staticmethod
    def __parserScalar(scalars):
        '''Parse the scalar list, and arrange the results.'''
        resDict = dict()
        if not scalars:
            return resDict
        else:
            for k in scalars[0]._asdict():
                resDict[k] = []
        for i in scalars:
            for k, v in i._asdict().items():
                resDict[k].append(v)
        for k, v in resDict.items():
            resDict[k] = np.asarray(v, dtype=np.float32)
        return resDict

    @staticmethod
    def __parserHistogram(histograms):
        '''Parse the histogram list, and arrange the results.'''
        resDict = dict()
        if not histograms:
            return resDict
        else:
            for k in histograms[0]._asdict():
                resDict[k] = []
        for i in histograms:
            for k, v in i._asdict().items():
                if k == 'histogram_value':
                    v = {
                        'x': np.asarray(v.bucket_limit, dtype=np.float32),
                        'n': np.asarray(v.bucket, dtype=np.float32),
                        'count': v.num
                    }
                resDict[k].append(v)
        for k, v in resDict.items():
            if k in ('wall_time', 'step'):
                resDict[k] = np.asarray(v, dtype=np.float32)
        return resDict

    @staticmethod
    def __parserDistribution(distributions):
        '''Parse the distribution list, and arrange the results.'''
        resDict = dict()
        if not distributions:
            return resDict
        else:
            for k in distributions[0]._asdict():
                resDict[k] = []
        for i in distributions:
            for k, v in i._asdict().items():
                if k == 'compressed_histogram_values':
                    x = []
                    val = []
                    for j in v:
                        x.append(j.basis_point)
                        val.append(j.value)
                    v = np.stack([x, val], axis=0)
                resDict[k].append(v)
        for k, v in resDict.items():
            resDict[k] = np.asarray(v, dtype=np.float32)
        return resDict

    def tohdf5(self, f, mode=None, compressed=True):
        '''Convert all data in a specific mode to HDF5 format.
        Arguments:
            f: a file path (would create a new file).
                or an h5py file object.
                or an h5py data group object.
            mode: the selected mode, if left None, would use the
                default mode.
            compressed: whether to apply the compression.
        '''
        if mode is None:
            mode = self.__curMode
        if mode not in (event_accumulator.SCALARS, 
            event_accumulator.HISTOGRAMS, 
            event_accumulator.COMPRESSED_HISTOGRAMS):
            raise ValueError('Your current mode is {0}, this type does'
                             'not support HDF5 conversion.'.format(mode))
        if isinstance(f, str):
            f = os.path.splitext(f)[0] + '.h5'
            f = h5py.File(f, 'w')
        name = f.filename
        f.attrs['type'] = mode
        for k, v in self.items():
            g = f.create_group(k)
            self.__recursive_writer(g=g, obj=v, compressed=compressed)
            print('Having dumped {0}.'.format(k))
        f.close()
        print('Having dumped the data {0} successfully.'.format(name))
        
    @classmethod
    def __recursive_writer(cls, g, obj, compressed=True):
        '''Recursive writer
        Should not be gotten accessed by users'''
        if isinstance(obj, dict):
            for k, v in obj.items():
                cls.__recursive_writer_work(g, k, v, compressed)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                cls.__recursive_writer_work(g, str(i), v, compressed)
        else:
            raise ValueError('The data part could not get parsed, check {0}'.format(obj))

    @classmethod 
    def __recursive_writer_work(cls, g, k, v, compressed=True):
        compression = 'gzip' if compressed else None
        if isinstance(v, (int, float)):
            g.create_dataset(k, data=float(v), dtype=np.float32)
        elif isinstance(v, np.ndarray):
            g.create_dataset(k, data=v, dtype=np.float32, chunks=(v.ndim>1), compression=compression)
        elif isinstance(v, (dict, list, tuple)):
            newg = g.create_group(k)
            cls.__recursive_writer(newg, obj=v, compressed=compressed)
        else:
            raise ValueError('The data part could not get parsed, check {0}: {1}'.format(k, v))

def launch(log_dir, ip=None, port=None):
    '''Tensorboard web interface launcher (function).
    Functional interface for launching a tensorboard.
    This class is equivalent to call TensorBoardTool.run() in this
    module.
    Arguments:
        log_dir: the path where we store the logs.
        ip [optional]: the IP address for the web interface.
        port [optional]: the port number for the web interface.
    '''
    osKey = 'GCS_READ_CACHE_DISABLED'
    getOS = os.environ.get(osKey, None)
    os.environ[osKey] = '1'
    tb = TensorBoardTool(log_dir, ip=ip, port=port)
    tb.run()
    input('Press Enter to ternimate this program.')
    if getOS is None:
        os.environ.pop(osKey)
    else:
        os.environ[osKey] = getOS

if __name__ == '__main__':
    os.chdir(sys.path[0])
    def test_thandle():
        th = TensorLogHandle('../../logs/test', 'scalars')
        #print(th['residual2d_transpose/alpha_0'])
        th.tohdf5('../../getscalar')

    #launch('../../logs/test', 'localhost', 8000)
    test_thandle()
