'''
################################################################
# Utilities - Extended visualization tools (mpl)
# @ Modern Deep Network Toolkits for Tensorflow-Keras
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6+
#   tensorflow r1.13+
#   matplotlib 3.1.1+
# Extended figure drawing tools. This module is based on
# matplotlib and provides some fast interfacies for drawing
# some specialized figures (like loss function).
# Version: 0.25 # 2019/12/05
# Comments:
#   1. Finish plot_distribution_curves.
#   2. Fix some bugs.
# Version: 0.20 # 2019/11/26
# Comments:
#   Finish plot_scatter, plot_training_records and
#   plot_error_curves.
# Version: 0.10 # 2019/11/23
# Comments:
#   Create this submodule, and finish such drawing tools:
#       setFontSize, useTex, fixLogAxis, plot_hist, plot_bar.
################################################################
'''

import itertools, functools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

class setFigure(object):
    '''setFigure decorator.
    A decorator class, which is used for changing the figure's
    configurations locally for a specific function.
    Arguments:
        style: the style of the figure. The available list 
            could be referred here:
            https://matplotlib.org/3.1.1/gallery/style_sheets/style_sheets_reference.html
        font_size: the local font size for the decorated function.
        use_tex: whether to use LaTeX backend for the output figure.
    An example is
    ```python
        @mdnt.utilities.draw.setFigure(font_size=12)
        def plot_curve():
            ...
    ```
    '''
    def __init__(self, style=None, font_size=None, use_tex=None):
        self.style = style
        self.font_size = font_size
        self.use_tex = use_tex

    def font_wrapped(self, foo, *args, **kwargs):
        if self.font_size:
            curFont = mpl.rcParams['font.size']
            mpl.rcParams['font.size'] = self.font_size
        try:
            res = foo(*args, **kwargs)
        finally:
            if self.font_size:
                mpl.rcParams['font.size'] = curFont
        return res

    def style_wrapped(self, foo, *args, **kwargs):
        if self.style:
            with plt.style.context(self.style):
                res = foo(*args, **kwargs)
        else:
            res = foo(*args, **kwargs)
        return res

    def tex_wrapped(self, foo, *args, **kwargs):
        if self.use_tex is not None:
            restore = dict()
            useafm = mpl.rcParams.get('ps.useafm', None)
            if useafm is not None:
                restore['ps.useafm'] = useafm
            use14corefonts = mpl.rcParams.get('pdf.use14corefonts', None)
            if use14corefonts is not None:
                restore['pdf.use14corefonts'] = use14corefonts
            usetex = mpl.rcParams.get('text.usetex', None)
            if usetex is not None:
                restore['text.usetex'] = usetex
            mpl.rcParams['ps.useafm'] = self.use_tex
            mpl.rcParams['pdf.use14corefonts'] = self.use_tex
            mpl.rcParams['text.usetex'] = self.use_tex
        try:
            res = foo(*args, **kwargs)
        finally:
            if self.use_tex is not None:
                for k, v in restore:
                    mpl.rcParams[k] = v
        return res

    @staticmethod
    def func_wrapper(foo, wrapped_func):
        def feed_func(*args, **kwargs):
            return wrapped_func(foo, *args, **kwargs)
        return feed_func

    def __call__(self, foo, *args, **kwargs):
        @functools.wraps(foo)
        def inner_func(*args, **kwargs):
            foo_font_size = self.func_wrapper(foo, self.font_wrapped)
            foo_tex       = self.func_wrapper(foo_font_size, self.tex_wrapped)
            foo_style     = self.func_wrapper(foo_tex, self.style_wrapped)
            return foo_style(*args, **kwargs)
        return inner_func

def useTex(flag=False):
    '''Switch the maplotlib backend to LaTeX.
    Arguments:
        flag: a bool, indicating whether to use the LaTeX backend
            for drawing figures.
    '''
    mpl.rcParams['ps.useafm'] = flag
    mpl.rcParams['pdf.use14corefonts'] = flag
    mpl.rcParams['text.usetex'] = flag

def fixLogAxis(ax=None, axis='y'):
    '''Control the log axis to be limited in 10^n ticks.
    Arguments:
        ax: the axis that requires to be controlled. If set None,
            the gca() would be used.
        axis: x, y or 'xy'.
    '''
    if ax is None:
        ax = plt.gca()
    if axis.find('y') != -1:
        ymin, ymax = np.log10(ax.get_ylim())
        ymin = np.floor(ymin) if ymin - np.floor(ymin) < 0.3 else ymin
        ymax = np.ceil(ymax) if np.ceil(ymax) - ymax < 0.3 else ymax
        ax.set_ylim(*np.power(10.0, [ymin, ymax]))
    if axis.find('x') != -1:
        xmin, xmax = np.log10(ax.get_xlim())
        xmin = np.floor(xmin) if xmin - np.floor(xmin) < 0.3 else xmin
        xmax = np.ceil(xmax) if np.ceil(xmax) - xmax < 0.3 else xmax
        ax.set_xlim(*np.power(10.0, [xmin, xmax]))

def plot_hist(gen, normalized=False, cumulative=False,
              xlabel='Value', ylabel='Number of samples',
              x_log=False, y_log=False,
              figure_size=(6, 5.5),
              legend_loc=None
             ):
    '''Plot a histogram for multiple distributions.
    Arguments:
        gen: a sample generator, each "yield" returns a sample. It
            allows users to provide an extra kwargs dict for each
            iteration. For each iteration it returns 1 1D data.
        normalized: whether to use normalization for each group
            when drawing the histogram.
        xlabel: the x axis label.
        ylabel: the y axis label.
        x_log: whether to convert the x axis into the log repre-
            sentation.
        y_log: whether to convert the y axis into the log repre-
            sentation.
        figure_size: the size of the output figure.
        legend_loc: the localtion of the legend. (The legend
            only works when passing `label` to each iteration)
    '''
    # Get iterator
    cit = itertools.cycle(mpl.rcParams['axes.prop_cycle'])
    # Set scale
    if x_log:
        plt.xscale('log')
    # Begin to parse data
    hasLabel = False
    for data in gen:
        c = next(cit)
        if isinstance(data, (tuple, list)) and len(data)>1 and isinstance(data[-1], dict):
            kwargs = data[-1]
            data = data[0]
        else:
            kwargs = dict()
        hasLabel = 'label' in kwargs
        kwargs.update(c)
        plt.hist(data, alpha=0.8, density=normalized, cumulative=cumulative, log=y_log, **kwargs)
    if ylabel:
        plt.ylabel(ylabel)
    if xlabel:
        plt.xlabel(xlabel)
    if figure_size:
        plt.gcf().set_size_inches(*figure_size)
    plt.tight_layout(rect=[0.03, 0, 0.97, 1])
    if hasLabel:
        plt.legend(loc=legend_loc, labelspacing=0.)

def plot_bar(gen, num,
             xlabel=None, ylabel='value',
             x_tick_labels=None, y_log=False,
             figure_size=(6, 5.5),
             legend_loc=None
            ):
    '''Plot a bar graph for multiple result groups.
    Arguments:
        gen: a sample generator, each "yield" returns a sample. It
            allows users to provide an extra kwargs dict for each
            iteration. For each iteration it returns 1 1D data.
        num: the total number of data samples thrown by the
            generator.
        xlabel: the x axis label.
        ylabel: the y axis label.
        x_tick_labels: the x tick labels that is used for
            overriding the original value [0, 1, 2, ...].
        y_log: whether to convert the y axis into the log repre-
            sentation.
        figure_size: the size of the output figure.
        legend_loc:  the localtion of the legend. (The legend
            only works when passing `label` to each iteration)
    '''
    # Get iterators
    cit = itertools.cycle(mpl.rcParams['axes.prop_cycle'])
    width = 0.75
    width = tuple(zip(np.linspace(-width/2, width/2, num+1)[:-1], width/num * np.ones(num)))
    wit = itertools.cycle(width)
    # Get tick labels
    if x_tick_labels is not None:
        x_tick_labels = list(x_tick_labels)
        x = np.arange(len(x_tick_labels))
    else:
        x = None
    # Set scale
    if y_log:
        plt.yscale('log')
    # Begin to parse data
    hasLabel = False
    for data in gen:
        c = next(cit)
        wp, w = next(wit)
        if isinstance(data, (tuple, list)) and len(data)>1 and isinstance(data[-1], dict):
            kwargs = data[-1]
            data = data[0]
        else:
            kwargs = dict()
        hasLabel = 'label' in kwargs
        kwargs.update(c)
        if x is None:
            x = np.arange(len(data))
        plt.bar(x + wp + w/2, data, w, **kwargs)
    if ylabel:
        plt.ylabel(ylabel)
    if xlabel:
        plt.xlabel(xlabel)
    if figure_size:
        plt.gcf().set_size_inches(*figure_size)
    if x_tick_labels is not None:
        ax = plt.gca()
        ax.set_xticks(x)
        ax.set_xticklabels(x_tick_labels)
    plt.tight_layout(rect=[0.03, 0, 0.97, 1])
    if hasLabel:
        plt.legend(loc=legend_loc, labelspacing=0.)

def plot_scatter(gen,
                 xlabel=None, ylabel='value',
                 x_log=None, y_log=False,
                 figure_size=(6, 5.5),
                 legend_loc=None
                ):
    '''Plot a scatter graph for multiple data groups.
    Arguments:
        gen: a sample generator, each "yield" returns a sample. It
            allows users to provide an extra kwargs dict for each
            iteration. For each iteration, it returns 2 1D arrays
            or 1 2D array.
        xlabel: the x axis label.
        ylabel: the y axis label.
        x_log: whether to convert the x axis into the log repre-
            sentation.
        y_log: whether to convert the y axis into the log repre-
            sentation.
        figure_size: the size of the output figure.
        legend_loc:  the localtion of the legend. (The legend
            only works when passing `label` to each iteration)
    '''
    # Get iterators
    cit = itertools.cycle(mpl.rcParams['axes.prop_cycle'])
    mit = itertools.cycle(['o', '^', 's', 'd', '*', 'P'])
    # Set scale
    if x_log:
        plt.xscale('log')
    if y_log:
        plt.yscale('log')
    # Begin to parse data
    hasLabel = False
    for data in gen:
        c, m = next(cit), next(mit)
        if isinstance(data, (tuple, list)) and len(data)>1 and isinstance(data[-1], dict):
            kwargs = data[-1]
            if len(data) == 3:
                l_m, l_d = data[:2]
            elif data[0].shape[0] == 2:
                l_m, l_d = data[0]
            else:
                raise ValueError('Input data should be two 1D arrays or '
                                 'one 2D array with a shape of [2, L]')
        else:
            kwargs = dict()
        hasLabel = 'label' in kwargs
        kwargs.update(c)
        plt.scatter(l_m, l_d, marker=m, **kwargs)
    if ylabel:
        plt.ylabel(ylabel)
    if xlabel:
        plt.xlabel(xlabel)
    if figure_size:
        plt.gcf().set_size_inches(*figure_size)
    plt.tight_layout(rect=[0.03, 0, 0.97, 1])
    if hasLabel:
        plt.legend( loc=legend_loc, labelspacing=0. )

def plot_training_records(gen,
                          xlabel=None, ylabel='value',
                          x_mark_num=None, y_log=False,
                          figure_size=(6, 5.5),
                          legend_loc=None,
                          legend_col=None,
                         ):
    '''Plot a scatter graph for multiple data groups.
    Arguments:
        gen: a sample generator, each "yield" returns a sample. It
            allows users to provide an extra kwargs dict for each
            iteration. For each iteration it returns 4 1D arrays, or
            2 2D arrays, or 2 1D arryas, or 1 4D array, or 1 2D
            array, or 1 1D array.
        xlabel: the x axis label.
        ylabel: the y axis label.
        y_log: whether to convert the y axis into the log repre-
            sentation.
        figure_size: the size of the output figure.
        legend_loc: the localtion of the legend. (The legend
            only works when passing `label` to each iteration)
        legend_col: the column of the legend.
    '''
    # Get iterators
    cit = itertools.cycle(mpl.rcParams['axes.prop_cycle'])
    mit = itertools.cycle(['o', '^', 's', 'd', '*', 'P'])
    # Set scale
    if y_log:
        plt.yscale('log')
    # Begin to parse data
    kwargs = dict()
    hasLabel = False
    for data in gen:
        c, m = next(cit), next(mit)
        hasValid = None
        if isinstance(data, (tuple, list)):
            if isinstance(data[-1], dict):
                *data, kwargs = data
            if len(data) == 4: # 4 1D data tuple.
                x, v, val_x, val_v = data
                hasValid = True
            elif len(data) == 2: # 2 data tuple.
                d1, d2 = data
                if d1.ndim == 2 and d2.ndim == 2 and d1.shape[0] == 2 and d2.shape[0] == 2:
                    # 2 2D data.
                    x, v = d1
                    val_x, val_v = d2
                    hasValid = True
                elif d1.ndim == 1 and d2.ndim == 1:
                    # 2 1D data.
                    x, v = d1, d2
                    hasValid = False
                else:
                    raise ValueError('The input data shape is invalid, when using'
                                     'data sequence, there should be 4 1D data, or'
                                     ' 2 2D data, or 2 1D data.')
            elif len(data) == 1:
                data = data[0]
        if hasValid is None:
            if data.ndim == 2:
                if len(data) == 4:
                    x, v, val_x, val_v = data
                    hasValid = True
                elif len(data) == 2:
                    x, v = data
                    hasValid = False
                else:
                    raise ValueError('The input data shape is invalid, when using'
                                     'a single array, it should be 4D data, or'
                                     ' 2D data, or 1D data.')
            elif data.ndim == 1:
                x = np.arange(0, len(data))
                v = data
                hasValid = False
            else:
                raise ValueError('The input data shape is invalid, when using'
                                 'a single array, it should be 4D data, or'
                                 ' 2D data, or 1D data.')
        hasLabel = 'label' in kwargs
        kwargs.update(c)
        baseLabel = kwargs.pop('label', None)
        getLabel = baseLabel
        if (baseLabel is not None) and hasValid:
            getLabel = baseLabel + ' (train)'
        if x_mark_num is not None:
            x_mark = np.round(np.linspace(0, len(x)-1, x_mark_num)).astype(np.int).tolist()
        plt.plot(x, v, marker=m, ms=7, label=getLabel, markevery=x_mark, **kwargs)
        if hasValid:
            if baseLabel is not None:
                getLabel = baseLabel + ' (valid)'
            plt.plot(val_x, val_v, linestyle='--', marker=m, ms=7, markevery=x_mark, label=getLabel, **kwargs)
    if ylabel:
        plt.ylabel(ylabel)
    if xlabel:
        plt.xlabel(xlabel)
    if figure_size:
        plt.gcf().set_size_inches(*figure_size)
    plt.tight_layout(rect=[0.03, 0, 0.97, 1])
    if hasLabel:
        plt.legend( loc =legend_loc, labelspacing=0., ncol=legend_col )

def plot_error_curves(gen, x_error_num=10,
                      y_error_method='std', plot_method='error',
                      xlabel=None, ylabel='value',
                      y_log=False,
                      figure_size=(6, 5.5),
                      legend_loc=None
                     ):
    '''Plot lines with error bars for multiple data groups.
    Arguments:
        gen: a sample generator, each "yield" returns a sample. It
            allows users to provide an extra kwargs dict for each
            iteration. For each iteration it returns 1D + 2D arrays,
            or a single 2D array.
        x_error_num: the number of displayed error bars.
        y_error_method: the method for calculating the error bar.
            (1) std: use standard error.
            (2) minmax: use the range of the data.
        plot_method: the method for plotting the figure.
            (1) error: use error bar graph.
            (2) fill:  use fill_between graph.
        xlabel: the x axis label.
        ylabel: the y axis label.
        y_log: whether to convert the y axis into the log repre-
            sentation.
        figure_size: the size of the output figure.
        legend_loc:  the localtion of the legend. (The legend
            only works when passing `label` to each iteration)
    '''
    # Get iterators
    cit = itertools.cycle(mpl.rcParams['axes.prop_cycle'])
    mit = itertools.cycle(['o', '^', 's', 'd', '*', 'P'])
    # Set scale
    if y_log:
        plt.yscale('log')
    # Begin to parse data
    hasLabel = False
    kwargs = dict()
    for data in gen:
        c, m = next(cit), next(mit)
        x = None
        if isinstance(data, (tuple, list)):
            if isinstance(data[-1], dict):
                *data, kwargs = data
            if len(data) == 2: # 4 1D data tuple.
                x, data = data
            elif len(data) == 1:
                data = data[0]
            else:
                raise ValueError('The input data list is invalid, it should'
                                 'contain 1D + 2D array or a 2D array.')
        if data.ndim == 2:
            if x is None:
                x = np.arange(0, len(data))
            v = data
            getValue = True
        else:
            raise ValueError('The input data list is invalid, it should'
                                'contain 1D + 2D array or a 2D array.')
        hasLabel = 'label' in kwargs
        kwargs.update(c)
        avg = np.mean(v, axis=1)
        if y_error_method == 'minmax':
            geterr = np.stack((avg-np.amin(v, axis=1), np.amax(v, axis=1)-avg), axis=0)
        else:
            geterr = np.repeat(np.expand_dims(np.std(v, axis=1), axis=0), 2, axis=0)
        if plot_method == 'fill':
            mark_every = np.round(np.linspace(0, len(x)-1, x_error_num)).astype(np.int).tolist()
            plt.plot(x, avg, marker=m, ms=7, markevery=mark_every, **kwargs)
            plt.fill_between(x, avg - geterr[0, ...], avg + geterr[1, ...], alpha=0.3, color=c['color'])
        else:
            error_every = len(x) // x_error_num
            plt.errorbar(x, avg, yerr=geterr, errorevery=error_every, marker=m, ms=5, markevery=error_every, **kwargs)
    if ylabel:
        plt.ylabel(ylabel)
    if xlabel:
        plt.xlabel(xlabel)
    if figure_size:
        plt.gcf().set_size_inches(*figure_size)
    plt.tight_layout(rect=[0.03, 0, 0.97, 1])
    if hasLabel:
        plt.legend( loc =legend_loc, labelspacing=0. )

def plot_distribution_curves(gen, method='mean', level=3, outlier=0.1,
                             xlabel=None, ylabel='value',
                             y_log=False,
                             figure_size=(6, 5.5),
                             legend_loc=None
                            ):
    '''Plot lines with multi-level distribution for multiple data groups.
    This function has similar meaning of plot_error_curves. It is
    used for compressing the time-series histograms. Its output is
    similar to tensorboard.distribution.
    Arguments:
        gen: a sample generator, each "yield" returns a sample. It
            allows users to provide an extra kwargs dict for each
            iteration. For each iteration it returns 1D + 2D arrays,
            or a single 2D array.
        method: the method for calculating curves, use 'mean' or 
            'middle'.
        level: the histogram level.
        outlier: outlier proportion, this part would be thrown when
            drawing the figures.
        xlabel: the x axis label.
        ylabel: the y axis label.
        y_log: whether to convert the y axis into the log repre-
            sentation.
        figure_size: the size of the output figure.
        legend_loc:  the localtion of the legend. (The legend
            only works when passing `label` to each iteration)
    '''
    if level < 1:
        raise TypeError('Histogram level should be at least 1.')
    if method not in ('mean', 'middle'):
        raise TypeError('The curve calculation method should be either \'mean\' or \'middle\'.')
    # Get iterators
    cit = itertools.cycle(mpl.rcParams['axes.prop_cycle'])
    mit = itertools.cycle(['o', '^', 's', 'd', '*', 'P'])
    # Set scale
    if y_log:
        plt.yscale('log')
    # Begin to parse data
    hasLabel = False
    kwargs = dict()
    for data in gen:
        c, m = next(cit), next(mit)
        x = None
        if isinstance(data, (tuple, list)):
            if isinstance(data[-1], dict):
                *data, kwargs = data
            if len(data) == 2: # 4 1D data tuple.
                x, data = data
            elif len(data) == 1:
                data = data[0]
            else:
                raise ValueError('The input data list is invalid, it should'
                                 'contain 1D + 2D array or a 2D array.')
        if data.ndim == 2:
            if x is None:
                x = np.arange(0, len(data))
            v = data
            getValue = True
        else:
            raise ValueError('The input data list is invalid, it should'
                                'contain 1D + 2D array or a 2D array.')
        hasLabel = 'label' in kwargs
        kwargs.update(c)
        vsort = np.sort(v, axis=1)
        N = v.shape[1]
        if method == 'middle':
            avg = vsort[:, N//2]
        else:
            avg = np.mean(v, axis=1)    
        # Calculate ranges according to levels:
        vu, vd = [], []
        for i in range(level):
            if method == 'middle':
                pos = max(1, int(np.round((outlier + (1-outlier)*((i-1)/level))*N)+0.1)), max(1, int(np.round((outlier + (1-outlier)*(i/level))*N)+0.1))
                vd.append(np.mean(vsort[:, pos[0]:pos[1]], axis=1))
                vu.append(np.mean(vsort[:, (-pos[1]):(-pos[0])], axis=1))
            else:
                pos = max(1, int(np.round((outlier + (1-outlier)*(i/level))*N)+0.1))
                vd.append(np.mean(vsort[:, :pos], axis=1))
                vu.append(np.mean(vsort[:, (-pos):], axis=1))
        # Draw distributions
        mark_every = np.round(np.linspace(0, len(x)-1, 10)).astype(np.int).tolist()
        plt.plot(x, avg, marker=m, ms=7, markevery=mark_every, **kwargs)
        for i in range(level):
            plt.fill_between(x, vd[i], vu[i], alpha=0.2, color=c['color'])
    if ylabel:
        plt.ylabel(ylabel)
    if xlabel:
        plt.xlabel(xlabel)
    if figure_size:
        plt.gcf().set_size_inches(*figure_size)
    plt.tight_layout(rect=[0.03, 0, 0.97, 1])
    if hasLabel:
        plt.legend( loc =legend_loc, labelspacing=0. )

if __name__ == '__main__':
    @setFigure(style='ggplot', font_size=14)
    def test_plot_hist():
        def func_gen():
            getbins = np.linspace(0,25,80)
            x1 = np.random.normal(loc=7.0, scale=1.0, size=100)
            yield x1, {'bins':getbins, 'label': '$x_1$'}
            x2 = np.random.normal(loc=12.0, scale=3.0, size=1000)
            yield x2, {'bins':getbins, 'label': '$x_2$'}
        plot_hist(func_gen(), xlabel='x', normalized=True, cumulative=False)
        plt.show()

    @setFigure(style='dark_background', font_size=14)
    def test_plot_bar():
        def func_gen():
            size = 5
            x1 = np.abs(np.random.normal(loc=6.0, scale=3.0, size=size))
            yield x1, {'label': '$x_1$'}
            x2 = np.abs(np.random.normal(loc=9.0, scale=6.0, size=size))
            yield x2, {'label': '$x_2$'}
            x3 = np.abs(np.random.normal(loc=12.0, scale=6.0, size=size))
            yield x3, {'label': '$x_3$'}
        plot_bar(func_gen(), num=3, ylabel='y', y_log=False,
                 x_tick_labels=['Jan.', 'Feb.', 'Mar.', 'Apr.', 'May'])
        plt.show()

    @setFigure(style='seaborn-darkgrid', font_size=16)
    def test_scatter():
        def func_gen():
            size = 100
            for i in range(3):
                center = -4.0 + 4.0 * np.random.rand(2)
                scale = 0.5 + 2.0 * np.random.rand(2)
                x1 = np.random.normal(loc=center[0], scale=scale[0], size=size)
                x2 = np.random.normal(loc=center[1], scale=scale[1], size=size)
                yield np.power(10, x1), np.power(10, x2), {'label': r'$x_{' + str(i+1) + r'}$'}
        plot_scatter(func_gen(), x_log=True, y_log=True,
                xlabel='Metric 1', ylabel='Metric 2')
        plt.show()
    
    @setFigure(style='Solarize_Light2', font_size=14)
    def test_training_records():
        def func_gen_batch():
            size = 100
            x = np.arange(start=0, stop=size)
            for i in range(3):
                begin = 1 + 99.0 * np.random.rand()
                end = 2 + 10 * np.random.rand()
                v = begin * np.exp((np.square((x-size)/size)-1.0)*end)
                yield x, v, {'label': r'$x_{' + str(i+1) + r'}$'}
        def func_gen_epoch():
            size = 10
            x = np.arange(start=0, stop=size)
            for i in range(3):
                begin = 1 + 99.0 * np.random.rand()
                end = 2 + 10 * np.random.rand()
                v = begin * np.exp((np.square((x-size)/size)-1.0)*end)
                val_v = begin * np.exp((np.square((x-size)/size)-1.0)*(end-1))
                data = np.stack([x, v, x, val_v], axis=0)
                yield data, {'label': r'$x_{' + str(i+1) + r'}$'}
        plot_training_records(func_gen_batch(), y_log=True, x_mark_num=10,
                xlabel='Step', ylabel=r'Batch $\mathcal{L}$')
        plt.show()
        plot_training_records(func_gen_epoch(), y_log=True, x_mark_num=10,
                xlabel='Step', ylabel=r'Epoch $\mathcal{L}$')
        plt.show()

    @setFigure(style='bmh', font_size=16)
    def test_error_bar():
        def func_gen():
            size = 100
            x = np.arange(start=0, stop=size)
            for i in range(3):
                begin = 1 + 99.0 * np.random.rand()
                end = 2 + 10 * np.random.rand()
                exp_v = np.square((x-size)/size)-1.0
                exp_vnoise = np.random.normal(0.0, np.expand_dims((size-x)/(10*size), axis=-1), (size, 50))
                v = begin * np.exp((np.expand_dims(exp_v, axis=-1)+exp_vnoise)*end)
                yield x, v, {'label': r'$x_{' + str(i+1) + r'}$'}
        plot_error_curves(func_gen(), y_log=True,
                y_error_method='minmax',
                xlabel='Step', ylabel=r'$\mathcal{L}$')
        plt.show()
        plot_error_curves(func_gen(), y_log=True,
                y_error_method='minmax', plot_method='fill',
                xlabel='Step', ylabel=r'$\mathcal{L}$')
        plt.show()

    @setFigure(style='classic', font_size=16)
    def test_distribution():
        def func_gen():
            size = 100
            x = np.arange(start=0, stop=size)
            for i in range(1):
                begin = 1 + 99.0 * np.random.rand()
                end = 2 + 10 * np.random.rand()
                exp_v = np.square((x-size)/size)-1.0
                exp_vnoise = np.random.normal(0.0, np.expand_dims((size-x)/(10*size), axis=-1), (size, 50))
                v = begin * np.exp((np.expand_dims(exp_v, axis=-1)+exp_vnoise)*end)
                yield x, v, {'label': r'$x_{' + str(i+1) + r'}$'}
        plot_distribution_curves(func_gen(), method='mean', level=5, outlier=0.05,
                                 xlabel='Step', ylabel=r'$\mathcal{L}$',
                                 y_log=True)
        plt.show()

    test_plot_hist()
    test_plot_bar()
    test_scatter()
    test_training_records()
    test_error_bar()
    test_distribution()