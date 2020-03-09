####### カラーバーサンプルの表示 #######
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import seaborn as sns
sns.set()


cmaps = OrderedDict()
cmaps['Perceptually Uniform Sequential'] = ['inferno', 'magma', 'plasma', 'viridis']
cmaps['Sequential'] = ['Blues', 'BuGn', 'BuPu', 'GnBu', 'Greens', 'Greys', 'OrRd', 'Oranges', 'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu', 'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']
cmaps['Sequential (2)'] = ['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink', 'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper']
cmaps['Diverging'] = ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
cmaps['Qualitative'] = ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c']
cmaps['Miscellaneous'] = ['flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv', 'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']
# 表示するデータとして (1, 256) の配列を作成する。
gradient = np.linspace(0, 1, 256).reshape(1, -1)
def plot_color_maps(cmap_category, cmap_list):
    num_cmaps = len(cmap_list)
    fig, axes = plt.subplots(num_cmaps, 2, figsize=(9, num_cmaps * 0.35))
    fig.subplots_adjust(wspace=0.4)
    axes[0][0].set_title(cmap_category + ' colormaps', fontsize=14, x=1.2)
    
    def plot_color_map(ax, gradient, name):
        ax.imshow(gradient, aspect='auto', cmap=name)
        ax.set_axis_off()
        ax.text(-10, 0, name, va='center', ha='right', fontsize=10)
    
    for [axL, axR], name in zip(axes, cmap_list):
        plot_color_map(axL, gradient, name)
        plot_color_map(axR, gradient, name + '_r')

for cmap_category, cmap_list in cmaps.items():
    plot_color_maps(cmap_category, cmap_list)
plt.show()



####### カラーマップサンプルの表示 #######
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import seaborn as sns
sns.set()

cmaps = OrderedDict()
cmaps['Sequential'] = ['Blues', 'BuGn', 'BuPu', 'GnBu', 'Greens', 'Greys', 'OrRd', 'Oranges', 'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu', 'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']
cmaps['Sequential (2)'] = ['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink', 'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper']
cmaps['Diverging'] = ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
cmaps['Qualitative'] = ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c']
cmaps['Miscellaneous'] = ['flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv', 'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']
# データを作成する。
mean, cov = [0, 2], [(1, 0.5), (0.5, 1)]
x, y = np.random.multivariate_normal(mean, cov, size=50).T
def density_plot(cmaps):
    cols = 4
    rows = np.ceil(len(cmaps) / 4)
    fig = plt.figure(figsize=(10, 2.5 * rows))
    for i, cmap in enumerate(cmaps, 1):
        ax = fig.add_subplot(rows, cols, i)
        sns.kdeplot(x, y, shade=True, ax=ax, cmap=cmap)
        ax.axis('off')
        ax.set_title(cmap)
    plt.show()

density_plot(cmaps['Sequential'])
density_plot(cmaps['Sequential (2)'])
density_plot(cmaps['Diverging'])
density_plot(cmaps['Qualitative'])
density_plot(cmaps['Miscellaneous'])











