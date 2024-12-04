import matplotlib.pyplot as plt
import numpy as np

def viewHiddenReps(opt):
    fig, axs = plt.subplots(3, 5)
    for i in range(15):
        stats = np.load('stats/' + opt.stats_fn+'.npy')
        
        colors = stats[:, opt.hidden_rep_dim].astype('int')
        stats = stats[:, :opt.hidden_rep_dim]
        stats = stats[np.logical_or(colors == 4, colors == 9)]
        colors = colors[np.logical_or(colors == 4, colors == 9)]
        
        c_arr = ['r', 'g', 'b', 'c', 'gray', 'orange', 'purple', 'black', 'yellow', 'pink']
        c = [c_arr[x] for x in colors]
        x = (stats[:500, 0] - np.mean(stats[:500, 0])) / np.std(stats[:500, 0])
        y = (stats[:500, 1] - np.mean(stats[:500, 1])) / np.std(stats[:500, 1])
        axs[i % 3, i // 3].scatter(stats[:500, 2 * i], stats[:500, 2 * i + 1], c=c[:500])
          
    plt.show()