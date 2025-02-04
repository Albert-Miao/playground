import math
import matplotlib.pyplot as plt
import numpy as np

import seaborn


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

    
def linearSepMetric(group1, group2):
    mu1 = np.mean(group1, axis=0)
    mu2 = np.mean(group2, axis=0)
    cov1 = np.cov(group1.T)
    cov2 = np.cov(group2.T)
 
    dm = mu1 - mu2
    s12 = (cov1 + cov2) / 2
    invmatr = np.linalg.inv(s12)
    
    tmp = np.core.dot(dm.T, invmatr)
    tmp = np.core.dot(tmp, dm)
    MH = math.sqrt(tmp)
    
    tmp = np.linalg.det(s12) / math.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2))
    tmp = math.log(tmp)
    B = MH / 8.0 + tmp / 2.0
    
    JM = math.sqrt(2 * (1 - math.exp(-B)))
        
    return JM


def classSeparabilityEval(opt):
    stats = np.load('stats/' + opt.stats_fn + '.npy')
    groups = []
    
    for i in range(10):
        groups.append(stats[stats[:, -1] == i][:, :30])
        
    classSeps = np.ones((10, 10)) * math.sqrt(2)
    for i in range(10):
        for j in range(i, 10):
            if j == i:
                continue
            classSeps[i, j] = linearSepMetric(groups[i], groups[j])
            classSeps[j, i] = classSeps[i, j]
            
    avgs = np.mean(classSeps, axis=0)
    print('   |  0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  |')
    
    for i in range(10):
        output = str(i) + '  |'
        for j in range(10):
            if classSeps[i, j] == math.sqrt(2):
                output += ' N/A |'
            else:
                output += str(classSeps[i, j])[1:6] + '|'
        print(output)
        
    output = "avg|"
    for i in range(10):
        output += str(avgs[i])[1:6] + '|'
    print(output)
    
    print('Entire Average: ' + str(np.mean(avgs)))
    
    seaborn.heatmap(classSeps)
    plt.show()
    