import os, sys
import time, glob

import random
import numpy as np
import matplotlib.pyplot as plt


def plot_1D( t, V, title=None, figsize=(6,4), xlabel='Time (ms)', ylabel='Membrane Potential (mV)', 
             label=None, xlim=None, ylim=None,
             save_path=None):
    
    fig, ax = plt.subplots(figsize=figsize)    
    fig.suptitle(title, fontsize=14)
    # ax.set_title('Simulation %d'%(simulationNo))
    if xlim !=None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim !=None:
        ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlabel(xlabel)
    plt.ylabel(ylabel)     
    ax.plot(t, V, label=label)   
    # textstr = "GNa : %1.4f\nGNaL : %1.4f\nGto : %1.4f\nPCa : %1.4f\nGKr : %1.4f\nGKs : %1.4f\nGK1 : %1.4f\nGf : %1.4f"%(GNa/g_fc[0], GNaL/g_fc[1], Gto/g_fc[2], PCa/g_fc[3], GKr/g_fc[4], GKs/g_fc[5], GK1/g_fc[6], Gf/g_fc[7])
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    #     ax.text(0.67, 0.60, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)    
    #     fig1 = plt.gcf()
    if label != None:
        ax.legend()
    plt.show()
    if save_path != None:
        fig.savefig(save_path, dpi=100)


def plot_stack( t, values, labels=(), title=None, figsize=(15,2), xlabel='Time (ms)', ylabel=None, 
                xlim=None, ylim=None,
                save_path=None):
    
    fig, ax = plt.subplots(figsize=figsize)    
    # fig.suptitle('New ORD-hERG - '+ ord._cell_types[cell_mode] +' cell', fontsize=16)
    # ax.set_title('Simulation %d'%(simulationNo))
    if xlim !=None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim !=None:
        ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)     
    ax.stackplot(t, values, labels=labels, alpha=0.8)
    if labels != ():
        ax.legend()
    plt.show()
    if save_path != None:
        fig.savefig(save_path, dpi=100)



if __name__=='__main__':
    
    start_time = time.time()
   
   

    print("--- %s seconds ---"%(time.time()-start_time))