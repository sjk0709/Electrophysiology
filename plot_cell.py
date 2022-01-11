import os, sys
import time, glob

import random
import numpy as np
import matplotlib.pyplot as plt


def plot_AP(title, times, AP, save_path=None):
    
    fig, ax = plt.subplots(figsize=(6,4))    
    fig.suptitle(title, fontsize=14)
    # ax.set_title('Simulation %d'%(simulationNo))
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')     
    ax.plot(times, AP, label='AP')   
    # textstr = "GNa : %1.4f\nGNaL : %1.4f\nGto : %1.4f\nPCa : %1.4f\nGKr : %1.4f\nGKs : %1.4f\nGK1 : %1.4f\nGf : %1.4f"%(GNa/g_fc[0], GNaL/g_fc[1], Gto/g_fc[2], PCa/g_fc[3], GKr/g_fc[4], GKs/g_fc[5], GK1/g_fc[6], Gf/g_fc[7])
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    #     ax.text(0.67, 0.60, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)    
    #     fig1 = plt.gcf()
    ax.legend()
    plt.show()
    if save_path != None:
        fig.savefig(save_path, dpi=100)


def plot_current(title, times, current_li, label_li):
    
    fig, ax = plt.subplots(figsize=(6,4))    
    fig.suptitle(title, fontsize=14)
    # ax.set_title('Simulation %d'%(simulationNo))
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')     

    for current, label in zip(current_li, label_li):
        ax.plot(times, current, label=label)   
    # textstr = "GNa : %1.4f\nGNaL : %1.4f\nGto : %1.4f\nPCa : %1.4f\nGKr : %1.4f\nGKs : %1.4f\nGK1 : %1.4f\nGf : %1.4f"%(GNa/g_fc[0], GNaL/g_fc[1], Gto/g_fc[2], PCa/g_fc[3], GKr/g_fc[4], GKs/g_fc[5], GK1/g_fc[6], Gf/g_fc[7])
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    #     ax.text(0.67, 0.60, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)    
    #     fig1 = plt.gcf()
    ax.legend()
    plt.show()
    # fig.savefig(os.path.join(result_folder, "AP.jpg"), dpi=100)


if __name__=='__main__':
    
    start_time = time.time()
   
   

    print("--- %s seconds ---"%(time.time()-start_time))