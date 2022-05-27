#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 05 09:15:08 2022

___e-mail__ = cesar_husillos@tutanota.com
__author__ = 'Cesar Husillos'

DESCRIPTION: It compares two text files that contain polarimetry results and plot results in PNG file format.

VERSION:
    1.0 Initial version
"""
import argparse
import os

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

def read_res(path):
    if not os.path.exists(path):
        return 1

    input_lines = [l for l in open(path).read().split('\n') if len(l) > 0]

    header = [tok.strip() for tok in input_lines[0].split(' ') if len(tok) > 0]
    real_header = []
    for h in header:
        if h.find('+-') != -1:
            real_header += [s.strip() for s in h.split('+-')]
        else:
            real_header.append(h)
    # print(f'HEADER = {real_header}')

    data_dict = defaultdict(list)

    for line in input_lines[1:]:
        toks = [t for t in line.split() if len(t) > 0]
        if len(toks) < len(real_header):
            # Append empty fields to complete information
            toks += [-99] * (len(real_header) - len(toks))
        for h, t in zip(real_header, toks):
            try:
                data_dict[h].append(float(t))
            except:
                data_dict[h].append(t)

    return data_dict


def make_plots(df_data_man, df_data_pipe, out_png_path, title='160110'):
    plt.rcParams['legend.fontsize'] = 3
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    fig.suptitle(title)

    ax1.tick_params(direction='out', length=6, width=2, colors='r', \
        grid_color='r', grid_alpha=0.5)
    ax1.yaxis.set_tick_params(labelsize=6)
    ax1.plot(df_data_man['RJD-50000'].values, df_data_man['P'].values, 'r.', alpha=0.5, label='MANUAL')
    ax1.plot(df_data_pipe['RJD-50000'].values, df_data_pipe['P'].values, 'b.', alpha=0.5, label='IOP3')
    #ax1.xaxis.set_tick_params(rotation=30, labelsize=8)
    #ax1.yaxis.set_tick_params(labelsize=6)
    ps = np.concatenate([df_data_man['P'].values, df_data_pipe['P'].values])
    ticks = np.linspace(start=ps.min(), stop=ps.max(), num=5)
    ax1.set_yticks(ticks)
    ax1.grid()
    ax1.legend()
    ax1.set_title('P')

    ax2.tick_params(direction='out', length=6, width=2, colors='r', \
        grid_color='r', grid_alpha=0.5)
    ax2.yaxis.set_tick_params(labelsize=6)
    ax2.plot(df_data_man['RJD-50000'].values, df_data_man['Theta'].values, 'r.', alpha=0.5, label='MANUAL')
    ax2.plot(df_data_pipe['RJD-50000'].values, df_data_pipe['Theta'].values, 'b.', alpha=0.5, label='IOP3')
    theta = np.concatenate([df_data_man['Theta'].values, df_data_pipe['Theta'].values])
    ticks = np.linspace(start=theta.min(), stop=theta.max(), num=7)
    ax2.set_yticks(ticks)
    ax2.grid()
    ax2.legend()
    ax2.set_title('Theta')

    ax3.tick_params(direction='out', length=6, width=2, colors='r', \
        grid_color='r', grid_alpha=0.5)
    ax3.yaxis.set_tick_params(labelsize=6)
    df_m_r = df_data_man[df_data_man['R'] > 0]
    ax3.plot(df_m_r['RJD-50000'].values, df_m_r['R'].values, 'r.', alpha=0.5, label='MANUAL')
    ax3.plot(df_data_pipe['RJD-50000'].values, df_data_pipe['R'].values, 'b.', alpha=0.5, label='IOP3')
    
    R = np.concatenate([df_m_r['R'], df_data_pipe['R'].values])
    ticks = np.linspace(start=R.min(), stop=R.max(), num=7)
    ax3.set_yticks(ticks)
    ax3.grid()
    ax3.xaxis.set_tick_params(rotation=30, labelsize=6)
    ax3.legend()
    ax3.set_title('R')

    # plot object names (for old and new polarimetry measurements)
    # manual procedure
    for x, y, n in zip(df_m_r['RJD-50000'].values, df_m_r['R'].values - 1.2, df_m_r['Object'].values):
        ax3.text(x, y, n, {'rotation': 'vertical', 'fontsize': 3, 'color': 'r', 'horizontalalignment': 'right'})
    
    # iop3 procedure
    for x, y, n in zip(df_data_pipe['RJD-50000'].values, df_data_pipe['R'].values + 0.5, df_data_pipe['Object'].values):
        ax3.text(x, y, n, {'rotation': 'vertical', 'fontsize': 3, 'color': 'b', 'horizontalalignment': 'left'})


    plt.savefig(out_png_path, dpi=300)
    plt.close()

    return 0
    


def main():
    parser = argparse.ArgumentParser(prog='compare_polarimetry_results.py', \
    conflict_handler='resolve',
    description='''Main program. It compares two text files (IOP3 .res format) 
    that contain polarimetry results and plot results in PNG file format.''',
    epilog='''''')
    parser.add_argument("manual_polarymetry_res", help=".res format file obtained by semi-manual processing.")
    parser.add_argument("iop3_polarymetry_res", help=".res format file obtained by automatic IOP3 pipeline processing.")
    parser.add_argument("out_png_path", help=".png output path plot.")
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    parser.add_argument('-v',
                        '--verbose',
                        action='count',
                        default=0,
                        help="Show running and progress information [default: %(default)s].")
    args = parser.parse_args()

    data_man = read_res(args.manual_polarymetry_res)
    data_pipe = read_res(args.iop3_polarymetry_res)
    
    print(data_man)
    for k, v in data_man.items():
        print(f'{k} -> {len(v)}')
    print('*' * 30)
    print(data_pipe)
    for k, v in data_pipe.items():
        print(f'{k} -> {len(v)}')
    
    print('*' * 30)
    
    df_data_man = pd.DataFrame.from_dict(data_man)
    df_data_pipe = pd.DataFrame.from_dict(data_pipe)

    print('*' * 30)
    print(df_data_man)
    print('*' * 30)
    print(df_data_pipe)

    # Plotting
    root, ext = os.path.splitext(args.iop3_polarymetry_res)
    title = root.split('_')[-1]
    make_plots(df_data_man, df_data_pipe, args.out_png_path, title=title)


if __name__ == '__main__':
    print(main())