#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
The first step of analysis is to explore the basic statistics of the Himawari-8 images.
The dimension of our dataset:
  - 3300 x 3300 pixels
  - 15' ~ 30'N / 114' ~ 129' E (~500m in real space)
  - And the data came in every 10 minutes if there is no missing.
We will scan through the dataset, sumarize the data values:
1. image by image
2. grid by grid
'''

import numpy as np
import pandas as pd
import os

__author__ = "Ting-Shuo Yo"
__copyright__ = "Copyright 2019~2020, DataQualia Lab Co. Ltd."
__credits__ = ["Ting-Shuo Yo"]
__license__ = "Apache License 2.0"
__version__ = "0.1.0"
__maintainer__ = "Ting-Shuo Yo"
__email__ = "tingyo@dataqualia.com"
__status__ = "development"
__date__ = '2020-07-20'


# Utility functions
def list_himawari8_files(dir, suffix='.btp', to_remove='.B13.PCCU.btp'):
    ''' To scan through the sapecified dir and get the corresponding file with suffix. '''
    xfiles = []
    for root, dirs, files in os.walk(dir): 
        for fn in files:
            if fn.endswith(suffix): 
                 xfiles.append({'timestamp':fn.replace(to_remove,''), 'xuri':os.path.join(root, fn)})
    return(pd.DataFrame(xfiles))

# Binary reader
def read_himawari8_btp(furi):
    ''' The method reads in a Himawari 8 image in binary format (.btp file). 
        the binary file contains 10,890,000 float point numbers (4-byte) represent 
        the brightness temperature. '''
    data = np.fromfile(furi, np.float32)
    return(data.reshape((3300,3300)))

def read_multiple_himawari8(flist, test_falg=False):
    data = []
    for f in flist:
        if test_flag:
            data.append(read_himawari8_btp(f)[1600:1700,1600:1700])
        else:
            data.append(read_himawari8_btp(f))
    return(np.array(data))

# Statistical summary
def summarize_single_image(img):
    ''' Calculate basic statistics of one Himawari-8 image. '''
    mean = np.mean(img.flatten())
    std = np.std(img.flatten())
    pt = np.percentile(img.flatten(), [0, 25, 50, 75,100])
    return({'mean':mean, 'stdev':std, 'min':pt[0],'pt25':pt[1],'median':pt[2],'pt75':pt[3], 'max':pt[4]})

def summarize_images_by_grid(data):
    ni, nx, ny = data.shape
    mean_image = np.zeros((nx,ny))
    std_image = np.zeros((nx,ny))
    pr_image = np.zeros((nx,ny,5))
    for i in range(nx):
        for j in range(ny):
            mean_image[i,j] = np.mean(data[:,i,j])
            std_image[i,j] = np.std(data[:,i,j])
            pr_image[i,j] = np.percentile(data[:,i,j], [0, 25, 50, 75,100])
    return({'mean':mean_image, 'stdev':std_image, 'percentile_rank':pr_image})

def summarize_himawari8_by_grid(flist, batch_size=None):
    ''' Calculate grid-by-grid statistics of a list of Himawari-8 images. '''
    # Read in data by batch
    if batch_size is None:
        data = read_multiple_himawari8(flist)
        summary = summarize_images_by_grid(data)
    else:
        pass
    # 
    return(summary)


def statistics_by_image(datainfo):
    ''' Given the data information, derive the statistics by image. '''
    list_stats = []
    for i in range(datainfo.shape[0]):
        row = datainfo.iloc[i,:]
        tmp = read_himawari8_btp(row['xuri'])
        stats = summarize_single_image(tmp)
        stats['timestamp'] = row['timestamp']
        list_stats.append(stats)
    return(pd.DataFrame(list_stats))





#-----------------------------------------------------------------------
def main():
    # Configure Argument Parser
    parser = argparse.ArgumentParser(description='Retrieve DBZ data for further processing.')
    parser.add_argument('--datapath', '-i', help='the directory containing Himawari data in btp format.')
    parser.add_argument('--output', '-o', help='the prefix of output files.')
    parser.add_argument('--logfile', '-l', default=None, help='the log file.')
    args = parser.parse_args()
    # Set up logging
    if not args.logfile is None:
        logging.basicConfig(level=logging.DEBUG, filename=args.logfile, filemode='w')
    else:
        logging.basicConfig(level=logging.DEBUG)
    logging.debug(args)
    # Get data files
    logging.info('Scanning data files.')
    datainfo = list_himawari8_files(args.datapath)
    datainfo.to_csv(args.output+'.file_info.csv', index=False)
    # Derive per-image statistics
    logging.info('Deriving statistics per image.')
    stats_by_image = statistics_by_image(datainfo)
    stats_by_image.to_csv(args.output+'.stats_by_image.csv', index=False)
    # Derive per-grid statistics
    #stats_by_grid = statistics_by_grid(datainfo)
    # done
    return(0)
    
#==========
# Script
#==========
if __name__=="__main__":
    main()

