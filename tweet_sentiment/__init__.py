# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 22:55:39 2019

The contains machine learning algorithms for meme detection.  For our research, we define internet memes as photos with superimposed text that are designed to use humor in order to connect a message with a given culture or sub-culture.

@author: dmbes
"""
import random
import urllib.request
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py as h5py
import cv2
import pandas as pd
from shutil import copyfile
import platform



def get_links(files):
    '''
    Extract image (photo) links from list of files
    '''
    import json
    import io, gzip
    
    if not isinstance(files, list):
       files = [files]
    
    seen = get_done()
   
    links = []
    for file in files:
        with io.TextIOWrapper(gzip.open(file, 'r')) as infile:
            for line in infile:
                if line != '\n':
                    tweet = json.loads(line)
                    if tweet['id_str'] in seen:
                        continue
#                    if 'possibly_sensitive' in tweet.keys():
#                        if tweet['possibly_sensitive']:
#                            continue
                    if 'media' in tweet['entities'].keys():
                        count = 0
                        for pic in tweet['entities']['media']:
                            if pic['type'] == 'photo':
                                u = pic['media_url']
                                links.append((u,tweet['id_str'] + '_' + str(count)))
                                count += 1
    print('Number of links:',len(links))
    return(links)
#%%
def get_done():
    '''
    checks if photos already downloaded
    '''
    import glob
    seen = {}
    if os.path.exists('img'):
        files = glob.glob('img/*')
        status_ids = [extract_id(x) for x in files]
        for i in status_ids:
            if i not in seen:
                seen[i] = True
    return(seen)
    
#%%
def download_link(link, ID):
    '''
    Given a link and tweet ID pair, this function will download the image and
    save it with the file name being the tweet ID followed by a four digit random 
    number.  The random number is added in case a tweet has more than one image.
    '''
    u = link
    Type = u.split('.')
    Type = Type[-1]
    Name = 'img/'+ str(ID) + '.' + Type
    try:
        urllib.request.urlretrieve(u, Name)
    except:
        return(u)
    
    return(None)
#%%
def download_parallel(links, my_cores = None):
    '''
    This downloads the images in parallel using all available cores.

    Use this for Linux and Mac systems.
    '''
    import multiprocessing
    import socket
    socket.setdefaulttimeout(15)

    cores =multiprocessing.cpu_count()
    
    if my_cores is not None:
        cores = my_cores
    pool = multiprocessing.Pool(processes = cores)
    output = pool.starmap(download_link, links)
    pool.close()
    return(output)

#%%
def download_normal(links):
    '''
    This downloads images in a single thread.  Recommended for Windows.
    '''
    import progressbar
    output= []
    bar = progressbar.ProgressBar()
    for link in bar(links):
        output.append(download_link(link[0],link[1]))
    return(output)
#%%
def download_files(files, parallel = False):
    '''
    This function downloads all images from a list of tweet files.

    Use parallel = True for Linux/Mac
    '''

    import os
    if not os.path.exists('img'):
        os.makedirs('img')

    links = get_links(files)

    if parallel:
        output = download_parallel(links)
    else:
        output = download_normal(links)
        
    return(output)

#%%

def file_as_bytes(file):
    with file:
        return file.read()

#%%
def get_md5(file):
    '''
    This function returns the md5 hash of a file
    '''
    import hashlib
    h = hashlib.md5(file_as_bytes(open(file, 'rb'))).hexdigest()
    return(h)


#%%
def check_white(file):
    '''
    This function tries to determine if a file is a document image.  To do this
    is calculates the mean of the RGB values.  If the mean is more then 220 
    (meaning that the image is mostly white), then the function returns True.
    '''
    img = cv2.imread(file)
    pixels = np.mean(cv2.mean(img)[:3])
    if pixels > 220:
        return(True)
    else:
        return(False)
#%%
def load_image(img_path, show=False):
    '''
    This loads an image, resizes it to 200x200, and convert it to a tensor 
    for prediction
    '''
    img = image.load_img(img_path, target_size=(200, 200))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    return img_tensor
#%%

def extract_id(string):
    system = platform.system()
    if system == 'Windows':
        temp = string.split('\\')[1]
        temp = temp.split('_')[0]
    else:   
        temp = string.split('/')[1]
        temp = temp.split('_')[0]
    return(str(temp))
#%%


def classify_memes( threshold = 0.1):
    '''
    This is the main function which is used to predict whether or not an image
    is a meme.  This function will look in the 'img' directory for images to classify.
    
    If it flags a meme, it will move the file to a new directory called 'memes'
    
    This will return a pandas data frame containing the prediction along with 
    the file name, the tweet ID, and the file md5 hash.
    '''
    import progressbar
    import glob
    import pkg_resources
    import re

    if not os.path.exists('memes'):
        os.makedirs('memes')

    files = glob.glob('img/*')

    model_path = pkg_resources.resource_filename('memetics', 'data/memes_basic_white_light_model_20190124.h5')
    model = load_model(model_path)

    prob = []
    all_hash = []
    seen = {}

    bar = progressbar.ProgressBar()
    for file in bar(files):

        md5_hash = get_md5(file)
        all_hash.append(md5_hash)

        if md5_hash in seen:
            prob.append(seen[md5_hash])
            continue

        if check_white(file):
            prob.append(1.0)
            seen[md5_hash] = 1.0
            continue

        # load a single image
        try:
            new_image = load_image(file)
    
            # check prediction
            pred = float(model.predict(new_image))
    
            prob.append(pred)
            seen[md5_hash] = pred
    
            if pred < threshold:
                os.rename(file, re.sub('img','memes',file))
        except:
            prob.append(1)
            seen[md5_hash] = 1
            continue

    df = pd.DataFrame({'file' : files, 'probability' : prob, 'md5_hash' : all_hash})
    df['meme'] = False
    df.loc[df['probability'] < threshold,'meme'] = True
    
    ## Get df from keras prediction file
    df['status_id'] = df['file'].apply(extract_id)
    
    return(df)
