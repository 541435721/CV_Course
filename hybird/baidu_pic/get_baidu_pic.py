# -*- coding: utf-8 -*-
# @Author: bxsh
# @Email:  xbc0809@gmail.com
# @File:  get_baidu_pic.py
# @Date:  2017/11/15 16:09


import sys
import requests
import os
import urllib
import time
import numpy as np


def Download(url, filename):
    if (os.path.exists(sys.argv[2]) == False):
        os.mkdir(sys.argv[2])
    download_path = sys.argv[2]
    names = os.listdir(download_path)
    Filename = filename
    # while Filename in names:
    #     Filename = str(np.random.randint(0, 100)) + filename
    filepath = os.path.join(sys.argv[2], '%s' % Filename)
    urllib.request.urlretrieve(url, filepath)
    return


def Request(param):
    searchurl = 'http://image.baidu.com/search/avatarjson'
    response = requests.get(searchurl, params=param)
    json = response.json()['imgs']

    for i in range(0, len(json)):
        try:
            filename = os.path.split(json[i]['objURL'])[1]
            print('Downloading from %s' % json[i]['objURL'])
            # time.sleep(1)
            Download(json[i]['objURL'], filename)
        except Exception as e:
            pass

    return


def Search():
    params = {
        'tn': 'resultjsonavatarnew',
        'ie': 'utf-8',
        'cg': '',
        'itg': '',
        'z': '0',
        'fr': '',
        'width': '',
        'height': '',
        'lm': '-1',
        'ic': '0',
        's': '0',
        'word': sys.argv[1],
        'st': '-1',
        'gsm': '',
        'rn': '30'
    }

    if (len(sys.argv) == 4):
        pages = int(sys.argv[3])
    else:
        pages = 1

    for i in range(0, pages):
        params['pn'] = '%d' % i
        Request(params)
    return


def CheckArgs():
    if (len(sys.argv) < 3):
        print('Usage: python ImgSearch.py [Keyword] [DownloadDir] [Pages=1]')
        return False
    return True


if __name__ == '__main__':
    if (CheckArgs() == False):
        sys.exit(-1)
    Search()
    pass
