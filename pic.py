'''
Created on 2013-4-24

@author: golden_zhang
'''
import Image
import random

def convertImg(im,threshold):
    out = im.convert("L")
    table = []
    for i in range(256):
        if i>threshold:
            table.append(255)
        else:
            table.append(0)
    bim = out.point(table,"F")
    #im.convert("1")
    return bim

def regulateImg(im):
    im.resize((32,32),Image.ANTIALIAS)
    return im

def readImg(im,num):
    data = list(im.getdata())
    sourcedata=[]
    cout = 0;
    for i in range(0,num,2):
        sum = 0
        for j in range(0,num,2):
            if data[i*num+j]==0:
                sum=sum+1
            if data[i*num+j+1]==0:
                sum = sum+1
            if data[(i+1)*num+1]==0:
                sum = sum+1
            if data[(i+1)*num+1]==0:
                sum = sum+1
            sourcedata.append(sum)
    return sourcedata
    