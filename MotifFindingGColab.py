



####Mount Drive
#from google.colab import drive
#drive.mount('/content/drive')

######install biopython
#!pip install biopython


#########@copy by sobhan siamak (GoogleColab Version)

import numpy as np
import pandas as pd
import scipy as sc
import math
import re
from Bio import SeqIO
import time
from datetime import datetime
from sklearn.model_selection import train_test_split

start_time = datetime.now()
####### Reading BioData by biopython
biotemp = []
se = SeqIO.parse("/content/drive/My Drive/Homo sapiens gene promoter list.txt", "fasta")
for i in se:
    biotemp.append(i.seq)
# print(biotemp[0])
m, n = np.shape(biotemp)
a = np.random.choice(m, 1999, replace=False)
bio = []
biotest = []
biotest.append(biotemp[0])  ##### test for motif with real motif

for i in range(1999):
    if i < 1900:
        bio.append(biotemp[a[i]])
    else: biotest.append(biotemp[a[i]])
# print(bio[1999])
# bioarray = np.array(bio).tostring()
# split train and test data
# bio = list(bio)
train, test = train_test_split(bio, test_size=0.1, random_state=82)
# print(np.size(train))
#### training phase


iterate = 100
th = 0.001
k = 10  # length of motif
seq = 600    # length of sequence
basecount = 4    # number of Base
seqnum = 1900    # number of rows
seqnumt = 100
psudocount = 0.05  #### we consider psudo count same for all nucleotides


###### create initial matrix of p and z
z = np.zeros([seqnum,seq-k])
zt = np.zeros([seqnumt,seq-k])
p = np.zeros([basecount,k+1])
moindex = np.zeros([seqnum,1])
motif = np.zeros([seqnum,k])
mo = []
moindext = np.zeros([seqnumt,1])
motift = np.zeros([seqnumt,k])
mot = []
pnorm = np.zeros([iterate])
# p[:,0] = 0.25
for i in range(basecount):
    for j in range(k+1):
        p[i,j] = np.random.uniform()
p[:,0] = 0.25
for i in range(k+1):
    a = sum(p[:,i])
    p[:,i] /= a

# print(sum(p[:,0]))
# print(p)
# for i in range(iteration):

def map(sequence):
    sequences = []
    for i in range(seq):
     if sequence[i] == 'A':
         sequences.append(0)
     elif sequence[i] == 'C':
         sequences.append(1)
     elif sequence[i] == 'G':
         sequences.append(2)
     elif sequence[i] == 'T':
         sequences.append(3)
    return sequences  ##### sequences is the mapping given sequence to 0,1,2, or 3

def decode(sequence):
    sequences = []
    for i in range(k):
     if sequence[i] == 0:
         sequences.append('A')
     elif sequence[i] == 1:
         sequences.append('C')
     elif sequence[i] == 2:
         sequences.append('G')
     elif sequence[i] == 3:
         sequences.append('T')
    return sequences  ##### sequences is the decoding given  motif sequence to A,C,G, or T
######
######calculate p for k != 0
def calcp(bio, z, k, base):
    m, n = np.shape(bio)
    nck1 = 0
    j= 0
    seq = 600
    for i in range(m):
        bmap = map(bio[i])
        for j in range(0, seq-k, j+k-1):
            if bmap[j] == base:
                nck1 += z[i,j]
    return  nck1

####### Counting the number of nucleotides in all training dataset
nA = 0
nC = 0
nG = 0
nT = 0
for i in range(seqnum):
    basemap = map(bio[i])
    for j in range(seq):
        if basemap[j] == 0:
            nA += 1
        elif basemap[j] == 1:
            nC += 1
        elif basemap[j] == 2:
            nG += 1
        elif basemap[j] == 3:
            nT += 1
#######
for iteration in range(iterate):
    for i in range(seqnum):###### Expexted Phase means created and calculate Z matrix by P matrix
        sequence = map(bio[i])
        for j in range(seq-k):
              prob = 0  ##### for calculating elements of z
              if j == 0:
                  for l in range(k):
                    prob += math.log(p[sequence[j+l+1],l+1],2)
                  for after in range(j+k, seq):
                      prob += math.log(p[sequence[after],0],2)
              if j != 0:
                  for befor in range(j,-1,-1):
                      prob += math.log(p[sequence[befor],0],2)
                  for after in range(j+k, seq):
                      prob += math.log(p[sequence[after],0],2)
                  for l in range(k):
                    prob += math.log(p[sequence[j+l+1],l+1],2)
              z[i,j] = (1/np.abs(prob))
         ###### Normalization
        z[i,:] = (z[i,:]/np.sum(z[i,:]))
        maxind = np.argmax(z[i,:])
        moindex[i,0] = maxind  #### moindex is motif index
        # motif[i,:] = bio[i,maxind:maxind+k]
        motif[i,:] = sequence[maxind:maxind+k]
        mo.append(decode(motif[1,:]))

    # print(motif[1:4,:])
    # print(mo)

    ######### Counting Base in motifs
    nAmotif = 0
    nCmotif = 0
    nGmotif = 0
    nTmotif = 0
    m, n = motif.shape
    for i in range(m):
        for j in range(n):
            if motif[i,j] == 0:
                nAmotif += 1
            elif motif[i,j] == 1:
                nCmotif += 1
            elif motif[i,j] == 2:
                nGmotif += 1
            elif motif[i,j] == 3:
                nTmotif += 1

    ####### Maximization Phase means created and calculate P matrix by z matrix
    for i in range(basecount):
        for j in range(k+1):
            if j == 0:  #### from background formula
                if i == 0:
                    ####Base = A
                   nck = nA - nAmotif
                   nbk = (nA + nC + nG + nT) - (4* psudocount)
                   p[i, j] = (nck + psudocount)/(nbk)
                elif i == 1:
                    ####Base = C
                    nck = nC - nCmotif
                    nbk = (nA + nC + nG + nT) - (4 * psudocount)
                    p[i, j] = (nck + psudocount) / (nbk)

                elif i == 2:
                    ####Base = G
                    nck = nG - nGmotif
                    nbk = (nA + nC + nG + nT) - (4 * psudocount)
                    p[i, j] = (nck + psudocount) / (nbk)
                elif i == 3:
                    ####Base = T
                    nck = nT - nTmotif
                    nbk = (nA + nC + nG + nT) - (4 * psudocount)
                    p[i, j] = (nck + psudocount) / (nbk)
            elif j != 0: #### from motif formula
                if i == 0:
                    nck1 = calcp(bio, z, k, i)
                    nbk1 = (nA + nC + nG + nT) - (4 * psudocount)
                    p[i, j] = (nck1 + psudocount) / (nbk1)
                elif i == 1:
                    nck1 = calcp(bio, z, k, i)
                    nbk1 = (nA + nC + nG + nT) - (4 * psudocount)
                    p[i, j] = (nck1 + psudocount) / (nbk1)
                elif i == 2:
                    nck1 = calcp(bio, z, k, i)
                    nbk1 = (nA + nC + nG + nT) - (4 * psudocount)
                    p[i, j] = (nck1 + psudocount) / (nbk1)
                elif i == 3:
                    nck1 = calcp(bio, z, k, i)
                    nbk1 = (nA + nC + nG + nT) - (4 * psudocount)
                    p[i, j] = (nck1 + psudocount) / (nbk1)


    m, n = np.shape(p)######## normalization P matrix
    for j in range(n):
         s = np.sum(p[:,j])
         p[:,j] = p[:,j]/s
    ######### converge condition
    # print(p)
    pnorm[iteration] = np.linalg.norm(p)
    if iteration > 1:
        diff = np.abs(pnorm[iteration] - pnorm[iteration-1])
        if diff <= th:
          break  # break here from iteration


print("Final P matrix is :", p)
print("Final Z matrix is:", z)
######testing phase

for i in range(seqnumt):  ###### Expexted Phase means created and calculate Z matrix by P matrix
    sequence = map(biotest[i])
    for j in range(seq - k):
        prob = 0  ##### for calculating elements of zt
        if j == 0:
            for l in range(k):
                prob += math.log(p[sequence[j + l + 1], l + 1], 2)
            for after in range(j + k, seq):
                prob += math.log(p[sequence[after], 0], 2)
        if j != 0:
            for befor in range(j, -1, -1):
                prob += math.log(p[sequence[befor], 0], 2)
            for after in range(j + k, seq):
                prob += math.log(p[sequence[after], 0], 2)
            for l in range(k):
                prob += math.log(p[sequence[j + l + 1], l + 1], 2)
        zt[i, j] = (1 / np.abs(prob))
    ###### Normalization
    zt[i, :] = (zt[i, :] / np.sum(zt[i, :]))
    maxind = np.argmax(zt[i, :])
    moindext[i, 0] = maxind  #### moindex is motif index
    # motif[i,:] = bio[i,maxind:maxind+k]
    motift[i, :] = sequence[maxind:maxind + k]
    mot.append(decode(motift[1, :]))

print("Final zt is :", zt)
print("Motif index for one sample start from test data is :", moindext[0])
print("Final index of motifs in test data is:", motift)


end_time = datetime.now()
print('Time for running this Code is: {}'.format(end_time - start_time))
