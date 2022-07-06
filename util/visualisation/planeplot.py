import numpy as np
import getopt
import cv2
import re
import scipy.linalg
import scipy.ndimage as ndi
import os 
import csv
import sys
import pandas as pd
#import utils
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse

from math import sqrt

def dot_product(x, y):
    return sum([x[i] * y[i] for i in range(len(x))])

def norm(x):
    return sqrt(dot_product(x, x))
    return [x[i] / norm(x) for i in range(len(x))]

def project_onto_plane(x, n):
    d = dot_product(x, n) / norm(n)
    p = [d * normalize(n)[i] for i in range(len(n))]
    return [x[i] - p[i] for i in range(len(x))]

BASEDIR=os.getcwd()

args = sys.argv[1:]
nstart = int(args[0])
nend = int(args[1])
nint = int(args[2])
spatial_res = int(args[3])
length_slice = int(args[4])
nfile = nend//nint
R = 11.
R2 = 13.
n = np.array([0.0, 1.0, 0.0])

folderlist = []
for filename in args[5::]:
  folderlist.append(filename)

list_array_phi, list_array_temp, list_array_mapvel, list_array_vel = [], [], [], []

for filename in folderlist:
  os.chdir(filename)

  filelist_col, filelist_mapvel, filelist_phi, filelist_vel, filelist_temp = [], [], [], [], []

  array_phi, array_temp, array_mapvel, array_vel = [], [], [], []

  for i in range(nstart,nend+nint,nint):
    filelist_temp.append('temperature-%08.0d.vtk' % i)
    filelist_phi.append('phi-%08.0d.vtk' % i)
    filelist_vel.append('vel-%08.0d.vtk' % i)
    filelist_mapvel.append('map_vel-%08.0d.vtk' % i)
    filelist_col.append('col-cds%08.0d.csv' % i)
  
  for t in range(nfile):
    m, rc = np.zeros(3), np.zeros(3)
    with open(filelist_col[t], 'r') as col:
      lines = col.readlines()
      rc[0], rc[1], rc[2] = float(lines[1].split(',')[1]), float(lines[1].split(',')[2]), float(lines[1].split(',')[3]) 
      m[0], m[1], m[2] = float(lines[1].split(',')[4]), float(lines[1].split(',')[5]), float(lines[1].split(',')[6]) 


    pfile = open(filelist_phi[t], 'r')
    lines = pfile.readlines()
    headerlines = np.array(lines[0:10])
    datalines = np.array(lines[10:])
    pfile.close()

    for line in headerlines:
      stub = line.strip().split(" ")     
      if stub[0] == "DIMENSIONS":
        NX, NY, NZ = int(stub[1]), int(stub[2]), int(stub[3])

    datalines = np.char.strip(datalines)
    phi = np.zeros((NX,NY,NZ), dtype = float)
    for il in range(NX):
      for jl in range(NY):
        for kl in range(NZ):
          index = il+jl*NX+kl*NX*NY
          stub = datalines[index]
          phi[il][jl][kl] = float(stub)

    tfile = open(filelist_temp[t], 'r')
    lines = tfile.readlines()
    headerlines = np.array(lines[0:10])
    datalines = np.array(lines[10:])
    tfile.close()

    datalines = np.char.strip(datalines)
    temp = np.zeros((NX,NY,NZ), dtype = float)
    for il in range(NX):
      for jl in range(NY):
        for kl in range(NZ):
          index = il+jl*NX+kl*NX*NY
          stub = datalines[index]
          temp[il][jl][kl] = float(stub)
   
    vfile = open(filelist_vel[t], 'r')
    lines = vfile.readlines()
    datalines = np.array(lines[9:])
    vfile.close()

    vel = np.zeros((NX,NY,NZ, 3), dtype = float)
    for il in range(NX):
      for jl in range(NY):
        for kl in range(NZ):
          index = il+jl*NX+kl*NX*NY
          vel[il][jl][kl][0] = float(datalines[index].split()[0])
          vel[il][jl][kl][1] = float(datalines[index].split()[1])
          vel[il][jl][kl][2] = float(datalines[index].split()[2])
 
    normal = np.cross(m,n)
  
    xs = np.linspace(-length_slice/2, +length_slice/2, spatial_res)
    ys = np.linspace(-length_slice/2, +length_slice/2, spatial_res)
    zs = np.linspace(-length_slice/2, +length_slice/2, spatial_res)
  
    coords = (m[:, None, None] * xs[None, :, None] + n[:, None, None] * ys[None, None, :])
    coords += rc[:,None,None]
    
    xbox = (np.floor(rc[0] + np.sqrt(2)*xs[0]).astype(int), np.ceil(rc[0] + np.sqrt(2)*xs[-1]).astype(int))
    ybox = (np.floor(rc[1] + np.sqrt(2)*ys[0]).astype(int), np.ceil(rc[1] + np.sqrt(2)*ys[-1]).astype(int))
    zbox = (np.floor(rc[2] + np.sqrt(2)*zs[0]).astype(int), np.ceil(rc[2] + np.sqrt(2)*zs[-1]).astype(int))
  
    interpolated_phi = ndi.map_coordinates(phi, coords, order = 3, mode = 'grid-wrap')
    interpolated_temp = ndi.map_coordinates(temp, coords, order = 3, mode = 'grid-wrap')
    interpolated_velx = ndi.map_coordinates(vel[:,:,:,0], coords, order = 3, mode = 'grid-wrap')
    interpolated_vely = ndi.map_coordinates(vel[:,:,:,1], coords, order = 3, mode = 'grid-wrap')
    interpolated_velz = ndi.map_coordinates(vel[:,:,:,2], coords, order = 3, mode = 'grid-wrap')
    
    interpolated_vel = np.stack((interpolated_velx, interpolated_vely, interpolated_velz), axis = -1)

    planevel = np.zeros( (len(xs), len(ys), 3) )
    normal_norm = np.sqrt(sum(normal**2))
    u = np.zeros((len(xs), len(ys)))
    v = np.zeros((len(xs), len(ys)))

    for i in range(len(xs)):
      for j in range(len(ys)):
          #Project interpolated_vel[] onto plane normal to "normal"
          vec = interpolated_vel[i, j]
          proj = (np.dot(vec, normal)/normal_norm**2)*normal

          planevel[i,j, 0] = vec[0] - proj[0]
          planevel[i,j, 1] = vec[1] - proj[1]
          planevel[i,j, 2] = vec[2] - proj[2]

          u[i,j] = np.dot(planevel[i,j,:], m)
          v[i,j] = np.dot(planevel[i,j,:], n)

    #plt.figure()
    #plt.quiver(u,v)
    #plt.savefig('vel{}.png'.format(t))
    
    # Each array_ is for each simulation (array over time)
    array_phi.append(interpolated_phi)
    array_temp.append(interpolated_temp)
    #array_mapvel.append(interpolated_mapvel)
    array_vel.append(interpolated_vel)
 
  # Each array is for each parent folder (array over folders) 
  list_array_phi.append(array_phi)
  list_array_temp.append(array_temp)
  #list_array_mapvel.append(array_mapvel)
  list_array_vel.append(array_vel)
  os.chdir(BASEDIR) 

list_array_phi = np.squeeze(np.array(list_array_phi), axis = 0)
list_array_temp = np.squeeze(np.array(list_array_temp), axis = 0)
list_array_vel = np.squeeze(np.array(list_array_vel), axis = 0)

t=np.arange(nstart, nend+nint, nint)
if 1:
  plt.figure()
  for i in range(len(list_array_temp)):
    plt.imshow(list_array_temp[i])
    plt.savefig("temp{}.png".format(i))

