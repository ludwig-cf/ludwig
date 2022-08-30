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

def normalize(x):
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

folderlist = []
for filename in args[5::]:
  folderlist.append(filename)

list_array_totphi, list_array_image, list_array_line, list_array_circle2, list_array_circle = [], [], [], [], []


for filename in folderlist:
  firsttime = True
  os.chdir(filename)

  filelist_col, filelist_dircol, filelist_phi, filelist_vel = [], [], [], []
  array_totphi, array_image, array_line, array_circle, array_circle2 = [], [], [], [], []

  for i in range(nstart,nend+nint,nint):
    filelist_col.append('col-cds%08.0d.csv' % i)
    filelist_phi.append('phi-%08.0d.vtk' % i)
    filelist_vel.append('vel-%08.0d.vtk' % i)
    filelist_dircol.append('dircol-cds%08.0d.csv' % i)
  
  for t in range(nfile):
    print("Working on "+filelist_dircol[t])
    m, n, rc = np.zeros(3), np.zeros(3), np.zeros(3)
    with open(filelist_dircol[t], 'r') as dircol:
      lines = dircol.readlines()
      rc[0], rc[1], rc[2] = float(lines[1].split(',')[0]), float(lines[1].split(',')[1]), float(lines[1].split(',')[2]) 
      m[0], m[1], m[2] = float(lines[1].split(',')[3]), float(lines[1].split(',')[4]), float(lines[1].split(',')[5]) 
      n[0], n[1], n[2] = float(lines[1].split(',')[6]), float(lines[1].split(',')[7]), float(lines[1].split(',')[8]) 

    pfile = open(filelist_phi[t], 'r')
    lines = pfile.readlines()
    headerlines = np.array(lines[0:10])
    datalines = np.array(lines[10:])
    pfile.close()

    if firsttime==True:
      for line in headerlines:
        stub = line.strip().split(" ")     
        if stub[0] == "DIMENSIONS":
          NX, NY, NZ = int(stub[1]), int(stub[2]), int(stub[3])
      firsttime=False 

    datalines = np.char.strip(datalines)
    phi = np.zeros((NX,NY,NZ), dtype = float)
    for il in range(NX):
      for jl in range(NY):
        for kl in range(NZ):
          index = il+jl*NX+kl*NX*NY
          stub = datalines[index]
          phi[il][jl][kl] = float(stub)
   

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
 
    totphi = np.sum(phi, axis = (0,1,2))
    df = pd.read_csv(filelist_col[t], index_col = 0) 
    sorted_df = df.sort_values('id') 
  
    normal = np.cross(m,n)
  
    xs = np.linspace(-length_slice/2, +length_slice/2, spatial_res)
    ys = np.linspace(-length_slice/2, +length_slice/2, spatial_res)
    zs = np.linspace(-length_slice/2, +length_slice/2, spatial_res)
    alphas = np.linspace(0, 2*np.pi, spatial_res)
  
    coords = (m[:, None, None] * xs[None, :, None] + n[:, None, None] * ys[None, None, :])
    coords_line = (m[:, None, None] * xs[None, :, None])
    coords_circle = (m[:, None, None] * R * np.cos(alphas)[None, :, None] +
			n[:, None, None] * R * np.sin(alphas)[None, :, None])
  
    coords_circle2 = (m[:, None, None] * R2 * np.cos(alphas)[None, :, None] +
			n[:, None, None] * R2 * np.sin(alphas)[None, :, None])

    coords += rc[:,None,None]
    coords_line += rc[:,None,None]
    coords_circle += rc[:,None,None]
    coords_circle2 += rc[:,None,None]
    
    xbox = (np.floor(rc[0] + np.sqrt(2)*xs[0]).astype(int), np.ceil(rc[0] + np.sqrt(2)*xs[-1]).astype(int))
    ybox = (np.floor(rc[1] + np.sqrt(2)*ys[0]).astype(int), np.ceil(rc[1] + np.sqrt(2)*ys[-1]).astype(int))
    zbox = (np.floor(rc[2] + np.sqrt(2)*zs[0]).astype(int), np.ceil(rc[2] + np.sqrt(2)*zs[-1]).astype(int))
  
    interpolated_phi = ndi.map_coordinates(phi, coords, order = 3, mode = 'grid-wrap')
    interpolated_phi_line = ndi.map_coordinates(phi, coords_line, order = 3, mode = 'grid-wrap')
    interpolated_phi_circle = ndi.map_coordinates(phi, coords_circle, order = 3, mode = 'grid-wrap')
    interpolated_phi_circle2 = ndi.map_coordinates(phi, coords_circle2, order = 3, mode = 'grid-wrap')
   
    interpolated_velx = ndi.map_coordinates(vel[:,:,:,0], coords, order = 3, mode = 'grid-wrap')
    interpolated_vely = ndi.map_coordinates(vel[:,:,:,1], coords, order = 3, mode = 'grid-wrap')
    interpolated_velz = ndi.map_coordinates(vel[:,:,:,2], coords, order = 3, mode = 'grid-wrap')
    
    interpolated_velx_line = ndi.map_coordinates(vel[:,:,:,0], coords_line, order = 3, mode = 'grid-wrap')
    interpolated_vely_line = ndi.map_coordinates(vel[:,:,:,1], coords_line, order = 3, mode = 'grid-wrap')
    interpolated_velz_line = ndi.map_coordinates(vel[:,:,:,2], coords_line, order = 3, mode = 'grid-wrap')

    interpolated_vel = np.stack((interpolated_velx, interpolated_vely, interpolated_velz), axis = -1)
    normal=np.cross(m,n)
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

    plt.quiver(u,v)
    plt.show()
    
    # Each array_ is for each simulation (array over time)
    array_totphi.append(totphi)
    array_image.append(interpolated_phi)
    array_line.append(interpolated_phi_line)
    array_circle.append(interpolated_phi_circle)
    array_circle2.append(interpolated_phi_circle2)
 
  # Each array is for each parent folder (array over folders) 
  list_array_totphi.append(array_totphi)
  list_array_image.append(array_image)
  list_array_line.append(array_line)
  list_array_circle.append(array_circle)
  list_array_circle2.append(array_circle2)

  os.chdir(BASEDIR) 

list_array_line = np.array(list_array_line)
list_array_circle = np.array(list_array_circle)
list_array_circle2 = np.array(list_array_circle2)

meanphis=np.mean(list_array_circle[:,:,spatial_res//4 : 3*spatial_res//4], axis = 2)
peakphis=list_array_circle[:,:,0]
deltaphis=peakphis-meanphis
gradphisss=deltaphis[:,-1]/12.

list_array_circle = np.array(list_array_circle)
list_array_circle2 = np.array(list_array_circle2)

hmin=np.min(list_array_circle)
hmax=np.max(list_array_circle)

hmin2=np.min(list_array_circle2)
hmax2=np.max(list_array_circle2)

labels=[0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
t=np.arange(nstart, nend+nint, nint)

if 0:
  plt.figure()
  for i in range(len(list_array_totphi)):
    plt.plot(list_array_totphi[i], label=labels[i])
  plt.xlabel("t")
  plt.ylabel("Total phi")
  plt.title("Total quantity of phi over time")
  plt.legend()
  plt.savefig("totalphiXt")

if 0:
  plt.figure()
  plt.plot(labels, gradphisss)
  plt.xlabel("phiprod")
  plt.ylabel("Grad Phi")
  plt.title("Gradphi steady state as a funcction of phi prod")
  plt.legend()
  plt.savefig("gradphiXphiprod")

if 0:
  y = np.loadtxt("normvssOverFolders", delimiter=",", unpack=True)
  plt.figure()
  plt.plot(gradphisss, y,'.')
  plt.xlabel("gradphis at steady-state")
  plt.ylabel("velocity at steady-state")
  plt.title("ss velocity as a function of gradphi")
  plt.legend()
  plt.savefig("vXgradphi")


if 0:
  y = np.loadtxt("normvOverTimeAndFolders", delimiter=",", unpack=True)
  plt.figure()
  for i, (yi, label) in enumerate(zip(y, labels)):
    plt.plot(t, yi, label=label, color=cm.gray(i/len(labels)))
  plt.xlabel("t")
  plt.ylabel("|v|")
  plt.title("norm of velocity over time")
  plt.legend()
  plt.savefig("normvXt")

if 0:
  y = np.loadtxt("normvssOverFolders", delimiter=",", unpack=True)
  plt.figure()
  plt.plot(labels, y, "*")
  plt.xlabel("PHIPROD")
  plt.ylabel("steady-state velocity")
  plt.title("steady-state velocity")
  plt.savefig("normvss")

if 0:
  y = np.loadtxt("alphaOverTimeAndFolders", delimiter=",", unpack=True)
  plt.figure()
  for i, (yi, label) in enumerate(zip(y, labels)):
    plt.plot(t, yi, label=label)
  plt.legend()
  plt.xlabel("t")
  plt.ylabel("alpha")
  plt.title("alpha over time")
  plt.savefig("alphaXt")

if 0:
  plt.figure()
  plt.gca(polar=True)
  plt.ylim(min(hmin,hmin2), max(hmax2, hmax))
  for i in range(len(list_array_circle)):
    plt.plot(alphas, list_array_circle[i, -1,:], label=labels[i], color=cm.Reds(i/len(labels)))
    plt.plot(alphas, list_array_circle2[i, -1,:], label=labels[i], color=cm.Blues(i/len(labels)))
  plt.title("Angular distribution of phi in steady state R = 14 (Red) and R=16 (Blue)")
  plt.legend()
  plt.savefig("phiOverCircle")

if 0:
  plt.figure()
  plt.gca(polar=True)
  plt.ylim(hmin2, hmax2)
  for i in range(len(list_array_circle)):
    plt.plot(alphas, list_array_circle2[i, -1,:], label=labels[i], color=cm.Reds(i/len(labels)))
  plt.title("Angular distribution of phi in steady state R = 16")
  plt.legend()
  plt.savefig("phiOverCircle2")
