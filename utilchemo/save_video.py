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
import matplotlib.pyplot as plt

# Import col, dircol and phi into numpy arrays
# Find position i:wq
# Write col and vtk files

path_images = './images/'
if not os.path.exists(path_images):
  os.makedirs(path_images)

savevideo=True
savefigs=True

args = sys.argv[1:]
nstart = int(args[0])
nend = int(args[1])
nint = int(args[2])
spatial_res = int(args[3])
length_slice = int(args[4])

nfile = nend//nint
R = 9.

filelist_col, filelist_dircol, filelist_phi = [], [], []
array_image, array_line, array_circle = [], [], []

for i in range(nstart,nend+nint,nint):
  filelist_col.append('col-cds%08.0d.csv' % i)
  filelist_phi.append('phi-%08.0d.vtk' % i)
  filelist_dircol.append('dircol-cds%08.0d.csv' % i)

for t in range(nfile):
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
 
  
  df = pd.read_csv(filelist_col[t], index_col = 0) 
  sorted_df = df.sort_values('id') 

  normal = np.cross(m,n)

  xs = np.linspace(-length_slice/2, +length_slice/2, spatial_res)
  ys = np.linspace(-length_slice/2, +length_slice/2, spatial_res)
  zs = np.linspace(-length_slice/2, +length_slice/2, spatial_res)
  alphas = np.linspace(0, 2*np.pi, spatial_res)

  coords = (n[:, None, None] * xs[None, :, None] + m[:, None, None] * ys[None, None, :])
  #coords_line = (m[:, None, None] * xs[None, :, None])
  #coords_circle = (m[:, None, None] * R * alphas[None, :, None])

  coords += rc[:,None,None]
  #coords_line += rc[:,None,None]
  #coords_circle += rc[:,None,None]
  
  xbox = (np.floor(rc[0] + np.sqrt(2)*xs[0]).astype(int), np.ceil(rc[0] + np.sqrt(2)*xs[-1]).astype(int))
  ybox = (np.floor(rc[1] + np.sqrt(2)*ys[0]).astype(int), np.ceil(rc[1] + np.sqrt(2)*ys[-1]).astype(int))
  zbox = (np.floor(rc[2] + np.sqrt(2)*zs[0]).astype(int), np.ceil(rc[2] + np.sqrt(2)*zs[-1]).astype(int))

  interpolated_phi = ndi.map_coordinates(phi, coords, order = 3, mode = 'grid-wrap')
  #interpolated_phi_line = ndi.map_coordinates(phi, coords_line, order = 3, mode = 'grid-wrap')
  #interpolated_phi_circle = ndi.map_coordinates(phi, coords_circle, order = 3, mode = 'grid-wrap')
  
  array_image.append(interpolated_phi)
  #array_line.append(interpolated_phi_line)
  #array_circle.append(interpolated_phi_circle)


# Create and save video 
if savevideo:
  hmin = min(list(map(np.min, array_image)))
  hmax = max(list(map(np.max, array_image)))
  height, width = array_image[0].shape
  for t, image in enumerate(array_image):
    plt.imsave('./images/{}.png'.format(t), image)
  images = [img for img in os.listdir(path_images) if img.endswith(".png")]
  images.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))
  frame = cv2.imread(os.path.join(path_images, images[0]))
  height, width, layers = frame.shape
  video = cv2.VideoWriter("phi_vesicle.avi", cv2.VideoWriter_fourcc(*'XVID'), 20, (width,height))
  for image in images:
    video.write(cv2.imread(os.path.join(path_images, image)))
  #os.system("rm -r images")
  cv2.destroyAllWindows()
  video.release()
