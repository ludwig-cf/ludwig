import csv
import numpy as np
import matplotlib.pyplot as plt
import utils

def file_to_array(filename):
  x, y, z = [], [], []

  f = open(filename, "r")
  for line in f.readlines()[:]:
    x.append(line.split()[0])
    y.append(line.split()[1])
    z.append(line.split()[2])

    x = list(map(float, x))
    y = list(map(float, y))
    z = list(map(float, z))
  coords = np.concatenate(([x],[y],[z]))
  return(np.array(coords))

# explicit function to normalize array
def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)   
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr

def rescale(arr, factor):
  arr = np.array(arr) 
  shape = np.shape(arr)
  crepe = np.array([factor * arr.flatten()[i] for i in range(len(arr.flatten())) ] )
  return crepe.reshape(shape)
  
def dist(x,y):
  dr = np.sqrt(np.sum((x-y)**2))
  return dr

def rotate(M_before, M_after):

  v = np.cross(M_after, M_before)
  s = np.sqrt(np.sum(v**2))
  c = np.dot(M_before, M_after)

  matvx = np.zeros((3,3))
  matvx[0][1] = -v[2]
  matvx[0][2] = v[1]
  matvx[1][0] = v[2]
  matvx[1][2] = -v[0]
  matvx[2][0] = -v[1]
  matvx[2][1] = v[0]
  matvx2 = np.matmul(matvx, matvx)
  R = np.eye(3) + matvx + matvx2 * (1/(1+c)) 

  return R


