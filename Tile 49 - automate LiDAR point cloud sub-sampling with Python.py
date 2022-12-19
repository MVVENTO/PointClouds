#!/usr/bin/env python
# coding: utf-8

# # How to automate LiDAR point cloud sub-sampling with Python

# # Step 1 : Setting up the environment 

# # Step 2: Load and prepare the data

# In[1]:


pip install laspy


# In[2]:


#libraries used
import numpy as np
import laspy as lp
import open3d as o3d


# In[3]:


#create paths and load data


#Load the file
dataname="Tile (49).las"
point_cloud = lp.read(dataname)


# In[4]:


# examine the avaliable features for the lidar file we have reead
list(point_cloud.point_format.dimension_names)


# In[5]:


set(list(point_cloud.classification))


# In[6]:


#Creating, filtering, and writing Point cloud data : Version 1
#store coordinates in "points", and colors in "colors" variable
points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
colors = np.vstack((point_cloud.red, point_cloud.green, point_cloud.blue)).transpose()


# In[7]:


#Creating, filtering, and writing Point cloud data : Version 2
# To create 3D point cloud data, we can stack together with the X, Y, Z dimensions using numpy

point_data = np.stack([point_cloud.X,point_cloud.Y,point_cloud.Z], axis = 0).transpose((1,0))


# In[8]:


# 3D Point Cloud Visualization 
# Laspy has no visualizationm methods so that we wil use the open3d library, we first create
# the open34D geometries and pass the point data we have created. Finally we use the 
# open3D visualization to draw geometries. 

geom = o3d.geometry.PointCloud()
geom.points = o3d.utility.Vector3dVector(point_data)
o3d.visualization.draw_geometries([geom])


# # Step 3: Choose a sub-sampling strategy

# In[9]:


# 1 Point Cloud Decimation


# In[10]:


#The decimation strategy, by setting a decimation factor
factor=160
decimated_points = points[::factor]
decimated_colors = colors[::factor]
len(decimated_points)


# In[11]:


# 2 Point Cloud voxel grid


# In[12]:


# Initialize the number of voxels to create to fill the space including every point
voxel_size=7
nb_vox=np.ceil((np.max(points, axis=0) - np.min(points, axis=0))/voxel_size)
#nb_vox.astype(int) #this gives you the number of voxels per axis


# In[13]:


# Compute the non empty voxels and keep a trace of indexes that we can relate to points in order to store points later on.
# Also Sum and count the points in each voxel.
non_empty_voxel_keys, inverse, nb_pts_per_voxel= np.unique(((points - np.min(points, axis=0)) // voxel_size).astype(int), axis=0, return_inverse=True, return_counts=True)
idx_pts_vox_sorted=np.argsort(inverse)
#len(non_empty_voxel_keys) # if you need to display how many no-empty voxels you have


# In[14]:


#Here, we loop over non_empty_voxel_keys numpy array to
#       > Store voxel indices as keys in a dictionnary
#       > Store the related points as the value of each key
#       > Compute each voxel barycenter and add it to a list
#       > Compute each voxel closest point to the barycenter and add it to a list

voxel_grid={}
grid_barycenter,grid_candidate_center=[],[]
last_seen=0

for idx,vox in enumerate(non_empty_voxel_keys):
  voxel_grid[tuple(vox)]=points[idx_pts_vox_sorted[last_seen:last_seen+nb_pts_per_voxel[idx]]]
  grid_barycenter.append(np.mean(voxel_grid[tuple(vox)],axis=0))
  grid_candidate_center.append(voxel_grid[tuple(vox)][np.linalg.norm(voxel_grid[tuple(vox)]-np.mean(voxel_grid[tuple(vox)],axis=0),axis=1).argmin()])
  last_seen+=nb_pts_per_voxel[idx]


# # Step 4: Vizualise and export the results

# In[15]:


import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.pyplot import figure

figure(figsize=(12, 10), dpi=80)

ax = plt.axes(projection='3d')
ax.scatter(decimated_points[:,0], decimated_points[:,1], decimated_points[:,2], c = decimated_colors/65535, s=0.01)
plt.show()


# In[16]:


#output_path = 
# %timeit np.savetxt(output_path+dataname+"_voxel-best_point_%s.xyz" % (voxel_size), grid_candidate_center, delimiter=";", fmt="%s")


# In[17]:


# Load data
import open3d as o3d
import laspy as lp

dataname="Tile (49).las"
point_cloud =  lp.read(dataname)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors/65535)
# pcd.normals = o3d.utility.Vector3dVector(normals)
o3d.visualization.draw_geometries([pcd])


# # Step 5 : Automate with functions

# In[18]:


#Define a function that takes as input an array of points, and a voxel size expressed in meters. It returns the sampled point cloud
def grid_subsampling(points, voxel_size):

  nb_vox=np.ceil((np.max(points, axis=0) - np.min(points, axis=0))/voxel_size)
  non_empty_voxel_keys, inverse, nb_pts_per_voxel= np.unique(((points - np.min(points, axis=0)) // voxel_size).astype(int), axis=0, return_inverse=True, return_counts=True)
  idx_pts_vox_sorted=np.argsort(inverse)
  voxel_grid={}
  grid_barycenter,grid_candidate_center=[],[]
  last_seen=0

  for idx,vox in enumerate(non_empty_voxel_keys):
    voxel_grid[tuple(vox)]=points[idx_pts_vox_sorted[last_seen:last_seen+nb_pts_per_voxel[idx]]]
    grid_barycenter.append(np.mean(voxel_grid[tuple(vox)],axis=0))
    grid_candidate_center.append(voxel_grid[tuple(vox)][np.linalg.norm(voxel_grid[tuple(vox)]-np.mean(voxel_grid[tuple(vox)],axis=0),axis=1).argmin()])
    last_seen+=nb_pts_per_voxel[idx]

  return grid_candidate_center


# In[ ]:


#Execute the function, and store the results in the grid_sampled_point_cloud variable
grid_sampled_point_cloud = grid_subsampling(point_cloud, 6)

#Save the variable to an ASCII file to open in a 3D Software
get_ipython().run_line_magic('timeit', 'np.savetxt(output_path+dataname+"_sampled.xyz", grid_sampled_point_cloud variable, delimiter=";", fmt="%s")')


# In[ ]:


# Load data
import open3d as o3d

dataname="Tile (49).las"
point_cloud= las = lp.read(dataname)

demo_crop_data = las = lp.read(dataname)
pcd = o3d.io.read_point_cloud(demo_crop_data)
vol = o3d.visualization.read_selection_polygon_volume(demo_crop_data.cropped_json_path)
Tile = vol.crop_point_cloud(pcd)

dists = pcd.compute_point_cloud_distance(Tile)
dists = np.asarray(dists)
ind = np.where(dists > 0.01)[0]
pcd_without_Tile = pcd.select_by_index(ind)
o3d.visualization.draw_geometries([pcd_without_Tile],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])


# In[ ]:


aabb = Tile.get_axis_aligned_bounding_box()
aabb.color = (0.5,0,0.6)
obb = Tile.get_oriented_bounding_box()
obb.color = (0.7,1,0.1)
o3d.visualization.draw_geometries([Tile, aabb, obb],
                                  zoom=0.7,
                                  front=[0.5439, -0.2333, -0.8060],
                                  lookat=[2.4615, 2.1331, 1.338],
                                  up=[-0.1781, -0.9708, 0.1608])


# In[ ]:


print("Convert mesh to a point cloud and estimate dimensions")
armadillo = o3d.data.ArmadilloMesh()
mesh = o3d.io.read_triangle_mesh(armadillo.path)
mesh.compute_vertex_normals()

pcd = mesh.sample_points_poisson_disk(5000)
diameter = np.linalg.norm(
    np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
o3d.visualization.draw_geometries([pcd])


# In[ ]:


print("Convert mesh to a point cloud and estimate dimensions")
armadillo = "Tile_1.las"
point_cloud= las = lp.read(dataname)
mesh = o3d.io.read_triangle_mesh(armadillo.path)
mesh.compute_vertex_normals()

pcd = mesh.sample_points_poisson_disk(5000)
diameter = np.linalg.norm(
    np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
o3d.visualization.draw_geometries([pcd])


# In[ ]:


# Point Cloud Data Segmentation C++
# https://www.youtube.com/watch?v=17zf9e5ZK3I

