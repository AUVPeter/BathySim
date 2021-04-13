import numpy as np
import random
import noise
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import utm

class BathySim:
  '''Bathymetric simulator

      Encapsulates a virtual 3D terrain and allows
      access to 2d views and sensor readings

  '''
  def __init__(self,feature_scale=300,depth_mid=100.,depth_range=20.,seed=None,range_accuracy=0.001,outlier_freq=0.):
    ''' Initialize grid and sensors

      Parameters
      ---------- 

      feature_scale : int
        the general 'size' of large scale features
    
      avg_depth : float
        mean depth of the generated bathymetry

      depth_range : float
        relative +/- range of the returned bathy

      seed : float, default None
        if not None, used for repeatability between instances
        
      range_accuracy : float
          additive noise component of sample, stdev of noise is: range_accuracy x measured_range
          
      outlier_frequency : float, [0,1]
          frequency of 'bad' beams, or outliers. 0 is never, 1 is always
    '''
    # grid params
    self._feature_scale = feature_scale
    self._depth_mid = depth_mid
    self._depth_range = depth_range
    if seed: random.seed(seed)
    self._xr = random.randrange(1e4)
    self._yr = random.randrange(1e4)

    # MB sensor params
    self._mb_num_beams = 120
    self._mb_swath_angle = 45
    self._mb_swath_factor = np.tan(np.radians(self._mb_swath_angle)/2.0)
    self._mb_range_accuracy = range_accuracy
    self._mb_outlier_frequency = outlier_freq
    self._mb_min_range = 0.0
    self._mb_max_range = 150.0


  def _sample_grid(self, x,y):
    ''' Utility method to return a single z from an x,y location
    '''
    depth = noise.pnoise2((x+self._xr)/self._feature_scale,(y+self._yr)/self._feature_scale,
                octaves=10, persistence=0.3,lacunarity=2.0)

    # shift depths to depth_mid +/- depth_range 
    z = (depth * self._depth_range/.4) + self._depth_mid
    return min(max(z,self._mb_min_range),self._mb_max_range)


  def generate_grid_view(self, origin=(0,0), shape=(500,500)):
    ''' Returns a 2D grid view of a selected region of the bathymetery

    Parameters
    ----------
    origin : tuple
      x,y position of grid origin
    width : int
      width in x-direction
    height : int
      height in y-direction

    Returns
    -------
    grid : 2d np array
    
    '''
    grid = np.zeros(shape)
    for x,y in np.ndindex(shape):
      grid[x][y] = self._sample_grid(x+origin[0],y+origin[1])
    print(origin,grid.mean(),grid.std())
    return grid


  def generate_ascii_grid(self, root_name, origin=(0,0), shape=(500,500)):
    ''' Generates an xyz asci file of selected region of the bathymetery

    Parameters
    ----------
    root_name : string
      generated file will be root_name.xyz
    origin : tuple, (float,float)
      x,y position of image bottom left corner in relation to bathymetry origin
    size : tuple, (int,int)
      width,height in meters
    '''
    try:
      out_file = open(root_name + '.xyz', 'w')
    except:
      print("Could not open " + root_name)
      return

    for x,y in np.ndindex(shape):
      z = self._sample_grid(x+origin[0],y+origin[1])
      out_file.write(f"{x},{y},{z:.3f}\n")
    out_file.close()


  def generate_moos_image(self, root_name, origin=(0,0), shape=(500,500), geo_origin=(42.,170.)):
    ''' Generates an image of a selected region of the bathymetery

    Parameters
    ----------
    root_name : string
      generated files will be root_name.tif, root_name.info
    origin : tuple, (float,float)
      x,y position of image bottom left corner in relation to bathymetry origin
    size : tuple, (int,int)
      width,height in meters
    geo_origin : tuple, (float,float)
      image center in geographic coordinates (lat, lon)
    
    '''
    grid = self.generate_grid_view(origin, shape)
    img_w = shape[0]/ 100
    img_h = shape[1] / 100
    fig= plt.figure(frameon=False)
    fig.set_size_inches(img_w, img_h)
    ax = plt.Axes(fig, [0.,0.,1.,1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(grid,aspect='auto')
    CS = ax.contour(grid,colors='w')
    ax.clabel(CS, CS.levels, inline=True,fmt='%.0f', fontsize=10)    

    fig.savefig(root_name + '.tif')

    easting,northing,zone_num,zone_let = utm.from_latlon(*geo_origin)
    lat_south, lon_west = utm.to_latlon(easting + origin[0],
                               northing + origin[1],
                               zone_num, zone_let)
    lat_north, lon_east = utm.to_latlon(easting + origin[0] + shape[0],
                               northing + origin[1] + shape[1],
                               zone_num, zone_let)

    print(lat_north, lat_south, lon_east, lon_west)
    with open(root_name + '.info','w') as info_file:
      info_file.write(f"lat_north={lat_north}\nlat_south={lat_south}\n")
      info_file.write(f"lon_east={lon_east}\nlon_west={lon_west}\n")


  def sb_sample(self,x,y,depth=0.0, sensor=True):
    '''Return single beam sample at the position
    
    Parameters
    ----------
    x : float
      x postion of sample
    y : float
      y position of sample
    depth : float, default 0.0
      depth of sensor
    sensor : bool, default True
      if True, data is sensor relative
      if False, data is global relative

    Returns
    -------
    d : float 
      depth value at location 
    '''
    
    d = self._sample_grid(x,y) * ( 1 + self._mb_range_accuracy * np.random.randn())
    if not sensor:
      d -= depth
    return d
  
  
  def mb_sample(self,x,y,hdg,depth=0.0,sensor=True):
    '''Return a multibeam sample array at the position/heading

    Parameters
    ----------
    x : float
      x postion of sample
    y : float
      y position of sample
    hdg : float
      North-up heading of sensor
    depth : float, default 0.0
      depth of sensor
    sensor : bool, default True
      if True, returns (array of relative depths, across track spacing)
      if False, returns array of x,y,z values for each beam

    Returns
    -------
    line : (num_beams,3) numpy array
      columns are x,y,z values
    '''

    rot = R.from_euler('z',[-hdg],degrees=True)

    # determine swath width at nadir and port/stbd extents
    nadir_alt = self._sample_grid(x,y) - depth
    swath_width = self._mb_swath_factor * nadir_alt

    pre_sample = np.zeros((2,3))
    pre_sample[:,0] =[-swath_width, swath_width]
    pre_sample = rot.apply(pre_sample)
    pre_sample[:,0] += x
    pre_sample[:,1] += y

    port_alt = self._sample_grid(pre_sample[0,0],pre_sample[0,1]) - depth
    stbd_alt = self._sample_grid(pre_sample[1,0],pre_sample[1,1]) - depth
    sw_p = self._mb_swath_factor * port_alt
    sw_s = self._mb_swath_factor * stbd_alt
    beams = np.linspace(-sw_p, sw_s, self._mb_num_beams)
    #unit sample vector
    line = np.zeros((len(beams),3))
    line[:,0] = beams

    #apply pose
    line = rot.apply(line)
    line[:,0] += x
    line[:,1] += y

    #collect samples
    #error = self._mb_range_accuracy * nadir_alt
    for i,(x,y) in enumerate(zip(line[:,0],line[:,1])):
      if np.random.rand() > self._mb_outlier_frequency:
        z = self._sample_grid(x,y) + (np.random.randn()*self._mb_range_accuracy*((x**2 + nadir_alt**2)**.5)) - depth
        if (z**2. + beams[i]**2.)**.5 > self._mb_max_range:
          z = 0.0
        line[i,2] = z
      elif np.random.rand() > 0.5:
        line[i,2] = self._mb_min_range
      else:
        line[i,2] = self._mb_max_range

    if not sensor:
      line[:,2] += depth
    else:
      line[:,0] = beams
      line[:,1] = [0] * self._mb_num_beams
    
    return line


  def mb_nav_sample(self, x,y,hdg, depth, x_err = 0.0, y_err = 0.0, hdg_err = 0.0, depth_err = 0.0): 
    line_true = self.mb_sample(x,y,hdg,depth,sensor=False)
    rot = R.from_euler('z',[-hdg_err],degrees=True)
    line_rot = rot.apply(line_true)
    line_err = line_rot + [x_err, y_err, depth_err]
    return line_err
    

if __name__ == '__main__':
  import sys

  if len(sys.argv) != 4:
    print("Usage" + sys.argv[0] + " seed width height")
    quit()

  s = float(sys.argv[1])
  w = int(sys.argv[2])
  h = int(sys.argv[3])

  bty = BathySim(seed=s)
  #print(bty.mb_sample(x=0,y=0,hdg=0, sensor=False))
  #print(bty.mb_sample(x=0,y=0,hdg=0, sensor=True))

  bty.generate_moos_image('test',shape=(w,h))
  bty.generate_ascii_grid('test',shape=(w,h))
