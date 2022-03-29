import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def remove_dups(points_spatial):
    '''
    Remove duplicates. (Sometimes it's convenient to generate a scenario in a way that creates an antenna multiple times. We can't have antennas right on top of each other!)
    '''
    return np.array(list(set(tuple(t) for t in points_spatial)))

def read_ant_positions(fname):
	'''
	Given a filename (including relative path to configs), read the antenna positions (in wavelengths) into an array (size number_of_ants x 2) called points_spatial.
	'''
	with open(fname, 'r') as f:
		x = pd.read_table(f, sep=',').to_numpy()
		points_spatial = remove_dups(x)
		return points_spatial

def get_baselines(points_spatial, redundant_baselines=True, sort=False):
	'''
	Compute the baselines (in wavelengths) from the antenna positions (in wavelengths). If redundant_baselines is False, keep only one observation of that baseline. If sort is True, sort the baselines (helpful for debugging perhaps).
	'''
	number_antennas = points_spatial.shape[0]
	number_visibilities = number_antennas**2

	first_antennas = np.repeat(points_spatial, number_antennas, axis=0)
	second_antennas = np.tile(points_spatial, (number_antennas,1))
	baselines = second_antennas - first_antennas

	if sort:
		sort_idx = 0 # sort by u coordinate (to sort by v coordinate, change sort_idx to 1)
		baselines = baselines[baselines[:,sort_idx].argsort()]

	if redundant_baselines:
		return baselines
	else:
		dc_baseline_idxs = np.where(np.sum(baselines == 0.0, 1) == 2)[0] # indices of "self" visibilities---an antenna to itself
		n_unique_baselines = len(set(tuple(b) for b in baselines)) # number of non-repeated baselines
		idxs = np.zeros((n_unique_baselines,), dtype=np.dtype(int))
		ss = set()
		cnt = 0
		for i,b in enumerate(baselines):
			if tuple(b) not in ss:
				idxs[cnt] = i
				ss.add(tuple(b))
				cnt += 1

		return baselines[idxs,:]

if __name__=='__main__':
	# Read all configurations in the directory /configs relative to the current working directory
	cwd = os.getcwd()
	configdir = cwd + "/configs"
	
	for filename in os.listdir(configdir):
		if filename.startswith("."): # ignore hidden files
			continue
		pathname = os.path.join(configdir, filename)
		points_spatial = read_ant_positions(pathname)
		
		scenario = filename[:-4] # remove ".txt"
		plt.figure()
		plt.scatter(points_spatial[:,1], points_spatial[:,0], c='g', marker='.', s=14, linewidth=0.1)
		plt.xlabel(r"$v$")
		plt.ylabel(r"$u$")
		plt.title(f"antenna positions, scenario {scenario}")
		plt.show()

		baselines = get_baselines(points_spatial)

		plt.figure()
		plt.scatter(baselines[:,1], baselines[:,0], c='g', marker='.', s=5, alpha=0.5, linewidth=0.1)
		plt.xlabel(r"$v$")
		plt.ylabel(r"$u$")
		plt.title(f"baselines, scenario {scenario}")
		plt.show()

