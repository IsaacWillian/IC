import tifffile
import numpy as np
import scipy.ndimage as ndi
import scipy
import scipy.ndimage.filters as fil
import matplotlib.pyplot as plt
import skimage.color as skicolor

def hessian_eigenvalues(img, sigma):
	'''Find the eigenvalues of the Hessian matrix of each pixel in the 3D image
	represented by img. The scale is set by parameter sigma.'''

	size_y,size_x = img.shape

	# Calculate image derivatives
	Ixx = (sigma**2.)*fil.gaussian_filter(img, sigma=sigma, order=[0,2])
	Iyy = (sigma**2.)*fil.gaussian_filter(img, sigma=sigma, order=[2,0])
	Ixy = (sigma**2.)*fil.gaussian_filter(img, sigma=sigma, order=[1,1])

    # Find the roots of the second degree characteristic polynomial of the Hessian matrix

	delta = (Ixx+Iyy)**2 - 4*(Ixx*Iyy-Ixy**2)
	delta_sqrt = scipy.sqrt(delta)

	Rplus = (Ixx+Iyy+delta_sqrt)/2
	Rminus = (Ixx+Iyy-delta_sqrt)/2

	eigvals = np.zeros([size_y,size_x,2])
	eigvals[:,:,0] = Rplus
	eigvals[:,:,1] = Rminus
			
	return eigvals
	
def sort_eigval(eig):
	'''Order the eigenvalues by absolute value'''

	[size_y,size_x,num_eigvals] = eig.shape
	
	ind = np.ogrid[0:size_y,0:size_x,0:num_eigvals]
	k = np.argsort(np.abs(eig),axis=2)
	
	eig_ord = eig[ind[0],ind[1],k]
	
	return eig_ord

def vesselness(eig, a, b, c):
	'''Calculate vesselness measurement from the given eigenvalues'''

	b2 = 2.*b**2.
	c2 = 2.*c**2.

	eig += 1e-10
		
	Rb2 = eig[:,:,0]/eig[:,:,1]
	Rb2 = np.power(Rb2,2.)
	
	S2 = np.power(eig[:,:,0],2.)
	S2 += np.power(eig[:,:,1],2.)

	V = np.exp(-Rb2/b2)
	V *= (1-np.exp(-S2/c2))
	
	ind = np.nonzero(eig[:,:,1]>0)
	V[ind] = 0.

	return V	
	
def frangi_2d(img,arr_sigma):

	# Open image and normalize it to the range [0,255]
	img = img.astype(float)
	# arr_sigma = np.linspace(4, 35, 10)
	size_y, size_x = img.shape
	# vess_scales[i] will contain the vessels enhanced at scale arr_sigma[i]
	vess_scales = np.zeros([arr_sigma.size, size_y, size_x], dtype=np.float32)

	for sigma_index, sigma in enumerate(arr_sigma):
		eigvals = hessian_eigenvalues(img, sigma)
		eig_ord = sort_eigval(eigvals)
		img_np_diffused = vesselness(eig_ord, 0.5, 0.5, 20)

		vess_scales[sigma_index] = img_np_diffused

	# Obtain maximum response at all scales
	vess_max = np.max(vess_scales, axis=0)

	return vess_max
