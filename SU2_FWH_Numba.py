import os, sys
from optparse import OptionParser
sys.path.append(os.environ['SU2_RUN'])
import SU2
import numpy as np
import pandas as pd
import glob
import pdb
import struct
import sys
import time
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import timeit
from numba import jit
#from memory_profiler import profile

@jit(nopython=True)
def compute_scalar_product(array1, array2, axes):
        """Computes the scalar product of 2 arrays."""
        #temp = array1*array2
        #result = np.sum(temp, axis=axes-1)
        return np.sum(array1*array2)

def read_binary_fwh(filename):

	print "\nReading file: %s"%filename

	start_time = time.time()

	# Not efficient way
	infile = open(filename, 'rb')
	data = infile.read()
	infile.close()
	
	print "Size of the file: %d in bytes"%len(data)

    # The first is a magic number that we can use to check for binary files (it is the hex
    # representation for "SU2").
	if (struct.unpack('i',data[:4])[0] != 535532):
		print "Magic number 535532 not found in the solution file %s" %filename
		sys.exit()

	# The second two values are number of variables and number of points (DoFs). 
	ntime = struct.unpack('i',data[4:8])[0]
	ndof = struct.unpack('i',data[8:12])[0]

	# Read data in one shoot
	start = 12
	end = start+ntime*ndof*4
	print ntime, ndof, len(data[start:end])
	array = np.asfarray(struct.unpack('%df'%(ntime*ndof),data[start:end]), dtype = np.float32)
	array = array.reshape(ntime,ndof)

	# Create dictionary
	data_file = {'data':array}

	time_interval = time.time() - start_time
	print "Elapsed time: %f seconds\n"%time_interval

	return data_file

def write_binary_fwh(data_file,filename="fwh_bin.dat"):
	"""

	"""
	fout = open(filename,'wb')

	# The first is a magic number that we can use to check for binary files (it is the hex
	# representation for "SU2").
	fout.write(struct.pack('i', 535532))

	# The second two values are number of time steps and number of points (DoFs).
	ntime = data_file['data'].shape[0]
	fout.write(struct.pack('i', ntime))
	ndof = data_file['data'].shape[1]
	fout.write(struct.pack('i', ndof))
	
	# Write the entire data in one shoot
	fout.write(struct.pack('%df'%(ntime*ndof), *data_file['data'].flatten()))

	# # Write ExtIter and Metadata
	# fout.write(struct.pack('i',data_file['ExtIter']))
	# for i in range(8):
	# 	fout.write(struct.pack('d', data_file['MetaData'][i]))
	fout.close()

	return None

def CSVToArray(csv_dir):

	infiles = glob.glob(csv_dir+"/surface_flow_*.csv")
	infiles.sort()

	array = None

	cont = -1

	for infile in infiles:
		print "Reading: ", infile
		cont += 1
		temp = np.loadtxt(infile, skiprows=1, delimiter=",")
		if (cont == 0):
			# Read the first csv and allocate the pressure matrix
			xyz = temp[:,1:4]
			pressure = temp[:,4]
			array = np.zeros((len(infiles),len(pressure)))
			array[0,:] = pressure
		else:
			array[cont,:] = temp[:,4]

	print array.shape[0]
	print array.shape[1]
	data_file = {'data':array}
	return data_file

#@jit(nopython=True)
def Compute_RadiationVec(analogy, formulation, surface_geo, nDim, coord, normals, Observer_Locations, nPanel, nObserver, FreeStreamMach):

	"""Compute radiation vector form source to observers."""

	Beta2 = 1.0 - FreeStreamMach**2.0

	for iPanel in range(nPanel):

		x = coord[iPanel,0]
		y = coord[iPanel,1]
		z = 0.0
		if ( nDim == 3):
			z = coord[iPanel,2]

		nx = normals[iPanel,0]
		ny = normals[iPanel,1]
		nz = 0.0
		if (nDim == 3):
			nz = normals[iPanel,2]

		#CheckNormal =  x*nx+y*ny+z*nz
		#if (CheckNormal<0):
		#	nx = -nx
		#	ny = -ny
		#	nz = -nz
		dS = normals[iPanel,3]
		Area_Factor = 1.0

		surface_geo[iPanel,0] = x
		surface_geo[iPanel,1] = y
		surface_geo[iPanel,2] = nx
		surface_geo[iPanel,3] = ny
		surface_geo[iPanel,4] = Area_Factor*dS
		
		if ( nDim == 3):
			surface_geo[iPanel,5] = z
			surface_geo[iPanel,6] = nz

		for iObserver in range(nObserver):
			r1 = Observer_Locations[iObserver,0]-x
			r2 = Observer_Locations[iObserver,1]-y
			r3 = Observer_Locations[iObserver,2]-z

			if (analogy == '1A' and formulation == 'solid'):
				
				r_mag = np.sqrt(r1*r1+r2*r2+r3*r3)
				r1    = r1/r_mag
				r2    = r2/r_mag
				r3    = r3/r_mag

			elif (analogy == '1A_WT' and formulation == 'solid'):
				
				r_star = np.sqrt(r1**2 + Beta2 * (r2**2 + r3**2))
				r_mag  = (-FreeStreamMach*r1 + r_star) / Beta2
				r1     = (-FreeStreamMach*r_star + r1 ) / (Beta2 * r_mag)
				r2     = r2/r_mag
				r3     = r3/r_mag


			surface_geo[iPanel,7+iObserver*4] = r1
			surface_geo[iPanel,8+iObserver*4] = r2
			surface_geo[iPanel,9+iObserver*4] = r3
			surface_geo[iPanel,10+iObserver*4] = r_mag

	return surface_geo

@jit(nopython=True)
def Extract_NoiseSources(Fr, data, nSample, nPanel, nObserver, FreeStreamPressure, surface_geo):

	for iSample in range(nSample):
		for iPanel in range(nPanel):
			# compute monopole and dipole source terms on the FWH surface

			nx = surface_geo[iPanel,2]
			ny = surface_geo[iPanel,3]
			nz = surface_geo[iPanel,6]

			Q = 0.0; rho=0.0; ux = 0.0; uy = 0.0; uz = 0.0
			p = data[iSample,iPanel]

			F1= rho*(ux*nx+uy*ny+uz*nz)*(ux)+(p-FreeStreamPressure)*nx
			F2= rho*(ux*nx+uy*ny+uz*nz)*(uy)+(p-FreeStreamPressure)*ny
			F3= rho*(ux*nx+uy*ny+uz*nz)*(uz)+(p-FreeStreamPressure)*nz
			for iObserver in range(nObserver):
				Fr[iObserver,iPanel,iSample]= F1*surface_geo[iPanel,7+iObserver*4]+F2*surface_geo[iPanel,8+iObserver*4] + \
												F3*surface_geo[iPanel,9+iObserver*4]
	return Fr

@jit(nopython=True)
def Extract_Mean(Fr, Fr_mean, nSample, nPanel, nObserver):

	for iPanel in range(nPanel):
		for iSample in range(nSample):
			for iObserver in range(nObserver):
				Fr[iObserver,iPanel,iSample] = Fr[iObserver,iPanel,iSample] - Fr_mean[iObserver,iPanel]
	return Fr

@jit(nopython=True)
def Compute_RetardedTime(pp_ret, Fr, surface_geo, nSample, nPanel, nObserver, FreeStreamDensity, dt, SamplingFreq, a_inf):

	dt_sampling = dt * SamplingFreq

	for iObserver in range(nObserver):
		for iPanel in range(nPanel):
			Un_dot = 0.0
			dS = surface_geo[iPanel,4]
			for iSample in range(nSample):
				
				if (iSample==0):
					Fr_dot = (-Fr[iObserver,iPanel,iSample+2]+4.0*Fr[iObserver,iPanel,iSample+1]-3.0*Fr[iObserver,iPanel,iSample])/2.0/dt_sampling
				elif (iSample==nSample-1):
					Fr_dot = (3.0*Fr[iObserver,iPanel,iSample]-4.0*Fr[iObserver,iPanel,iSample-1]+Fr[iObserver,iPanel,iSample-2])/2.0/dt_sampling
				else:
					Fr_dot = (Fr[iObserver,iPanel,iSample+1]-Fr[iObserver,iPanel,iSample-1])/2.0/dt_sampling

				#pp_ret[iObserver][iPanel][iSample] = FreeStreamDensity*Un_dot/surface_geo[iPanel][10+iObserver*4] + Fr_dot/surface_geo[iPanel][10+iObserver*4]/a_inf + \
				#									Fr[iObserver][iPanel][iSample]/surface_geo[iPanel][10+iObserver*4]/surface_geo[iPanel][10+iObserver*4]
				pp_ret[iObserver,iPanel,iSample] = Fr_dot/(surface_geo[iPanel,10+iObserver*4]*a_inf) + \
													 Fr[iObserver,iPanel,iSample]/(surface_geo[iPanel,10+iObserver*4]**2)
				pp_ret[iObserver,iPanel,iSample] = pp_ret[iObserver,iPanel,iSample]*dS/(4.0*np.pi)

	return pp_ret

@jit(nopython=True)
def Compute_RetardedTime_WT(pp_ret, data, surface_geo, nSample, nPanel, nObserver, FreeStreamPressure, FreeStreamDensity, FreeStreamMach, M_i, a_inf, dt, SamplingFreq):

	Lr = np.zeros((nObserver, nPanel, nSample), dtype = np.float32)
	Lm = np.zeros((nPanel,nSample), dtype = np.float32)

	for iSample in range(nSample):
		for iPanel in range(nPanel):
			# compute monopole and dipole source terms on the FWH surface

			nx = surface_geo[iPanel,2]
			ny = surface_geo[iPanel,3]
			nz = surface_geo[iPanel,6]

			Q = 0.0; rho=0.0; ux = 0.0; uy = 0.0; uz = 0.0
			p = data[iSample,iPanel]

			L1= (p-FreeStreamPressure)*nx
			L2= (p-FreeStreamPressure)*ny
			L3= (p-FreeStreamPressure)*nz
			
			L_temp = np.array([L1,L2,L3])

			for iObserver in range(nObserver):
				R_temp = np.array([surface_geo[iPanel,7+iObserver*4], surface_geo[iPanel,8+iObserver*4], surface_geo[iPanel,9+iObserver*4]])
				#pdb.set_trace()
				Lr[iObserver,iPanel,iSample] = compute_scalar_product(L_temp, R_temp, 3)

			Lm[iPanel,iSample] = compute_scalar_product(L_temp, M_i, 3)

	dt_sampling = dt * SamplingFreq

	for iObserver in range(nObserver):
		for iPanel in range(nPanel):
			Un_dot = 0.0
			dS = surface_geo[iPanel,4]
			for iSample in range(nSample):
				
				if (iSample==0):
					Lr_dot = (-Lr[iObserver,iPanel,iSample+2]+4.0*Lr[iObserver,iPanel,iSample+1]-3.0*Lr[iObserver,iPanel,iSample])/2.0/dt_sampling
				elif (iSample==nSample-1):
					Lr_dot = (3.0*Lr[iObserver,iPanel,iSample]-4.0*Lr[iObserver,iPanel,iSample-1]+Lr[iObserver,iPanel,iSample-2])/2.0/dt_sampling
				else:
					Lr_dot = (Lr[iObserver,iPanel,iSample+1]-Lr[iObserver,iPanel,iSample-1])/2.0/dt_sampling

				#pp_ret[iObserver][iPanel][iSample] = FreeStreamDensity*Un_dot/surface_geo[iPanel][10+iObserver*4] + Fr_dot/surface_geo[iPanel][10+iObserver*4]/a_inf + \
				#									Fr[iObserver][iPanel][iSample]/surface_geo[iPanel][10+iObserver*4]/surface_geo[iPanel][10+iObserver*4]
				
				R_temp = np.array([surface_geo[iPanel,7+iObserver*4], surface_geo[iPanel,8+iObserver*4], surface_geo[iPanel,9+iObserver*4]])
				Mr = compute_scalar_product(M_i,R_temp,3)

				R = surface_geo[iPanel,10+iObserver*4]
				
				pp_ret[iObserver,iPanel,iSample] = Lr_dot / (a_inf * R * (1.0 - Mr)**2) + \
													 (Lr[iObserver,iPanel,iSample] - Lm[iPanel,iSample])/(R**2 * (1.0 - Mr)**2) + \
													 (Lr[iObserver,iPanel,iSample]*(Mr - FreeStreamMach**2))/(R**3 * (1.0 - Mr)**3)

				pp_ret[iObserver,iPanel,iSample] = pp_ret[iObserver,iPanel,iSample]*dS/(4.0*np.pi)

	return pp_ret


@jit(nopython=True)
def Compute_ObserverTime( t_interp, t_Obs, surface_geo, nSample, nPanel, nObserver, dt, SamplingFreq, start_iter, a_inf):

	r_minmax = np.zeros((nObserver,2))
	for iObserver in range(nObserver):
		r_min = 10.0e31
		r_max = 0.0
		for iPanel in range(nPanel):
			r = surface_geo[iPanel,10+iObserver*4]
			if (r>r_max):
				r_max = r
			if (r<r_min):
				r_min = r

			for iSample in range(nSample):
				t_src = dt*(start_iter+iSample*SamplingFreq)
				t_Obs[iObserver,iPanel,iSample]=t_src + r/a_inf

		r_minmax[iObserver,0]= r_min
		r_minmax[iObserver,1]= r_max
		#print "Time Shift INFO: ", r_min, ", ", r_max

	for iObserver in range(nObserver):
		t_interp_start = dt*(start_iter)+r_minmax[iObserver,1]/a_inf
		t_interp_end   = dt*(start_iter+nSample*SamplingFreq-1)+r_minmax[iObserver,0]/a_inf
		dt_interp = (t_interp_end - t_interp_start)/(nSample-1)

		for iSample in range(nSample):
			t_interp[iObserver,iSample] = t_interp_start + dt_interp*iSample

	return t_interp, t_Obs

@jit(nopython=True)
def Integrate_Sources(pp_TimeDomain, pp_interp, nSample, nPanel, nObserver):

	for iObserver in range(nObserver):
		for iSample in range(nSample):
			for iPanel in range(nPanel):
				pp_TimeDomain[iObserver,iSample] = pp_TimeDomain[iObserver,iSample] + pp_interp[iObserver,iPanel,iSample]
	return pp_TimeDomain

#@jit(nopython=True)
def Interp_PressureSignal(pp_interp, t_interp, pp_ret, t_Obs, nSample, nPanel, nObserver):
	
	for iObserver in range(nObserver):
		for iPanel in range(nPanel):
			aux_x = t_Obs[iObserver,iPanel,:]
			aux_y = pp_ret[iObserver,iPanel,:]			
			f = interp1d(aux_x, aux_y, kind = 'cubic')
			pp_interp[iObserver,iPanel,:] = f(t_interp[iObserver,:])
	return pp_interp

def SU2_SetSpline(x, y,n, yp1, ypn, y2):

	u = np.zeros(n)

	if (yp1 > 0.99e30):			
		y2[0]= 0.0
		u[0] = 0.0			   
	else: 				        
		y2[0] = -0.5
		u[0]=(3.0/(x[1]-x[0]))*((y[1]-y[0])/(x[1]-x[0])-yp1)

	for i in range(2,n-1):

		sig=(x[i-1]-x[i-2])/(x[i]-x[i-2])
		p=sig*y2[i-2]+2.0		
		y2[i-1]=(sig-1.0)/p					
		a1 = (y[i]-y[i-1])/(x[i]-x[i-1])
		
		if (x[i] == x[i-1]):
			a1 = 1.0
		a2 = (y[i-1]-y[i-2])/(x[i-1]-x[i-2])
		
		if (x[i-1] == x[i-2]):
			a2 = 1.0;
		u[i-1]= a1 - a2;
		u[i-1]=(6.0*u[i-1]/(x[i]-x[i-2])-sig*u[i-2])/p
    
	if (ypn > 0.99e30):
		qn=un=0.0
	else:
		qn=0.5
		un=(3.0/(x[n-1]-x[n-2]))*(ypn-(y[n-1]-y[n-2])/(x[n-1]-x[n-2]))

	y2[n-1]=(un-qn*u[n-2])/(qn*y2[n-2]+1.0)

	k_inv = range(1,n-1)[::-1]
	for k in k_inv:
		y2[k-1]=y2[k-1]*y2[k]+u[k-1]

	return y2

def SU2_GetSpline(xa, ya, y2a, n, x):
	
	if (x < xa[0]):
		x = xa[0]       #Clip max and min values
	
	if (x > xa[n-1]):
		x = xa[n-1]

	klo = 1			                # We will find the right place in the table by means of
	khi = n		                    # bisection. This is optimal if sequential calls to this
	while (khi-klo > 1):            # routine are at random values of x. If sequential calls
		k = (khi+klo) >> 1   	    # are in order, and closely spaced, one would do better
		if (xa[k-1] > x): 
			khi = k	                # to store previous values of klo and khi and test if
		else:
			klo=k				    # they remain appropriate on the next call. klo and khi now bracket the input value of x
	h = xa[khi-1] - xa[klo-1]
	if (h == 0.0): 
		h = EPS;                    # The xa?s must be distinct.
	a = (xa[khi-1]-x)/h
	b = (x-xa[klo-1])/h		   # Cubic spline polynomial is now evaluated.
	y = a*ya[klo-1]+b*ya[khi-1]+((a*a*a-a)*y2a[klo-1]+(b*b*b-b)*y2a[khi-1])*(h*h)/6.0
  
	return y

#@jit(nopython=True)
def Interp_PressureSignal_Fast(pp_interp, t_interp, pp_ret, t_Obs, nSample, nPanel, nObserver):

	x = np.zeros(nSample)
	t = np.zeros(nSample)
	derivative = np.zeros(nSample)
	yp1 = 10.0e31
	ypn = 10.0e31

	for iObserver in range(nObserver):
		for iPanel in range(nPanel):
			x = t_Obs[iObserver][iPanel][:]
			t = pp_ret[iObserver][iPanel][:]
			derivative = SU2_SetSpline(t, x, nSample, yp1, ypn, derivative)
			for iSample in range(nSample):
				pp_interp[iObserver][iPanel][iSample] = SU2_GetSpline(t, x, derivative, nSample, t_interp[iObserver][iSample])

	return pp_interp
# -------------------------------------------------

#@profile
def main():
	# Command line options
	parser=OptionParser()
	parser.add_option("--configFile", dest="config_file", help="Read config from FILE", metavar="FILE")
	parser.add_option("--fwhFile", dest="fwh_file", help="Read FHW data FILE", metavar="FILE")
	parser.add_option("--csvFile", dest="csv_file", help="Read surface_csv FILE", metavar="FILE")
	parser.add_option("--nDim", dest="nDim", default=3, help="Define the number of DIMENSIONS",
					metavar="DIMENSIONS")
	parser.add_option("--samplingFreq", dest="SamplingFreq", default=1, help="Define the sampling FREQUENCY",
					metavar="FREQUENCY")

	parser.add_option("--analogy", type="string", dest="analogy", default="1A" )
	parser.add_option("--type", type="string", dest="formulation", default= "solid")


	(options, args) = parser.parse_args()
	options.nDim  = int( options.nDim )

	analogy = options.analogy
	formulation = options.formulation


	print "\nFfowcs-Williams & Hawkings Analogy Solver\n"

	# Assert inputs here!
  	
  	# Config
	config = SU2.io.Config(options.config_file)

	FreeStreamPressure = 5895.49 # Check this number or calculate from config file
	FreeStreamDensity  = 0.0688412
	FreeStreamVelocity = 44.321
	FreeStreamMach = 0.128
	a_inf = FreeStreamVelocity / FreeStreamMach
	l_sim = 3.0
	l_discard = 0.0 # 0.75
	
	U0_i = np.array([FreeStreamVelocity,0.0,0.0])
	M_i = np.array([-U0_i[0]/a_inf,-U0_i[1]/a_inf,-U0_i[2]/a_inf])

	dt = float(config['UNST_TIMESTEP']) * float(options.SamplingFreq)
	SamplingFreq = 1.0
	M_PI = np.pi

	# Load Surface Coordinates and Normals
	coord_aux = np.loadtxt('CoordinatesNormals.dat')

	# Load observers
	Observer_Locations = np.loadtxt('Observers.dat')
	nObserver          = len(Observer_Locations)

	# Interpolate the normals.
	# I need to do this because the surface_csv files does not have this information.
	coord        = np.loadtxt(options.csv_file, skiprows=1, delimiter=",")[:,1:4]
	
	if (len(coord) != len(coord_aux)):
		raise ValueError(str_error("Please check input files. The number of panels are differents."))

	indx = np.nonzero((l_discard < coord[:,2]) & ((l_sim - l_discard) > coord[:,2] ))[0]
	# indx = np.nonzero((l_discard > coord[:,2]) | ((l_sim - l_discard) < coord[:,2] ))[0]
	#pdb.set_trace()
	coord = coord[indx]
	print coord[:,2].min(), coord[:,2].max()

	normals      = np.zeros((len(coord),4))
	normals[:,0] = griddata(coord_aux[:,0:3], coord_aux[:,3], coord, method='nearest')
	normals[:,1] = griddata(coord_aux[:,0:3], coord_aux[:,4], coord, method='nearest')
	normals[:,2] = griddata(coord_aux[:,0:3], coord_aux[:,5], coord, method='nearest')
	normals[:,3] = griddata(coord_aux[:,0:3], coord_aux[:,6], coord, method='nearest')

	# Load binary FWH data
	data_file = read_binary_fwh(options.fwh_file)
	data_file['data'] = data_file['data'][:,indx]
	#pdb.set_trace()
	nSample   = data_file['data'].shape[0]
	nPanel    = data_file['data'].shape[1]

	# Now following Beckett implementation:
	# Allocating some variables
	# Note that the input pressure data is already float32
	surface_geo        = np.zeros((nPanel,2*options.nDim+1+4*nObserver), dtype = np.float32)
	pp_ret             = np.zeros((nObserver,nPanel,nSample)           , dtype = np.float32)
	pp_interp          = np.zeros((nObserver,nPanel,nSample)           , dtype = np.float32)
	t_Obs              = np.zeros((nObserver,nPanel,nSample)           , dtype = np.float32)
	t_interp           = np.zeros((nObserver,nSample)                  , dtype = np.float32)
	pp_TimeDomain      = np.zeros((nObserver,nSample)                  , dtype = np.float32)
	Q                  = np.full((nPanel,nSample), FreeStreamDensity * FreeStreamVelocity, dtype = np.float32)
	# pp_TimeDomain_root = np.zeros((nObserver,nSample))
	
	fwh_start = timeit.default_timer()

	print 'Creating surface metrics and radiation vector.\n'
	nDim = options.nDim
	surface_geo = Compute_RadiationVec(analogy, formulation, surface_geo, nDim, coord, normals, Observer_Locations, nPanel, nObserver, FreeStreamMach)

	print "Observer, Radiation vector (min), Radiation vector (max)"
	for iObserver in range(nObserver):
		print iObserver, surface_geo[:,10+iObserver*4].min(), surface_geo[:,10+iObserver*4].max()

	if analogy == '1A':
		print 'Extracting noise sources.\n'
		Lr = np.zeros((nObserver,nPanel,nSample))
		Lr = Extract_NoiseSources(Lr, data_file['data'], nSample, nPanel, nObserver, FreeStreamPressure, surface_geo)

		# Fluctuation on zero mean.
		print "Fluctuation on zero mean.\n"
		Lr_mean = np.mean(Lr, axis = 2) # axis = 2 is the sample axis
		Lr = Extract_Mean(Lr, Lr_mean, nSample, nPanel, nObserver)

		# From Compute_TimeDomainPanelSignal
		print "Computing retarted time.\n"
		pp_ret = Compute_RetardedTime(pp_ret, Lr, surface_geo, nSample, nPanel, nObserver, FreeStreamDensity, dt, SamplingFreq, a_inf)

		# Delete - In python it is not neccessary however this program will eat all your available memory.
		del Lr_mean
		del Lr

	elif analogy == '1A_WT':
		print "Computing retarted time for WT formulation.\n"
		pp_ret = Compute_RetardedTime_WT(pp_ret, data_file['data'], surface_geo, nSample, nPanel, nObserver, FreeStreamPressure, FreeStreamDensity, FreeStreamMach, M_i, a_inf, dt, SamplingFreq)
	

	# From Compute_ObserverTime
	#pdb.set_trace()
	print "Computing Oberserver time.\n"
	start_iter = int(config['UNST_RESTART_ITER'])
	t_interp, t_Obs = Compute_ObserverTime( t_interp, t_Obs, surface_geo, nSample, nPanel, nObserver, dt, SamplingFreq, start_iter, a_inf)
	#pdb.set_trace()
	# From Interpolate_PressureSignal - Check with Beckett the order of the interpolator.
	print "Interpolating the pressure signal.\n"
	pp_interp = Interp_PressureSignal(pp_interp, t_interp, pp_ret, t_Obs, nSample, nPanel, nObserver)
	# pp_interp = Interp_PressureSignal_Fast(pp_interp, t_interp, pp_ret, t_Obs, nSample, nPanel, nObserver)
	# pp_interp = pp_ret

	# Delete 
	del t_Obs
	del pp_ret

	# From Integrated_Sources
	print "Integrating sources.\n"
	pp_TimeDomain = Integrate_Sources(pp_TimeDomain, pp_interp, nSample, nPanel, nObserver)

	# Delete
	del pp_interp

	print "Writing the results.\n"
	header = "#"
	for iObserver in range(nObserver):
		header += " Oberserver %d"%iObserver
	
	#pdb.set_trace()
	np.savetxt("Observer_Noise.dat", np.column_stack((t_interp.transpose(),pp_TimeDomain.transpose()))  , fmt='%.18e', delimiter=' ', newline='\n', header=header)


	# Delete
	del surface_geo
	del t_interp
	del pp_TimeDomain

	# ======== end cpu time
	fwh_end = timeit.default_timer()
	fwh_time = fwh_end - fwh_start
	
	print "Total time: %.3f min."%(fwh_time/60)

	return None

if __name__ == '__main__':

	
	main()

	# To Extract pressure data to CSV files please comment main() and uncomment this 3 lines

	#csv_dir = '/media/esmolina/Backup/Tandem/Lz3p0_Dz0p02/PyWrapper/Data'
	#data_file = CSVToArray(csv_dir)
	#write_binary_fwh(data_file)
	