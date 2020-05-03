from SU2_FWH_Numba import write_binary_fwh
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import glob

def ReadArrayFromVtu(file_name, array_name):
    
    # Read the source file.
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(file_name)
    reader.Update()  # Needed because of GetScalarRange
    data = reader.GetOutput()
    
    points = data.GetPoints()
    npts = points.GetNumberOfPoints()
    x = vtk_to_numpy(points.GetData())
    
    n_arrays = reader.GetNumberOfPointArrays()
    for i in range(n_arrays):
        if (reader.GetPointArrayName(i) == array_name):
            break
    
    return vtk_to_numpy(data.GetPointData().GetArray(i))

def VtuToArray(vtu_dir):

	infiles = glob.glob(vtu_dir+"/surface_*.vtu")
	infiles.sort()

	array = None

	cont = -1

	for infile in infiles:
		print "Reading: ", infile
		cont += 1
		pressure = ReadArrayFromVtu(infile, "Pressure")
		if (cont == 0):
			# Read the first csv and allocate the pressure matrix
			array = np.zeros((len(infiles),len(pressure)))
			
		array[cont,:] = pressure

	print "Number of iterations: ", array.shape[0]
	print "Number of points: ", array.shape[1]
	print "Mean: ", np.mean(array)
	data_file = {'data':array}
	return data_file

if __name__ == '__main__':

	
	vtu_dir = '/home/emolina/temp/Lagoon/WMLES'
	data_file = VtuToArray(vtu_dir)
	write_binary_fwh(data_file)
