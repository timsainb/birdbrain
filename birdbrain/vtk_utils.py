import k3d
import vtk


def vox2vtk(voxels, zero_point=None):
    """ converts voxels to vkt mesh object
    reduce_poly: 0-1, less to more simplification
    zero_point: if a zero point is provided, the extent of the vtk file is set so that the zero point is in the center
    """
    # import voxels
    xs, ys, zs = voxels.shape
    dataImporter = vtk.vtkImageImport()
    data_string = voxels.tostring()
    dataImporter.CopyImportVoidPointer(data_string, len(data_string))
    dataImporter.SetDataScalarTypeToUnsignedChar()
    dataImporter.SetNumberOfScalarComponents(1)

    # whole extent needs to be relative to original
    dataImporter.SetDataExtent(0, zs - 1, 0, ys - 1, 0, xs - 1)
    if zero_point is None:
        dataImporter.SetWholeExtent(0, xs - 1, 0, ys - 1, 0, zs - 1)
    else:
        dataImporter.SetWholeExtent(
            -zero_point[0], zs - 1, -zero_point[1], ys - 1, -zero_point[2], xs - 1
        )

    # convert to mesh
    dmc = vtk.vtkDiscreteMarchingCubes()
    dmc.SetInputConnection(dataImporter.GetOutputPort())
    dmc.GenerateValues(1, 1, 1)
    dmc.Update()

    return dmc


def vtk_reduce(dmc, polygon_simplification=0.1, verbose=False):
    """ reduces the number of polygons in a mesh model
    """
    # reduce number of polygons
    decimate = vtk.vtkDecimatePro()
    decimate.SetInputData(dmc.GetOutput())
    decimate.SetTargetReduction(polygon_simplification)
    decimate.Update()

    return decimate


def write_to_stl(vtk_obj, filename):
    writer = vtk.vtkSTLWriter()
    writer.SetInputConnection(vtk_obj.GetOutputPort())
    writer.SetFileTypeToBinary()
    writer.SetFileName(filename)
    writer.Write()


def generate_stl(filename, voxels, zero_point=None, polygon_simplification=0.1):
    """ 
    zero point: the center point of the voxels (e.g. the y sinus in voxels)
    polygon_simplification: how much to reduce the number of polygons (1 = more)
    
    """
    dmc = vox2vtk(voxels, zero_point)
    decimatedPoly = vtk_reduce(dmc, polygon_simplification=polygon_simplification)
    # print(dmc, decimatedPoly)
    write_to_stl(decimatedPoly, filename)
