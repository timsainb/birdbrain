import copy
import numpy as np
from tqdm.autonotebook import tqdm
import scipy.ndimage
from IPython.display import display
from ipywidgets import widgets
from birdbrain import utils


import k3d
import vtk

def vox2vtk(voxels, zero_point=None):
    """ converts voxels to vkt mesh object
    reduce_poly: 0-1, less to more simplification
    """
    # import voxels
    xs, ys, zs = voxels.shape
    dataImporter = vtk.vtkImageImport()
    data_string = voxels.tostring()
    dataImporter.CopyImportVoidPointer(data_string, len(data_string))
    dataImporter.SetDataScalarTypeToUnsignedChar()
    dataImporter.SetNumberOfScalarComponents(1)

    # whole extent needs to be relative to original
    dataImporter.SetDataExtent(0, xs - 1, 0, ys - 1, 0, zs - 1)
    if zero_point is None:
        dataImporter.SetWholeExtent(0, xs - 1, 0, ys - 1, 0, zs - 1)
    else:
        dataImporter.SetWholeExtent(
            -zero_point["ventral-dorsal"],
            xs - 1,
            -zero_point["posterior-anterior"],
            ys - 1,
            -zero_point["medial-lateral"],
            zs - 1,
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
    decimatedPoly = vtk.vtkPolyData()
    decimatedPoly.ShallowCopy(decimate.GetOutput())

    if verbose:
        print(
            "pct decimated:",
            1 - (decimatedPoly.GetNumberOfPolys() / dmc.GetOutput().GetNumberOfPolys()),
        )
    return decimatedPoly


def plot_regions_3d(
    atlas,
    regions_to_plot=[["HVC", "Nuclei"]],
    downsample_pct=1,
    polygon_simplification=0,
    verbose=False,
):
    """ plots brain regions on top of brain
    """
    # collect data
    brain_data = atlas.voxel_data.loc["Brain", "voxels"]

    # get axis boundaries
    bounds = np.array(
        [
            atlas.ventral - atlas.y_sinus_um_transform["ventral-dorsal"],
            atlas.dorsal - atlas.y_sinus_um_transform["ventral-dorsal"],
            atlas.posterior - atlas.y_sinus_um_transform["posterior-anterior"],
            atlas.anterior - atlas.y_sinus_um_transform["posterior-anterior"],
            atlas.lateral_neg - atlas.y_sinus_um_transform["medial-lateral"],
            atlas.lateral_pos - atlas.y_sinus_um_transform["medial-lateral"],
        ]
    ).astype("int")

    # downsample
    if downsample_pct < 1:
        brain_data = scipy.ndimage.zoom(brain_data, downsample_pct)

    # make a volume plot of the brain
    brain_volume = k3d.volume(
        brain_data,
        color_range=[0, 1],
        color_map=k3d.basic_color_maps.Binary,
        samples=128,
        alpha_coef=0.25,
        bounds=bounds,
        compression_level=9,
    )

    # the zero point in voxels, relative to y sinus
    zero_point = atlas.region_vox.loc["y_sinus", "coords_vox"]

    color_pal = atlas.label_cmap.colors

    # loop through regions
    regs = []
    for ri, (reg, type_) in enumerate(tqdm(regions_to_plot, leave=False)):
        color = (np.array(color_pal[ri % len(color_pal)]) * 255).astype("int")
        # get voxel_data
        vox_data = np.array(
            atlas.voxel_data.loc[type_, "voxels"]
            == atlas.brain_labels.loc[reg, "label"]
        )
        # convert to vtk format
        vtk_dat = vox2vtk(vox_data, zero_point=zero_point)
        # simplify polygon
        if polygon_simplification > 0:
            vtk_dat = vtk_reduce(
                vtk_dat.GetOutput(),
                polygon_simplification=polygon_simplification,
                verbose=verbose,
            )
        # shape of voxel data
        xs, ys, zs = vox_data.shape
        region_bounds = [
            0,
            (bounds[1] - bounds[0]) / xs,
            0,
            (bounds[3] - bounds[2]) / ys,
            0,
            (bounds[5] - bounds[4]) / zs,
        ]
        # create mesh plot
        region = k3d.vtk_poly_data(
            vtk_dat.GetOutput(),
            color=utils.rgb2hex(color[0], color[1], color[2]),
            bounds=region_bounds,
        )

        regs.append(region)

    origins = [-5000, 0, 0, 0, -5000, 0, 0, 0, -5000]
    vectors = [10000, 0, 0, 0, 10000, 0, 0, 0, 10000]
    colors = [0xFF0000, 0xFF0000, 0x00FF00, 0x00FF00, 0x0000FF, 0x0000FF]
    vec = k3d.vectors(
        origins, vectors, colors=colors, line_width=100, use_head=True, head_size=1000
    )

    plot = k3d.plot(height=1024, background_color=0xFEFEFE)
    plot += brain_volume

    # plot regions
    for region in regs:
        plot += region

    plot += vec
    plot.display()

    plot.camera_auto_fit = False
    plot.grid_visible = False
    # camera loc, center or rocation, angle (?)
    plot.camera = [9000, 4000, -20000, 0, 0, 0, 0.95, 0.1, 0.36]

    dorsal_ventral = widgets.FloatSlider(
        value=0, min=bounds[0], max=bounds[1], step=100, description="ventral-dorsal:"
    )
    anterior_posterior = widgets.FloatSlider(
        value=0,
        min=bounds[2],
        max=bounds[3],
        step=100,
        description="posterior-anterior:",
    )
    medial_lateral = widgets.FloatSlider(
        value=0, min=bounds[4], max=bounds[5], step=100, description="medial-lateral:"
    )
    region_dropdown = widgets.Dropdown(
        options=list(np.array(regions_to_plot)[:, 0]) + ["y_sinus"],
        value="y_sinus",
        description="Region:",
    )

    def dv(change):
        """ controls dorsal ventral widget
        """
        if change["type"] == "change" and change["name"] == "value":

            orig = copy.deepcopy(vec.origins)
            vec.origins = [
                change["new"] - 5000,
                orig[1],
                orig[2],
                change["new"],
                orig[4],
                orig[5],
                change["new"],
                orig[7],
                orig[8],
            ]

            # move the camera rotational axis
            pc = copy.deepcopy(plot.camera)
            plot.camera = pc[:3] + [change["new"]] + pc[-5:]

    def ap(change):
        """ controls dorsal ventral widget
        """
        if change["type"] == "change" and change["name"] == "value":

            orig = copy.deepcopy(vec.origins)

            vec.origins = [
                orig[0],
                change["new"],
                orig[2],
                orig[3],
                change["new"] - 5000,
                orig[5],
                orig[6],
                change["new"],
                orig[8],
            ]

            # move the rotational axis
            pc = copy.deepcopy(plot.camera)
            plot.camera = pc[:4] + [change["new"]] + pc[-4:]

    def ml(change):
        """ controls dorsal ventral widget
        """
        if change["type"] == "change" and change["name"] == "value":

            orig = copy.deepcopy(vec.origins)

            vec.origins = [
                orig[0],
                orig[1],
                change["new"],
                orig[3],
                orig[4],
                change["new"],
                orig[6],
                orig[7],
                change["new"] - 5000,
            ]

            # move the rotational axis
            pc = copy.deepcopy(plot.camera)
            plot.camera = pc[:5] + [change["new"]] + pc[-3:]

    def reg_dd(change):
        # move location to the center of that region

        if change["type"] == "change" and change["name"] == "value":

            change_loc = utils.vox_to_um(
                atlas.voxel_data.loc["Brain", "affine"],
                utils.loc_dict_to_list(atlas.region_vox.loc[change["new"], "coords_vox"]),
                atlas.um_mult,
                atlas.y_sinus_um_transform,
            )
            dv_loc = change_loc["ventral-dorsal"]
            ap_loc = change_loc["posterior-anterior"]
            ml_loc = -change_loc["medial-lateral"]

            vec.origins = [
                dv_loc - 5000,
                ap_loc,
                ml_loc,
                dv_loc,
                ap_loc - 5000,
                ml_loc,
                dv_loc,
                ap_loc,
                ml_loc - 5000,
            ]
            # set sliders
            anterior_posterior.value = ap_loc
            dorsal_ventral.value = dv_loc
            medial_lateral.value = ml_loc

            # move the rotational axis
            pc = copy.deepcopy(plot.camera)
            plot.camera = pc[:3] + [dv_loc, ap_loc, ml_loc] + pc[-3:]

    dorsal_ventral.observe(dv)
    display(dorsal_ventral)
    anterior_posterior.observe(ap)
    display(anterior_posterior)
    medial_lateral.observe(ml)
    display(medial_lateral)

    region_dropdown.observe(reg_dd)
    display(region_dropdown)

    plot.camera = [9000, 4000, -20000, 0, 0, 0, 0.95, 0.1, 0.36]

    return plot, vec
