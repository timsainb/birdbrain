import numpy as np
import k3d
import scipy.ndimage
from birdbrain import utils
from birdbrain.vtk_utils import vox2vtk, vtk_reduce
from tqdm.autonotebook import tqdm
import copy

from birdbrain.visualization.plotting_3d import widget_controllers


def get_axis_boundaries(atlas):
    bounds = np.array(
        [atlas.xmin, atlas.xmax, atlas.ymin, atlas.ymax, atlas.zmin, atlas.zmax]
    ).astype("int")
    return bounds


def get_voxel_zero_point(atlas):
    zero_point = utils.um_to_vox(
        [0, 0, 0],
        atlas.voxel_data.loc["Brain", "affine"],
        atlas.um_mult,
        atlas.y_sinus_um_transform,
    )
    return zero_point


def make_brain_volume(
    atlas,
    bounds,
    zero_point=None,
    downsample_frac=1,
    alpha_coef=0.25,
    color_map=k3d.basic_color_maps.Binary,
    display_half=True,
):

    # collect data
    brain_data = copy.deepcopy(
        np.array(np.swapaxes(atlas.voxel_data.loc["Brain", "voxels"], 0, 2))
    )
    import matplotlib.pyplot as plt

    if display_half:
        brain_data[:, :, zero_point[0] :] = 0

    # downsample to speed up rendering
    if downsample_frac < 1:
        brain_data = scipy.ndimage.zoom(brain_data, downsample_frac)

    # make a volume plot of the brain
    brain_volume = k3d.volume(
        brain_data.astype(float),
        color_range=[0, 1],
        color_map=color_map,
        samples=128,
        alpha_coef=alpha_coef,
        bounds=bounds,
        compression_level=9,
        name="Brain",
        colorLegend=False,
    )
    return brain_volume


def make_volume(atlas, vol):
    """ add a volume with an affine transform (that is not brain)
    TODO: is brain volume function need to be seperate from this one?
    """
    bg_image_data = atlas.voxel_data.loc[vol, "voxels"]
    affine = atlas.voxel_data.loc[vol, "affine"]

    xm, ym, zm = utils.vox_to_um(
        [0, 0, 0], affine, atlas.um_mult, atlas.y_sinus_um_transform
    )
    xma, yma, zma = utils.vox_to_um(
        list(np.shape(bg_image_data)), affine, atlas.um_mult, atlas.y_sinus_um_transform
    )

    img_bg_extent = np.concatenate(np.array([[xm, xma], [ym, yma], [zm, zma]]))

    bg_image_data = np.uint8(utils.norm01(np.swapaxes(bg_image_data, 0, 2)) * 256)

    addl_vol = k3d.volume(
        bg_image_data,
        color_range=[70, 100],
        color_map=k3d.matplotlib_color_maps.Greys,
        samples=128,
        alpha_coef=10.0,
        bounds=img_bg_extent,
        compression_level=9,
    )
    return addl_vol


def label_text(atlas, reg, color=(0, 0, 0), size=1):
    # the center of the region in voxels
    cv = list(atlas.region_vox.loc[reg].coords_vox)
    # add an offset to put text above region
    cv[2] += (
        atlas.region_vox.loc[reg].region_bounds[5]
        - atlas.region_vox.loc[reg].region_bounds[4]
    ) / 2

    text_x, text_y, text_z = utils.vox_to_um(
        cv,
        atlas.voxel_data.loc["Brain", "affine"],
        atlas.um_mult,
        atlas.y_sinus_um_transform,
    )
    color = [int(i * 0.5) for i in color]
    reg = reg.replace("_", "\, ")
    plt_text = k3d.text(
        reg,
        position=[text_x, text_y, text_z],
        color=utils.rgb2hex(color[0], color[1], color[2]),
        size=size,
        reference_point="cb",
        name=reg + "_label",
    )
    return plt_text


def make_poly(
    atlas,
    lab,
    reg,
    bounds,
    type_,
    color,
    zero_point,
    polygon_simplification=0,
    verbose=False,
):
    # get voxel_data
    vox_data = np.swapaxes(np.array(atlas.voxel_data.loc[type_, "voxels"] == lab), 0, 2)
    # convert to vtk format
    vtk_dat = vox2vtk(vox_data, zero_point=zero_point)

    # simplify polygon
    if polygon_simplification > 0:
        vtk_dat = vtk_reduce(
            vtk_dat, polygon_simplification=polygon_simplification, verbose=verbose
        )
    # shape of voxel data
    xs, ys, zs = vox_data.shape
    region_bounds = [
        0,
        (bounds[1] - bounds[0]) / zs,
        0,
        (bounds[3] - bounds[2]) / ys,
        0,
        (bounds[5] - bounds[4]) / xs,
    ]
    # create mesh plot
    region = k3d.vtk_poly_data(
        vtk_dat.GetOutput(),
        color=utils.rgb2hex(color[0], color[1], color[2]),
        bounds=region_bounds,
        name=reg,
    )
    return region


def axis_vector():
    origins = [-5000, 0, 0, 0, -5000, 0, 0, 0, -5000]
    vectors = [10000, 0, 0, 0, 10000, 0, 0, 0, 10000]
    colors = [0xFF0000, 0xFF0000, 0x00FF00, 0x00FF00, 0x0000FF, 0x0000FF]
    vec = k3d.vectors(
        origins,
        vectors,
        colors=colors,
        line_width=100,
        use_head=True,
        head_size=1000,
        name="center point",
    )
    return vec


def plot_regions_3d(
    atlas,
    regions_to_plot=[["HVC", "Nuclei"]],
    downsample_frac=1,
    polygon_simplification=0,
    additional_volumes=[],
    verbose=False,
    height=1024,
    show_labels=True,
    camera_starting_point=[-22977, 18052, 8696, 0, 0, 0, 0.14, -0.16, 0.997],
    display_half_brain=True,
):
    """ plots brain regions on top of brain
    """
    # get colormap from atlas
    color_pal = atlas.label_cmap.colors

    # get axis boundaries
    bounds = get_axis_boundaries(atlas)

    # the zero point in voxels, relative to y sinus
    zero_point = get_voxel_zero_point(atlas)

    # make a volume plot of the brain
    brain_volume = make_brain_volume(
        atlas, bounds, display_half=display_half_brain, zero_point=zero_point
    )

    # add non-brain volumes (like skull)
    addl_vols = []
    for vol in additional_volumes:
        addl_vol = make_volume(atlas, vol)
        addl_vols.append(addl_vol)

    # set atlas a region for y sinus
    atlas.region_vox.loc["y_sinus"] = ["y_sinus", "y_sinus", np.nan, np.nan, np.nan]
    atlas.region_vox.loc["y_sinus", "coords_vox"] = zero_point

    # add regions as polygons and labels
    regs = []
    text_labels = []
    for ri, (reg, type_) in enumerate(tqdm(regions_to_plot, leave=False)):
        # subset color
        color = (np.array(color_pal[ri % len(color_pal)]) * 255).astype("int")
        # get label name
        lab = atlas.brain_labels[atlas.brain_labels.type_ == type_].loc[reg, "label"]
        # add label
        if show_labels:
            plt_text = label_text(atlas, reg, color=color, size=1)
            text_labels.append(plt_text)

        # add region polygon (e.g. HVC)
        region = make_poly(
            atlas, lab, reg, bounds, type_, color, zero_point, polygon_simplification=0
        )
        regs.append(region)

    plot = k3d.plot(
        height=height,
        background_color=0xFEFEFE,
        axes=["medial-lateral", "anterior-posterior", "dorsal-ventral"],
    )
    plot += brain_volume

    for vol in addl_vols:
        plot += vol

    # plot labels
    for lab in text_labels:
        plot += lab

    # plot regions
    for region in regs:
        plot += region

    # vector axis in center
    vec = axis_vector()

    plot += vec

    # display the plot
    plot.display()

    plot.camera_auto_fit = False
    # plot.grid_visible = False

    # camera loc, center or rocation, angle (?)
    plot.camera = camera_starting_point

    widget_controllers(atlas, vec, plot, bounds, regions_to_plot)

    return plot, vec
