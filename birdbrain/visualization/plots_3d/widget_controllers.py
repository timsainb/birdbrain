import copy
import numpy as np
from ipywidgets import widgets
import k3d
from IPython.display import display
from tqdm.autonotebook import tqdm
import time

from birdbrain import utils


def widget_controllers(atlas, vec, plot, bounds, regions_to_plot):
    x_slider = widgets.FloatSlider(
        value=0, min=bounds[0], max=bounds[1], step=100, description="medial-lateral:"
    )
    y_slider = widgets.FloatSlider(
        value=0,
        min=bounds[2],
        max=bounds[3],
        step=100,
        description="posterior-anterior:",
    )
    z_slider = widgets.FloatSlider(
        value=0, min=bounds[4], max=bounds[5], step=100, description="ventral-dorsal:"
    )
    region_dropdown = widgets.Dropdown(
        options=list(np.array(regions_to_plot)[:, 0]) + ["y_sinus"],
        value="y_sinus",
        description="Region:",
    )

    def s_x(change):
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

    def s_y(change):
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

    def s_z(change):
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
                atlas.region_vox.loc[change["new"], "coords_vox"],
                atlas.voxel_data.loc["Brain", "affine"],
                atlas.um_mult,
                atlas.y_sinus_um_transform,
            )
            print(change_loc)
            dv_loc, ap_loc, ml_loc = change_loc

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
            y_slider.value = ap_loc
            x_slider.value = dv_loc
            z_slider.value = ml_loc

            # move the rotational axis
            pc = copy.deepcopy(plot.camera)
            plot.camera = pc[:3] + [dv_loc, ap_loc, ml_loc] + pc[-3:]

    x_slider.observe(s_x)
    display(x_slider)
    y_slider.observe(s_y)
    display(y_slider)
    z_slider.observe(s_z)
    display(z_slider)

    region_dropdown.observe(reg_dd)
    display(region_dropdown)


def rotate_plot(plot, hide_vectors=True, n_frames=10, fr=5, nrot=1, radius=8000):
    # screenshots = []
    vector_locs = np.where([type(i) == k3d.objects.Vectors for i in plot.objects])[0]
    # hide vectors
    if hide_vectors:
        for vec_i in vector_locs:
            plot.objects[vec_i].visible = False
        pg = False
        if plot.grid_visible == True:
            pg = True
            plot.grid_visible = False

    # make circle
    camera_loc = copy.deepcopy(plot.camera)
    for rad in tqdm(np.linspace(0, 2 * np.pi * nrot, n_frames * nrot), leave=False):
        # move the camera
        plot.camera = (
            list(
                np.array([3 * radius * np.sin(rad), 3 * radius * np.cos(rad), 0])
                + np.array(camera_loc[3:6])
            )
            + camera_loc[3:6]
            + [0, 0, 1]
        )
        time.sleep(1 / fr)

    # show vectors
    if hide_vectors:
        for vec_i in vector_locs:
            plot.objects[vec_i].visible = True
        if pg:
            plot.grid_visible = True
