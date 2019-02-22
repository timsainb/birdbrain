import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from birdbrain.utils import vox_to_um, um_to_vox, get_axis_bounds, inverse_dict


def plt_box_bg(background_dims, ax, extent, box_size=20, mn=0.2, mx=0.3):
    """ plots a box background for showing image transparency
    """
    d0 = background_dims[0] // box_size // 2
    d1 = background_dims[1] // box_size // 2
    bg = np.array(([mn, mx] * (d0) + [mx, mn] * (d0)) * d1).reshape(d0 * 2, d1 * 2)
    ax.matshow(
        bg, cmap=plt.cm.gray, interpolation="nearest", extent=extent, vmin=0, vmax=1
    )


def plot_transection(
    atlas,
    regions_to_plot="Brainregions",
    region_alpha=0.25,
    img_ds="T2",
    n_slice=8,
    zoom=4,
):
    """ plot transections across brain data
    """
    data = atlas.voxel_data.loc[regions_to_plot, "voxels"]
    cmap = atlas.label_cmap

    # load brain data and making sure not to slice regions that don't have any brain
    brain_data = atlas.voxel_data.loc["Brain", "voxels"]
    brain_exists_saggital = np.where(brain_data.sum(axis=1).sum(axis=1) > 0)[0]
    brain_exists_coronal = np.where(brain_data.sum(axis=0).sum(axis=1) > 0)[0]
    brain_exists_transversal = np.where(brain_data.sum(axis=0).sum(axis=0) > 0)[0]

    # load img data and make sure not to plot image where the brain isn't
    img_data = atlas.voxel_data.loc[img_ds, "voxels"]
    img_data[brain_data == 0] = 0

    fig, axs = plt.subplots(3, n_slice, figsize=[n_slice * zoom, 3 * zoom])

    # loop through axis to view
    for brain_exists, axis in [
        [brain_exists_saggital, 0],
        [brain_exists_coronal, 1],
        [brain_exists_transversal, 2],
    ]:
        # loop through slices
        for slc_i, slc_loc in enumerate(
            np.linspace(brain_exists[0], brain_exists[-1], n_slice).astype(int)
        ):

            slc = np.rot90(np.take(data, indices=slc_loc, axis=axis).squeeze())

            # plt_box_bg(
            #    background_dims=np.shape(slc),
            #    ax=axs[axis, slc_i],
            #    box_size=20,
            # )

            img = np.rot90(np.take(img_data, slc_loc, axis=axis).squeeze()).astype(
                "int32"
            )

            if np.sum(img) > 0:
                axs[axis, slc_i].matshow(img ** 0.5, cmap=atlas.img_cmap, vmin=1e-4)

            if np.sum(slc) > 0:
                axs[axis, slc_i].matshow(
                    slc, alpha=region_alpha, cmap=cmap, vmin=0.1, vmax=np.max(slc)
                )
    return fig


def plot_box_bg(ax, extent, box_size_um=2000, mn=0.2, mx=0.3):
    """ 
    mn, mx: darkness of blocks
    """
    # number of boxes based on extent
    d0 = int(np.ceil((extent[1] - extent[0]) / (box_size_um * 2)))
    d1 = int(np.ceil((extent[3] - extent[2]) / (box_size_um * 2)))

    # make grid
    bg = np.vstack(
        [
            np.array(([mn, mx] * (d0) + [mx, mn] * (d0))).reshape(2, d0 * 2)
            for i in range(d1)
        ]
    )

    # compute new extent of d0,d1 boxes to keep um 1000
    new_extent = [
        extent[0],
        extent[0] + (d0 * 2 * box_size_um),
        extent[2],
        extent[2] + (d1 * 2 * box_size_um),
    ]
    # plot grid
    ax.matshow(
        bg, cmap=plt.cm.gray, interpolation="nearest", extent=new_extent, vmin=0, vmax=1
    )


def plot_2d_coordinates(
    atlas,
    point_in_voxels=None,
    point_in_um=None,
    regions_to_plot="Brainregions",
    img_ds="T2",
    region_alpha=0.25,
    line_alpha=0.5,
    zoom=6,
):
    """ Produce a 2d plot of brain data
    """
    # get the point in either um or voxels
    if point_in_um is None:
        point_in_um = np.array(
            vox_to_um(
                point_in_voxels,
                atlas.voxel_data.loc["Brain", "affine"],
                atlas.um_mult,
                atlas.y_sinus_um_transform,
            )
        )
        point_in_voxels = np.array(point_in_voxels)
    if point_in_voxels is None:
        point_in_voxels = np.array(
            um_to_vox(
                point_in_um,
                atlas.voxel_data.loc["Brain", "affine"],
                atlas.um_mult,
                atlas.y_sinus_um_transform,
            )
        )
        point_in_um = np.array(point_in_um)

    # print the point in um
    print({inverse_dict(atlas.axes_dict)[i]: int(point_in_um[i]) for i in range(3)})

    # the type of image to plot
    label_data = atlas.voxel_data.loc[regions_to_plot, "voxels"]
    img_data = atlas.voxel_data.loc[img_ds, "voxels"]
    brain_data = atlas.voxel_data.loc["Brain", "voxels"]

    # set voxels where there is no brain to 0
    img_data[brain_data == 0] = 0

    # get image data
    x_img = img_data[point_in_voxels[0], :, :].squeeze()
    y_img = img_data[:, point_in_voxels[1], :].squeeze()
    z_img = img_data[:, :, point_in_voxels[2]].squeeze()

    # get label data
    x_lab = label_data[point_in_voxels[0], :, :].squeeze()
    y_lab = label_data[:, point_in_voxels[1], :].squeeze()
    z_lab = label_data[:, :, point_in_voxels[2]].squeeze()

    # subset labels dataframe for only the ones appearing here
    unique_labels = np.unique(
        list(x_lab.flatten()) + list(y_lab.flatten()) + list(z_lab.flatten())
    )

    # subset brain_labels dataframe for only the labels shown here
    regions_plotted = atlas.brain_labels[
        (atlas.brain_labels.type_ == regions_to_plot)
        & ([label in unique_labels for label in atlas.brain_labels.label.values])
    ]
    # normalize colors to regions being plotted
    cmin = np.min(regions_plotted.label.values) - 1e-4
    cmax = np.max(regions_plotted.label.values)
    cnorm = matplotlib.colors.Normalize(vmin=cmin, vmax=cmax)
    # set colors specific to labels
    regions_plotted["colors"] = list(
        atlas.label_cmap(cnorm(regions_plotted.label.values))
    )

    # make a dataframe corresponding to each axis (which images belong in which axis)
    ax_list = [
        [0, 0, "anterior-posterior"],
        [0, 1, "medial-lateral"],
        [1, 0, "dorsal-ventral"],
    ]

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(2 * zoom, 2 * zoom))
    for ax0, ax1, axis in ax_list:
        ax = axs[ax0, ax1]
        # get bounds
        shadow = np.rot90(img_data.sum(axis=atlas.axes_dict[axis]) > 0)

        # axis extents
        xlims, ylims = get_axis_bounds(atlas, axis=atlas.axes_dict[axis])

        # plot grid background
        plot_box_bg(ax, np.concatenate([xlims, ylims]))

        # get extent from voxels
        extent = np.concatenate(
            np.array(
                [
                    [atlas.xmin, atlas.xmax],
                    [atlas.ymin, atlas.ymax],
                    [atlas.zmin, atlas.zmax],
                ]
            )[np.arange(3) != atlas.axes_dict[axis]]
        )

        # plot shadow background
        ax.matshow(
            shadow * 255,
            cmap=atlas.img_cmap,
            extent=extent,
            vmin=1e-4,
            vmax=1,
            alpha=0.5,
        )

        # plot image
        img = [x_img, y_img, z_img][atlas.axes_dict[axis]]
        ax.matshow(np.rot90(img ** 0.5), cmap=atlas.img_cmap, extent=extent, vmin=1e-4)
        # plot labels
        labels = [x_lab, y_lab, z_lab][atlas.axes_dict[axis]]
        ax.matshow(
            np.rot90(labels),
            cmap=atlas.label_cmap,
            extent=extent,
            vmin=cmin,
            vmax=cmax,
            alpha=region_alpha,
        )

        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

        # plot line at y sinus
        ax.axhline(0, color=(1, 1, 1, line_alpha), ls="dashed")
        ax.axvline(0, color=(1, 1, 1, line_alpha), ls="dashed")

        # plot line at specified location
        x_um_loc, y_um_loc = point_in_um[np.arange(3) != atlas.axes_dict[axis]]
        ax.axhline(y_um_loc, color=(0, 0, 0, line_alpha), ls="dashed")
        ax.axvline(x_um_loc, color=(0, 0, 0, line_alpha), ls="dashed")

    # label regions
    patches = []
    for idx, row in regions_plotted.iterrows():
        patches.append(mpatches.Patch(color=row.colors, label=row.region))
    axs[1, 1].legend(handles=patches, loc="center", ncol=2)
    plt.tight_layout()
    axs[1, 1].axis("off")

    return fig


from IPython.display import display, clear_output
from ipywidgets import widgets


def widget_controllers_2d(atlas, regions_to_plot="Nuclei", **kwargs):
    def reg_dd(change):
        # move location to the center of that region

        if change["type"] == "change" and change["name"] == "value":

            change_loc = vox_to_um(
                atlas.region_vox.loc[change["new"], "coords_vox"],
                atlas.voxel_data.loc["Brain", "affine"],
                atlas.um_mult,
                atlas.y_sinus_um_transform,
            )
            reset_controls(change_loc)

    def reset_controls(change_loc=[0, 0, 0]):
        x_slider.value, y_slider.value, z_slider.value = change_loc

    def on_button_click(b):
        clear_output()
        plot_2d_coordinates(
            atlas,
            point_in_um=[x_slider.value, y_slider.value, z_slider.value],
            regions_to_plot=regions_to_plot,
            **kwargs
        )
        display(x_slider)
        display(y_slider)
        display(z_slider)
        display(region_dropdown)
        display(button)

    regions = [
        [reg, "Nuclei"]
        for reg in list(
            atlas.brain_labels[
                atlas.brain_labels.type_ == regions_to_plot
            ].region.values
        )
    ]

    region_dropdown = widgets.Dropdown(
        options=list(np.array(regions)[:, 0]) + ["y_sinus"],
        value="y_sinus",
        description="Region:",
    )

    x_slider = widgets.FloatSlider(
        value=(atlas.xmax - atlas.ymin) / 2,
        min=atlas.xmin,
        max=atlas.xmax,
        step=100,
        description="medial-lateral:",
    )
    y_slider = widgets.FloatSlider(
        value=(atlas.ymax - atlas.ymin) / 2,
        min=atlas.ymin,
        max=atlas.ymax,
        step=100,
        description="posterior-anterior:",
    )
    z_slider = widgets.FloatSlider(
        value=(atlas.zmax - atlas.zmin) / 2,
        min=atlas.zmin,
        max=atlas.zmax,
        step=100,
        description="ventral-dorsal:",
    )

    button = widgets.Button(description="Generate graph")

    display(x_slider)
    display(y_slider)
    display(z_slider)
    display(region_dropdown)
    display(button)

    reset_controls()

    region_dropdown.observe(reg_dd)
    button.on_click(on_button_click)
