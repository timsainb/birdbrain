import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from birdbrain import utils



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
    for brain_exists, extent, axis in [
        [brain_exists_saggital, atlas.saggital_extent, 0],
        [brain_exists_coronal, atlas.coronal_extent, 1],
        [brain_exists_transversal, atlas.transversal_extent, 2],
    ]:
        # loop through slices
        for slc_i, slc_loc in enumerate(
            np.linspace(brain_exists[0], brain_exists[-1], n_slice).astype(int)
        ):

            slc = np.rot90(np.take(data, indices=slc_loc, axis=axis).squeeze())

            plt_box_bg(
                background_dims=np.shape(slc),
                ax=axs[axis, slc_i],
                extent=extent,
                box_size=20,
            )

            img = np.rot90(np.take(img_data, slc_loc, axis=axis).squeeze()).astype(
                "int32"
            )

            if np.sum(img) > 0:
                axs[axis, slc_i].matshow(
                    img ** 0.5, cmap=atlas.img_cmap, extent=extent, vmin=1e-4
                )

            if np.sum(slc) > 0:
                axs[axis, slc_i].matshow(
                    slc,
                    alpha=region_alpha,
                    cmap=cmap,
                    vmin=0.1,
                    vmax=np.max(slc),
                    extent=extent,
                )
    return fig


def plot_location_coordinates(
    atlas,
    point,
    regions_to_plot,
    points_to_plot=None,
    img_ds="T2",
    region_alpha=0.25,
    zoom=3,
):

    # the type of image to plot
    label_data = atlas.voxel_data.loc[regions_to_plot, "voxels"]

    img_data = atlas.voxel_data.loc[img_ds, "voxels"]

    brain_data = atlas.voxel_data.loc["Brain", "voxels"]

    # set voxels where there is no brain to 0
    img_data[brain_data == 0] = 0

    # get the image data
    sagittal_img = np.rot90(img_data[point["medial-lateral"], :, :].squeeze()).astype(
        "int32"
    )
    coronal_img = np.rot90(
        img_data[:, point["posterior-anterior"], :].squeeze()
    ).astype("int32")
    transversal_img = np.rot90(
        img_data[:, :, point["ventral-dorsal"]].squeeze()
    ).astype("int32")

    # get labels data
    sagittal_labels = np.rot90(
        label_data[point["medial-lateral"], :, :].squeeze()
    ).astype("int32")
    coronal_labels = np.rot90(
        label_data[:, point["posterior-anterior"], :].squeeze()
    ).astype("int32")
    transversal_labels = np.rot90(
        label_data[:, :, point["ventral-dorsal"]].squeeze()
    ).astype("int32")

    # subset labelsu
    unique_labels = np.unique(
        list(sagittal_labels.flatten())
        + list(coronal_labels.flatten())
        + list(transversal_labels.flatten())
    )

    regions_plotted = atlas.brain_labels[
        (atlas.brain_labels.type_ == regions_to_plot)
        & ([label in unique_labels for label in atlas.brain_labels.label.values])
    ]
    cmin = np.min(regions_plotted.label.values) - 1e-4
    cmax = np.max(regions_plotted.label.values)

    # for normalizing colors to range
    cnorm = matplotlib.colors.Normalize(vmin=cmin, vmax=cmax)
    regions_plotted["colors"] = list(
        atlas.label_cmap(cnorm(regions_plotted.label.values))
    )

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(2 * zoom, 2 * zoom))

    axlist = [
        [
            sagittal_img,
            0,
            1,
            atlas.sagittal_shadow,
            atlas.saggital_extent,
            "saggital",
            sagittal_labels,
            atlas.saggital_extent_plt,
        ],
        [
            coronal_img,
            0,
            0,
            atlas.coronal_shadow,
            atlas.coronal_extent,
            "coronal",
            coronal_labels,
            atlas.coronal_extent_plt,
        ],
        [
            transversal_img,
            1,
            0,
            atlas.transversal_shadow,
            atlas.transversal_extent,
            "transversal",
            transversal_labels,
            atlas.transversal_extent_plt,
        ],
    ]

    for img, ax0, ax1, shadow, extent, title, labels, extent_plt in axlist:
        plt_box_bg(
            background_dims=np.shape(img),
            ax=axs[ax0, ax1],
            extent=extent_plt,
            box_size=20,
        )
        axs[ax0, ax1].matshow(
            shadow * 255,
            cmap=atlas.img_cmap,
            extent=extent,
            vmin=1e-4,
            vmax=1,
            alpha=0.5,
        )
        axs[ax0, ax1].matshow(img ** 0.5, cmap=atlas.img_cmap, extent=extent, vmin=1e-4)
        axs[ax0, ax1].matshow(
            labels,
            cmap=atlas.label_cmap,
            extent=extent,
            vmin=cmin,
            vmax=cmax,
            alpha=region_alpha,
        )
        axs[ax0, ax1].set_title(title)

    axs[1, 0].set_title("transversal")
    axs[1, 0].set_xlabel("micrometers")
    axs[1, 0].set_ylabel("micrometers")
    axs[0, 0].set_ylabel("micrometers")

    # if there are points to plot, plot them in all 3 axes
    if points_to_plot is not None:
        points_to_plot = np.array(points_to_plot)
        kws = {"color": "red", "s": 10}
        axs[0, 1].scatter(points_to_plot[:, 1], points_to_plot[:, 2], **kws)
        axs[0, 0].scatter(points_to_plot[:, 0], points_to_plot[:, 2], **kws)
        axs[1, 0].scatter(points_to_plot[:, 0], points_to_plot[:, 1], **kws)

    point_list = utils.loc_dict_to_list(point)
    coords = utils.vox_to_um(
        atlas.voxel_data.loc[regions_to_plot, "affine"],
        point_list,
        atlas.um_mult,
        atlas.y_sinus_um_transform,
    )

    # this seems like a bug... I'm not sure why this is getting flipped
    coords["medial-lateral"] *= -1

    print(
        " | ".join([key + ": " + str(round(val)) + "um" for key, val in coords.items()])
    )

    # set limits
    axs[0, 1].set_xlim(
        [
            atlas.axis_bounds_min["posterior-anterior"],
            atlas.axis_bounds_max["posterior-anterior"],
        ]
    )
    axs[0, 1].set_ylim(
        [
            atlas.axis_bounds_min["ventral-dorsal"],
            atlas.axis_bounds_max["ventral-dorsal"],
        ]
    )

    axs[0, 0].set_xlim(
        [
            atlas.axis_bounds_min["medial-lateral"],
            atlas.axis_bounds_max["medial-lateral"],
        ]
    )
    axs[0, 0].set_ylim(
        [
            atlas.axis_bounds_min["ventral-dorsal"],
            atlas.axis_bounds_max["ventral-dorsal"],
        ]
    )

    axs[1, 0].set_xlim(
        [
            atlas.axis_bounds_min["medial-lateral"],
            atlas.axis_bounds_max["medial-lateral"],
        ]
    )
    axs[1, 0].set_ylim(
        [
            atlas.axis_bounds_min["posterior-anterior"],
            atlas.axis_bounds_max["posterior-anterior"],
        ]
    )

    # draw the line through
    line_alpha = 0.5
    axs[0, 1].axvline(
        coords["posterior-anterior"], color=(0, 0, 0, line_alpha), ls="dashed"
    )
    axs[0, 1].axhline(
        coords["ventral-dorsal"], color=(0, 0, 0, line_alpha), ls="dashed"
    )

    axs[0, 0].axvline(
        coords["medial-lateral"], color=(0, 0, 0, line_alpha), ls="dashed"
    )
    axs[0, 0].axhline(
        coords["ventral-dorsal"], color=(0, 0, 0, line_alpha), ls="dashed"
    )

    axs[1, 0].axvline(
        coords["medial-lateral"], color=(0, 0, 0, line_alpha), ls="dashed"
    )
    axs[1, 0].axhline(
        coords["posterior-anterior"], color=(0, 0, 0, line_alpha), ls="dashed"
    )

    axs[1, 1].axis("off")

    # draw dashed midline
    for ax in axs.flatten()[:-1]:
        ax.axvline(0, color=(1, 1, 1, 0.25), ls="dashed")
        ax.axhline(0, color=(1, 1, 1, 0.25), ls="dashed")
        ax.set_facecolor("#333333")

    # label regions
    patches = []
    for idx, row in regions_plotted.iterrows():
        patches.append(mpatches.Patch(color=row.colors, label=row.region))
    axs[1, 1].legend(handles=patches, loc="center", ncol=2)
    plt.tight_layout()

    plt.show()
    return fig
