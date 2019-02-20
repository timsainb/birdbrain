import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob
from nibabel.affines import apply_affine
from birdbrain import utils
from birdbrain.utils import um_to_vox, vox_to_um, loc_dict_to_list


class atlas(object):
    def __init__(
        self, dset_dir, label_cmap=None, um_mult=100, img_cmap=None, smoothing=[]
    ):

        # path of delineations
        delin_path = os.path.join(os.path.abspath(dset_dir), "delineations/")

        # path of images
        img_files = glob(os.path.join(delin_path, "*.img")) + glob(
            os.path.join(dset_dir, "*.img")
        )
        # path of labels
        text_files = glob(os.path.join(delin_path, "*.txt"))

        # images from each type of scan, as well as transcribed locations ['type_', 'src', 'voxels']
        self.voxel_data = utils.get_voxel_data(img_files)

        # smooth the whole brain because the atlas is a bit noisy
        for img in smoothing:
            self.voxel_data.loc[img, "voxels"] = utils.smooth_voxels(
                self.voxel_data.loc[img, "voxels"], sigma=2
            )

        # transcription labels ['label', 'region', 'type_']
        self.brain_labels = utils.get_brain_labels(text_files)

        # for some reason, units are um/100 in this dataset
        self.um_mult = um_mult

        # make a shadow background for plots
        self.create_shadows()

        # set the colormap for labels
        if label_cmap is None:
            self.label_cmap = label_cmap = plt.cm.tab20
            self.label_cmap.set_under(color=(0, 0, 0, 0))
        else:
            self.label_cmap = label_cmap

        # set the colormap for images
        if img_cmap is None:
            self.img_cmap = img_cmap = plt.cm.Greys
            self.img_cmap.set_under(color=(0, 0, 0, 0))
        else:
            self.img_cmap = img_cmap

        # unless the y sinus is updated from the original location (from the files), there is no transform
        self.y_sinus_um_transform = {
            "medial-lateral": 0,
            "posterior-anterior": 0,
            "ventral-dorsal": 0,
        }

        # voxels for individual regions ['region', 'type_', 'nucleus_ID', 'region_bounds', 'coords_vox', 'voxels']
        self.region_vox = utils.get_region_voxels(self)

        # set the bounds in terms of resoltuion
        self.set_um_bounds()

        # determine where the brain exists (for plotting)
        self.determine_brain_limits()

        print("Atlas created")

    def determine_brain_limits(self):
        """only plot in regions where the brain exists
        """
        # get axis minima and maxima
        axis2_min, axis2_max = np.where(
            self.voxel_data.loc["Brain", "voxels"].sum(axis=0).sum(axis=0) > 0
        )[0][[0, -1]]
        axis0_min, axis0_max = np.where(
            self.voxel_data.loc["Brain", "voxels"].sum(axis=1).sum(axis=1) > 0
        )[0][[0, -1]]
        axis1_min, axis1_max = np.where(
            self.voxel_data.loc["Brain", "voxels"].sum(axis=0).sum(axis=1) > 0
        )[0][[0, -1]]

        # the boundaries for where the brain shows in each dimension
        self.axis_bounds_min = vox_to_um(
            self.voxel_data.loc["Brain", "affine"],
            loc_dict_to_list(
                {
                    "medial-lateral": axis0_max,
                    "posterior-anterior": axis1_min,
                    "ventral-dorsal": axis2_min,
                }
            ),
            self.um_mult,
            self.y_sinus_um_transform,
        )
        self.axis_bounds_max = vox_to_um(
            self.voxel_data.loc["Brain", "affine"],
            loc_dict_to_list(
                {
                    "medial-lateral": axis0_min,
                    "posterior-anterior": axis1_max,
                    "ventral-dorsal": axis2_max,
                }
            ),
            self.um_mult,
            self.y_sinus_um_transform,
        )

        self.axis_bounds_min = {
            key: value - 1000 for key, value in self.axis_bounds_min.items()
        }
        self.axis_bounds_max = {
            key: value + 1000 for key, value in self.axis_bounds_max.items()
        }

        # set extents for what is plotted
        self.saggital_extent_plt = [
            self.axis_bounds_min["posterior-anterior"],
            self.axis_bounds_max["posterior-anterior"],
            self.axis_bounds_min["ventral-dorsal"],
            self.axis_bounds_max["ventral-dorsal"],
        ]

        self.coronal_extent_plt = [
            self.axis_bounds_min["medial-lateral"],
            self.axis_bounds_max["medial-lateral"],
            self.axis_bounds_min["ventral-dorsal"],
            self.axis_bounds_max["ventral-dorsal"],
        ]

        self.transversal_extent_plt = [
            self.axis_bounds_min["medial-lateral"],
            self.axis_bounds_max["medial-lateral"],
            self.axis_bounds_min["posterior-anterior"],
            self.axis_bounds_max["posterior-anterior"],
        ]

        # set the y sinus location in voxels
        # get location of y_sinus in voxels if region_vox exists
        y_sinus_vox = um_to_vox(
            self.voxel_data.loc["Brain", "affine"],
            {"medial-lateral": 0, "posterior-anterior": 0, "ventral-dorsal": 0},
            self.um_mult,
            self.y_sinus_um_transform,
        )

        self.region_vox.loc["y_sinus"] = [
            "y_sinus",
            "y_sinus",
            np.nan,
            np.nan,
            y_sinus_vox,
        ]

    def create_shadows(self):
        """ create backgrounds for visualizing transections
        """
        self.coronal_shadow = (
            np.rot90(np.sum(self.voxel_data.loc["Brain", "voxels"], axis=1) > 0)
            * np.shape(self.voxel_data.loc["Brain", "voxels"])[0]
        )
        self.transversal_shadow = (
            np.rot90(np.sum(self.voxel_data.loc["Brain", "voxels"], axis=2) > 0)
            * np.shape(self.voxel_data.loc["Brain", "voxels"])[1]
        )
        self.sagittal_shadow = (
            np.rot90(np.sum(self.voxel_data.loc["Brain", "voxels"], axis=0) > 0)
            * np.shape(self.voxel_data.loc["Brain", "voxels"])[2]
        )

    def update_y_sinus(self, updated_y_sinus):
        """update y sinus voxel location


        Arguments:
            updated_y_sinus {[list]} -- [list in voxels]
        """
        # update the y_sinus
        self.updated_y_sinus = updated_y_sinus

        # move all of the bounds
        self.lateral_pos, self.posterior, self.ventral = loc_dict_to_list(vox_to_um(
            self.voxel_data.loc["Brain", "affine"],
            [0,0,0],
            self.um_mult,
            self.y_sinus_um_transform,
        ))

        self.lateral_neg, self.anterior, self.dorsal = loc_dict_to_list(vox_to_um(
            self.voxel_data.loc["Brain", "affine"],
            np.array(np.shape(self.voxel_data.loc["Brain", "voxels"])) - 1,
            self.um_mult,
            self.y_sinus_um_transform,
        ))
        
        """self.lateral_pos, self.posterior, self.ventral = (
                                                            apply_affine(self.voxel_data.loc["Brain", "affine"], [0, 0, 0])
                                                            * self.um_mult
                                                        )
                                self.lateral_neg, self.anterior, self.dorsal = (
                                    apply_affine(
                                        self.voxel_data.loc["Brain", "affine"],
                                        np.array(np.shape(self.voxel_data.loc["Brain", "voxels"])) - 1,
                                    )
                                    * self.um_mult
                                                        )"""

        print(self.lateral_pos, self.posterior, self.ventral)

        # determine (in um) how much the y_sinus has moved
        self.y_sinus_um_transform = vox_to_um(
            self.voxel_data.loc["Brain", "affine"],
            utils.loc_dict_to_list(updated_y_sinus),
            self.um_mult,
            self.y_sinus_um_transform,
        )


        

        self.saggital_extent = [
            self.posterior - self.y_sinus_um_transform["posterior-anterior"],
            self.anterior - self.y_sinus_um_transform["posterior-anterior"],
            self.ventral - self.y_sinus_um_transform["ventral-dorsal"],
            self.dorsal - self.y_sinus_um_transform["ventral-dorsal"],
        ]
        self.coronal_extent = [
            self.lateral_neg - self.y_sinus_um_transform["medial-lateral"],
            self.lateral_pos - self.y_sinus_um_transform["medial-lateral"],
            self.ventral - self.y_sinus_um_transform["ventral-dorsal"],
            self.dorsal - self.y_sinus_um_transform["ventral-dorsal"],
        ]
        self.transversal_extent = [
            self.lateral_neg - self.y_sinus_um_transform["medial-lateral"],
            self.lateral_pos - self.y_sinus_um_transform["medial-lateral"],
            self.posterior - self.y_sinus_um_transform["posterior-anterior"],
            self.anterior - self.y_sinus_um_transform["posterior-anterior"],
        ]

        # everytime a UM measurement is called for (vox_to_um),
        # reorient in terms of the updated y sinus
        self.determine_brain_limits()

    def set_um_bounds(self, loc=[0, 0, 0]):
        """ gets the boundaries in um for the brain dataset, for convenience

        """
        lateral_pos, posterior, ventral = (
            apply_affine(self.voxel_data.loc["Brain", "affine"], loc) * self.um_mult
        )
        lateral_neg, anterior, dorsal = (
            apply_affine(
                self.voxel_data.loc["Brain", "affine"],
                np.array(np.shape(self.voxel_data.loc["Brain", "voxels"])) - 1,
            )
            * self.um_mult
        )
        self.saggital_extent = [posterior, anterior, ventral, dorsal]
        self.coronal_extent = [lateral_neg, lateral_pos, ventral, dorsal]
        self.transversal_extent = [lateral_neg, lateral_pos, posterior, anterior]

 