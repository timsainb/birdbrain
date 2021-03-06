import os
import matplotlib as mpl
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

from birdbrain.atlas import atlas
from birdbrain.visualization.plotting_2d import plot_transection, plot_2d_coordinates


def test_make_atlas():

    # where to look for the dataset
    dset_dir = "../../data/processed/starling/"
    # create the atlas
    starling_atlas = atlas(
        species="starling",
        dset_dir=dset_dir,
        um_mult=100,
        smoothing=[],  # ['Brain', 'Brainregions']
    )
    # plot 3d
    plot_2d_coordinates(starling_atlas, point_in_um=[0, 0, 0])

    # update y sinus
    updated_y_sinus = [0, 1500, -200]
    starling_atlas.update_y_sinus(updated_y_sinus=updated_y_sinus)

    # plot transections
    plot_transection(starling_atlas, regions_to_plot="Brainregions", n_slice=8)
