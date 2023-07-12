import os

import cupy as cp
import cupyx.scipy as spx
import netCDF4 as nc
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
from cucim.skimage.exposure import rescale_intensity, equalize_adapthist
from cucim.skimage.feature import canny
from cucim.skimage.segmentation import chan_vese
from cucim.skimage.morphology import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_opening,
    disk,
)
from skimage.io import imsave
from skimage.measure import find_contours, label, regionprops_table

# NOTES:
# Since I was using contour based methods anyway, I tried using Chan-Vese
# which is a active contour based segmentation method. It alone does not
# work perfectly, underestimating the area. Combining it with my previous
# contour based methods seems to work nicely. This is a test.


DATA_PATH = (
    "/home/col/AC_Data/UNC_data/20230621/green-P18-day12-2023-06-21/left_stack.nc"
)
SAVE_PATH = "/home/col/AC-Code/UNC_organoids/"
MAX_X = 8
MAX_Y = 6


def segment_organoid_boundary_from_projection_image(image, channel):
    # change to cp array and take the min projection over color channels
    organoid = cp.array(image, dtype=cp.float32) / 255
    organoid = spx.ndimage.median_filter(organoid, size=25)
    organoid = equalize_adapthist(
        organoid, nbins=256, clip_limit=0.01, kernel_size=organoid.shape[0] // 4
    )

    # if type == "max":
    #     organoid_min = cp.max(organoid, axis=2)
    # elif type == "min":
    #     organoid_min = cp.min(organoid, axis=2)

    # rescale the intensity setting the 0 at the 25th percentile
    # and 1 at the 75th percentile
    p5, p95 = np.percentile(organoid.get(), (5, 95))
    organoid_rs = rescale_intensity(organoid, (p5, p95))

    # median blur the rescaled image
    organoid_mb = spx.ndimage.median_filter(organoid_rs, size=25)

    # apply Chan-Vese and get the region boundary of that segmentation
    cv = chan_vese(
        organoid_mb,
        mu=0.4,
        lambda1=1,
        lambda2=0.99,
        tol=0.001,
        max_num_iter=500,
        dt=0.5,
        init_level_set="checkerboard",
        extended_output=False,
    )
    cv = ~cv
    footprint = disk(25)
    cv = binary_opening(cv, footprint=footprint)
    cv_label = label(cv.get(), connectivity=2)
    cv[cv_label == 1] = False
    cvmask = canny(cv, sigma=1, use_quantiles=True, mode="mirror")

    # get the edges directly from the image
    edges_rs = canny(
        organoid_mb, sigma=25, low_threshold=0.2, high_threshold=0.5, use_quantiles=True
    )

    # add them both to get the best of both worlds (hopefully)
    th = cvmask + edges_rs
    th[th > 1] = 1

    # apply dilation for a round of contour detection as before
    footprint = disk(1)
    th = binary_dilation(th, footprint=footprint)

    # Hopefully, after the step above, the organoid is bound by a closed loop
    # Applying contour detection will identify that path (and other things)
    # we filter out contours of less than 4000 pixels
    # find_contours is not in cucim at the time of writing

    cfin = th.get()
    contours = find_contours(cfin, 0.8)

    # Create an empty image to store the masked array
    r_mask = np.zeros_like(cfin, dtype="bool")

    # Create a contour image by using the contour coordinates rounded to their
    # nearest integer value
    for contour in contours:
        # reject small or open contours
        if len(contour) > 2000 and np.all(contour[0] == contour[-1]):
            r_mask[
                np.round(contour[:, 0]).astype("int"),
                np.round(contour[:, 1]).astype("int"),
            ] = 1

    footprint = disk(11)
    r_mask = cp.array(r_mask)
    r_mask = binary_closing(r_mask, footprint=footprint)
    return r_mask


def fill_boundary_hole(r_mask):
    # Fill in the hole created by the contour boundary
    # This is our segmentation
    r_mask = ndimage.binary_fill_holes(r_mask.get())
    footprint = disk(19)
    r_mask = cp.array(r_mask)
    # opening to get rid of small friovolous detections
    r_mask = binary_opening(r_mask, footprint=footprint)
    footprint = disk(5)
    r_mask = binary_erosion(r_mask, footprint=footprint)
    return r_mask.get()


def get_properties_of_the_segmented_area(segmentation, organoid):
    """Given the obtained segmentation and the original file, label and measure
    the region properties and save in a csv file"""
    # TODO save a csv file
    org_label = label(segmentation, connectivity=2)
    regions = regionprops_table(
        org_label, organoid, properties=("label", "bbox", "area")
    )
    data = pd.DataFrame(regions)
    data = data.sort_values(by=["area"], ascending=False)
    try:
        return data["area"].iloc[0]
    except (KeyError, IndexError):
        pass


def make_product_image_from_stack(color_stack):
    """Given a z-stack, take the product of the min and max
    projection and return it.
    """

    min_stack = np.min(color_stack, axis=0) / 255
    max_stack = np.max(color_stack, axis=0) / 255
    product = min_stack * max_stack
    return (product * 255).astype(np.uint8)


if __name__ == "__main__":
    seg_save_folder = SAVE_PATH + "saved_masks/seg"
    boundary_save_folder = SAVE_PATH + "saved_masks/boundary"

    if not os.path.exists(seg_save_folder):
        os.makedirs(seg_save_folder)

    if not os.path.exists(boundary_save_folder):
        os.makedirs(boundary_save_folder)

    data = nc.Dataset(DATA_PATH)
    cam_data = data.variables["images"]

    good_ones = 0
    not_good = []
    for x in range(MAX_X):
        for y in range(MAX_Y):
            product = make_product_image_from_stack(cam_data[:, x, y, :, :])
            mask = segment_organoid_boundary_from_projection_image(product, 0)

            imsave(
                boundary_save_folder + f"/mask{x}_{y}.png",
                (mask * 255).get().astype(np.uint8),
                check_contrast=False,
            )

            mask = fill_boundary_hole(mask)
            imsave(
                seg_save_folder + f"/mask{x}_{y}.png",
                (mask * 255).astype(np.uint8),
                check_contrast=False,
            )

            prop = get_properties_of_the_segmented_area(mask, product)
            print(f"cam{x}_{y}: {prop}")
            if prop is not None and 2000000 >= prop >= 500000 :
                good_ones += 1
            else:
                not_good.append(f"cam{x}_{y}")

print(f"Potentially {good_ones}/48 good regions")
print("not good list: ", not_good)
