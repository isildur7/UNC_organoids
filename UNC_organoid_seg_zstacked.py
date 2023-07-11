import os

import cupy as cp
import cupyx.scipy as spx
import cv2
import netCDF4 as nc
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
from cucim.skimage.exposure import rescale_intensity
from cucim.skimage.feature import multiscale_basic_features
from cucim.skimage.morphology import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_opening,
    disk,
    remove_small_objects,
)
from cucim.skimage.filters import threshold_otsu
from skimage.io import imsave
from skimage.measure import find_contours, label, regionprops_table

# NOTES:
# CAM1_4 and CAM6_5 for green-p18-day12 have dust and other artefacts
# Dilation is disk of rad 3
# 1. Product Stack gave 35/47 accurate regions
# 2. Max Stack gave 26/47 accurate regions
# 3. Max Stack + dilation gave 40/47 accurate regions
# 4. Product Stack + dilation gave 44/47 accurate regions
# 5. Product Stack + dilation + texture 47/47 accurate regions
# 6. Product Stack + smaller dilation gave 42/47 accurate regions
# 7. Product Stack + small dilation + texture + other 43/47 accurate regions
# BUT, extraneous areas are detected

DATA_PATH = (
    "/home/col/AC_Data/UNC_data/20230621/green-P18-day12-2023-06-21/left_stack.nc"
)
SAVE_PATH = "/home/col/AC-Code/UNC_organoids/"
MAX_X = 8
MAX_Y = 6


def segment_organoid_boundary_from_projection_image(image, channel):
    # change to cp array and take the min projection over color channels
    organoid = cp.array(image, dtype=cp.float32) / 255

    # if type == "max":
    #     organoid_min = cp.max(organoid, axis=2)
    # elif type == "min":
    #     organoid_min = cp.min(organoid, axis=2)

    # rescale the intensity setting the 0 at the 25th percentile
    # and 1 at the 75th percentile
    p25, p75 = np.percentile(organoid.get(), (25, 75))
    organoid_rs = rescale_intensity(organoid, (p25, p75))

    # median blur the rescaled image
    organoid_mb = spx.ndimage.median_filter(organoid_rs, size=25)

    # get edges from multiscale features
    edges_rs = multiscale_basic_features(
        organoid_mb, intensity=False, edges=True, texture=False
    )
    # sum it all up to get an indication of where edges are
    # edges_rs = cp.sum(edges_rs, axis=2)
    edges_rs = cp.average(edges_rs, weights=[1, 1.4, 1.8, 2.2, 2.6, 3], axis=2)

    # threshold features to get the edges out
    edges_th = edges_rs > threshold_otsu(edges_rs)
    # dilate a little to be able to detect contours
    # this dilation is referenced in the notes
    footprint = disk(1)
    edges_th = binary_dilation(edges_th, footprint=footprint)
    edges_th = remove_small_objects(edges_th, min_size=2000)

    # Hopefully, after the step above, the organoid is bound by a closed loop
    # Applying contour detection will identify that path (and other things)
    # we filter out contours of less than 4000 pixels
    # find_contours is not in cucim at the time of writing

    cfin = edges_th.get()
    contours = find_contours(cfin, 0.8)

    # Create an empty image to store the masked array
    r_mask = np.zeros_like(cfin, dtype="bool")

    # Create a contour image by using the contour coordinates rounded to their
    # nearest integer value
    for contour in contours:
        if len(contour) > 4000:
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
    except KeyError:
        pass


def make_color_z_stack_from_mcam_nc_data(nc_image_data, x, y):
    """Pass in the 'images' key from the MCAM nc dataset and the
    camera x, y coordinates

    Returns:
    RGB z-stack
    """
    stack = nc_image_data[:, x, y]

    color_stack = []
    for i in range(stack.shape[0]):
        rgb = cv2.cvtColor(stack[i], cv2.COLOR_BayerGBRG2BGR)
        color_stack.append(rgb)

    return stack  # np.stack(color_stack)


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
            if prop is not None and prop >= 500000:
                good_ones += 1
            else:
                not_good.append(f"cam{x}_{y}")

print(f"Potentially {good_ones}/48 good regions")
print("not good list: ", not_good)
