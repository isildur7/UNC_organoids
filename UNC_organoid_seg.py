# Segmenting organoids from baffled MCAM acquisition
# Amey Chaware

# Requirements:
# numpy
# scikit-image
# scipy
# matplotlib
# pandas

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as spx
import skimage
from skimage.color import label2rgb, rgb2gray
from skimage.exposure import equalize_adapthist, rescale_intensity
from skimage.filters import threshold_otsu
from skimage.io import imread, imsave
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, disk, opening

organoid = imread("cam6_0_baffle.bmp") / 255
organoid = np.array(organoid, dtype=np.float32)
# TODO does color help?
organoid = rgb2gray(organoid)

print("Improving Contrast...")
# contrast improvement through histogram eqaulization and intensity rescaling
organoid_aeq = equalize_adapthist(
    organoid, kernel_size=451, nbins=256, clip_limit=0.005
)
p25, p95 = np.percentile(organoid_aeq, (25, 95))
organoid_itn_rescale = rescale_intensity(organoid_aeq, (p25, p95))

print("Thresholding the image...")
# Thresholding
organoid_rescaled_mb = spx.ndimage.median_filter(organoid_itn_rescale, size=51)
organoid_otsu_th = organoid_rescaled_mb < threshold_otsu(
    organoid_rescaled_mb, nbins=256
)

print("Applying Morphology")
# Morphology
footprint = disk(51)
organoid_th_closed = opening(organoid_otsu_th, footprint=footprint)
footprint = disk(11)
organoid_th_closed = binary_dilation(organoid_th_closed, footprint=footprint)

print("Labeling Regions...")
# labeling regions
org_label = label(organoid_th_closed, connectivity=2)
image_label_overlay = label2rgb(org_label, image=organoid_itn_rescale, bg_label=0)
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(image_label_overlay)

for region in regionprops(org_label):
    # take regions with large enough areas
    if region.area >= 1000:
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle(
            (minc, minr),
            maxc - minc,
            maxr - minr,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
        ax.add_patch(rect)

ax.set_axis_off()
plt.tight_layout()
plt.show()

print("Populating Region Properties ...")


# analyzing region properties, can include metrics such as area, perimeter etc
# documentation for regionprops will have a list of all the metrics available
def pixelcount(regionmask):
    return np.sum(regionmask)


print("First five largest regions:")
regions = skimage.measure.regionprops_table(
    org_label, organoid_itn_rescale, extra_properties=(pixelcount,)
)
data = pd.DataFrame(regions)

# sort regions by the number of pixels they occupy
# select the second entry for the organoid
data = data.sort_values(by=["pixelcount"], ascending=False)
print(data.head)
