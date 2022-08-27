#%% Imports
from skimage import color, io, measure, img_as_ubyte
from skimage.measure import profile_line
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom
# %% Exercise 2

# Directory containing data and images
in_dir = "data/"
# X-ray image
im_name = "metacarpals.png"
# Read the image.
# Here the directory and the image name is concatenated
# by "+" to give the full path to the image.
im_org = io.imread(in_dir + im_name)

# dimensions
print(im_org.shape)

# type
print(im_org.dtype)

# %% Exercise 3

io.imshow(im_org)
plt.title('Metacarpal image')
io.show()

# %% Exercise 4

io.imshow(im_org, cmap="jet")
plt.title('Metacarpal image (with colormap)')
io.show()

# Exercise 5
io.imshow(im_org, vmin=20, vmax=170)
plt.title('Metacarpal image (with gray level scaling)')
io.show()

# %% Exercise 6

io.imshow(im_org, vmin=20, vmax=170)
plt.title('Metacarpal image (with gray level scaling)')
io.show()

# %% Exercise 7

plt.hist(im_org.ravel(), bins=256)
plt.title('Image histogram')
io.show()

h = plt.hist(im_org.ravel(), bins=256)

bin_no = 100
count = h[0][bin_no]
print(f"There are {count} pixel values in bin {bin_no}")

bin_left = h[1][bin_no]
bin_right = h[1][bin_no + 1]
print(f"Bin edges: {bin_left} to {bin_right}")

y, x, _ = plt.hist(im_org.ravel(), bins=256)

# %% Exercise 8

hmaxy = max(y)
hmaxx = max(x)
print(f"max y: {hmaxy}, max x: {hmaxx}")

# Pixel values and image coordinate systems

# %% Exercise 9

# Get pixel value at specific point
r = 110
c = 90
im_val = im_org[r, c]
print(f"The pixel value at (r,c) = ({r}, {c}) is: {im_val}")

# %% Exercise 10
im_org[:30] = 0
io.imshow(im_org)
io.show()
# Set the first 30 colums to pixel value 0

# %% Exercise 11
# todo NOT WORKING
mask = im_org > 150
io.imshow(mask)
io.show()

im_org[mask] = 255
io.imshow(im_org)
io.show()

# Color images

# %% Exercise 12

# X-ray image
im_name = "ardeche.jpg"
# Read the image.
# Here the directory and the image name is concatenated
# by "+" to give the full path to the image.
im_org = io.imread(in_dir + im_name)

# dimensions
print(im_org.shape)

print(im_org.dtype)

# Number of rows
rows = im_org.shape[0]
# Number of columns
columns = im_org.shape[1]

print(f"Rows: {rows}, Columns: {columns}")

# %% Exercise 13

r = 110
c = 90
im_org[r, c] = [255, 0, 0]

io.imshow(im_org)
plt.title('Ardeche image')
io.show()

r_2 = rows // 2

io.imshow(im_org)
plt.title('Metacarpal image Green')
io.show()

im_org[:r_2] = [0, 255, 0]
io.imshow(im_org)
plt.title('Metacarpal image Green')
io.show()

# Working with your own image
# %% Exercise 14

im_name = "EdS.jpg"
im_org = io.imread(in_dir + im_name)
image_rescaled = rescale(im_org, 0.25, anti_aliasing=True, multichannel=True)

# dimensions
print(im_org.shape)

print(im_org.dtype)

# Number of rows
rows = im_org.shape[0]
# Number of columns
columns = im_org.shape[1]

io.imshow(im_org)
plt.title('Ed S')
io.show()

image_resized = resize(im_org, (im_org.shape[0] // 4, im_org.shape[1] // 6), anti_aliasing=True)

# Number of rows
rows_resized = image_resized.shape[0]
# Number of columns
columns_resized = image_resized.shape[1]

print(f"Rows: {rows_resized}, Columns: {columns_resized}")

io.imshow(im_org)
plt.title('Ed S')
io.show()


# %% Exercise 15
def resize_image(image):
    new_image = resize(image, (image.shape[0] // 4, 400), anti_aliasing=True)
    # Number of rows
    rows_2 = new_image.shape[0]
    # Number of columns
    cols = new_image.shape[1]
    print(f"Rows: {rows_2}, Columns: {cols}")
    return new_image


new_img = resize_image(im_org)

# %% Exercise 16
im_gray = color.rgb2gray(im_org)
io.imshow(im_gray)
plt.title('Ed S Gray')
io.show()

im_byte = img_as_ubyte(im_gray)
io.imshow(im_byte)
plt.title('Ed S Gray')
io.show()

# %% Exercise 17

plt.hist(im_org.ravel(), bins=256)
plt.title('Image histogram')
io.show()

h = plt.hist(im_org.ravel(), bins=256)

bin_no = 100
count = h[0][bin_no]
print(f"There are {count} pixel values in bin {bin_no}")

bin_left = h[1][bin_no]
bin_right = h[1][bin_no + 1]
print(f"Bin edges: {bin_left} to {bin_right}")

y, x, _ = plt.hist(im_org.ravel(), bins=256)

# Color channels #todo NOT WORKING
# %% Exercise 19

im_name = "DTUSign1.jpg"
im_org = io.imread(in_dir + im_name)

# dimensions
print(im_org.shape)

io.imshow(im_org)
plt.title('DTU sign')
io.show()

# %% Visualize red color component
r_comp = im_org[:, :, 0]
io.imshow(r_comp)
plt.title('DTU sign image (Red)')
io.show()

# %% Visualize green color component
g_comp = im_org[:, 0, :]
io.imshow(g_comp)
plt.title('DTU sign image (Green)')
io.show()
# %% Visualize blue color component

b_comp = im_org[0, :, :]
io.imshow(b_comp)
plt.title('DTU sign image (Blue)')
io.show()

## Simple Image Manipulations
# %% Exercise 20
im_name = "DTUSign1.jpg"
im_org = io.imread(in_dir + im_name)

# dimensions
print(im_org.shape)

io.imshow(im_org)
plt.title('DTU sign')
io.show()
im_org[500:1000, 800:1500, :] = 0
io.imshow(im_org)
plt.title('DTU sign')
io.show()

io.imsave('data/DTUSign1-marked.png', im_org)

# %% Exercise 21
# TODO find red area??

# %% Exercise 22
# TODO
im_name = "metacarpals.png"
im_org = io.imread(in_dir + im_name)

io.imshow(im_org)
plt.title('metacarpals')
io.show()

im_org = color.rgb2gray(im_org)

io.imshow(im_org)
plt.title('metacarpals')
io.show()

# Advanced Image Visualisation
# %% Exercise 23
##  grey-level profile
im_name = "metacarpals.png"
im_org = io.imread(in_dir + im_name)

io.imshow(im_org)
plt.title('metacarpals')
io.show()

p = profile_line(im_org, (342, 77), (320, 160))
plt.plot(p)
plt.ylabel('Intensity')
plt.xlabel('Distance along line')
plt.show()

#%% Exercise 24
## Landscape
im_name = "road.png"
im_org = io.imread(in_dir + im_name)
im_gray = color.rgb2gray(im_org)
ll = 200
im_crop = im_gray[40:40 + ll, 150:150 + ll]
xx, yy = np.mgrid[0:im_crop.shape[0], 0:im_crop.shape[1]]
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(xx, yy, im_crop, rstride=1, cstride=1, cmap=plt.cm.jet, linewidth=0)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

## DICOM images
#%% Exercise 25
im_name = "1-442.dcm"
ds = dicom.dcmread(in_dir + im_name)
print(ds)

#%% Exercise 26
im = ds.pixel_array

print(im.shape)

#%% Exercise 27
io.imshow(im, vmin=-500, vmax=500, cmap='gray')
io.show()