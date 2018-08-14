# import all the necessary libraries
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import piexif
import utm
from PIL import Image

from classifier import classifier

# Kernel for morphological operations
kernel = np.ones((3, 3), np.uint8)

# Counter for counting the true-positives
positives = 0

# Object for classifier class
obj = classifier()


# Function to read image exif data and rotate it
def rotate_jpeg(filename):

    image = Image.open(filename)
    if "exif" in image.info:
        exif_dict = piexif.load(image.info["exif"])

        if piexif.ImageIFD.Orientation in exif_dict["0th"]:
            orientation = exif_dict["0th"].pop(piexif.ImageIFD.Orientation)
            exif_bytes = piexif.dump(exif_dict)

            if orientation == 2:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 3:
                image = image.rotate(180)
            elif orientation == 4:
                image = image.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 5:
                image = image.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 6:
                image = image.rotate(-90, expand=True)
            elif orientation == 7:
                image = image.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
            image.save(filename, exif=exif_bytes)


# Function to check whether a bounding box contains a point in it
def rect_contains(rect, pt):
    logic = rect[0] < pt[0] < rect[0] + rect[2] and rect[1] < pt[1] < rect[1] + rect[3]
    return logic


# Function to do color thresholding in an RGB-colorspace with the threshold value of (rgb_t, rgb_t, rgb_t)
def rgb_threshold(rgb, rgb_t=180):
    low_rgb = np.array([rgb_t, rgb_t, rgb_t])
    high_rgb = np.array([255, 255, 255])
    black_white = cv2.inRange(rgb, low_rgb, high_rgb)
    return black_white


# Function to do thresholding in Differential-RGB-colorspace with threshold value of (drgb_t, drgb_t, drgb_t)
# This process takes lot of time and I believe improvements could be made by using PCA
def drgb_threshold(img, drgb_t=30):
    blank_drgb = np.zeros(img.shape, np.uint8)
    h, w, c = img.shape
    for i in range(h):
        for j in range(w):
            b = int(img[i, j, 0])
            g = int(img[i, j, 1])
            r = int(img[i, j, 2])
            blank_drgb[i, j, 0] = abs(b - r)
            blank_drgb[i, j, 1] = abs(g - b)
            blank_drgb[i, j, 2] = abs(r - g)
    # windows("drgb")
    # cv2.imshow("drgb", blank)
    # cv2.waitKey(0)
    lower = np.array([0, 0, 0])
    higher = np.array([drgb_t, drgb_t, drgb_t])
    mask = cv2.inRange(blank_drgb, lower, higher)
    return mask


# To perform morphological operations on a black-white image
def morphology(black_white):
    closing = cv2.morphologyEx(black_white, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    return opening


# To extract contours from black_white-image
def extract_contours(black_white):
    black_white, contours, h = cv2.findContours(black_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


# Checks for convexity of contours and append verified contours to the list
# Replacement needed for "0.01 * cv2.arcLength(contour, True)", so as to enforce contour-convexity check more rigidly
def convexity(contours):
    """
    Checks for convexity for a list of contours, and appends it to the new list if they are concave.
    .. note::
        The argument that specifies approximation, i.e. "0.01 * cv2.arcLength(contour, True)" could be changed
        based on the requirement.

    :param contours: List of contours to be checked for convexity
    :type contours: list
    :return: New list of contours that are concave
    """
    concave = []
    approx_contours = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        if ~cv2.isContourConvex(approx):
            concave.append(contour)
            approx_contours.append(approx)
    return concave


def windows(name):
    """
    Creates a re-sizable Image window with a given name

    :param str name: Name of the window to be created
    :return: None
    """
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)


# Checks for contour area and append contours to a list
# The upper limits and the lower limits are hard-coded and could be improvised by geo-information
def check_contour_area(contours):
    area = []
    for contour in contours:
        if 0 <= cv2.contourArea(contour) <= 850:
            area.append(contour)
    return area


# Checks for bounding-box area and aspect ratio
# The limits of area could be improvised from Geo-Information
# The tolerance of +- 10 is not efficient enough to get the whole image correctly, improvements are needed
def box_area(contours, img):
    bbox = []
    height, width, channels = img.shape
    for contour in contours:
        x, y, h, w = cv2.boundingRect(contour)
        if 50 <= h * w <= 1500 and (abs(h - w) <= max(h / 2, w / 2)):
            bbox.append((max(x - 10, 0), max(y - 10, 0), min(h + 20, height - y + 10), min(w + 20, width - x + 10)))
    return bbox


# Generates an RGB-color ROIs from a list of bounding boxes
def extract_roi(rgb, bbox):
    rois = []
    for x, y, w, h in bbox:
        ROI = rgb[y:y + h - 1, x:x + h - 1]
        rois.append(ROI)
    return rois, bbox


# Extracts edges from the ROI by removing false edge pixels
def extract_edges(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray, (3, 3))
    h, w, c = roi.shape
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(blur, ret / 3, ret)

    # The value of 200, could be changed, automatically, using geo information,
    # by taking into account the perimeter of the GCP in the image
    while cv2.countNonZero(edges) <= 200 and ret >= 150:
        ret = ret - 5
        edges = cv2.Canny(blur, ret / 3, ret)

    thresh1 = rgb_threshold(roi, 160)
    thresh1 = morphology(thresh1)
    thresh2 = drgb_threshold(roi)
    thresh2 = morphology(thresh2)
    thresh = cv2.bitwise_and(thresh1, thresh2)

    contours = extract_contours(thresh)
    contours = check_contour_area(contours)
    contours = convexity(contours)
    bbox = box_area(contours, roi)
    blank = np.zeros(edges.shape, np.uint8)
    if len(bbox) > 1:
        return blank

    idx = 0
    max_area = 0
    index = -1
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            index = idx
            max_area = area
        idx = idx + 1
    points = []
    if index == -1:
        return np.zeros(edges.shape, np.uint8)

    for i in contours[index]:
        points.append(i[0])

    blank = np.zeros((h, w), np.uint8)
    cv2.fillPoly(blank, np.int32([points]), 255)
    blank = cv2.dilate(blank, kernel, 2)
    new_edges = cv2.bitwise_and(edges, blank)

    for i in range(h):
        for j in range(w):
            if edges[i][j]:
                dist = cv2.pointPolygonTest(contours[index], (j, i), True)
                if abs(dist) <= 3:
                    new_edges[i][j] = 255

    # print max_area,
    return new_edges


# Calculates differential data for edge vector calculation
def quiver_data(roi):
    height, width, channels = roi.shape
    edges = extract_edges(roi)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    xs = []
    ys = []
    dxs = []
    dys = []
    for i in range(height):
        for j in range(width):
            if edges[i][j]:
                xs.append(j)
                ys.append(i)
                dxs.append(scharr_x[i][j])
                dys.append(scharr_y[i][j])
    return xs, ys, dxs, dys, edges


# Calculates orientation of edge
def angles(dxs, dys):
    angles = []
    for (dx, dy) in zip(dxs, dys):
        if dx > 0 and dy >= 0:
            angle = np.arctan(dy / dx) * 180 / np.pi
        elif dx == 0 and dy > 0:
            angle = 90
        elif dx < 0 <= dy:
            angle = 180 + np.arctan(dy / dx) * 180 / np.pi
        elif dx < 0 and dy < 0:
            angle = 180 + np.arctan(dy / dx) * 180 / np.pi
        elif dx == 0 and dy < 0:
            angle = 270
        elif dx > 0 >= dy:
            angle = 360 + np.arctan(dy / dx) * 180 / np.pi
        else:
            continue
        angles.append(angle)
    return angles


# Function to create a resizable-windows for display purpose
def windows(name):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)


# Function to convert from degrees to decimal
def degree(value):
    d0 = value[0][0]
    d1 = value[0][1]
    d = float(d0) / float(d1)
    m0 = value[1][0]
    m1 = value[1][1]
    m = float(m0) / float(m1)
    s0 = value[2][0]
    s1 = value[2][1]
    s = float(s0) / float(s1)
    return d + (m / 60.0) + (s / 3600.0)


# Returns an image list from a GCP-cordinate within vicinity of 100m
def get_img_list(cordinates):
    x, y, z, zl = utm.from_latlon(cordinates[0], cordinates[1])
    img_list = []
    path = "/home/aman/Desktop/SkyLark Drones/Cleanmax_Flight_M1_F1.2/Positives/"
    file_names = os.listdir(path)
    for file_name in file_names:
        img = Image.open(path + file_name)
        if "exif" in img.info:
            exif_dict = piexif.load(img.info["exif"])
            lat, lon = degree(exif_dict['GPS'][piexif.GPSIFD.GPSLatitude]), degree(
                exif_dict['GPS'][piexif.GPSIFD.GPSLongitude])
            x1, y1, z1, zl1 = utm.from_latlon(lat, lon)
            if ((x - x1) ** 2 + (y - y1) ** 2) ** 0.5 <= 100:
                img_list.append(path + file_name)
    return img_list


# Extracts possible regions in images that contains GCPs
def bbox_detector(file_name):

    content = ""
    # rotate the image from exif information and read it in opencv format
    rotate_jpeg(file_name)
    img = cv2.imread(file_name)

    # calculate avg-intensity of the image
    h, w, c = img.shape
    avg = sum(np.ravel(img)) / (h * w * c)
    content = content + str(avg) + ","
    # print avg,
    # cv2.imshow("Image", img)

    thresh_rgb = rgb_threshold(img, 160)
    thresh_rgb = morphology(thresh_rgb)
    # cv2.imshow("rgb_threshold", thresh_rgb)

    thresh_Drgb = drgb_threshold(img)
    thresh_Drgb = morphology(thresh_Drgb)
    # cv2.imshow("Drgb_threshold", thresh_Drgb)

    thresh = cv2.bitwise_and(thresh_rgb, thresh_Drgb)
    # cv2.imshow("Mask", thresh)

    contours = extract_contours(thresh)
    blank = np.zeros(img.shape, np.uint8)
    cv2.drawContours(blank, contours, -1, (255, 0, 0), 1, 8)
    # cv2.imshow("Contours", blank)

    contours1 = check_contour_area(contours)
    blank = np.zeros(img.shape, np.uint8)
    cv2.drawContours(blank, contours1, -1, (0, 255, 0), 1, 8)
    # cv2.imshow("ContourArea", blank)

    contours2 = convexity(contours1)
    blank = np.zeros(img.shape, np.uint8)
    cv2.drawContours(blank, contours2, -1, (0, 255, 0), 1, 8)
    # cv2.imshow("Convexity", blank)

    bbox = box_area(contours2, img)

    img_copy = np.zeros(img.shape, np.uint8)
    img_copy = cv2.bitwise_or(img, img_copy)

    # Draw rectangle on a copy of image
    for x, y, w, h in bbox:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 0), 2, 8)
    # print len(bbox)
    content = content + str(len(bbox)) + ","
    # cv2.imshow("BBOX", img_copy)
    # if (cv2.waitKey(0) == 27):
    #   break

    return bbox, content


# Establishes whether  a bounding box contains GCP or not
def bbox_verification(bbox, file_name, cordinates):

    content = ""
    img = cv2.imread(file_name)
    (rois, bboxs) = extract_roi(img, bbox)
    # for every bbox in a image, verify its a GCP or not
    for (roi, bbox) in zip(rois, bboxs):
        xs, ys, dxs, dys, edges = quiver_data(roi)
        orientations = angles(dxs, dys)
        # print orientations

        # draw a 36 bin histogram of edge orientations
        bins = np.zeros((36, 1), np.uint)
        fig, [ax1, ax2, ax3] = plt.subplots(1, 3)
        im = ax1.imshow(roi)
        plt.colorbar(im)
        ax1.quiver(xs, ys, dxs, dys)
        ax1.set(aspect=1, title='Quiver Plot')
        data_ = ax3.hist(orientations, 36, (0, 360))
        bins = np.transpose(bins)

        # Smooth the histogram-frequency plot
        for i in range(36):
            if i == 0:
                bins[0][i] = data_[0][i] + data_[0][35] + data_[0][i + 1]
            if i == 35:
                bins[0][i] = data_[0][i - 1] + data_[0][i] + data_[0][0]
            else:
                bins[0][i] = data_[0][i - 1] + data_[0][i] + data_[0][i + 1]
        ax2.plot(bins[0])

        # Find the 4-peaks that are nearly 90 degrees apart
        ans = np.zeros((1, 9), np.uint)
        for i in range(9):
            a = np.zeros((1, 36), np.uint)
            a[0][i] = 100
            a[0][9 + i] = 100
            a[0][18 + i] = 100
            a[0][27 + i] = 100
            ans[0][i] = sum(a[0] * bins[0])
        peak = int(np.argmax(ans[0]))
        stages = 0
        # print peak
        # plt.show()

        # Check for the status of peaks detected
        if np.max(bins[0][max(peak - 2, 0): peak + 2]) >= np.max(bins[0][max(0, peak - 4):peak + 4]) >= 5:
            # print np.max(bins[0][max(0, peak - 4):peak + 4]), np.max(bins[0][max(peak - 2, 0) : peak + 2])
            stages = stages + 1
        if np.max(bins[0][peak + 9 - 2: peak + 9 + 2]) >= np.max(bins[0][peak + 9 - 4: peak + 9 + 4]) >= 5:
            # print np.max(bins[0][peak + 9 - 4: peak + 9 + 4]), np.max(bins[0][peak + 9 - 2 : peak + 9 + 2])
            stages = stages + 1
        if np.max(bins[0][peak + 18 - 2: peak + 18 + 2]) >= np.max(bins[0][peak + 18 - 4: peak + 18 + 4]) >= 5:
            # print np.max(bins[0][peak + 18 - 4: peak + 18 + 4]), np.max(bins[0][peak + 18 - 2 : peak + 18 + 2])
            stages = stages + 1
        if np.max(bins[0][peak + 27 - 2: min(peak + 27 + 2, 36)]) >= np.max(
                bins[0][peak + 27 - 4: min(peak + 27 + 4, 36)]) >= 5:
            # print np.max(bins[0][peak + 27 - 4 : min(peak + 27 + 4, 36)]),
            # np.max(bins[0][peak + 27 - 2 : min(peak + 27 + 2, 36)])
            stages = stages + 1

        # If it catches all the real-peaks, give it a weight of 1, if it catches 3 peaks, give it a weight of 0.75
        # and so on and similarly for the output probability of Machine Learning
        edges = cv2.resize(edges, (28, 28))
        #probability = obj.classify(edges)
        #answer = 0.5 * stages / 4 + 0.5 * probability[0][0]

        if stages == 4: #stages / 4 >= 0.75 and probability[0][0] >= 0.75 and answer >= 0.875:
            if rect_contains(bbox, cordinates):
                status = "True-Positive"
                content = content + '({}, {}, {})'.format(np.array(bbox), stages, status) + ","
                print '({}, {}, {})'.format(np.array(bbox), stages, status),

            else:
                status = "False-Positive"
                content = content + '({}, {}, {})'.format(np.array(bbox), stages, status) + ","
                print '({}, {}, {})'.format(np.array(bbox), stages, status),

        else:
            if rect_contains(bbox, cordinates):
                status = "False-Negative"
                content = content + '({}, {}, {})'.format(np.array(bbox), stages, status) + ","
                print '({}, {}, {})'.format(np.array(bbox), stages, status),

        plt.close('all')
    return content
