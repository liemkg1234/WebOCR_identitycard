import numpy as np
import cv2
import torch
from PIL import Image

# Xoa cac box chong` len nhau
def non_max_suppression_fast(boxes, labels, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    #
    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type

    final_labels = [labels[idx] for idx in pick]
    final_boxes = boxes[pick].astype("int")

    return final_boxes, final_labels


# crop anh
def perspective_transoform(image, source_points, point):
    x = point[0]
    y = point[1]
    dest_points = np.float32([[0, 0], [x, 0], [0, y], [x, y]])
    M = cv2.getPerspectiveTransform(source_points, dest_points)
    dst = cv2.warpPerspective(image, M, (x, y))

    return dst


# Can bang do sang va do tuong phan
def automatic_brightness_and_contrast(gray, clip_hist_percent):

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    auto_result = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)


# Image Processing
def Img_Processing(img):
    # Rescale the image, if needed.
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    # Converting to gray scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Removing Shadows
    rgb_planes = cv2.split(img)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        result_planes.append(diff_img)
    img = cv2.merge(result_planes)
    #Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)#increases the white region in the image
    img = cv2.erode(img, kernel, iterations=1) #erodes away the boundaries of foreground object
    # Apply blur to smooth out the edges
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # Can bang do sang va do tuong phan
    img, alpha, beta = automatic_brightness_and_contrast(img,1)
    return img


# ORC (results, imgs) phu hop voi du lieu
# result = results.xyxy[i]; result_pandas = results.pandas().xyxy[i] ; img = imgs[i]
def OCR(result, result_pandas, img, detector):
    if len(result) == 0:  # Loai truong hop khong tim duoc gi
        return None
    # 1. input overlap (Tensor 4*4, labels)
    # Tensor
    tensor = torch.tensor([])
    for i in range(len(result)):
        tensor_i = result[i][0:4].reshape(1, 4)
        tensor = torch.cat((tensor, tensor_i), 0)
    # Label
    label = list(result_pandas['name'])
    # 2. Overlap
    final_boxes, final_labels = non_max_suppression_fast(tensor.numpy(), label, 0.3)
    # print(final_boxes)
    # print(final_labels)
    dic = dict()  # luu tru thong tin
    y = 0  ## Luu vi tri y max
    # 3. Crop Imgaes trong box
    for i in range(len(final_boxes)):
        xmin = final_boxes[i][0]
        ymin = final_boxes[i][1]
        xmax = final_boxes[i][2]
        ymax = final_boxes[i][3]
        point = [xmax - xmin, ymax - ymin]
        source_points = np.float32([[xmin, ymin],  # [[xmin,ymin] [xmax,ymin], [xmin,ymax],[xmax,ymax]]
                                    [xmax, ymin],
                                    [xmin, ymax],
                                    [xmax, ymax]])

        new_img = perspective_transoform(img, source_points, point)  # Cat anh cua tung box
        # print(final_labels[i])

        # 4. Image Processing
        new_img = Img_Processing(new_img)

        # 5. Predict thong tin #######################
        new_img = Image.fromarray(new_img)
        information = detector.predict(new_img)  # Predict duoc thong tin
        # print(information)
        if final_labels[i] in dic:
            if ymax > y:
                string = dic[final_labels[i]] + ", " + information
                # print(string)
                dic[final_labels[i]] = string
                y = ymax
            else:
                string = information + ", " + dic[final_labels[i]]
                # print(string)
                dic[final_labels[i]] = string

        else:
            dic[final_labels[i]] = information
            y = ymax

    return dic
