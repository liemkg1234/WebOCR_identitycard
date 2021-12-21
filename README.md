# Website nhận diện và trích xuất thông tin từ Chứng Minh Nhân Dân
Trong project này, mình xây dựng các mô hình nhận diện bằng YOLOv5 và dùng mô hình TransformerOCR để trích xuất thông tin, ngoài ra áp dụng thêm các phương pháp căn chỉnh và tiền xử lý ảnh để tăng độ chính xác khi nhận diện và trích xuất. Cuối cùng mình xây dựng website bằng Flask và ngrok để người dùng cuối có thể sử dụng chức năng trích xuất.

**Các thư viện sử dụng**
- [YOLOv5](https://github.com/ultralytics/yolov5)
- [VietOCR](https://github.com/pbcquoc/vietocr)
- [LabelImg](https://github.com/tzutalin/labelImg)
- [Flask](https://flask.palletsprojects.com/en/2.0.x/)
- [pyngrok](https://pypi.org/project/pyngrok/)

**Môi trường**
- [Google Colaboratory](https://research.google.com/colaboratory/)
- Python 3.7.12
- CUDA Version: 11.2

## THIẾT KẾ VÀ CÀI ĐẶT

### Sơ đồ tổng quát
![samples](https://github.com/liemkg1234/WebOCR_identitycard/blob/master/image/sodo1.png)
### Tập dữ liệu

Tập dữ liệu của các mô hình là mặt trước của CMND, mình thu thập được 280 ảnh tương ứng với 28 người khác nhau. Ảnh thỏa mãn các điều kiền như: được chụp ở nhiều góc với nhiều kích thước khác nhau, độ phân phải của ảnh có thể cao hoặc thấp, ảnh bị mất góc. ảnh bị lóe sáng hoặc thiếu sáng, khung nền của ảnh phải khác nhau, ... 
Mình chia tập dữ liệu thành 3 phần:
train: 16 người
val: 8 người
test: 4 người

Do tính bảo mật của khách hàng nên mình xin phép không chia sẽ tập dữ liệu.
### Cropper
**Huấn luyện mô hình nhận diện 4 góc (model_crop.pt):**
```
!python train.py --img 640 --batch 8 --epochs 250 --data coco128.yaml --weights yolov5x.pt
```
**Căn chỉnh hình ảnh:**
```
def CropImg(result,result_pandas,img):
  #1. input overlap (Tensor 4*4, labels)
  #Tensor
  tensor = torch.tensor([])
  for i in range(len(result)):
    tensor_i = result[i][0:4].reshape(1,4)
    tensor = torch.cat((tensor, tensor_i), 0)
  #Label
  label = list(result_pandas['name'])
  #2. Overlap
  final_boxes, final_labels = non_max_suppression_fast(tensor.numpy(), label, 0.3)
  #3. Tim midpoint cua cac boxes
  final_points = list(map(get_center_point, final_boxes))
  label_boxes = dict(zip(final_labels, final_points))
  #3.5 Neu thieu 1 goc -> Noi suy goc con lai
  if (len(label_boxes) == 3):
    label_boxes = calculate_missed_coord_corner(label_boxes)
  elif (len(label_boxes) < 3):
    return None
  #4. Crop anh
  source_points = np.float32([label_boxes['top_left'], label_boxes['top_right'], label_boxes['bottom_right'], label_boxes['bottom_left']])
  crop = perspective_transoform(img, source_points)
  return crop
```
![cropper](https://github.com/liemkg1234/WebOCR_identitycard/blob/master/image/cropper.jpg)
### Detector
**Huấn luyện mô hình nhận diện thông tin (model_detect.pt):**
```
!python train.py --img 640 --batch 8 --epochs 70 --data crop.yaml --weights yolov5l.pt
```
![Detector](https://github.com/liemkg1234/WebOCR_identitycard/blob/master/image/detector.jpg)
### Reader
**Tiền xử lý ảnh**
```
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
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1) 
    # Apply blur to smooth out the edges
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # Can bang do sang va do tuong phan
    img, alpha, beta = automatic_brightness_and_contrast(img,1)
    return img
```
**OCR**
```
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
```
![Reader](https://github.com/liemkg1234/WebOCR_identitycard/blob/master/image/reader.jpg)
## KIỂM THỬ VÀ ĐÁNH GIÁ
### Kết quả trên tập test của tập dữ liệu 
**Cropper**
|   | Nhóm ảnh tốt | Nhóm ảnh xấu |
| ------------- | ------------- | ------------- |
| Căn chỉnh  | 26  | 10  |
| Không căn chỉnh  | 0  | 4  |

**Detector**
|   | Nhóm ảnh tốt | Nhóm ảnh xấu |
| ------------- | ------------- | ------------- |
| Cắt hết  | 26  | 10  |
| Cắt thiếu  | 0  | 0  |
| Không cắt  | 0  | 4  |

**Reader**
|   | Nhóm ảnh tốt | Nhóm ảnh xấu |
| ------------- | ------------- | ------------- |
| Dự đoán đúng  | 24  | 5  |
| Dự đoán sai  | 2  | 5  |
| Không dự đoán  | 0  | 4  |

### Trang web
![Web](https://github.com/liemkg1234/WebOCR_identitycard/blob/master/image/web.jpg)
## NHẬN XÉT
- Tập dữ liệu quá ít để huấn luyện các mô hình nhận diện
- Tiền xử lý ảnh trước khi trích xuất chưa tối ưu, khiến kết quả trích xuất giảm độ chính xác
![IMGProcessing](https://github.com/liemkg1234/WebOCR_identitycard/blob/master/image/img_processing.jpg)
