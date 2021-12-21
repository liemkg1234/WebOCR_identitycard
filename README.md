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

## Thiết kế và cài đặt

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
- Huấn luyện mô hình nhận diện 4 góc (model_crop.pt):
```
!python train.py --img 640 --batch 8 --epochs 250 --data coco128.yaml --weights yolov5x.pt
```
- Căn chỉnh hình ảnh:
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
### Detector

### Reader

## Kiểm thử và đánh giá

## Tổng kết
