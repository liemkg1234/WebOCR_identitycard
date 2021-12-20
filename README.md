# Website nhận diện và trích xuất thông tin từ Chứng Minh Nhân Dân (OCR_identitycard)
Trong project này, mình xây dựng các mô hình nhận diện bằng YOLOv5 và dùng mô hình TransformerOCR để trích xuất thông tin, ngoài ra mình áp dụng thêm các phương pháp căn chỉnh và tiền xử lý ảnh để tăng độ chính xác khi nhận diện và trích xuất. Cuối cùng mình xây dựng website bằng Flask và ngrok để người dùng cuối có thể sử dụng các mô hình.

**Các thư viện sử dụng**
- [YOLOv5](https://github.com/ultralytics/yolov5)
- [VietOCR](https://github.com/pbcquoc/vietocr)
- [LabelImg](https://github.com/tzutalin/labelImg)
