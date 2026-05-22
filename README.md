# SGLATrack TensorRT - Jetson Xavier NX

Triển khai SGLATrack inference bằng C++ TensorRT trên Jetson Xavier NX.

## Yêu cầu

- Jetson Xavier NX với JetPack 4.6.2
- TensorRT 8.2.1 (có sẵn trong JetPack)
- CUDA 10.2 (có sẵn trong JetPack)
- OpenCV 4.x (có sẵn trong JetPack)
- CMake >= 3.10

## Cấu trúc thư mục

```
sglatrack_trt/
├── CMakeLists.txt
├── build.sh
├── README.md
├── src/
│   └── main.cpp
└── models/
    └── model.engine    ← copy file engine vào đây
```

## Hướng dẫn

### 1. Copy project sang Jetson

Từ PC, dùng scp:

```bash
scp -r sglatrack_trt/ jetson@<JETSON_IP>:~/
```

Hoặc dùng USB/SD card copy thư mục `sglatrack_trt` sang Jetson.

### 2. Copy file engine sang Jetson

```bash
scp model.engine jetson@<JETSON_IP>:~/sglatrack_trt/models/
```

**Lưu ý quan trọng:** File `.engine` được build trên máy nào thì chỉ chạy được trên máy đó (phụ thuộc GPU architecture). Nếu bạn build engine trên PC (GPU desktop), bạn cần build lại engine trên Jetson từ file ONNX:

```bash
# Trên Jetson, convert ONNX -> Engine
/usr/src/tensorrt/bin/trtexec \
    --onnx=model.onnx \
    --saveEngine=models/model.engine \
    --fp16 \
    --workspace=1024
```

### 3. Build trên Jetson

```bash
cd ~/sglatrack_trt
chmod +x build.sh
./build.sh
```

### 4. Chạy

```bash
./build/sgla_tracker --engine models/model.engine --video /path/to/video.mp4
```

- Khi chạy, cửa sổ hiện lên frame đầu tiên → dùng chuột kéo chọn object cần track → nhấn Enter/Space
- Nhấn `q` để thoát

## Ghi chú

- FPS được hiển thị trên góc trái video
- Model input: template 128x128, search 256x256
- Dùng `--fp16` khi convert engine để tận dụng FP16 trên Xavier NX (tăng tốc ~2x)
- Nếu gặp lỗi binding name, kiểm tra tên input/output trong ONNX model phải là: `template`, `search`, `pred_boxes`




### Problem


- TemplateFactor, searchFactor is fixed
- sample_target
- How the network works?
- Post processing
- Output
- Why miss track
- margin = 10 (is fixed)