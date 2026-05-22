/*
 * sgla_eval_trt.cpp
 *
 * Evaluate a TensorRT engine against ground-truth bounding boxes.
 *
 * Directory layout expected:
 *   Test_10_videos/
 *     <seq_name>/
 *       <seq_name>.mp4
 *       ground_truth.txt      (one line per frame: x,y,w,h  — comma or space separated)
 *
 * Outputs (written next to each sequence folder):
 *   <seq_name>_predictions.txt   — frame-by-frame predicted boxes
 *   <seq_name>_errors.txt        — per-frame errors + aggregate statistics
 *
 * Build (example):
 *   g++ -std=c++17 -O2 sgla_eval_trt.cpp \
 *       -I/usr/local/cuda/include \
 *       -I/usr/include/opencv4 \
 *       -L/usr/local/cuda/lib64 -lcudart \
 *       -lnvinfer -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_videoio \
 *       -o sgla_eval_trt
 *
 * Usage:
 *   ./sgla_eval_trt --engine model.engine --dataset ./Test_10_videos [--output_dir ./results]
 */
 
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <filesystem>
 
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
 
namespace fs = std::filesystem;
 
// ─────────────────────────────────────────────────────────────────────────────
//  TensorRT logger
// ─────────────────────────────────────────────────────────────────────────────
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << "\n";
    }
};
static Logger gLogger;
 
// ─────────────────────────────────────────────────────────────────────────────
//  Data types
// ─────────────────────────────────────────────────────────────────────────────
struct BBox { float x, y, w, h; };   // top-left origin, pixel coords
 
struct FrameRecord {
    int   frameIdx;
    BBox  pred;           // TRT prediction (absolute pixel coords)
    BBox  gt;             // ground truth
    float err_x;          // |pred.x  - gt.x|
    float err_y;          // |pred.y  - gt.y|
    float err_w;          // |pred.w  - gt.w|
    float err_h;          // |pred.h  - gt.h|
    float center_err;     // Euclidean distance between centres
    float iou;            // Intersection-over-Union (pred vs gt)
};
 
// ─────────────────────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────────────────────
static float iou(const BBox& a, const BBox& b) {
    float ax2 = a.x + a.w, ay2 = a.y + a.h;
    float bx2 = b.x + b.w, by2 = b.y + b.h;
    float ix1 = std::max(a.x, b.x), iy1 = std::max(a.y, b.y);
    float ix2 = std::min(ax2, bx2),  iy2 = std::min(ay2, by2);
    float inter = std::max(0.f, ix2 - ix1) * std::max(0.f, iy2 - iy1);
    float uni   = a.w * a.h + b.w * b.h - inter;
    return (uni <= 0.f) ? 0.f : inter / uni;
}
 
static BBox clipBox(const BBox& box, int H, int W, int margin = 10) {
    float x1 = std::min(std::max(0.f, box.x),          (float)(W - margin));
    float y1 = std::min(std::max(0.f, box.y),          (float)(H - margin));
    float x2 = std::min(std::max((float)margin, box.x + box.w), (float)W);
    float y2 = std::min(std::max((float)margin, box.y + box.h), (float)H);
    return { x1, y1, std::max((float)margin, x2 - x1),
                     std::max((float)margin, y2 - y1) };
}
 
// ─────────────────────────────────────────────────────────────────────────────
//  Image preprocessing  (identical to sglatrack_trt)
// ─────────────────────────────────────────────────────────────────────────────
static cv::Mat sampleTarget(const cv::Mat& im, const BBox& bb, float factor,
                             int outSz, float& resizeFactor) {
    float cx = bb.x + 0.5f * bb.w;
    float cy = bb.y + 0.5f * bb.h;
    float cropSz = std::ceil(std::sqrt(bb.w * bb.h) * factor);
 
    int x1 = (int)std::round(cx - cropSz * 0.5f);
    int y1 = (int)std::round(cy - cropSz * 0.5f);
    int x2 = x1 + (int)cropSz;
    int y2 = y1 + (int)cropSz;
 
    int x1p = std::max(0, -x1),       x2p = std::max(0, x2 - im.cols + 1);
    int y1p = std::max(0, -y1),       y2p = std::max(0, y2 - im.rows + 1);
 
    cv::Mat crop = im(cv::Rect(x1 + x1p, y1 + y1p,
                               (x2 - x2p) - (x1 + x1p),
                               (y2 - y2p) - (y1 + y1p)));
    cv::Mat padded;
    cv::copyMakeBorder(crop, padded, y1p, y2p, x1p, x2p,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
 
    resizeFactor = (float)outSz / cropSz;
    cv::Mat resized;
    cv::resize(padded, resized, cv::Size(outSz, outSz));
    return resized;
}
 
static void preprocess(const cv::Mat& img, float* buf,
                       int C, int H, int W) {
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float stdv[3] = {0.229f, 0.224f, 0.225f};
    for (int c = 0; c < C; c++) {
        int bgr_c = 2 - c;
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++) {
                float px = img.at<cv::Vec3b>(h, w)[bgr_c] / 255.f;
                buf[c * H * W + h * W + w] = (px - mean[c]) / stdv[c];
            }
    }
}
 
// ─────────────────────────────────────────────────────────────────────────────
//  Ground-truth parser
//  Supports:  "x,y,w,h"   or   "x y w h"   per line
//  Returns empty vector on failure.
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<BBox> loadGroundTruth(const std::string& path) {
    std::vector<BBox> out;
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[WARN] Cannot open ground truth: " << path << "\n";
        return out;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        // replace commas with spaces for uniform parsing
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream ss(line);
        BBox b{};
        if (ss >> b.x >> b.y >> b.w >> b.h)
            out.push_back(b);
    }
    return out;
}
 
// ─────────────────────────────────────────────────────────────────────────────
//  TRT engine loader
// ─────────────────────────────────────────────────────────────────────────────
static nvinfer1::ICudaEngine* loadEngine(const std::string& path,
                                          nvinfer1::IRuntime* rt) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) { std::cerr << "Cannot open engine: " << path << "\n"; return nullptr; }
    f.seekg(0, std::ios::end); size_t sz = f.tellg(); f.seekg(0);
    std::vector<char> buf(sz);
    f.read(buf.data(), sz);
    return rt->deserializeCudaEngine(buf.data(), sz, nullptr);
}
 
// ─────────────────────────────────────────────────────────────────────────────
//  Statistics helper
// ─────────────────────────────────────────────────────────────────────────────
struct Stats {
    float mean, med, min_, max_, range, std_;
};
 
static Stats computeStats(std::vector<float> v) {
    Stats s{};
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.min_  = v.front();
    s.max_  = v.back();
    s.range = s.max_ - s.min_;
    s.mean  = std::accumulate(v.begin(), v.end(), 0.f) / v.size();
    s.med   = (v.size() % 2 == 0)
              ? 0.5f * (v[v.size()/2 - 1] + v[v.size()/2])
              : v[v.size()/2];
    float var = 0.f;
    for (float x : v) var += (x - s.mean) * (x - s.mean);
    s.std_ = std::sqrt(var / v.size());
    return s;
}
 
// ─────────────────────────────────────────────────────────────────────────────
//  Write per-frame predictions
// ─────────────────────────────────────────────────────────────────────────────
static void writePredictions(const std::string& path,
                              const std::vector<FrameRecord>& recs) {
    std::ofstream f(path);
    f << "# frame_idx  pred_x  pred_y  pred_w  pred_h\n";
    f << std::fixed << std::setprecision(4);
    for (auto& r : recs)
        f << r.frameIdx << "\t"
          << r.pred.x << "\t" << r.pred.y << "\t"
          << r.pred.w << "\t" << r.pred.h << "\n";
}
 
// ─────────────────────────────────────────────────────────────────────────────
//  Write per-frame errors + aggregate statistics
// ─────────────────────────────────────────────────────────────────────────────
static void writeErrors(const std::string& path,
                         const std::vector<FrameRecord>& recs,
                         const std::string& seqName) {
    std::ofstream f(path);
    f << std::fixed << std::setprecision(4);
 
    f << "# Sequence: " << seqName << "\n";
    f << "# Total frames: " << recs.size() << "\n";
    f << "#\n";
    f << "# Columns: frame_idx | gt_x gt_y gt_w gt_h | pred_x pred_y pred_w pred_h"
         " | err_x err_y err_w err_h | center_err | IoU\n";
    f << "#\n";
 
    for (auto& r : recs) {
        f << r.frameIdx
          << "\t" << r.gt.x   << " " << r.gt.y   << " " << r.gt.w   << " " << r.gt.h
          << "\t" << r.pred.x << " " << r.pred.y << " " << r.pred.w << " " << r.pred.h
          << "\t" << r.err_x  << " " << r.err_y  << " " << r.err_w  << " " << r.err_h
          << "\t" << r.center_err
          << "\t" << r.iou
          << "\n";
    }
 
    // ── aggregate ─────────────────────────────────────────────────────────────
    auto collect = [&](auto field) {
        std::vector<float> v; v.reserve(recs.size());
        for (auto& r : recs) v.push_back(field(r));
        return v;
    };
 
    auto printStats = [&](const std::string& label, Stats s) {
        f << "  " << std::left << std::setw(18) << label
          << "  mean=" << std::setw(9) << s.mean
          << "  median=" << std::setw(9) << s.med
          << "  min="  << std::setw(9) << s.min_
          << "  max="  << std::setw(9) << s.max_
          << "  range=" << std::setw(9) << s.range
          << "  std="  << s.std_ << "\n";
    };
 
    f << "\n";
    f << "════════════════════════════════════════════════════════════════\n";
    f << "  AGGREGATE STATISTICS\n";
    f << "════════════════════════════════════════════════════════════════\n";
 
    printStats("err_x (px)",     computeStats(collect([](auto& r){ return r.err_x; })));
    printStats("err_y (px)",     computeStats(collect([](auto& r){ return r.err_y; })));
    printStats("err_w (px)",     computeStats(collect([](auto& r){ return r.err_w; })));
    printStats("err_h (px)",     computeStats(collect([](auto& r){ return r.err_h; })));
    printStats("center_err (px)",computeStats(collect([](auto& r){ return r.center_err; })));
    printStats("IoU",            computeStats(collect([](auto& r){ return r.iou; })));
 
    // success rate at common thresholds
    auto countBelow = [&](float thr) {
        return (int)std::count_if(recs.begin(), recs.end(),
                                  [thr](auto& r){ return r.center_err < thr; });
    };
    auto countIoU = [&](float thr) {
        return (int)std::count_if(recs.begin(), recs.end(),
                                  [thr](auto& r){ return r.iou >= thr; });
    };
    int N = (int)recs.size();
    f << "\n  Precision (center_err < threshold):\n";
    for (float thr : {5.f, 10.f, 20.f, 50.f})
        f << "    < " << std::setw(4) << thr << " px : "
          << countBelow(thr) << " / " << N
          << "  (" << std::setprecision(1) << 100.f*countBelow(thr)/N << "%)\n";
 
    f << "\n  Success (IoU >= threshold):\n";
    for (float thr : {0.25f, 0.5f, 0.75f})
        f << "    >= " << thr << " IoU : "
          << countIoU(thr) << " / " << N
          << "  (" << std::setprecision(1) << 100.f*countIoU(thr)/N << "%)\n";
}
 
// ─────────────────────────────────────────────────────────────────────────────
//  Process one sequence
// ─────────────────────────────────────────────────────────────────────────────
static bool processSequence(const std::string& seqDir,
                             const std::string& seqName,
                             const std::string& outputDir,
                             nvinfer1::IExecutionContext* context,
                             nvinfer1::ICudaEngine* engine,
                             // CUDA resources (pre-allocated, reused across seqs)
                             void* d_template, void* d_search, void* d_output,
                             std::vector<float>& h_template,
                             std::vector<float>& h_search,
                             std::vector<float>& h_output,
                             void** bindings,
                             cudaStream_t stream) {
    const int templateC = 3, templateH = 128, templateW = 128;
    const int searchC   = 3, searchH   = 256, searchW   = 256;
    const float templateFactor = 2.0f;
    const float searchFactor   = 4.0f;
 
    // ── video ──────────────────────────────────────────────────────────────────
    std::string videoPath = seqDir + "/" + seqName + ".mp4";
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "[SKIP] Cannot open video: " << videoPath << "\n";
        return false;
    }
 
    // ── ground truth ───────────────────────────────────────────────────────────
    std::vector<BBox> gt = loadGroundTruth(seqDir + "/ground_truth.txt");
    if (gt.empty()) {
        std::cerr << "[SKIP] No ground truth for: " << seqName << "\n";
        return false;
    }
 
    // ── first frame & initial state from GT[0] ─────────────────────────────────
    cv::Mat firstFrame;
    cap >> firstFrame;
    if (firstFrame.empty()) { std::cerr << "[SKIP] Empty first frame.\n"; return false; }
 
    BBox state = gt[0];   // initialise tracker with first GT box
 
    float resizeFactorZ;
    cv::Mat templatePatch = sampleTarget(firstFrame, state, templateFactor, 128, resizeFactorZ);
    preprocess(templatePatch, h_template.data(), templateC, templateH, templateW);
    size_t templateBytes = templateC * templateH * templateW * sizeof(float);
    cudaMemcpyAsync(d_template, h_template.data(), templateBytes, cudaMemcpyHostToDevice, stream);
 
    std::vector<FrameRecord> records;
    // Frame 0: prediction == GT (we initialise here, nothing to infer)
    {
        FrameRecord r0{};
        r0.frameIdx   = 0;
        r0.pred       = state;
        r0.gt         = gt[0];
        r0.err_x = r0.err_y = r0.err_w = r0.err_h = 0.f;
        r0.center_err = 0.f;
        r0.iou        = 1.f;
        records.push_back(r0);
    }
 
    cv::Mat frame;
    int frameIdx = 1;
    size_t searchBytes = searchC * searchH * searchW * sizeof(float);
    size_t outputBytes = 4 * sizeof(float);
 
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
 
        int H = frame.rows, W = frame.cols;
 
        float resizeFactor;
        cv::Mat searchPatch = sampleTarget(frame, state, searchFactor, 256, resizeFactor);
        preprocess(searchPatch, h_search.data(), searchC, searchH, searchW);
 
        cudaMemcpyAsync(d_search, h_search.data(), searchBytes, cudaMemcpyHostToDevice, stream);
        context->enqueueV2(bindings, stream, nullptr);
        cudaMemcpyAsync(h_output.data(), d_output, outputBytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
 
        // ── decode prediction ──────────────────────────────────────────────────
        float pred_cx = h_output[0] * 256.f / resizeFactor;
        float pred_cy = h_output[1] * 256.f / resizeFactor;
        float pred_w  = h_output[2] * 256.f / resizeFactor;
        float pred_h  = h_output[3] * 256.f / resizeFactor;
 
        float cx_prev  = state.x + 0.5f * state.w;
        float cy_prev  = state.y + 0.5f * state.h;
        float halfSide = 0.5f * 256.f / resizeFactor;
 
        float cx_abs = pred_cx + (cx_prev - halfSide);
        float cy_abs = pred_cy + (cy_prev - halfSide);
 
        BBox newBox = { cx_abs - 0.5f * pred_w, cy_abs - 0.5f * pred_h, pred_w, pred_h };
        state = clipBox(newBox, H, W, 10);
 
        // ── compare with GT ────────────────────────────────────────────────────
        FrameRecord rec{};
        rec.frameIdx = frameIdx;
        rec.pred     = state;
 
        if (frameIdx < (int)gt.size()) {
            rec.gt = gt[frameIdx];
        } else {
            // GT shorter than video — mark as missing
            rec.gt = { -1, -1, -1, -1 };
        }
 
        if (rec.gt.x >= 0) {
            rec.err_x = std::abs(state.x - rec.gt.x);
            rec.err_y = std::abs(state.y - rec.gt.y);
            rec.err_w = std::abs(state.w - rec.gt.w);
            rec.err_h = std::abs(state.h - rec.gt.h);
 
            float pcx = state.x + 0.5f * state.w;
            float pcy = state.y + 0.5f * state.h;
            float gcx = rec.gt.x + 0.5f * rec.gt.w;
            float gcy = rec.gt.y + 0.5f * rec.gt.h;
            rec.center_err = std::sqrt((pcx-gcx)*(pcx-gcx) + (pcy-gcy)*(pcy-gcy));
            rec.iou        = iou(state, rec.gt);
        }
 
        records.push_back(rec);
        frameIdx++;
    }
 
    cap.release();
 
    // ── write outputs ──────────────────────────────────────────────────────────
    fs::create_directories(outputDir);
    std::string predPath  = outputDir + "/" + seqName + "_predictions.txt";
    std::string errorPath = outputDir + "/" + seqName + "_errors.txt";
    writePredictions(predPath,  records);
    writeErrors     (errorPath, records, seqName);
 
    std::cout << "[DONE] " << seqName
              << "  frames=" << frameIdx
              << "  -> " << predPath << "\n"
              << "         -> " << errorPath << "\n";
    return true;
}
 
// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    std::string enginePath, datasetDir, outputDir = "./eval_results";
 
    for (int i = 1; i < argc; i++) {
        std::string a(argv[i]);
        if (a == "--engine"     && i+1 < argc) enginePath = argv[++i];
        else if (a == "--dataset"    && i+1 < argc) datasetDir = argv[++i];
        else if (a == "--output_dir" && i+1 < argc) outputDir  = argv[++i];
    }
 
    if (enginePath.empty() || datasetDir.empty()) {
        std::cerr << "Usage: ./sgla_eval_trt --engine <model.engine>"
                     " --dataset <Test_10_videos> [--output_dir <./eval_results>]\n";
        return -1;
    }
 
    // ── load engine ────────────────────────────────────────────────────────────
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    nvinfer1::ICudaEngine* engine = loadEngine(enginePath, runtime);
    if (!engine) return -1;
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
 
    // ── binding indices ────────────────────────────────────────────────────────
    int templateIdx = engine->getBindingIndex("template");
    int searchIdx   = engine->getBindingIndex("search");
    int outputIdx   = engine->getBindingIndex("pred_boxes");
 
    if (templateIdx < 0 || searchIdx < 0 || outputIdx < 0) {
        std::cerr << "ERROR: Cannot find expected binding names.\nAvailable:\n";
        for (int i = 0; i < engine->getNbBindings(); i++)
            std::cerr << "  [" << i << "] " << engine->getBindingName(i) << "\n";
        return -1;
    }
 
    // ── CUDA allocations (reused across sequences) ─────────────────────────────
    const int tC=3, tH=128, tW=128, sC=3, sH=256, sW=256;
    size_t tBytes = tC*tH*tW*sizeof(float);
    size_t sBytes = sC*sH*sW*sizeof(float);
    size_t oBytes = 4*sizeof(float);
 
    void *d_template, *d_search, *d_output;
    cudaMalloc(&d_template, tBytes);
    cudaMalloc(&d_search,   sBytes);
    cudaMalloc(&d_output,   oBytes);
 
    void* bindings[3];
    bindings[templateIdx] = d_template;
    bindings[searchIdx]   = d_search;
    bindings[outputIdx]   = d_output;
 
    cudaStream_t stream;
    cudaStreamCreate(&stream);
 
    std::vector<float> h_template(tC*tH*tW);
    std::vector<float> h_search(sC*sH*sW);
    std::vector<float> h_output(4);
 
    // ── iterate over sequences ─────────────────────────────────────────────────
    int seqOk = 0, seqFail = 0;
    for (auto& entry : fs::directory_iterator(datasetDir)) {
        if (!entry.is_directory()) continue;
        std::string seqName = entry.path().filename().string();
        std::string seqDir  = entry.path().string();
 
        std::cout << "\n[SEQ] " << seqName << "\n";
        bool ok = processSequence(seqDir, seqName, outputDir,
                                  context, engine,
                                  d_template, d_search, d_output,
                                  h_template, h_search, h_output,
                                  bindings, stream);
        ok ? seqOk++ : seqFail++;
    }
 
    // ── cleanup ────────────────────────────────────────────────────────────────
    cudaStreamDestroy(stream);
    cudaFree(d_template);
    cudaFree(d_search);
    cudaFree(d_output);
    context->destroy();
    engine->destroy();
    runtime->destroy();
 
    std::cout << "\n══════════════════════════════════════════\n";
    std::cout << "  Evaluation complete.\n";
    std::cout << "  Sequences processed: " << seqOk << "  |  failed: " << seqFail << "\n";
    std::cout << "  Results written to:  " << outputDir << "\n";
    std::cout << "══════════════════════════════════════════\n";
    return 0;
}
 