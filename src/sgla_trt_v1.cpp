/*
 * sgla_eval_trt.cpp
 *
 * Chạy TensorRT engine trên toàn bộ sequences trong Test_10_videos,
 * ghi raw output của model ra file .txt.
 *
 * Layout:
 *   Test_10_videos/
 *     <seq_name>/
 *       <seq_name>.mp4
 *       ground_truth.txt    (dòng đầu dùng làm initial bbox)
 *
 * Output:
 *   <output_dir>/<seq_name>_predictions.txt
 *   Mỗi dòng: val0 val1 val2 val3   (raw float output từ pred_boxes)
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
#include <iomanip>
#include <sys/stat.h>
#include <dirent.h>

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

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
//  Types
// ─────────────────────────────────────────────────────────────────────────────
struct BBox { float x, y, w, h; };

// ─────────────────────────────────────────────────────────────────────────────
//  POSIX helpers
// ─────────────────────────────────────────────────────────────────────────────
static void mkdirP(const std::string& path) {
    mkdir(path.c_str(), 0755);
}

static bool isDirectory(const std::string& path) {
    struct stat st{};
    return (stat(path.c_str(), &st) == 0) && S_ISDIR(st.st_mode);
}

static std::vector<std::string> listSubDirs(const std::string& dir) {
    std::vector<std::string> result;
    DIR* d = opendir(dir.c_str());
    if (!d) { std::cerr << "Cannot open dataset dir: " << dir << "\n"; return result; }
    struct dirent* ent;
    while ((ent = readdir(d)) != nullptr) {
        std::string name(ent->d_name);
        if (name == "." || name == "..") continue;
        if (isDirectory(dir + "/" + name))
            result.push_back(name);
    }
    closedir(d);
    std::sort(result.begin(), result.end());
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Ground truth parser 
// ─────────────────────────────────────────────────────────────────────────────
static bool loadInitBox(const std::string& path, BBox& out) {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    std::string line;
    if (!std::getline(f, line)) return false;
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream ss(line);
    return (bool)(ss >> out.x >> out.y >> out.w >> out.h);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Preprocessing
// ─────────────────────────────────────────────────────────────────────────────
static cv::Mat sampleTarget(const cv::Mat& im, const BBox& bb,
                             float factor, int outSz, float& resizeFactor) {
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

static void preprocess(const cv::Mat& img, float* buf, int C, int H, int W) {
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

static BBox clipBox(const BBox& box, int H, int W, int margin = 10) {
    float x1 = std::min(std::max(0.f, box.x),               (float)(W - margin));
    float y1 = std::min(std::max(0.f, box.y),               (float)(H - margin));
    float x2 = std::min(std::max((float)margin, box.x+box.w),(float)W);
    float y2 = std::min(std::max((float)margin, box.y+box.h),(float)H);
    return { x1, y1, std::max((float)margin, x2-x1),
                     std::max((float)margin, y2-y1) };
}

// ─────────────────────────────────────────────────────────────────────────────
//  Engine loader
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
//  Process one sequence
// ─────────────────────────────────────────────────────────────────────────────
static bool processSequence(const std::string& seqDir,
                             const std::string& seqName,
                             const std::string& outputDir,
                             nvinfer1::IExecutionContext* context,
                             void* d_template, void* d_search, void* d_output,
                             std::vector<float>& h_template,
                             std::vector<float>& h_search,
                             std::vector<float>& h_output,
                             void** bindings,
                             cudaStream_t stream) {
    const int tC=3, tH=128, tW=128;
    const int sC=3, sH=256, sW=256;
    const float templateFactor = 2.0f;
    const float searchFactor   = 4.0f;

    // ── mở video ──────────────────────────────────────────────────────────────
    std::string videoPath = seqDir + "/" + seqName + ".mp4";
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "[SKIP] Cannot open: " << videoPath << "\n";
        return false;
    }

    // ── đọc init bbox từ dòng đầu ground_truth.txt ────────────────────────────
    BBox state{};
    if (!loadInitBox(seqDir + "/groundtruth_rect.txt", state)) {
        std::cerr << "[SKIP] Cannot read init bbox for: " << seqName << "\n";
        return false;
    }

    // ── frame đầu → template ──────────────────────────────────────────────────
    cv::Mat firstFrame;
    cap >> firstFrame;
    if (firstFrame.empty()) { std::cerr << "[SKIP] Empty first frame.\n"; return false; }

    float rzZ;
    cv::Mat templatePatch = sampleTarget(firstFrame, state, templateFactor, tH, rzZ);
    preprocess(templatePatch, h_template.data(), tC, tH, tW);
    cudaMemcpyAsync(d_template, h_template.data(),
                    tC*tH*tW*sizeof(float), cudaMemcpyHostToDevice, stream);

    // ── bắt đầu ghi file ──────────────────────────────────────────────────────
    std::string outPath = outputDir + "/" + seqName + "_predictions.txt";
    std::ofstream fout(outPath);
    fout << std::fixed << std::setprecision(8);

    // ghi frame 0: dùng lại init box (chưa có inference)
    fout << state.x << " " << state.y << " " << state.w << " " << state.h << "\n";

    // ── loop frames ───────────────────────────────────────────────────────────
    cv::Mat frame;
    int frameIdx = 1;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        int H = frame.rows, W = frame.cols;

        float rz;
        cv::Mat searchPatch = sampleTarget(frame, state, searchFactor, sH, rz);
        preprocess(searchPatch, h_search.data(), sC, sH, sW);

        cudaMemcpyAsync(d_search, h_search.data(),
                        sC*sH*sW*sizeof(float), cudaMemcpyHostToDevice, stream);
        context->enqueueV2(bindings, stream, nullptr);
        cudaMemcpyAsync(h_output.data(), d_output,
                        4*sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // ghi raw output (cx_norm, cy_norm, w_norm, h_norm) thẳng ra file
        fout << h_output[0] << " "
             << h_output[1] << " "
             << h_output[2] << " "
             << h_output[3] << "\n";

        // cập nhật state để crop search region frame tiếp theo
        float pred_cx = h_output[0] * 256.f / rz;
        float pred_cy = h_output[1] * 256.f / rz;
        float pred_w  = h_output[2] * 256.f / rz;
        float pred_h  = h_output[3] * 256.f / rz;

        float cx_prev  = state.x + 0.5f * state.w;
        float cy_prev  = state.y + 0.5f * state.h;
        float halfSide = 0.5f * 256.f / rz;

        BBox newBox = { (pred_cx + cx_prev - halfSide) - 0.5f * pred_w,
                        (pred_cy + cy_prev - halfSide) - 0.5f * pred_h,
                        pred_w, pred_h };
        state = clipBox(newBox, H, W, 10);

        frameIdx++;
    }

    cap.release();
    std::cout << "[DONE] " << seqName << "  frames=" << frameIdx
              << "  -> " << outPath << "\n";
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    std::string enginePath, datasetDir, outputDir = "./eval_results";

    for (int i = 1; i < argc; i++) {
        std::string a(argv[i]);
        if      (a == "--engine"     && i+1 < argc) enginePath = argv[++i];
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

    // ── bindings ───────────────────────────────────────────────────────────────
    int templateIdx = engine->getBindingIndex("template");
    int searchIdx   = engine->getBindingIndex("search");
    int outputIdx   = engine->getBindingIndex("pred_boxes");

    if (templateIdx < 0 || searchIdx < 0 || outputIdx < 0) {
        std::cerr << "ERROR: binding names not found. Available:\n";
        for (int i = 0; i < engine->getNbBindings(); i++)
            std::cerr << "  [" << i << "] " << engine->getBindingName(i) << "\n";
        return -1;
    }

    // ── CUDA alloc ─────────────────────────────────────────────────────────────
    void *d_template, *d_search, *d_output;
    cudaMalloc(&d_template, 3*128*128*sizeof(float));
    cudaMalloc(&d_search,   3*256*256*sizeof(float));
    cudaMalloc(&d_output,   4*sizeof(float));

    void* bindings[3];
    bindings[templateIdx] = d_template;
    bindings[searchIdx]   = d_search;
    bindings[outputIdx]   = d_output;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::vector<float> h_template(3*128*128);
    std::vector<float> h_search(3*256*256);
    std::vector<float> h_output(4);

    // ── chạy từng sequence ─────────────────────────────────────────────────────
    mkdirP(outputDir);
    int ok = 0, fail = 0;
    for (const std::string& seqName : listSubDirs(datasetDir)) {
        std::cout << "\n[SEQ] " << seqName << "\n";
        bool r = processSequence(datasetDir + "/" + seqName, seqName, outputDir,
                                 context,
                                 d_template, d_search, d_output,
                                 h_template, h_search, h_output,
                                 bindings, stream);
        r ? ok++ : fail++;
    }

    // ── cleanup ────────────────────────────────────────────────────────────────
    cudaStreamDestroy(stream);
    cudaFree(d_template);
    cudaFree(d_search);
    cudaFree(d_output);
    context->destroy();
    engine->destroy();
    runtime->destroy();

    std::cout << "\nDone. OK=" << ok << "  FAIL=" << fail
              << "  results -> " << outputDir << "\n";
    return 0;
}
