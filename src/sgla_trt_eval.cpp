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
#include <chrono>
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
static cv::Mat sampleTarget(const cv::Mat& im, const BBox& bb,     //H = W ?
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

static BBox clipBox(const BBox& box, int H, int W, int margin = 10) {  //margin = 10 ??
    float x1 = std::min(std::max(0.f, box.x),               (float)(W - margin));
    float y1 = std::min(std::max(0.f, box.y),               (float)(H - margin));
    float x2 = std::min(std::max((float)margin, box.x+box.w),(float)W);
    float y2 = std::min(std::max((float)margin, box.y+box.h),(float)H);
    return { x1, y1, std::max((float)margin, x2-x1),
                     std::max((float)margin, y2-y1) };
}

static std::string baseNameNoExt(const std::string& path) {
    size_t slash = path.find_last_of("/\\");
    std::string name = (slash == std::string::npos) ? path : path.substr(slash + 1);
    size_t dot = name.find_last_of('.');
    return (dot == std::string::npos) ? name : name.substr(0, dot);
}

static double msSince(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
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
static bool processSequence(const std::string& seqPath,
                             bool isFileInput,
                             const std::string& seqName,
                             const std::string& outputDir,
                             nvinfer1::IExecutionContext* context,
                             void* d_template, void* d_search, void* d_output,
                             std::vector<float>& h_template,
                             std::vector<float>& h_search,
                             std::vector<float>& h_output,
                             void** bindings,
                             cudaStream_t stream,
                             cudaEvent_t inferStart,
                             cudaEvent_t inferEnd,
                             bool hasInitBox = false,
                             const BBox& initBox = BBox{}) {
    const int tC=3, tH=128, tW=128;
    const int sC=3, sH=256, sW=256;
    const float templateFactor = 2.0f;
    const float searchFactor   = 4.0f;

    // ── mở video ──────────────────────────────────────────────────────────────
    std::string videoPath = isFileInput ? seqPath : seqPath + "/" + seqName + ".mp4";
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "[SKIP] Cannot open: " << videoPath << "\n";
        return false;
    }

    // ── đọc init bbox từ dòng đầu ground_truth.txt hoặc dùng init_box
    BBox state{};
    if (hasInitBox) {
        state = initBox;
    } else {
        std::string gtPath;
        if (isFileInput) {
            size_t slash = videoPath.find_last_of("/\\");
            gtPath = (slash == std::string::npos) ? "groundtruth_rect.txt"
                                                  : videoPath.substr(0, slash + 1) + "groundtruth_rect.txt";
        } else {
            gtPath = seqPath + "/groundtruth_rect.txt";
        }
        if (!loadInitBox(gtPath, state)) {
            std::cerr << "[SKIP] Cannot read init bbox for: " << seqName << "\n";
            return false;
        }
    }

    // ── frame đầu → template ──────────────────────────────────────────────────
    cv::Mat firstFrame;
    cap >> firstFrame;
    if (firstFrame.empty()) { std::cerr << "[SKIP] Empty first frame.\n"; return false; }

    float rzZ;
    auto tTemplateSampleStart = std::chrono::steady_clock::now();
    cv::Mat templatePatch = sampleTarget(firstFrame, state, templateFactor, tH, rzZ);
    double templateSampleMs = msSince(tTemplateSampleStart);

    auto tTemplatePreStart = std::chrono::steady_clock::now();
    preprocess(templatePatch, h_template.data(), tC, tH, tW);
    double templatePreMs = msSince(tTemplatePreStart);

    auto tTemplateUploadStart = std::chrono::steady_clock::now();
    cudaMemcpyAsync(d_template, h_template.data(),
                    tC*tH*tW*sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    double templateUploadMs = msSince(tTemplateUploadStart);

    // ── bắt đầu ghi file ──────────────────────────────────────────────────────
    std::string outPath = outputDir + "/" + seqName + "_predictions.txt";
    std::string timePath = outputDir + "/" + seqName + "_timing.txt";
    std::ofstream fout(outPath);
    std::ofstream t_out(timePath);
    fout << std::fixed << std::setprecision(8);
    t_out << "# sequence: " << seqName << "\n";
    t_out << "# columns: frame sample_ms preprocess_ms upload_ms infer_ms output_copy_ms post_decode_ms clip_ms total_ms\n";

    // ghi frame 0: dùng lại init box (chưa có inference)
    fout << "# sequence: " << seqName << "\n";
    fout << "# prediction columns: x y w h\n";
    fout << state.x << " " << state.y << " " << state.w << " " << state.h << "\n";
    std::cout << "[TEMPLATE] sample=" << templateSampleMs
              << " preprocess=" << templatePreMs
              << " upload=" << templateUploadMs << " ms\n";
    std::cout << "[OUTPUT] predictions -> " << outPath << "\n";
    std::cout << "[OUTPUT] timing      -> " << timePath << "\n";

    // ── loop frames ───────────────────────────────────────────────────────────
    cv::Mat frame;
    int frameIdx = 1;
    double totalSampleMs = 0.0, totalPreMs = 0.0, totalUploadMs = 0.0;
    double totalInferMs = 0.0, totalCopyMs = 0.0, totalDecodeMs = 0.0, totalClipMs = 0.0;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        int H = frame.rows, W = frame.cols;

        float rz;
        auto tSampleStart = std::chrono::steady_clock::now();
        cv::Mat searchPatch = sampleTarget(frame, state, searchFactor, sH, rz);
        double sampleMs = msSince(tSampleStart);

        auto tPreStart = std::chrono::steady_clock::now();
        preprocess(searchPatch, h_search.data(), sC, sH, sW);
        double preprocessMs = msSince(tPreStart);

        auto tUploadStart = std::chrono::steady_clock::now();
        cudaMemcpyAsync(d_search, h_search.data(),
                        sC*sH*sW*sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
        double uploadMs = msSince(tUploadStart);

        cudaEventRecord(inferStart, stream);
        context->enqueueV2(bindings, stream, nullptr);
        cudaEventRecord(inferEnd, stream);
        cudaEventSynchronize(inferEnd);
        float inferMs = 0.f;
        cudaEventElapsedTime(&inferMs, inferStart, inferEnd);

        auto tCopyStart = std::chrono::steady_clock::now();
        cudaMemcpyAsync(h_output.data(), d_output,
                        4*sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        double outputCopyMs = msSince(tCopyStart);

        auto tDecodeStart = std::chrono::steady_clock::now();
        fout << h_output[0] << " "
             << h_output[1] << " "
             << h_output[2] << " "
             << h_output[3] << "\n";

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
        double postDecodeMs = msSince(tDecodeStart);

        auto tClipStart = std::chrono::steady_clock::now();
        state = clipBox(newBox, H, W, 10);
        double clipMs = msSince(tClipStart);

        double totalMs = sampleMs + preprocessMs + uploadMs + inferMs + outputCopyMs + postDecodeMs + clipMs;
        t_out << frameIdx << " " << sampleMs << " " << preprocessMs << " "
              << uploadMs << " " << inferMs << " "
              << outputCopyMs << " " << postDecodeMs << " "
              << clipMs << " " << totalMs << "\n";

        std::cout << "[FRAME " << frameIdx << "] pred="
                  << h_output[0] << " " << h_output[1] << " "
                  << h_output[2] << " " << h_output[3]
                  << " time_ms=" << totalMs
                  << " ([sample=" << sampleMs
                  << " preprocess=" << preprocessMs
                  << " upload=" << uploadMs
                  << " infer=" << inferMs
                  << " copy=" << outputCopyMs
                  << " decode=" << postDecodeMs
                  << " clip=" << clipMs << "])\n";

        totalSampleMs += sampleMs;
        totalPreMs += preprocessMs;
        totalUploadMs += uploadMs;
        totalInferMs += inferMs;
        totalCopyMs += outputCopyMs;
        totalDecodeMs += postDecodeMs;
        totalClipMs += clipMs;
        frameIdx++;
    }

    cap.release();
    int processedFrames = frameIdx - 1;
    if (processedFrames > 0) {
        std::cout << "[SUMMARY] " << seqName
                  << " frames=" << processedFrames
                  << " avg_sample=" << (totalSampleMs / processedFrames)
                  << " avg_preprocess=" << (totalPreMs / processedFrames)
                  << " avg_upload=" << (totalUploadMs / processedFrames)
                  << " avg_infer=" << (totalInferMs / processedFrames)
                  << " avg_copy=" << (totalCopyMs / processedFrames)
                  << " avg_decode=" << (totalDecodeMs / processedFrames)
                  << " avg_clip=" << (totalClipMs / processedFrames)
                  << " ms\n";
    }

    std::cout << "[DONE] " << seqName << "  frames=" << frameIdx
              << "  -> " << outPath << "  / " << timePath << "\n";
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    std::string enginePath, datasetDir, videoPath, outputDir = "./eval_results";
    bool hasInitBox = false;
    BBox initBox{};

    for (int i = 1; i < argc; i++) {
        std::string a(argv[i]);
        if      (a == "--engine"      && i+1 < argc) enginePath = argv[++i];
        else if (a == "--dataset"     && i+1 < argc) datasetDir = argv[++i];
        else if (a == "--video"       && i+1 < argc) videoPath  = argv[++i];
        else if (a == "--output_dir"  && i+1 < argc) outputDir  = argv[++i];
        else if (a == "--init_bbox"   && i+4 < argc) {
            initBox.x = std::stof(argv[++i]);
            initBox.y = std::stof(argv[++i]);
            initBox.w = std::stof(argv[++i]);
            initBox.h = std::stof(argv[++i]);
            hasInitBox = true;
        }
    }

    if (enginePath.empty() || (datasetDir.empty() && videoPath.empty())) {
        std::cerr << "Usage: ./sgla_eval_trt --engine <model.engine>"
                     " --dataset <Test_10_videos> | --video <video.mp4> [--init_bbox x y w h]"
                     " [--output_dir <./eval_results>]\n";
        return -1;
    }
    if (!datasetDir.empty() && !videoPath.empty()) {
        std::cerr << "ERROR: specify only one of --dataset or --video\n";
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
    cudaMalloc(&d_template, 3*128*128*sizeof(float));   // 128 ???  Outputscore
    cudaMalloc(&d_search,   3*256*256*sizeof(float));   // 256 ??
    cudaMalloc(&d_output,   4*sizeof(float));

    void* bindings[3];
    bindings[templateIdx] = d_template;
    bindings[searchIdx]   = d_search;
    bindings[outputIdx]   = d_output;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t inferStart, inferEnd;
    cudaEventCreate(&inferStart);
    cudaEventCreate(&inferEnd);

    std::vector<float> h_template(3*128*128);
    std::vector<float> h_search(3*256*256);
    std::vector<float> h_output(4);

    // ── chạy từng sequence ─────────────────────────────────────────────────────
    mkdirP(outputDir);
    int ok = 0, fail = 0;
    if (!videoPath.empty()) {
        std::string seqName = baseNameNoExt(videoPath);
        std::cout << "\n[SEQ] " << seqName << "\n";
        bool r = processSequence(videoPath, true, seqName, outputDir,
                                 context,
                                 d_template, d_search, d_output,
                                 h_template, h_search, h_output,
                                 bindings, stream,
                                 inferStart, inferEnd,
                                 hasInitBox, initBox);
        r ? ok++ : fail++;
    } else {
        for (const std::string& seqName : listSubDirs(datasetDir)) {
            std::cout << "\n[SEQ] " << seqName << "\n";
            bool r = processSequence(datasetDir + "/" + seqName, false, seqName, outputDir,
                                     context,
                                     d_template, d_search, d_output,
                                     h_template, h_search, h_output,
                                     bindings, stream,
                                     inferStart, inferEnd);
            r ? ok++ : fail++;
        }
    }

    // ── cleanup ────────────────────────────────────────────────────────────────
    cudaEventDestroy(inferStart);
    cudaEventDestroy(inferEnd);
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
