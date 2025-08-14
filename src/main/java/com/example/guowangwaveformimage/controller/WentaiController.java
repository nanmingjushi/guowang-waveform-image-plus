package com.example.guowangwaveformimage.controller;

/* 
    @author nanchao 
    @date 2025/8/14
*/


import org.bytedeco.javacpp.Loader;
import org.bytedeco.opencv.opencv_java;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Rect;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.util.*;
import java.util.stream.Collectors;


@RestController
@RequestMapping("/wentai")
public class WentaiController {


    // opencv 加载
    static {
        try {
            // 让 JavaCPP 自动解包并加载正确的本地库
            Loader.load(opencv_java.class);
            System.out.println("OpenCV native libs loaded by JavaCPP.");
        } catch (Throwable e) {
            e.printStackTrace();
            // 失败时抛出，方便尽早发现问题（不要只吞掉异常）
            throw new RuntimeException("Failed to load OpenCV native libraries", e);
        }
    }


    // 配置
    private static final Rect ROI_A = new Rect(55, 56, 1400, 310);
    private static final Rect ROI_B = new Rect(55, 370, 1400, 310);
    private static final Rect ROI_C = new Rect(55, 683, 1400, 310);

    private static final double PER_SEGMENT_VALUE = 200000.0;   // 每段代表的幅值(V)
    private static final int    DASH_FALLBACK_PIXELS = 300;      // 找不到刻度时的兜底像素
    private static final double STEADY_WINDOW_RIGHT_PORTION = 0.40; // 只取右侧40%做稳态
    private static final int    HSV_S_THRESH = 40;               // 波形像素筛选：S阈
    private static final int    HSV_V_THRESH = 40;               // 波形像素筛选：V阈

    //返回结构
    public static class PhaseResult {
        public String phase;            // A/B/C
        public Double steadyPeakV;      // 稳态峰值(相对中线的幅值, V)
        public Double steadyRmsV;       // 稳态RMS(=峰值/√2 假设正弦)
        public Double sampleRmsV;       // 采样RMS(从稳态窗口样本直接算)
        public String  error;           // 错误信息（若有）
        public Map<String, Object> debug = new LinkedHashMap<>();
    }

    public static class FileResult {
        public String file;
        public List<PhaseResult> phases = new ArrayList<>();
    }


    @PostMapping("/upload")
    public ResponseEntity<?> uploadImages(@RequestParam("files") MultipartFile[] files) {
        List<FileResult> out = Arrays.stream(files).map(this::analyzeOneFile)
                .collect(Collectors.toList());
        return ResponseEntity.ok(out);
    }


    /* ==================== 主流程：单文件 -> 三相稳态 ==================== */
    private FileResult analyzeOneFile(MultipartFile file) {
        FileResult r = new FileResult();
        r.file = file.getOriginalFilename();

        Mat img = null;
        try {
            byte[] bytes = file.getBytes();
            Mat buf = new MatOfByte(bytes);
            img = Imgcodecs.imdecode(buf, Imgcodecs.IMREAD_COLOR);
            if (img == null || img.empty()) {
                r.phases.add(errPhase("A", "图片解码失败"));
                r.phases.add(errPhase("B", "图片解码失败"));
                r.phases.add(errPhase("C", "图片解码失败"));
                return r;
            }

            r.phases.add(analyzeSteadyOnePhase(img.submat(ROI_A), "A"));
            r.phases.add(analyzeSteadyOnePhase(img.submat(ROI_B), "B"));
            r.phases.add(analyzeSteadyOnePhase(img.submat(ROI_C), "C"));

        } catch (Exception e) {
            r.phases.add(errPhase("A", e.getMessage()));
            r.phases.add(errPhase("B", e.getMessage()));
            r.phases.add(errPhase("C", e.getMessage()));
        } finally {
            if (img != null) img.release();
        }
        return r;
    }

    /* ==================== 单相稳态分析 ==================== */
    private PhaseResult analyzeSteadyOnePhase(Mat roi, String phaseName) {
        PhaseResult pr = new PhaseResult();
        pr.phase = phaseName;

        // 1) 找三条横向黑实线（上/中/下）
        List<Integer> blackLines = detectHorizontalBlackLines(roi, 0.6);
        if (blackLines.size() < 3) {
            pr.error = "检测到的黑实线不足3条";
            pr.debug.put("blackLines", blackLines);
            return pr;
        }
        Collections.sort(blackLines);
        int y1 = blackLines.get(0), y2 = blackLines.get(blackLines.size()/2), y3 = blackLines.get(blackLines.size()-1);

        // 2) 找横向虚线刻度（用于像素->电压的标定）
        List<Integer> dashYs = detectHorizontalDashLines(roi, y1, y3);

        // 3) 逐列跟踪波形 y(x)，只在 y1~y3 范围
        int[] yTrace = traceWaveYPerColumn(roi, y1, y3);

        // 4) 取右侧稳态窗口
        int w = roi.width();
        int xStart = (int)Math.round(w * (1.0 - STEADY_WINDOW_RIGHT_PORTION));
        xStart = Math.max(0, Math.min(w-1, xStart));
        List<Integer> ys = new ArrayList<>();
        for (int x = xStart; x < w; x++) {
            if (yTrace[x] >= 0) ys.add(yTrace[x]);
        }
        if (ys.size() < 10) {
            pr.error = "稳态窗口有效样本不足";
            pr.debug.put("xStart", xStart);
            return pr;
        }

        // 5) 在窗口内找峰值/谷值（距中线更大的那个视为稳态峰值）
        int ymax = ys.stream().max(Integer::compareTo).orElse(y2);
        int ymin = ys.stream().min(Integer::compareTo).orElse(y2);
        // 与中线的像素偏差
        int devTop    = Math.abs(ymin - y2); // 上峰（像素更小）
        int devBottom = Math.abs(ymax - y2); // 下峰
        int peakY = (devTop >= devBottom) ? ymin : ymax;

        // 6) 像素 -> 电压（分段+线性插值；找不到刻度就用兜底像素）
        double steadyPeakV = pixelToVoltByDashes(peakY, y2, dashYs);

        // 7) RMS(假设正弦) = 峰值/√2
        double steadyRms = steadyPeakV / Math.sqrt(2.0);

        // 8) 采样RMS（把窗口内每个 y 映射成电压，再直接算 RMS，抗失真）
        double[] vSamples = new double[w - xStart];
        int idx = 0;
        for (int x = xStart; x < w; x++) {
            int yy = yTrace[x];
            if (yy >= 0) vSamples[idx++] = pixelToVoltByDashes(yy, y2, dashYs);
        }
        double sampleRms = calcRms(vSamples, idx); // 只对有效样本计

        pr.steadyPeakV = steadyPeakV;
        pr.steadyRmsV  = steadyRms;
        pr.sampleRmsV  = sampleRms;

        // debug信息便于调参
        pr.debug.put("y1y2y3", Arrays.asList(y1,y2,y3));
        pr.debug.put("dashYs", dashYs);
        pr.debug.put("xStart", xStart);
        pr.debug.put("windowSamples", idx);
        return pr;
    }

    /* ==================== 工具函数 ==================== */

    // 基于水平投影+二值化找“长黑横线”
    private List<Integer> detectHorizontalBlackLines(Mat roi, double runRatioThresh) {
        Mat gray = new Mat(); Imgproc.cvtColor(roi, gray, Imgproc.COLOR_BGR2GRAY);
        Mat bin  = new Mat(); Imgproc.threshold(gray, bin, 0, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);
        int h = bin.rows(), w = bin.cols();
        byte[] data = new byte[h*w];
        bin.get(0,0,data);

        List<Integer> lines = new ArrayList<>();
        for (int y=0; y<h; y++) {
            int rowOff = y*w;
            int blackRun = 0, maxRun = 0;
            for (int x=0; x<w; x++) {
                int v = data[rowOff+x] & 0xFF;
                if (v > 0) { // 黑
                    blackRun++;
                    if (blackRun > maxRun) maxRun = blackRun;
                } else blackRun = 0;
            }
            // 若该行存在较长连续黑段（例如占宽度 60%）
            if (maxRun >= w * runRatioThresh) lines.add(y);
        }
        gray.release(); bin.release();
        return lines;
    }

    // 在 [y1,y3] 范围内找“虚线刻度”：按行统计黑像素密度，取多个峰值行
    private List<Integer> detectHorizontalDashLines(Mat roi, int y1, int y3) {
        Mat gray = new Mat(); Imgproc.cvtColor(roi, gray, Imgproc.COLOR_BGR2GRAY);
        Mat bin  = new Mat(); Imgproc.adaptiveThreshold(gray, bin, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C,
                Imgproc.THRESH_BINARY_INV, 15, 10);
        int h = bin.rows(), w = bin.cols();
        y1 = Math.max(0, y1); y3 = Math.min(h-1, y3);

        byte[] data = new byte[h*w];
        bin.get(0,0,data);

        double[] rowSum = new double[h];
        for (int y=y1; y<=y3; y++) {
            int off=y*w, cnt=0;
            for (int x=0; x<w; x++) if ((data[off+x]&0xFF) > 0) cnt++;
            rowSum[y] = cnt;
        }
        // 平滑
        double[] sm = new double[h];
        int win=5;
        for (int y=y1; y<=y3; y++) {
            int L=Math.max(y1,y-win), R=Math.min(y3,y+win);
            double s=0; int c=0;
            for(int k=L;k<=R;k++){ s+=rowSum[k]; c++; }
            sm[y]=s/c;
        }
        // 峰值
        double mean = 0; int c=0;
        for (int y=y1; y<=y3; y++){ mean += sm[y]; c++; }
        mean/=Math.max(1,c);

        List<Integer> peaks = new ArrayList<>();
        for (int y=y1+1; y<y3; y++) {
            if (sm[y] > sm[y-1] && sm[y] > sm[y+1] && sm[y] > mean*1.2) peaks.add(y);
        }
        // 合并近邻
        List<Integer> merged = new ArrayList<>();
        int tol=3;
        for (int y: peaks){
            if (merged.isEmpty() || y-merged.get(merged.size()-1)>tol) merged.add(y);
            else {
                int prev=merged.get(merged.size()-1);
                merged.set(merged.size()-1,(prev+y)/2);
            }
        }
        gray.release(); bin.release();
        return merged;
    }

    // 逐列提取“波形代表y”：在 y1~y3 中寻找 S/V 足够的像素（近似波形颜色）
    private int[] traceWaveYPerColumn(Mat roi, int y1, int y3) {
        Mat hsv = new Mat(); Imgproc.cvtColor(roi, hsv, Imgproc.COLOR_BGR2HSV);
        int h = roi.rows(), w = roi.cols();
        y1 = Math.max(0,y1); y3 = Math.min(h-1,y3);

        int[] ys = new int[w];
        Arrays.fill(ys, -1);

        for (int x=0; x<w; x++) {
            int found=-1;
            for (int y=y1; y<=y3; y++) {
                double[] p = hsv.get(y,x);
                if (p==null) continue;
                double S=p[1], V=p[2];
                if (S > HSV_S_THRESH && V > HSV_V_THRESH) { found=y; break; }
            }
            ys[x]=found;
        }
        // 简单平滑
        int k=3;
        int[] sm = new int[w];
        for (int x=0; x<w; x++) {
            int L=Math.max(0,x-k), R=Math.min(w-1,x+k);
            int cnt=0, sum=0;
            for (int i=L;i<=R;i++){
                if (ys[i]>=0){ cnt++; sum+=ys[i]; }
            }
            sm[x] = cnt==0 ? -1 : (sum/cnt);
        }
        hsv.release();
        return sm;
    }

    // 把某个 y 映射到电压值（V）：以中线 y2 为 0，上方为正
    private double pixelToVoltByDashes(int y, int y2, List<Integer> dashYs) {
        if (y < 0) return Double.NaN;
        if (y < y2) { // 上方：正
            int prev = y2; int section = 0;
            List<Integer> up = new ArrayList<>();
            for (int d : dashYs) if (d < y2) up.add(d);
            Collections.sort(up, Comparator.reverseOrder()); // 从近到远：y2上方最近的开始
            for (int d : up) {
                if (y <= d) {
                    double ratio = (prev - y) * 1.0 / (prev - d);
                    return section * PER_SEGMENT_VALUE + ratio * PER_SEGMENT_VALUE;
                }
                prev = d; section++;
            }
            // 超出最上面一段
            if (!up.isEmpty()) {
                double ratio = (prev - y) * 1.0 / (prev - up.get(up.size() - 1));
                return section * PER_SEGMENT_VALUE + ratio * PER_SEGMENT_VALUE;
            } else {
                return (y2 - y) * PER_SEGMENT_VALUE / DASH_FALLBACK_PIXELS;
            }
        } else if (y == y2) {
            return 0.0;
        } else { // 下方：负
            int prev = y2; int section = 0;
            List<Integer> down = new ArrayList<>();
            for (int d : dashYs) if (d > y2) down.add(d);
            Collections.sort(down); // 从近到远
            for (int d : down) {
                if (y >= d) {
                    double ratio = (y - prev) * 1.0 / (d - prev);
                    return -(section * PER_SEGMENT_VALUE + ratio * PER_SEGMENT_VALUE);
                }
                prev = d; section++;
            }
            if (!down.isEmpty()) {
                double ratio = (y - prev) * 1.0 / (down.get(down.size() - 1) - prev);
                return -(section * PER_SEGMENT_VALUE + ratio * PER_SEGMENT_VALUE);
            } else {
                return -(y - y2) * PER_SEGMENT_VALUE / DASH_FALLBACK_PIXELS;
            }
        }
    }

    // 采样RMS（对有效样本前 idx 个）
    private double calcRms(double[] arr, int validCount) {
        if (validCount <= 0) return Double.NaN;
        double s=0; int c=0;
        for (int i=0; i<validCount; i++){
            double v = arr[i];
            if (!Double.isNaN(v)) { s += v*v; c++; }
        }
        return c==0 ? Double.NaN : Math.sqrt(s/c);
    }

    private PhaseResult errPhase(String phase, String msg) {
        PhaseResult p = new PhaseResult();
        p.phase = phase; p.error = msg;
        return p;
    }

}
