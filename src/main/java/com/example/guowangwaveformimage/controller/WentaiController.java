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

    static {
        try {
            Loader.load(opencv_java.class);
            System.out.println("OpenCV native libs loaded by JavaCPP.");
        } catch (Throwable e) {
            e.printStackTrace();
            throw new RuntimeException("Failed to load OpenCV native libraries", e);
        }
    }

    // 三相 ROI
    private static final Rect ROI_A = new Rect(55, 56, 1400, 310);
    private static final Rect ROI_B = new Rect(55, 370, 1400, 310);
    private static final Rect ROI_C = new Rect(55, 683, 1400, 310);

    // 刻度：电压每段200000（V），电流每段500（A）
    private static final double VOLT_PER_SEG = 200000.0;
    private static final double CURR_PER_SEG = 500.0;

    private static final int    DASH_FALLBACK_PIXELS = 300;
    private static final double STEADY_WINDOW_RIGHT_PORTION = 0.40;   // 右侧40%
    private static final int    HSV_S_THRESH = 40;
    private static final int    HSV_V_THRESH = 40;

    // 返回结构
    public static class PhaseResult {
        public String phase;            // A/B/C
        public Double steadyPeakV;      // 稳态峰值（已按单位换算：电压=kV，电流=A）
        public Double steadyRmsV;       // 稳态RMS（同上单位，且为正）
        public Double sampleRmsV;       // 采样RMS（同上单位，且为正）
        public String  error;
        public Map<String, Object> debug = new LinkedHashMap<>();
    }
    public static class FileResult {
        public String file;
        public String mode;             // "voltage" / "current"
        public String unit;             // "kV" / "A"
        public List<PhaseResult> phases = new ArrayList<>();
    }

    @PostMapping("/upload")
    public ResponseEntity<?> uploadImages(@RequestParam("files") MultipartFile[] files,
                                          @RequestParam(value = "mode", defaultValue = "voltage") String mode) {

        final boolean isVoltage = !"current".equalsIgnoreCase(mode);
        final double perSegmentValue = isVoltage ? VOLT_PER_SEG : CURR_PER_SEG;
        final double displayScale = isVoltage ? (1.0 / 1000.0) : 1.0; // 电压转kV；电流保持A

        List<FileResult> out = Arrays.stream(files)
                .map(f -> analyzeOneFile(f, perSegmentValue, isVoltage, displayScale, mode))
                .collect(Collectors.toList());
        return ResponseEntity.ok(out);
    }

    /* ==================== 主流程：单文件 -> 三相稳态 ==================== */
    private FileResult analyzeOneFile(MultipartFile file,
                                      double perSegmentValue,
                                      boolean isVoltage,
                                      double displayScale,
                                      String mode) {
        FileResult r = new FileResult();
        r.file = file.getOriginalFilename();
        r.mode = mode;
        r.unit = isVoltage ? "kV" : "A";

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

            r.phases.add(analyzeSteadyOnePhase(img.submat(ROI_A), "A", perSegmentValue, isVoltage, displayScale));
            r.phases.add(analyzeSteadyOnePhase(img.submat(ROI_B), "B", perSegmentValue, isVoltage, displayScale));
            r.phases.add(analyzeSteadyOnePhase(img.submat(ROI_C), "C", perSegmentValue, isVoltage, displayScale));

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
    private PhaseResult analyzeSteadyOnePhase(Mat roi,
                                              String phaseName,
                                              double perSegmentValue,
                                              boolean isVoltage,
                                              double displayScale) {
        PhaseResult pr = new PhaseResult();
        pr.phase = phaseName;

        // 1) 三条黑实线
        List<Integer> blackLines = detectHorizontalBlackLines(roi, 0.6);
        if (blackLines.size() < 3) {
            pr.error = "检测到的黑实线不足3条";
            pr.debug.put("blackLines", blackLines);
            return pr;
        }
        Collections.sort(blackLines);
        int y1 = blackLines.get(0), y2 = blackLines.get(blackLines.size()/2), y3 = blackLines.get(blackLines.size()-1);

        // 2) 虚线刻度
        List<Integer> dashYs = detectHorizontalDashLines(roi, y1, y3);

        // 3) 跟踪波形 y(x)
        int[] yTrace = traceWaveYPerColumn(roi, y1, y3);

        // 4) 右侧 40% 窗口
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

        // 5) 找峰/谷，取离中线更远的
        int ymax = ys.stream().max(Integer::compareTo).orElse(y2);
        int ymin = ys.stream().min(Integer::compareTo).orElse(y2);
        int devTop    = Math.abs(ymin - y2);
        int devBottom = Math.abs(ymax - y2);
        int peakY = (devTop >= devBottom) ? ymin : ymax;
        boolean isUp = (devTop >= devBottom);

        // 6) 像素 -> 物理量（V 或 A）
        double peakVal = pixelToValueByDashes(peakY, y2, dashYs, perSegmentValue, DASH_FALLBACK_PIXELS, isUp);

        // 7) 有效值（正）：理论 RMS = |peak|/√2
        double steadyRms = Math.abs(peakVal) / Math.sqrt(2.0);

        // 8) 采样 RMS（正）：窗口内逐点换算，再算 RMS
        double[] samples = new double[w - xStart];
        int idx = 0;
        for (int x = xStart; x < w; x++) {
            int yy = yTrace[x];
            if (yy >= 0) {
                boolean upHere = yy < y2;
                double val = pixelToValueByDashes(yy, y2, dashYs, perSegmentValue, DASH_FALLBACK_PIXELS, upHere);
                samples[idx++] = Math.abs(val); // 取绝对值，保证RMS为正
            }
        }
        double sampleRms = calcRms(samples, idx);

        // 9) 按显示单位输出：电压->kV，电流->A
        pr.steadyPeakV = peakVal * displayScale;
        pr.steadyRmsV  = steadyRms * displayScale;
        pr.sampleRmsV  = sampleRms * displayScale;

        // 调参信息
        pr.debug.put("y1y2y3", Arrays.asList(y1,y2,y3));
        pr.debug.put("dashYs", dashYs);
        pr.debug.put("xStart", xStart);
        pr.debug.put("windowSamples", idx);
        return pr;
    }

    /* ==================== 工具函数 ==================== */

    // 找长黑横线（实线）
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
            if (maxRun >= w * runRatioThresh) lines.add(y);
        }
        gray.release(); bin.release();
        return lines;
    }

    // 找虚线刻度
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

    // 按列追踪波形 y
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

    // y -> 物理量（上方正，下方负）
    private double pixelToValueByDashes(int y, int y2, List<Integer> dashYs,
                                        double perSeg, int fallbackPixels, boolean isUp) {
        if (y < 0) return Double.NaN;

        List<Integer> all = new ArrayList<>(dashYs);
        all.add(y2);
        Collections.sort(all);

        if (isUp) {
            List<Integer> up = new ArrayList<>();
            for (int d: all) if (d < y2) up.add(d);
            up.sort(Collections.reverseOrder());
            int prev = y2, section = 0;
            for (int d : up) {
                if (y <= d) {
                    double ratio = (prev - y) * 1.0 / (prev - d);
                    return section * perSeg + ratio * perSeg;
                }
                prev = d; section++;
            }
            if (!up.isEmpty()) {
                double ratio = (prev - y) * 1.0 / (prev - up.get(up.size()-1));
                return section * perSeg + ratio * perSeg;
            } else {
                return (y2 - y) * perSeg / fallbackPixels;
            }
        } else {
            List<Integer> down = new ArrayList<>();
            for (int d: all) if (d > y2) down.add(d);
            Collections.sort(down);
            int prev = y2, section = 0;
            for (int d : down) {
                if (y >= d) {
                    double ratio = (y - prev) * 1.0 / (d - prev);
                    return -(section * perSeg + ratio * perSeg);
                }
                prev = d; section++;
            }
            if (!down.isEmpty()) {
                double ratio = (y - prev) * 1.0 / (down.get(down.size()-1) - prev);
                return -(section * perSeg + ratio * perSeg);
            } else {
                return -(y - y2) * perSeg / fallbackPixels;
            }
        }
    }

    // RMS（对前 validCount 个样本；样本值已取绝对）
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
