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
@RequestMapping("/pinlv")
public class PinlvController {

    static {
        try {
            Loader.load(opencv_java.class);
            System.out.println("OpenCV native libs loaded by JavaCPP.");
        } catch (Throwable e) {
            e.printStackTrace();
            throw new RuntimeException("Failed to load OpenCV native libraries", e);
        }
    }

    // 三相 ROI（与前面保持一致）
    private static final Rect ROI_A = new Rect(55, 56, 1400, 310);
    private static final Rect ROI_B = new Rect(55, 370, 1400, 310);
    private static final Rect ROI_C = new Rect(55, 683, 1400, 310);

    // 使用右侧 60% 计算频率（避开最左的暂态）
    private static final double RIGHT_PORTION = 0.60;

    // 纵向“长黑线”判定阈值：一列中黑色连续像素长度占高度的比例
    private static final double VLINE_RUN_RATIO = 0.55;   // 可调 0.5~0.7
    private static final int VLINE_MERGE_PX = 4;          // 竖线去重时的合并容忍像素

    // 彩色像素阈值（HSV）
    private static final int HSV_S_THRESH = 40;
    private static final int HSV_V_THRESH = 40;

    // 每两条竖实线之间的时间（秒）
    private static final double SECONDS_PER_GRID = 0.025;

    // 工频合理范围（秒）用于自相关搜索窗口：40–70 Hz ≈ 25–14.3 ms，再稍微放宽
    private static final double PERIOD_MIN_SEC = 0.012;   // 12 ms
    private static final double PERIOD_MAX_SEC = 0.030;   // 30 ms

    // 自相关最小相关阈值（过低则启用兜底法）
    private static final double AUTOCORR_MIN_SCORE = 0.15;

    /* ---------- 输出结构 ---------- */

    public static class PhaseFreq {
        public String phase;        // A/B/C
        public Double freqHz;       // 频率（Hz）
        public Double periodMs;     // 周期（ms）
        public String  error;       // 异常信息（若有）
        public Map<String,Object> debug = new LinkedHashMap<>();
    }

    public static class FileFreqResult {
        public String file;
        public List<PhaseFreq> phases = new ArrayList<>();
    }

    /* ---------- 接口 ---------- */

    @PostMapping("/upload")
    public ResponseEntity<?> uploadImages(@RequestParam("files") MultipartFile[] files) {
        List<FileFreqResult> out = Arrays.stream(files)
                .map(this::analyzeOneFile)
                .collect(Collectors.toList());
        return ResponseEntity.ok(out);
    }

    /* ---------- 单文件 -> 三相 ---------- */

    private FileFreqResult analyzeOneFile(MultipartFile file) {
        FileFreqResult r = new FileFreqResult();
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

            r.phases.add(analyzePhaseFreq(img.submat(ROI_A), "A"));
            r.phases.add(analyzePhaseFreq(img.submat(ROI_B), "B"));
            r.phases.add(analyzePhaseFreq(img.submat(ROI_C), "C"));

        } catch (Exception e) {
            r.phases.add(errPhase("A", e.getMessage()));
            r.phases.add(errPhase("B", e.getMessage()));
            r.phases.add(errPhase("C", e.getMessage()));
        } finally {
            if (img != null) img.release();
        }
        return r;
    }

    /* ---------- 核心：单相频率 ---------- */

    private PhaseFreq analyzePhaseFreq(Mat roi, String phaseName) {
        PhaseFreq out = new PhaseFreq();
        out.phase = phaseName;

        int h = roi.rows(), w = roi.cols();

        // 1) 竖实线识别（整幅 ROI 内做）
        List<Integer> vlines = detectVerticalBlackLines(roi, VLINE_RUN_RATIO);
        if (vlines.size() < 2) {
            out.error = "竖实线检测不足，无法标定时间刻度";
            out.debug.put("vlines", vlines);
            return out;
        }
        Collections.sort(vlines);

        // 去掉极端边界（靠近0或w-1的）
        if (!vlines.isEmpty() && vlines.get(0) < 5) vlines.remove(0);
        if (!vlines.isEmpty() && vlines.get(vlines.size()-1) > w - 6) vlines.remove(vlines.size()-1);
        if (vlines.size() < 2) {
            out.error = "有效竖实线不足";
            out.debug.put("vlines_trim", vlines);
            return out;
        }

        // 2) 网格像素（用中位数） -> 秒/像素
        List<Integer> diffs = new ArrayList<>();
        for (int i=1;i<vlines.size();i++){
            int d = vlines.get(i) - vlines.get(i-1);
            if (d > 1) diffs.add(d);
        }
        if (diffs.isEmpty()) {
            out.error = "竖线间距异常";
            out.debug.put("vline_diffs", diffs);
            return out;
        }
        Collections.sort(diffs);
        double pixelsPerGrid = diffs.get(diffs.size()/2);
        double secondsPerPixel = SECONDS_PER_GRID / Math.max(1.0, pixelsPerGrid);

        // 3) 右侧 60% 窗口
        int xStart = (int)Math.round(w * (1.0 - RIGHT_PORTION));
        xStart = Math.max(0, Math.min(w-2, xStart));

        // 4) 跟踪波形 y(x)：取每列彩色像素上下边中点 + 平滑
        int[] yTrace = traceWaveYCenterPerColumn(roi, 0, h-1);
        int[] yWin = Arrays.copyOfRange(yTrace, xStart, w);

        // 4.1 转成连续有效样本并去均值
        double[] sig = compactValid(yWin);
        if (sig.length < 30) {
            out.error = "稳态窗口有效样本不足";
            out.debug.put("xStart", xStart);
            return out;
        }
        // 平滑（移动平均 3～5）
        sig = movingAverage(sig, 3);
        // 去均值（自相关用）
        double mean = Arrays.stream(sig).average().orElse(0);
        for (int i=0;i<sig.length;i++) sig[i] -= mean;

        // 5) 自相关估计周期（限制在 12–30ms 转成像素的范围）
        int minLagPx = Math.max(3, (int)Math.round(PERIOD_MIN_SEC / secondsPerPixel));
        int maxLagPx = Math.min(sig.length/2, (int)Math.round(PERIOD_MAX_SEC / secondsPerPixel));
        if (minLagPx >= maxLagPx) {
            out.error = "可搜索的周期像素范围无效";
            out.debug.put("minLagPx", minLagPx);
            out.debug.put("maxLagPx", maxLagPx);
            return out;
        }

        AutoCorrResult ar = bestAutocorrLag(sig, minLagPx, maxLagPx);
        double periodPx;
        if (ar.score >= AUTOCORR_MIN_SCORE) {
            periodPx = ar.lag;
        } else {
            // 兜底：用同类型极值的相邻距离（峰/谷），再取中位数
            int[] peaks = findExtremaIndices(sig); // 在平滑、去均值后的信号上找极值
            List<Integer> deltas = new ArrayList<>();
            for (int i=1;i<peaks.length;i++){
                int d = peaks[i] - peaks[i-1];
                if (d >= minLagPx && d <= maxLagPx) deltas.add(d);
            }
            if (deltas.isEmpty()) {
                out.error = "极值不足或间距异常";
                out.debug.put("autocorr_score", ar.score);
                return out;
            }
            Collections.sort(deltas);
            periodPx = deltas.get(deltas.size()/2);
        }

        double Tsec = periodPx * secondsPerPixel;
        double freq = (Tsec > 0) ? (1.0 / Tsec) : Double.NaN;

        out.periodMs = sanitizeNumber(Tsec * 1000.0);
        out.freqHz   = sanitizeNumber(freq);

        // debug
        out.debug.put("vlines", vlines);
        out.debug.put("pixelsPerGrid", pixelsPerGrid);
        out.debug.put("secondsPerPixel", secondsPerPixel);
        out.debug.put("xStart", xStart);
        out.debug.put("sig_len", sig.length);
        out.debug.put("auto_score", ar.score);
        out.debug.put("auto_lag_px", ar.lag);
        return out;
    }

    /* ---------- 辅助：竖实线检测（逐列最大黑连通长度） ---------- */
    private List<Integer> detectVerticalBlackLines(Mat roi, double runRatio) {
        Mat gray = new Mat(); Imgproc.cvtColor(roi, gray, Imgproc.COLOR_BGR2GRAY);
        Mat bin  = new Mat(); Imgproc.threshold(gray, bin, 0, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);
        int h = bin.rows(), w = bin.cols();

        byte[] data = new byte[h*w];
        bin.get(0,0,data);

        List<Integer> xs = new ArrayList<>();
        int minRun = (int)(h * runRatio);
        for (int x=0;x<w;x++){
            int maxRun=0, run=0;
            for (int y=0;y<h;y++){
                int v = data[y*w + x] & 0xFF;
                if (v > 0) { run++; if (run>maxRun) maxRun=run; }
                else run=0;
            }
            if (maxRun >= minRun) xs.add(x);
        }
        // 合并近邻列
        List<Integer> merged = new ArrayList<>();
        Integer curStart = null, curEnd = null;
        for (int x : xs){
            if (curEnd == null || x - curEnd <= VLINE_MERGE_PX){
                if (curStart == null) curStart = x;
                curEnd = x;
            } else {
                merged.add((curStart + curEnd)/2);
                curStart = x; curEnd = x;
            }
        }
        if (curStart != null) merged.add((curStart + curEnd)/2);

        gray.release(); bin.release();
        return merged;
    }

    /* ---------- 辅助：按列取“彩色像素上下边中点” ---------- */
    private int[] traceWaveYCenterPerColumn(Mat roi, int y1, int y3) {
        Mat hsv = new Mat(); Imgproc.cvtColor(roi, hsv, Imgproc.COLOR_BGR2HSV);
        int h = roi.rows(), w = roi.cols();
        y1 = Math.max(0, y1); y3 = Math.min(h-1, y3);

        int[] ys = new int[w];
        Arrays.fill(ys, -1);

        for (int x=0; x<w; x++) {
            int top=-1, bottom=-1;
            for (int y=y1; y<=y3; y++) {
                double[] p = hsv.get(y,x);
                if (p == null) continue;
                double S=p[1], V=p[2];
                if (S > HSV_S_THRESH && V > HSV_V_THRESH) {
                    if (top == -1) top = y;
                    bottom = y;
                }
            }
            if (top != -1) ys[x] = (top + bottom) / 2;
        }
        hsv.release();

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
        return sm;
    }

    /* ---------- 自相关估计最佳滞后 ---------- */
    private static class AutoCorrResult { double score; int lag; }
    private AutoCorrResult bestAutocorrLag(double[] s, int minLag, int maxLag){
        AutoCorrResult r = new AutoCorrResult();
        r.score = -1e9; r.lag = minLag;

        // 归一化能量
        double energy = 0;
        for (double v : s) energy += v*v;
        if (energy <= 1e-12) { r.score = 0; r.lag=minLag; return r; }

        for (int k=minLag; k<=maxLag; k++){
            double acc=0;
            int n = s.length - k;
            for (int i=0;i<n;i++) acc += s[i] * s[i+k];
            // 归一化到 [−1,1] 近似
            double score = acc / Math.sqrt(energy * energy);
            if (score > r.score){
                r.score = score; r.lag = k;
            }
        }
        return r;
    }

    /* ---------- 极值兜底 ---------- */
    private int[] findExtremaIndices(double[] s){
        // 简易导数符号法 + 最小间距（用长度的3%）
        int n = s.length;
        int[] sgn = new int[n];
        for (int i=1;i<n;i++){
            double d = s[i] - s[i-1];
            sgn[i] = (d>0)?1:((d<0)?-1:0);
        }
        List<Integer> maxima = new ArrayList<>();
        List<Integer> minima = new ArrayList<>();
        for (int i=1;i<n-1;i++){
            if (sgn[i-1] > 0 && sgn[i] < 0) maxima.add(i);
            if (sgn[i-1] < 0 && sgn[i] > 0) minima.add(i);
        }
        List<Integer> chosen = (maxima.size() >= minima.size()) ? maxima : minima;

        int minGap = Math.max(3, (int)(n * 0.03));
        List<Integer> filtered = new ArrayList<>();
        int last = -10000;
        for (int p : chosen){
            if (p - last >= minGap) { filtered.add(p); last = p; }
        }
        return filtered.stream().mapToInt(i->i).toArray();
    }

    /* ---------- 小工具 ---------- */
    private double[] compactValid(int[] y){
        // 去掉 -1 无效值，并转 double（反转使“上为正”不是必须，这里只做相对变化）
        List<Double> v = new ArrayList<>();
        for (int value : y) if (value >= 0) v.add((double) value);
        double[] out = new double[v.size()];
        for (int i=0;i<v.size();i++) out[i] = v.get(i);
        return out;
    }

    private double[] movingAverage(double[] s, int win){
        if (win <= 1) return s;
        int n = s.length;
        double[] out = new double[n];
        double sum = 0;
        int half = win/2;
        for (int i=0;i<n;i++){
            int L = Math.max(0, i-half);
            int R = Math.min(n-1, i+half);
            sum = 0;
            for (int k=L;k<=R;k++) sum += s[k];
            out[i] = sum / (R-L+1);
        }
        return out;
    }

    private static Double sanitizeNumber(double v){
        if (Double.isNaN(v) || Double.isInfinite(v)) return null;
        return v;
    }

    private PhaseFreq errPhase(String phase, String msg) {
        PhaseFreq p = new PhaseFreq();
        p.phase = phase; p.error = msg;
        return p;
    }
}
