package com.example.guowangwaveformimage.controller;

/*
    功率计算（带调试输出）：
    - 输入：一张三相电压图 + 一张三相电流图（同一时间窗口）
    - 每相：右侧60%稳态窗口，分别提取 v(x)、i(x)，做幅值标定并时间对齐
    - 结果：Vrms(kV)、Irms(A)、P(kW)、S(kVA)、PF
    - 调试：打印黑实线/虚线、窗口、映射后的时间序列、电压电流RMS与功率
*/

import org.bytedeco.javacpp.Loader;
import org.bytedeco.opencv.opencv_java;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Rect;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.*;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/gonglv")
public class GonglvController {

    /** 打开后端调试输出 */
    private static final boolean DEBUG = true;

    static {
        try {
            Loader.load(opencv_java.class);
            if (DEBUG) System.out.println("OpenCV native libs loaded by JavaCPP.");
        } catch (Throwable e) {
            e.printStackTrace();
            throw new RuntimeException("Failed to load OpenCV native libraries", e);
        }
    }

    // 三相 ROI（与你项目中的保持一致）
    private static final Rect ROI_A = new Rect(55, 56, 1400, 310);
    private static final Rect ROI_B = new Rect(55, 370, 1400, 310);
    private static final Rect ROI_C = new Rect(55, 683, 1400, 310);

    // 右侧稳态窗口比例
    private static final double RIGHT_PORTION = 0.60;

    // 每段物理量（虚线刻度）：电压=200000，电流=500
    private static final double VOLT_PER_SEG = 200000.0;
    private static final double CURR_PER_SEG = 500.0;

    // 找虚线/实线等的参数
    private static final int HSV_S_THRESH = 40;
    private static final int HSV_V_THRESH = 40;
    private static final double HLINE_RUN_RATIO = 0.60;   // 水平黑实线占宽度阈值
    private static final int    HLINE_MERGE_PX   = 10;
    private static final int    DASH_SMOOTH_WIN  = 5;
    private static final double DASH_PEAK_GAIN   = 1.2;   // 相对均值阈

    /* ===== 输出结构 ===== */
    public static class PhasePower {
        public String phase;     // A/B/C
        public Double vrms_kV;   // 电压有效值 (kV)
        public Double irms_A;    // 电流有效值 (A)
        public Double P_kW;      // 有功功率 (kW)
        public Double S_kVA;     // 视在功率 (kVA)
        public Double PF;        // 功率因数 (0~1)
        public String error;     // 错误信息
        public Map<String,Object> debug = new LinkedHashMap<>();
    }
    public static class PairResult {
        public String filePair;         // "voltageName | currentName"
        public List<PhasePower> phases = new ArrayList<>();
    }

    @PostMapping("/upload")
    public ResponseEntity<?> upload(@RequestParam("voltageFiles") MultipartFile[] voltageFiles,
                                    @RequestParam("currentFiles") MultipartFile[] currentFiles) {
        int n = Math.min(voltageFiles.length, currentFiles.length);
        if (n == 0) return ResponseEntity.ok(Collections.emptyList());

        List<PairResult> out = new ArrayList<>();
        for (int i=0; i<n; i++) {
            PairResult pr = analyzeOnePair(voltageFiles[i], currentFiles[i]);
            out.add(pr);
        }
        return ResponseEntity.ok(out);
    }

    /* ===== 单对文件：三相 ===== */
    private PairResult analyzeOnePair(MultipartFile vFile, MultipartFile iFile) {
        PairResult r = new PairResult();
        r.filePair = (vFile.getOriginalFilename() + " | " + iFile.getOriginalFilename());

        Mat vImg = null, iImg = null;
        try {
            vImg = Imgcodecs.imdecode(new MatOfByte(vFile.getBytes()), Imgcodecs.IMREAD_COLOR);
            iImg = Imgcodecs.imdecode(new MatOfByte(iFile.getBytes()), Imgcodecs.IMREAD_COLOR);
            if (vImg == null || vImg.empty() || iImg == null || iImg.empty()) {
                r.phases.add(errPhase("A", "图片解码失败"));
                r.phases.add(errPhase("B", "图片解码失败"));
                r.phases.add(errPhase("C", "图片解码失败"));
                return r;
            }

            if (DEBUG) {
                System.out.println("============================================");
                System.out.println("[PAIR] " + r.filePair);
                System.out.printf("Voltage size: %dx%d, Current size: %dx%d%n",
                        vImg.width(), vImg.height(), iImg.width(), iImg.height());
            }

            r.phases.add(analyzePhase(vImg.submat(ROI_A), iImg.submat(ROI_A), "A"));
            r.phases.add(analyzePhase(vImg.submat(ROI_B), iImg.submat(ROI_B), "B"));
            r.phases.add(analyzePhase(vImg.submat(ROI_C), iImg.submat(ROI_C), "C"));

        } catch (Exception e) {
            r.phases.add(errPhase("A", e.getMessage()));
            r.phases.add(errPhase("B", e.getMessage()));
            r.phases.add(errPhase("C", e.getMessage()));
        } finally {
            if (vImg != null) vImg.release();
            if (iImg != null) iImg.release();
        }
        return r;
    }

    /* ===== 核心：单相功率（带调试输出） ===== */
    private PhasePower analyzePhase(Mat vROI, Mat iROI, String phase) {
        PhasePower out = new PhasePower();
        out.phase = phase;

        // 1) 找三条水平黑实线，用于确定中线与上下边界
        List<Integer> vLines = detectHorizontalBlackLines(vROI, HLINE_RUN_RATIO);
        List<Integer> iLines = detectHorizontalBlackLines(iROI, HLINE_RUN_RATIO);
        if (vLines.size() < 3 || iLines.size() < 3) {
            out.error = "黑实线不足(电压或电流)";
            out.debug.put("vLines", vLines);
            out.debug.put("iLines", iLines);
            if (DEBUG) {
                System.out.printf("[Phase %s] ERROR: 黑实线不足 -> vLines=%s, iLines=%s%n", phase, vLines, iLines);
            }
            return out;
        }
        Collections.sort(vLines);
        Collections.sort(iLines);
        int vy1=vLines.get(0), vy2=vLines.get(vLines.size()/2), vy3=vLines.get(vLines.size()-1);
        int iy1=iLines.get(0), iy2=iLines.get(iLines.size()/2), iy3=iLines.get(iLines.size()-1);
        if (DEBUG) {
            System.out.printf("[Phase %s] vLines(y1,y2,y3)=(%d,%d,%d), iLines(y1,y2,y3)=(%d,%d,%d)%n",
                    phase, vy1, vy2, vy3, iy1, iy2, iy3);
        }

        // 2) 找虚线刻度（像素->物理量）
        List<Integer> vDash = detectHorizontalDashLines(vROI, vy1, vy3);
        List<Integer> iDash = detectHorizontalDashLines(iROI, iy1, iy3);
        if (DEBUG) {
            System.out.printf("[Phase %s] dash count: V=%d, I=%d%n", phase, vDash.size(), iDash.size());
        }

        // 3) 逐列跟踪波形中心 y(x)
        int[] vY = traceWaveYCenterPerColumn(vROI, vy1, vy3);
        int[] iY = traceWaveYCenterPerColumn(iROI, iy1, iy3);

        // 4) 取右侧稳态窗口
        int vw = vROI.width(), iw = iROI.width();
        int vx0 = Math.max(0, Math.min(vw-2, (int)Math.round(vw*(1.0 - RIGHT_PORTION))));
        int ix0 = Math.max(0, Math.min(iw-2, (int)Math.round(iw*(1.0 - RIGHT_PORTION))));
        int[] vWin = Arrays.copyOfRange(vY, vx0, vw);
        int[] iWin = Arrays.copyOfRange(iY, ix0, iw);
        if (DEBUG) {
            System.out.printf("[Phase %s] window x-start: V=%d/%d, I=%d/%d%n", phase, vx0, vw, ix0, iw);
        }

        // 5) 像素 -> 物理量：电压(V)，电流(A)
        double[] vSig = mapToValueArray(vWin, vy2, vDash, VOLT_PER_SEG, true);
        double[] iSig = mapToValueArray(iWin, iy2, iDash, CURR_PER_SEG, false);

        vSig = compactValid(vSig);
        iSig = compactValid(iSig);
        if (DEBUG) {
            System.out.printf("[Phase %s] valid samples (before smooth): V=%d, I=%d%n", phase, vSig.length, iSig.length);
        }
        if (vSig.length < 30 || iSig.length < 30) {
            out.error = "有效样本不足(电压或电流)";
            out.debug.put("v_samples", vSig.length);
            out.debug.put("i_samples", iSig.length);
            if (DEBUG) System.out.printf("[Phase %s] ERROR: 有效样本不足%n", phase);
            return out;
        }

        vSig = movingAvg(vSig, 3);
        iSig = movingAvg(iSig, 3);

        // DEBUG: 打印序列前后各50个点，快速确认是否“随着时间变化”
        if (DEBUG) {
            debugDumpSeries("V(kV) window(mapped, smooth)", phase, vSig, 50, 1000.0); // 以kV输出方便看量级
            debugDumpSeries("I(A) window(mapped, smooth)",  phase, iSig, 50, 1.0);
        }

        // 6) 对齐长度（线性重采样到共同长度）
        int N = Math.min(vSig.length, iSig.length);
        double[] vS = (vSig.length == N) ? vSig : resampleLinear(vSig, N);
        double[] iS = (iSig.length == N) ? iSig : resampleLinear(iSig, N);
        if (DEBUG) {
            System.out.printf("[Phase %s] resampled length: %d%n", phase, N);
        }

        // 7) 计算 Vrms / Irms / P / S / PF
        double Vrms = rms(vS);
        double Irms = rms(iS);
        double P = meanProduct(vS, iS);  // W
        double S = Vrms * Irms;          // VA
        double PF = (S > 1e-12) ? clamp(P / S, 0.0, 1.0) : 0.0;

        out.vrms_kV = Vrms / 1000.0;     // kV
        out.irms_A  = Irms;              // A
        out.P_kW    = P / 1000.0;        // kW
        out.S_kVA   = S / 1000.0;        // kVA
        out.PF      = PF;

        if (DEBUG) {
            System.out.printf("[Phase %s] RESULT -> Vrms=%.3f kV, Irms=%.3f A, P=%.3f kW, S=%.3f kVA, PF=%.3f%n",
                    phase, out.vrms_kV, out.irms_A, out.P_kW, out.S_kVA, out.PF);
        }
        return out;
    }

    /* ====== 基础工具：线/刻度/波形提取 ====== */
    private List<Integer> detectHorizontalBlackLines(Mat roi, double runRatio) {
        Mat gray = new Mat(); Imgproc.cvtColor(roi, gray, Imgproc.COLOR_BGR2GRAY);
        Mat bin  = new Mat(); Imgproc.threshold(gray, bin, 0, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);
        int h = bin.rows(), w = bin.cols();
        byte[] data = new byte[h*w]; bin.get(0,0,data);

        List<Integer> ys = new ArrayList<>();
        int minRun = (int)(w * runRatio);
        for (int y=0; y<h; y++){
            int run=0, maxRun=0;
            for (int x=0; x<w; x++){
                int v = data[y*w + x] & 0xFF;
                if (v > 0) { run++; if (run > maxRun) maxRun = run; } else run=0;
            }
            if (maxRun >= minRun) ys.add(y);
        }
        // 合并近邻
        List<Integer> merged = new ArrayList<>();
        Integer s=null,e=null;
        for (int y : ys){
            if (e == null || y - e <= HLINE_MERGE_PX){ if (s==null) s=y; e=y; }
            else { merged.add((s+e)/2); s=y; e=y; }
        }
        if (s != null) merged.add((s+e)/2);
        gray.release(); bin.release();
        return merged;
    }

    private List<Integer> detectHorizontalDashLines(Mat roi, int y1, int y3) {
        Mat gray = new Mat(); Imgproc.cvtColor(roi, gray, Imgproc.COLOR_BGR2GRAY);
        Mat bin  = new Mat(); Imgproc.adaptiveThreshold(gray, bin, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C,
                Imgproc.THRESH_BINARY_INV, 15, 10);
        int h = bin.rows(), w = bin.cols();
        y1 = Math.max(0, y1); y3 = Math.min(h-1, y3);

        byte[] data = new byte[h*w]; bin.get(0,0,data);
        double[] rowSum = new double[h];

        for (int y=y1; y<=y3; y++){
            int off=y*w, cnt=0; for (int x=0;x<w;x++) if ((data[off+x]&0xFF)>0) cnt++;
            rowSum[y] = cnt;
        }
        // 平滑
        double[] sm = new double[h];
        for (int y=y1; y<=y3; y++){
            int L=Math.max(y1,y-DASH_SMOOTH_WIN), R=Math.min(y3,y+DASH_SMOOTH_WIN);
            double s=0; int c=0; for (int k=L;k<=R;k++){ s+=rowSum[k]; c++; }
            sm[y]=s/Math.max(1,c);
        }
        // 阈值选峰
        double mean=0; int c=0; for (int y=y1;y<=y3;y++){ mean+=sm[y]; c++; } mean/=Math.max(1,c);
        List<Integer> peaks = new ArrayList<>();
        for (int y=y1+1;y<y3;y++){
            if (sm[y]>sm[y-1] && sm[y]>sm[y+1] && sm[y] > mean*DASH_PEAK_GAIN) peaks.add(y);
        }
        // 合并近邻
        List<Integer> merged = new ArrayList<>();
        int tol=3;
        for (int y: peaks){
            if (merged.isEmpty() || y-merged.get(merged.size()-1)>tol) merged.add(y);
            else {
                int prev = merged.get(merged.size()-1);
                merged.set(merged.size()-1, (prev+y)/2);
            }
        }
        gray.release(); bin.release();
        return merged;
    }

    private int[] traceWaveYCenterPerColumn(Mat roi, int y1, int y3) {
        Mat hsv = new Mat(); Imgproc.cvtColor(roi, hsv, Imgproc.COLOR_BGR2HSV);
        int h = roi.rows(), w = roi.cols();
        y1 = Math.max(0, y1); y3 = Math.min(h-1, y3);

        int[] ys = new int[w];
        Arrays.fill(ys, -1);

        for (int x=0;x<w;x++){
            int top=-1, bottom=-1;
            for (int y=y1; y<=y3; y++){
                double[] p = hsv.get(y,x);
                if (p==null) continue;
                double S=p[1], V=p[2];
                if (S>HSV_S_THRESH && V>HSV_V_THRESH){
                    if (top==-1) top=y;
                    bottom=y;
                }
            }
            if (top!=-1) ys[x]=(top+bottom)/2;
        }
        hsv.release();

        // 简单平滑
        int k=3; int[] sm = new int[w];
        for (int x=0;x<w;x++){
            int L=Math.max(0,x-k), R=Math.min(w-1,x+k);
            int cnt=0,sum=0;
            for (int i=L;i<=R;i++){
                if (ys[i]>=0){ cnt++; sum+=ys[i]; }
            }
            sm[x] = cnt==0 ? -1 : (sum/cnt);
        }
        return sm;
    }

    // 像素数组 -> 物理量数组；up/down 自动由与 y2 的关系决定
    private double[] mapToValueArray(int[] yArr, int y2, List<Integer> dashYs, double perSeg, boolean isVoltage) {
        List<Double> out = new ArrayList<>();
        for (int y : yArr){
            if (y < 0) { out.add(Double.NaN); continue; }
            boolean isUp = y < y2;
            out.add(pixelToValueByDashes(y, y2, dashYs, perSeg, isUp));
        }
        double[] a = new double[out.size()];
        for (int i=0;i<a.length;i++) a[i] = out.get(i);
        return a;
    }

    // 单个像素 y -> 物理量（分段+线性插值），上正下负
    private double pixelToValueByDashes(int y, int y2, List<Integer> dashYs, double perSeg, boolean isUp) {
        List<Integer> all = new ArrayList<>(dashYs);
        all.add(y2); Collections.sort(all);
        if (isUp){
            List<Integer> up = new ArrayList<>();
            for (int d: all) if (d < y2) up.add(d);
            up.sort(Collections.reverseOrder());
            int prev = y2, section = 0;
            for (int d: up){
                if (y <= d){
                    double ratio = (prev - y) * 1.0 / (prev - d);
                    return section * perSeg + ratio * perSeg;
                }
                prev = d; section++;
            }
            if (!up.isEmpty()){
                double ratio = (prev - y) * 1.0 / (prev - up.get(up.size()-1));
                return section * perSeg + ratio * perSeg;
            } else {
                return (y2 - y) * perSeg / 300.0; // 兜底
            }
        } else {
            List<Integer> down = new ArrayList<>();
            for (int d: all) if (d > y2) down.add(d);
            Collections.sort(down);
            int prev = y2, section = 0;
            for (int d: down){
                if (y >= d){
                    double ratio = (y - prev) * 1.0 / (d - prev);
                    return -(section * perSeg + ratio * perSeg);
                }
                prev = d; section++;
            }
            if (!down.isEmpty()){
                double ratio = (y - prev) * 1.0 / (down.get(down.size()-1) - prev);
                return -(section * perSeg + ratio * perSeg);
            } else {
                return -(y - y2) * perSeg / 300.0; // 兜底
            }
        }
    }

    /* ====== 数学小工具 ====== */
    private double[] compactValid(double[] a){
        return Arrays.stream(a).filter(v -> !Double.isNaN(v)).toArray();
    }
    private double[] movingAvg(double[] s, int win){
        if (win<=1) return s;
        int n=s.length; double[] out=new double[n];
        for (int i=0;i<n;i++){
            int L=Math.max(0,i-win/2), R=Math.min(n-1,i+win/2);
            double sum=0; int c=0;
            for (int k=L;k<=R;k++){ sum+=s[k]; c++; }
            out[i]=sum/Math.max(1,c);
        }
        return out;
    }
    private double[] resampleLinear(double[] s, int N){
        int n=s.length; if (n==N) return s.clone();
        double[] out=new double[N];
        for (int i=0;i<N;i++){
            double t = i*(n-1.0)/(N-1.0);
            int t0 = (int)Math.floor(t), t1 = Math.min(n-1, t0+1);
            double a = t - t0;
            out[i] = (1-a)*s[t0] + a*s[t1];
        }
        return out;
    }
    private double rms(double[] s){
        double sum=0; for (double v: s) sum+=v*v;
        return Math.sqrt(sum/Math.max(1, s.length));
    }
    private double meanProduct(double[] a, double[] b){
        int n=Math.min(a.length,b.length); if (n==0) return 0;
        double s=0; for (int i=0;i<n;i++) s+=a[i]*b[i];
        return s/n;
    }
    private double clamp(double v, double lo, double hi){ return Math.max(lo, Math.min(hi, v)); }

    private PhasePower errPhase(String phase, String msg){
        PhasePower p = new PhasePower();
        p.phase = phase; p.error = msg; return p;
    }

    /* ====== 调试打印：序列前后各一些点 ====== */
    private void debugDumpSeries(String title, String phase, double[] s, int edgeCount, double scale) {
        if (!DEBUG) return;
        System.out.printf("[Phase %s] %s, len=%d%n", phase, title, s.length);
        int n = s.length;
        int m = Math.min(edgeCount, n);
        // 前 m 个
        StringBuilder sb1 = new StringBuilder();
        for (int i=0; i<m; i++) {
            sb1.append(String.format("%.3f", s[i]/scale));
            if (i<m-1) sb1.append(", ");
        }
        // 后 m 个
        StringBuilder sb2 = new StringBuilder();
        for (int i=Math.max(0, n-m); i<n; i++) {
            sb2.append(String.format("%.3f", s[i]/scale));
            if (i<n-1) sb2.append(", ");
        }
        System.out.println("  head: [" + sb1 + "]");
        System.out.println("  tail: [" + sb2 + "]");
    }
}
