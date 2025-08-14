package com.example.guowangwaveformimage.controller;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.opencv.opencv_java;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Rect;
import org.opencv.imgcodecs.Imgcodecs;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.InputStream;
import java.util.*;

@RestController
@RequestMapping("/zantai")
public class ZantaiController {
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


    @PostMapping("/upload")
    public ResponseEntity<?> uploadImages(@RequestParam("files") MultipartFile[] files,@RequestParam(value="mode", defaultValue="voltage") String mode) {

        // ★改动：根据模式选择每段物理量（电压=200000，电流=500）
        final double perSegmentValue = "current".equalsIgnoreCase(mode) ? 500.0 : 200000.0;

        List<Map<String, Object>> allResults = new ArrayList<>();

        for (MultipartFile file : files) {
            try {
                // 1. 读取为 OpenCV Mat
                InputStream in = file.getInputStream();
                byte[] bytes = in.readAllBytes();
                Mat img = Imgcodecs.imdecode(new MatOfByte(bytes), Imgcodecs.IMREAD_COLOR);

                // 2. 获取尺寸
                int width = img.width();
                int height = img.height();
                System.out.println("--------------------------------------------------");
                System.out.println("图片尺寸: " + width + "x" + height);

                // 3.裁剪区域（直接内存裁剪三相）
                Rect rect1 = new Rect(55, 56, 1400, 310);
                Rect rect2 = new Rect(55, 370, 1400, 310);
                Rect rect3 = new Rect(55, 683, 1400, 310);

                Mat img1 = new Mat(img, rect1);
                Mat img2 = new Mat(img, rect2);
                Mat img3 = new Mat(img, rect3);

                String baseName = file.getOriginalFilename();
                if (baseName == null) baseName = "unknown";
                baseName = baseName.replaceAll("\\.[^.]+$", "");

                String[] names = {"A", "B", "C"};
                Mat[] images = {img1, img2, img3};
                List<Map<String, Object>> phaseResults = new ArrayList<>();
                for (int i = 0; i < 3; i++) {
                    // ★改动：把 perSegmentValue 传入
                    Map<String, Object> r = analyzePhase(images[i], names[i], perSegmentValue);
                    phaseResults.add(r);
                    // 控制台输出
                    System.out.println("文件 " + baseName + " 相" + names[i] +
                            "：最大值=" + r.get("value") + ", 最高点y=" + r.get("wave_top_y"));
                }

                Map<String, Object> fileResult = new LinkedHashMap<>();
                fileResult.put("file", file.getOriginalFilename());
                fileResult.put("phases", phaseResults);
                allResults.add(fileResult);

            } catch (Exception e) {
                System.out.println("处理图片异常: " + e.getMessage());
            }
        }
        // 只返回结果结构
        return ResponseEntity.ok(allResults);
    }

    public Map<String, Object> analyzePhase(Mat part, String phaseName, double perSegmentValue) {
        // 检测三条黑实线
        List<Integer> lineY = detectHorizontalBlackLines(part, 0.6);
        if (lineY.size() < 3) {
            System.out.println("相" + phaseName + "：检测到的黑实线不足3条，实际行y=" + lineY);
            return Map.of("phase", phaseName, "error", "检测到的黑实线不足3条", "lines", lineY);
        }
        Collections.sort(lineY);
        int y1 = lineY.get(0), y2 = lineY.get(lineY.size() / 2), y3 = lineY.get(lineY.size() - 1);

        // 检测所有“虚线”y
        List<Integer> dashLines = detectHorizontalDashLines(part, 400, 800);
        Collections.sort(dashLines);

        // 只在y1~y3之间检测彩色波形最高点
        int waveTopY = findWaveformTopY(part, y1, y3);
        boolean isUp = waveTopY < y2;

        double value = calcMaxValueByDashes(y2, dashLines, waveTopY, isUp, perSegmentValue);

        return Map.of(
                "phase", phaseName,
                "wave_top_y", waveTopY,
                "value", value
        );
    }

    public List<Integer> detectHorizontalBlackLines(Mat img, double totalRunRatio) {
        List<Integer> lines = new ArrayList<>();
        int minTotalRun = (int) (img.cols() * totalRunRatio);
        for (int y = 0; y < img.rows(); y++) {
            int totalRun = 0, currRun = 0;
            for (int x = img.cols() - 1; x >= 0; x--) {
                double[] color = img.get(y, x);
                if (isBlack(color)) {
                    currRun++;
                } else {
                    totalRun += currRun;
                    currRun = 0;
                }
            }
            totalRun += currRun;
            if (totalRun > minTotalRun) lines.add(y);
        }
        List<Integer> uniq = new ArrayList<>();
        int last = -1000;
        for (int y : lines) {
            if (y - last > 10) uniq.add(y);
            last = y;
        }
        return uniq;
    }

    public boolean isBlack(double[] color) {
        return color[0] < 150 && color[1] < 150 && color[2] < 150;
    }

    // 只在实线y1~y3之间找“最靠上”且有彩色像素的那一行y
    public int findWaveformTopY(Mat img, int y1, int y3) {
        for (int y = y1 + 1; y < y3; y++) {
            for (int x = 0; x < img.cols(); x++) {
                double[] color = img.get(y, x);
                if (isColorful(color) && !isBlack(color)) {
                    return y;
                }
            }
        }
        return y3;
    }

    public boolean isColorful(double[] color) {
        double max = Math.max(color[0], Math.max(color[1], color[2]));
        double min = Math.min(color[0], Math.min(color[1], color[2]));
        return (max - min) > 40 && max > 80;
    }

    public List<Integer> detectHorizontalDashLines(Mat img, int minTotal, int maxTotal) {
        List<Integer> lines = new ArrayList<>();
        for (int y = 0; y < img.rows(); y++) {
            int total = 0, curr = 0;
            for (int x = 0; x < img.cols(); x++) {
                double[] color = img.get(y, x);
                if (isBlack(color)) {
                    curr++;
                } else {
                    total += curr;
                    curr = 0;
                }
            }
            total += curr;
            if (total >= minTotal && total < maxTotal) lines.add(y);
        }
        List<Integer> uniq = new ArrayList<>();
        int last = -100;
        for (int y : lines) {
            if (y - last > 8) uniq.add(y);
            last = y;
        }
        return uniq;
    }

    /**
     * 计算最大值，基于0轴（中间实线）、所有虚线（升序）、波形最高点
     */
    public double calcMaxValueByDashes(int y2, List<Integer> dashLines, int waveTopY, boolean isUp,double perSegmentValue) {
        // 0轴和虚线全部合到一起，排序
        List<Integer> allY = new ArrayList<>(dashLines);
        allY.add(y2);
        Collections.sort(allY);

        if (isUp) {
            List<Integer> up = new ArrayList<>();
            for (int y : allY) if (y < y2) up.add(y);
            Collections.sort(up, Collections.reverseOrder());

            int prev = y2;
            int section = 0;
            for (int y : up) {
                if (waveTopY <= y) {
                    double ratio = (prev - waveTopY) * 1.0 / (prev - y);
                    return section * perSegmentValue + ratio * perSegmentValue;
                }
                prev = y;
                section++;
            }
            // 超出最上面一段
            if (!up.isEmpty()) {
                double ratio = (prev - waveTopY) * 1.0 / (prev - up.get(up.size() - 1));
                return section * perSegmentValue + ratio * perSegmentValue;
            } else {
                return (y2 - waveTopY) * perSegmentValue / (y2 - 0+1e-9); // 没虚线就兜底
            }
        } else {
            List<Integer> down = new ArrayList<>();
            for (int y : allY) if (y > y2) down.add(y);
            Collections.sort(down);

            int prev = y2;
            int section = 0;
            for (int y : down) {
                if (waveTopY >= y) {
                    double ratio = (waveTopY - prev) * 1.0 / (y - prev);
                    return -(section * perSegmentValue + ratio * perSegmentValue);
                }
                prev = y;
                section++;
            }
            // 超出最下面一段
            if (!down.isEmpty()) {
                double ratio = (waveTopY - prev) * 1.0 / (down.get(down.size() - 1) - prev);
                return -(section * perSegmentValue + ratio * perSegmentValue);
            } else {
                return -(waveTopY - y2) * perSegmentValue / (300.0); // 兜底，假设最下面实线与0轴相距300像素
            }
        }
    }

}
