package com.example.guowangwaveformimage.controller;

/*
    @author nanchao
    @date 2025/8/14
*/


import org.bytedeco.javacpp.Loader;
import org.bytedeco.opencv.opencv_java;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;


@RestController
@RequestMapping("/pinlv")
public class PinlvController {

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
    public ResponseEntity<?> uploadImages(@RequestParam("files") MultipartFile[] files) {


        return null;
    }
}
