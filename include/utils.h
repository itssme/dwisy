//
// Created by itssme on 11/02/2020.
//

#ifndef DWISY_UTILS_H
#define DWISY_UTILS_H

#include <opencv2/opencv.hpp>
#include <vector>
using namespace cv;

struct Config {
    double fps;
    double width;
    double height;
};

void process_buffer(std::vector<Mat> buffer, const int& buffer_maxsize, int buffer_index, int frames, const Config& config, const Size& size) {
    CascadeClassifier face_detection("res/haarcascade_frontalface_default.xml");
    VideoWriter writer{VideoWriter("/tmp/test.avi", CV_FOURCC('M', 'J', 'P', 'G'), config.fps, size)};

    Mat gray_frame;
    std::vector<Rect_<int>> faces;
    for (const Mat& process_frame: buffer) {
        cvtColor(process_frame, gray_frame, COLOR_BGR2GRAY);
        face_detection.detectMultiScale(gray_frame, faces, 1.1, 5);

        for (const Rect_<int>& face: faces) {
            rectangle(process_frame,
                    Point(face.x, face.y),
                    Point(face.x + face.width, face.y + face.height),
                    Scalar(255, 0, 0));
        }
        writer.write(process_frame);
    }

    writer.release();
    std::cout << "processed buffer" << std::endl;
}

void write_frames(std::vector<Mat> buffer, const int& buffer_maxsize, const Config& config, const Size& size) {
    VideoWriter writer{VideoWriter("/tmp/test.avi", CV_FOURCC('M', 'J', 'P', 'G'), config.fps, size)};

    for (int i = 0; i < buffer_maxsize; ++i) {
        writer.write(buffer.at(i));
    }

    writer.release();
}

#endif //DWISY_UTILS_H
