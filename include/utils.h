//
// Created by itssme on 11/02/2020.
//

#ifndef DWISY_UTILS_H
#define DWISY_UTILS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include <ctime>
#include "face_db.h"

using namespace cv;

struct Config {
    double fps;
    double width;
    double height;
};

void process_buffer(std::vector<Mat> buffer, const int& buffer_maxsize, int buffer_index, int frames, CascadeClassifier* face_detection, const Config& config, const Size& size, FaceDB* face_db) {
    std::time_t current_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    String filename = "/tmp/" + String(std::ctime(&current_time)) + ".avi";
    VideoWriter writer{VideoWriter(filename, VideoWriter::fourcc('M', 'J', 'P', 'G'), config.fps, size)};

    Mat gray_frame{};
    std::vector<Rect_<int>> faces;
    for (const Mat& process_frame: buffer) {
        cvtColor(process_frame, gray_frame, COLOR_BGR2GRAY);
        face_detection->detectMultiScale(gray_frame, faces, 1.1, 5);

        for (const Rect_<int>& face: faces) {
            rectangle(process_frame,
                    Point(face.x, face.y),
                    Point(face.x + face.width, face.y + face.height),
                    Scalar(255, 0, 0));
            face_db->store_unidentified(face_db->prepare_face(process_frame, face));
        }
        writer.write(process_frame);
    }

    writer.release();
    std::cout << "processed buffer" << std::endl;
}

#endif //DWISY_UTILS_H
