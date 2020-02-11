#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>
#include "utils.h"

#define MOVEMENT_DETECTION_IMAGE_SIZE 500
#define MOVEMENT_THRESHOLD 20
#define MOVEMENT_CHECK_FRAMES 5

using namespace cv;


int main() {
    VideoCapture video_stream = VideoCapture(0);

    Config config = Config();
    config.fps = video_stream.get(CAP_PROP_FPS);
    config.width = video_stream.get(CAP_PROP_FRAME_WIDTH);
    config.height = video_stream.get(CAP_PROP_FRAME_HEIGHT);

    Mat frame;
    Mat config_frame;
    Mat config_frame_in_movement;
    Mat compared_frame;
    Mat compared_frame_in_movement;
    Mat color_frame;
    int in_movement_threshold_counter{0};
    const int renew_config_interval{(int) config.fps};
    int renew_config_frame{renew_config_interval};
    Size size(config.width/config.height * MOVEMENT_DETECTION_IMAGE_SIZE, MOVEMENT_DETECTION_IMAGE_SIZE);
    bool in_movement{false};
    const int buffer_maxsize{(int) config.fps * 5};
    std::vector<Mat> buffer;
    Scalar color = Scalar(0, 0, 255);
    CascadeClassifier face_detection("res/haarcascade_frontalface_default.xml");
    for (int i = 0; i < buffer_maxsize; ++i) buffer.emplace_back(Mat());
    int buffer_counter{0};

    std::cout << "Waiting for config frame" << std::endl;

    { // find first frame for later comparison
        bool config_frame_found = false;
        int warm_up{(int) config.fps * 2}; // wait for first camera adjustment for 2 seconds
        while (not config_frame_found) {
            video_stream >> frame;

            if (not frame.empty()) {
                if (not warm_up) {
                    resize(frame, config_frame, size);
                    cvtColor(config_frame, config_frame, COLOR_BGR2GRAY);
                    GaussianBlur(config_frame, config_frame, Size_<int>(21, 21), 0);
                    config_frame_found = true;
                } else {
                    warm_up--;
                }
            }
        }
    }

    std::cout << "Found config frame" << std::endl;

    std::vector<Vec4i> hierarchy;
    std::vector<std::vector<Point> > contours;
    while (true) {
        video_stream >> color_frame;

        if (color_frame.empty()) {
            break;
        }

        resize(color_frame, color_frame, size);
        cvtColor(color_frame, frame, COLOR_BGR2GRAY);
        GaussianBlur(frame, frame, Size_<int>(21, 21), 0);

        buffer.at(buffer_counter) = color_frame.clone();
        buffer_counter++;

        if (buffer_counter == buffer_maxsize) {
            std::cout << "full" << std::endl;
            buffer_counter = 0;
            std::thread writer(process_buffer, std::vector<Mat>(buffer), buffer_maxsize, 0, 0, &face_detection, config, size);
            writer.detach();
        }

        if (not in_movement) {
            absdiff(frame, config_frame, compared_frame);
            threshold(compared_frame, compared_frame, MOVEMENT_THRESHOLD, 255, THRESH_BINARY);
            findContours(compared_frame, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

            for (unsigned long i = 0; i < contours.size(); i++) {
                in_movement = true;
                drawContours(color_frame, contours, (int) i, color, 2, 8, hierarchy, 0, Point());
            }

            if (in_movement) {
                config_frame.copyTo(config_frame_in_movement);
            }
        }

        if (in_movement_threshold_counter == MOVEMENT_CHECK_FRAMES) {
            absdiff(frame, config_frame_in_movement, compared_frame_in_movement);
            threshold(compared_frame_in_movement, compared_frame_in_movement, MOVEMENT_THRESHOLD, 255, THRESH_BINARY);
            findContours(compared_frame_in_movement, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

            in_movement = false;
            for (unsigned long i = 0; i < contours.size(); i++) {
                in_movement = true;
                drawContours(color_frame, contours, (int) i, color, 2, 8, hierarchy, 0, Point());
            }

            if (not in_movement) {
                config_frame_in_movement.copyTo(config_frame);
            }

            frame.copyTo(config_frame_in_movement);
            in_movement_threshold_counter = 0;
        } else if (in_movement) {
            in_movement_threshold_counter++;
        }

        if (not in_movement) {
            if (not renew_config_frame) {
                renew_config_frame = renew_config_interval;
                frame.copyTo(config_frame);
            } else {
                renew_config_frame -= 1;
            }
        }

        imshow("Image", color_frame);
        imshow("Frame", frame);
        imshow("config_frame", config_frame);
        imshow("Compared", compared_frame);

        if (in_movement) {
            std::cout << "In movement: true" << std::endl;
        } else {
            std::cout << "In movement: false" << std::endl;
        }

        char c = (char) waitKey(25);
        if (c == 27) // esc key
            break;
    }

    video_stream.release();
    destroyAllWindows();

    return 0;
}