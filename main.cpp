#include <iostream>
#include <opencv2/opencv.hpp>

#define MOVEMENT_DETECTION_IMAGE_SIZE 500

using namespace cv;

struct Config {
    double fps;
    double width;
    double height;
};

int main() {
    VideoCapture video_stream = VideoCapture(0);

    Config config = Config();
    config.fps = video_stream.get(CAP_PROP_FPS);
    config.width = video_stream.get(CAP_PROP_FRAME_WIDTH);
    config.height = video_stream.get(CAP_PROP_FRAME_HEIGHT);

    Mat frame;
    Mat config_frame;
    Mat compared_frame;
    Size size(config.width/config.height * MOVEMENT_DETECTION_IMAGE_SIZE, MOVEMENT_DETECTION_IMAGE_SIZE);

    std::cout << "Waiting for config frame" << std::endl;

    { // find first frame for later comparison
        bool config_frame_found = false;
        while (not config_frame_found) {
            video_stream >> frame;

            if (not frame.empty()) {
                resize(frame, config_frame, size);
                cvtColor(frame, frame, COLOR_BGR2GRAY);
                GaussianBlur(frame, frame, Size_<int>(21, 21), 0);
                config_frame_found = true;
            }
        }
    }

    std::cout << "Found config frame" << std::endl;

    while (true) {
        video_stream >> frame;

        if (frame.empty()) {
            break;
        }

        resize(frame, frame, size);
        cvtColor(frame, frame, COLOR_BGR2GRAY);
        GaussianBlur(frame, frame, Size_<int>(21, 21), 0);

        absdiff(frame, config_frame, compared_frame);
        //threshold(compared_frame, compared_frame, 20, 255, THRESH_BINARY);
        //dilate(compared_frame, compared_frame, Mat());

        resize(frame, frame, size);
        imshow("UsbImage", frame);
        char c = (char) waitKey(25);
        if (c == 27)
            break;

    }

    video_stream.release();
    destroyAllWindows();

    return 0;
}