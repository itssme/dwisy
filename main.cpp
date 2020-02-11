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
    Mat color_frame;
    const int renew_config_interval{(int) config.fps};
    int renew_config_frame{renew_config_interval};
    Size size(config.width/config.height * MOVEMENT_DETECTION_IMAGE_SIZE, MOVEMENT_DETECTION_IMAGE_SIZE);
    bool in_movement{false};

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

    while (true) {
        video_stream >> color_frame;

        if (color_frame.empty()) {
            break;
        }

        resize(color_frame, color_frame, size);
        cvtColor(color_frame, frame, COLOR_BGR2GRAY);
        GaussianBlur(frame, frame, Size_<int>(21, 21), 0);

        absdiff(frame, config_frame, compared_frame);
        threshold(compared_frame, compared_frame, 20, 255, THRESH_BINARY);
        dilate(compared_frame, compared_frame, Mat());

        std::vector<Vec4i> hierarchy;
        std::vector<std::vector<Point> > contours;
        findContours(compared_frame, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        in_movement = false;
        for(unsigned long i = 0; i < contours.size(); i++ ) {
            in_movement = true;
            Scalar color = Scalar(0, 0, 255);
            drawContours(color_frame, contours, (int) i, color, 2, 8, hierarchy, 0, Point());
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

        char c = (char) waitKey(25);
        if (c == 27) // esc key
            break;
    }

    video_stream.release();
    destroyAllWindows();

    return 0;
}