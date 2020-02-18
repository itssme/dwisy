/* 
 * author: Joel Klimont
 * filename: face_db.h
 * date: 14/02/20
*/

#ifndef DWISY_FACE_DB_H
#define DWISY_FACE_DB_H

#define FACE_SIZE_OFFSET 25
#define FACE_QUALITY 0.0002

#include <opencv2/opencv.hpp>
#include <chrono>
#include <algorithm>

using namespace cv;

struct IdentifiedFace {
    Mat face;
    String name;
};

class FaceDB {
public:
    String working_dir{"/tmp"};
    bool store(Mat face, String name);
    bool store(IdentifiedFace id_face);
    bool store_unidentified(const Mat& face);
    static Mat prepare_face(const Mat& face, const Rect& detected_face);
    String identify(Mat face);
    std::vector<IdentifiedFace> readFaces();
};

#endif //DWISY_FACE_DB_H
