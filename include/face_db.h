/* 
 * author: Joel Klimont
 * filename: face_db.h
 * date: 14/02/20
*/

#ifndef DWISY_FACE_DB_H
#define DWISY_FACE_DB_H

#include <opencv2/opencv.hpp>

using namespace cv;

struct IdentifiedFace {
    Mat face;
    String name;
};

class FaceDB {
public:
    String working_dir;
    bool store(Mat face, String name);
    bool store(IdentifiedFace id_face);
    String identify(Mat face);
    std::vector<IdentifiedFace> readFaces();
};

#endif //DWISY_FACE_DB_H
