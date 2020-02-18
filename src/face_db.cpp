/* 
 * author: Joel Klimont
 * filename: face_db.cpp
 * date: 14/02/20
*/

#include "face_db.h"

Mat FaceDB::prepare_face(const Mat& face, const Rect_<int>& detected_face) {
    cv::Rect face_roi(max(0, detected_face.x - FACE_SIZE_OFFSET),
            max(0, detected_face.y - FACE_SIZE_OFFSET),
            (face.size().width < detected_face.width + detected_face.x + FACE_SIZE_OFFSET * 2) ? face.size().width - detected_face.x : detected_face.width + FACE_SIZE_OFFSET * 2,
            (face.size().height < detected_face.height + detected_face.y + FACE_SIZE_OFFSET * 2) ? face.size().height - detected_face.y : detected_face.height + FACE_SIZE_OFFSET * 2);

    cv::Mat cut_face = face(face_roi);

    return cut_face;
}

bool FaceDB::store_unidentified(const Mat& face) {
    std::chrono::milliseconds ms = std::chrono::duration_cast< std::chrono::milliseconds >(
            std::chrono::system_clock::now().time_since_epoch()
    );
    return imwrite(this->working_dir + "/unidentified/" + std::to_string(ms.count()) + ".png", face);
}
