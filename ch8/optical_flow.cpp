#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace cv;

string file_1 = "./LK1.png";
string file_2 = "./LK2.png";

class OpticalFlowTracker{
public:
    OpticalFlowTracker(
        const Mat &img1_,
        const Mat &img2_,
        const vector<KeyPoint> &kp1_,
        vector<KeyPoint> &kp2_,
        vector<bool> &success_,
        bool inverse_ = true, bool has_initial_ = false) :
        img1(img1_), img2(img2_), kp1(kp1_), kp2(kp2_), success(success_), inverse(inverse_), has_initial(has_initial_){}
    void calculateOpticalFlow(const Range &range);

private:
    const Mat &img1;
    const Mat &img2;
    const vector<KeyPoint> &kp1;
    vector<KeyPoint> &kp2;
    vector<bool> &success;
    bool inverse = true;
    bool has_initial = false;
};

void OpticalFlowSingleLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse = false,
    bool has_initial_gauss = false
);

void OpticalFlowMultiLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse = false
);

inline float GetPixelValue(const cv::Mat &img, float x, float y){
    if(x < 0) x = 0;
    if(y < 0) y = 0;
    if(x >= img.cols - 1) x = img.cols - 2;
    if(y >= img.rows - 1) y = img.rows - 2;

    float xx = x - floor(x);
    float yy = y - floor(y);
    int x_a1 = std::min(img.cols - 1, int(x) + 1);
    int y_a1 = std::min(img.rows - 1, int(y) + 1);
    
    return (1 - xx) * (1 - yy) * img.at<uchar>(y, x)
    + xx * (1 - yy) * img.at<uchar>(y, x_a1)
    + (1 - xx) * yy * img.at<uchar>(y_a1, x)
    + xx * yy * img.at<uchar>(y_a1, x_a1);
}

int main(int argc, char **argv){
    Mat img1 = imread(file_1, 0);
    Mat img2 = imread(file_2, 0);

    vector<KeyPoint> kp1;
    Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20);
    detector->detect(img1, kp1);

    vector<KeyPoint> kp2_single;
    vector<bool> success_single;
    OpticalFlowSingleLevel(img1, img2, kp1, kp2_single, success_single);

    vector<KeyPoint> kp2_multi;
    vector<bool> success_multi;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    OpticalFlowMultiLevel(img1, img2, kp1, kp2_multi, success_multi, true);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optical flow by gauss-newton: " << time_used.count() << endl;

    vector<Point2f> pt1, pt2;
    for(auto &kp : kp1) pt1.push_back(kp.pt);
    vector<uchar> status;
    vector<uchar> 
    return 0;
}

void OpticalFlowSingleLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse,
    bool has_initial_gauss
){
    kp2.resize(kp1.size());
    success.resize(kp1.size());
    OpticalFlowTracker tracker(img1, img2, kp1, kp2, success, inverse, has_initial_gauss);
    parallel_for_(Range(0, kp1.size()),
        std::bind(&OpticalFlowTracker::calculateOpticalFlow, &tracker, placeholders::_1));
}

void OpticalFlowTracker::calculateOpticalFlow(const Range &range){
    int half_patch_size = 4;
    int iterations = 10;
    for(size_t i = range.start; i < range.end; i++){
        auto kp = kp1[i];
        double dx = 0, dy = 0;
        if(has_initial){
            dx = kp2[i].pt.x - kp.pt.x;
            dy = kp2[i].pt.y - kp.pt.y;
        }

        double cost = 0, lastCost = 0;
        bool succ = true;

        Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
        Eigen::Vector2d b = Eigen::Vector2d::Zero();
        Eigen::Vector2d J;
        for(int iter = 0; iter < iterations; iter++){
            if(inverse == false){
                H = Eigen::Matrix2d::Zero();
                b = Eigen::Vector2d::Zero();
            }
            else{
                b = Eigen::Vector2d::Zero();
            }
            
            cost = 0;
            for(int x = -half_patch_size; x < half_patch_size; x++)
            for(int y = -half_patch_size; y < half_patch_size; y++){
                double error = GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y) - 
                    GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy);
                if(inverse == false){
                    J = -1.0 * Eigen::Vector2d(
                        0.5 * (GetPixelValue(img2, kp.pt.x + dx + x + 1, kp.pt.y + dy + y) -
                            GetPixelValue(img2, kp.pt.x + dx + x - 1, kp.pt.y + dy + y)),
                        0.5 * (GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y + 1) -
                            GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y - 1))
                    );
                } else if(iter == 0){
                    J = -1.0 * Eigen::Vector2d(
                        0.5 * (GetPixelValue(img1, kp.pt.x + x + 1, kp.pt.y + y) - 
                            GetPixelValue(img1, kp.pt.x + x - 1, kp.pt.y + y)),
                        0.5 * (GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y + 1) -
                            GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y - 1))
                    );
                }
                b += -error * J;
                cost += error * error;
                if(inverse == false || iter == 0){
                    H += J * J.transpose();
                }
            }
            Eigen::Vector2d update = H.ldlt().solve(b);

            if(std::isnan(update[0])){
                cout << "update is nan" << endl;
                succ = false;
                break;
            }

            if(iter > 0 && cost > lastCost){
                break;
            }

            dx += update[0];
            dy += update[1];
            lastCost = cost;
            succ = true;

            if(update.norm() < 1e-2){
                break;
            }
        }

        success[i] = succ;

        kp2[i].pt = kp.pt + Point2f(dx, dy);
    }
}

void OpticalFlowMultiLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse
){
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    vector<Mat> pyr1, pyr2;
    for(int i = 0; i < pyramids; i++){
        if(i == 0){
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        } else{
            Mat img1_pyr, img2_pyr;
            cv::resize(pyr1[i - 1], img1_pyr,
                cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
            cv::resize(pyr2[i - 1], img2_pyr,
                cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optimization costs time: " << time_used.count() << "seconds." << endl;
}