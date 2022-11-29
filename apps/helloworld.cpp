#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <librealsense2/rs.hpp>
// #include <opencv2/xfeatures2d.hpp>
// #include <opencv2/highgui.hpp>
// #include <opencv2/imgproc.hpp>
#include <vector>

using namespace std;
using namespace cv;
int main()
{
    std::cout << "hello" << std::endl;
    cv::Ptr<cv::SIFT> siftDetector = cv::SIFT::create(500);
    rs2::pipeline p;
    // rs2::frameset frames = p
	// Ptr<SIFT> detector = SIFT::create(500);
    // cv::Ptr<cv::feature
    // cv::Ptr<cv::Feature2D> sift = cv::Feature2D:: SIFT::create();
    return 0;
}