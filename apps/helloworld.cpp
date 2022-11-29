#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <librealsense2/rs.hpp> // Include Intel RealSense Cross Platform API
// #include <opencv2/xfeatures2d.hpp>
// #include <opencv2/highgui.hpp>
// #include <opencv2/imgproc.hpp>
#include <vector>

int main()
{
    std::cout << "hello" << std::endl;
    cv::Ptr<cv::SIFT> siftDetector = cv::SIFT::create(500);

    // Create a Pipeline - this serves as a top-level API for streaming and processing frames
    rs2::pipeline p;
    // Configure and start the pipeline
    p.start();
    // Block program until frames arrive
    rs2::frameset frames = p.wait_for_frames();
    // Try to get a frame of a depth image
    rs2::depth_frame depth = frames.get_depth_frame();
    // Get the depth frame's dimensions
    float width = depth.get_width();
    float height = depth.get_height();
    // Query the distance from the camera to the object in the center of the image
    float dist_to_center = depth.get_distance(width / 2, height / 2);
    // Print the distance
    std::cout << "The camera is facing an object " << dist_to_center << " meters away " << "\n";

	// Ptr<SIFT> detector = SIFT::create(500);
    // cv::Ptr<cv::feature
    // cv::Ptr<cv::Feature2D> sift = cv::Feature2D:: SIFT::create();
    return 0;
}