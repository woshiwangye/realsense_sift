// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/opencv.hpp>   // Include OpenCV API

int main(int argc, char * argv[]) try
{
    // Declare depth colorizer for pretty visualization of depth data
    rs2::colorizer color_map;

    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;
    // Start streaming with default recommended configuration
    rs2::config cfg;
    // cfg.enable_stream(RS2_STREAM_INFRARED);
    cfg.enable_all_streams();
    // cfg.enable_stream(RS2_STREAM_DEPTH);
    // cfg.enable_stream(RS2_STREAM_COLOR);

    pipe.start(cfg);

    using namespace cv;
    const auto window_name = "Display Image";
    namedWindow(window_name, WINDOW_AUTOSIZE);
    
    while (waitKey(1) < 0 && getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0)
    {
        rs2::frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera

        rs2::video_frame left_frame = data.get_infrared_frame(1);
        rs2::video_frame right_frame = data.get_infrared_frame(2);

        // auto pf = left_frame.get_profile().as<rs2::video_stream_profile>();
        // auto w = pf.width();
        // auto h = pf.height();
        int w = 640, h = 480;
        // rs2::frame color = data.get_color_frame();
        // rs2::frame depth = data.get_depth_frame().apply_filter(color_map);

        // Query frame size (width and height)
        // const int w = depth.as<rs2::video_frame>().get_width();
        // const int h = depth.as<rs2::video_frame>().get_height();
        Mat image1(Size(w, h), CV_8UC1, (void*)left_frame.get_data());
        Mat image2(Size(w, h), CV_8UC1, (void*)right_frame.get_data());
        // imshow("color image", image2);

        // Create OpenCV matrix of size (w,h) from the colorized depth data
        // Mat image(Size(w, h), CV_8UC3, (void*)depth.get_data(), Mat::AUTO_STEP);
        // Mat pic_left(Size(w, h), CV_8UC1, (void*)left_frame.get_data());
        // Update the window with new data
        imshow(window_name, image1);
        imshow("right", image2);
        // imshow("left image", pic_left);

    }
    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}



