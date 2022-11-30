#include <iostream>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp> 
#include<opencv2/xfeatures2d.hpp>
#include<opencv2/core/core.hpp>
#include <stdio.h>  
#include <cv.h>
#include <highgui.h>

using namespace cv;
using namespace std;
//using namespace cv::xfeatures2d;                    //只有加上这句命名空间，SiftFeatureDetector and SiftFeatureExtractor才可以使用


int main()
{
    //Create SIFT class pointer
    Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();

    //Loading images
    Mat img1 = imread("1.jpg");
    Mat img2 = imread("2.jpg");

    if (!img1.data || !img2.data)
    {
        cout << "Reading picture error！" << endl;
        return false;
    }

    //Detect the 特征点
    double t0 = getTickCount();                         //当前时间
    vector<KeyPoint> m_LeftKey, m_RightKey;
    f2d->detect(img1, m_LeftKey);
    f2d->detect(img2, m_RightKey);

    cout << "The keypoints number of img1 is:" << m_LeftKey.size() << endl;
    cout << "The keypoints number of img2 is:" << m_RightKey.size() << endl;

    //Calculate descriptors (feature vectors)		描述子
    Mat descriptors_1, descriptors_2;
    f2d->compute(img1, m_LeftKey, descriptors_1);
    f2d->compute(img2, m_RightKey, descriptors_2);

    double freq = getTickFrequency();
    double tt = ((double)getTickCount() - t0) / freq;
    cout << "Extract SIFT Time:" << tt << "ms" << endl;

    //画关键点
    Mat img_keypoints_1, img_keypoints_2;
    drawKeypoints(img1, m_LeftKey, img_keypoints_1, Scalar::all(-1), 0);
    drawKeypoints(img2, m_RightKey, img_keypoints_2, Scalar::all(-1), 0);
    imshow("img_keypoints_1", img_keypoints_1);
    imshow("img_keypoints_2", img_keypoints_2);

    //Matching descriptor vector using BFMatcher   全部匹配点  暴力匹配
    BFMatcher matcher;
    vector<DMatch> matches;					//存储匹配出的对应点索引和距离
    matcher.match(descriptors_1, descriptors_2, matches);
    cout << "The number of match:" << matches.size() << endl;

    /*
    //绘制匹配出的全部关键点
    Mat img_matches;
    drawMatches(img1, m_LeftKey, img2, m_RightKey, matches, img_matches);
    imshow("Match image", img_matches);
    */

    //计算匹配结果中距离的最大和最小值  
    //距离是指两个特征向量间的欧式距离，表明两个特征的差异，值越小表明两个特征点越接近  
    double max_dist = 0;
    double min_dist = 100;
    for (int i = 0; i < matches.size(); i++)
    {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }
    cout << "最大距离：" << max_dist << endl;
    cout << "最小距离：" << min_dist << endl;

    //筛选出较好的匹配点  
    vector<DMatch> goodMatches;
    for (int i = 0; i < matches.size(); i++)
    {
        if (matches[i].distance < 0.2 * max_dist)
        {
            goodMatches.push_back(matches[i]);
        }
    }
    cout << "goodMatch个数：" << goodMatches.size() << endl;

    //画出匹配结果  
    Mat img_matches;
    //红色连接的是匹配的特征点对，绿色是未匹配的特征点  
    drawMatches(img1, m_LeftKey, img2, m_RightKey, goodMatches, img_matches,Scalar::all(-1), CV_RGB(0, 255, 0), Mat(), 2);

    imshow("MatchSIFT筛选后", img_matches);
    IplImage result = img_matches;

    waitKey(0);


    //RANSAC匹配过程 使用筛选后的sift匹配点
    vector<DMatch> m_Matches = goodMatches;
    // 分配空间
    int ptCount = (int)m_Matches.size();

    Mat p1(ptCount, 2, CV_32F);         //存储左图匹配点坐标
    Mat p2(ptCount, 2, CV_32F);         //存储右图匹配点坐标

    // 把Keypoint转换为Mat
    Point2f pt;
    for (int i = 0; i < ptCount; i++)
    {
        pt = m_LeftKey[m_Matches[i].queryIdx].pt;
        p1.at<float>(i, 0) = pt.x;
        p1.at<float>(i, 1) = pt.y;

        pt = m_RightKey[m_Matches[i].trainIdx].pt;
        p2.at<float>(i, 0) = pt.x;
        p2.at<float>(i, 1) = pt.y;
    }

    // 用RANSAC方法计算F
    Mat m_Fundamental;
    vector<uchar> m_RANSACStatus;       // 这个变量用于存储RANSAC后每个点的状态
    findFundamentalMat(p1, p2, m_RANSACStatus, FM_RANSAC);

    // 计算野点个数  未匹配点

    int OutlinerCount = 0;
    for (int i = 0; i < ptCount; i++)
    {
        if (m_RANSACStatus[i] == 0)    // 状态为0表示野点
        {
            OutlinerCount++;
        }
    }
    int InlinerCount = ptCount - OutlinerCount;   // 计算内点个数
    cout << "内点数为：" << InlinerCount << endl;


    // 这三个变量用于保存匹配点（内点）和匹配关系
    vector<Point2f> m_LeftInlier;           //左图匹配点
    vector<Point2f> m_RightInlier;          //右图匹配点
    vector<DMatch> m_InlierMatches;         //匹配关系

    m_InlierMatches.resize(InlinerCount);
    m_LeftInlier.resize(InlinerCount);
    m_RightInlier.resize(InlinerCount);
    InlinerCount = 0;
    float inlier_minRx = img1.cols;        //用于存储内点中右图最小横坐标，以便后续融合

    for (int i = 0; i < ptCount; i++)
    {
        if (m_RANSACStatus[i] != 0)
        {
            //左图匹配点坐标
            m_LeftInlier[InlinerCount].x = p1.at<float>(i, 0);
            m_LeftInlier[InlinerCount].y = p1.at<float>(i, 1);
            //右图匹配点坐标
            m_RightInlier[InlinerCount].x = p2.at<float>(i, 0);
            m_RightInlier[InlinerCount].y = p2.at<float>(i, 1);
            //匹配关系
            m_InlierMatches[InlinerCount].queryIdx = InlinerCount;
            m_InlierMatches[InlinerCount].trainIdx = InlinerCount;
            
            //存储匹配点中右图最小横坐标
            if (m_RightInlier[InlinerCount].x < inlier_minRx) 
                inlier_minRx = m_RightInlier[InlinerCount].x;   

            InlinerCount++;
        }
    }

    // 把内点转换为drawMatches可以使用的格式
    vector<KeyPoint> key1(InlinerCount);
    vector<KeyPoint> key2(InlinerCount);
    KeyPoint::convert(m_LeftInlier, key1);
    KeyPoint::convert(m_RightInlier, key2);

    // 显示计算F过后的内点匹配
    Mat OutImage;
    drawMatches(img1, key1, img2, key2, m_InlierMatches, OutImage);
  //  cvNamedWindow("Match features", 1);
//    cvShowImage("Match features", OutImage);
 //   waitKey(0);

    cvDestroyAllWindows();

    //矩阵H用以存储RANSAC得到的单应矩阵
    Mat H = findHomography(m_LeftInlier, m_RightInlier, RANSAC);        //两平面之间的转换矩阵

    //存储左图四角，及其变换到右图位置
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point(0, 0); 
    obj_corners[1] = Point(img1.cols, 0);
    obj_corners[2] = Point(img1.cols, img1.rows); 
    obj_corners[3] = Point(0, img1.rows);

    std::vector<Point2f> scene_corners(4);      //左图四角变换到右图的位置
    
    //转换
    perspectiveTransform(obj_corners, scene_corners, H);                //投影到右侧的新视角  按匹配的特征点位置进行转换投影

    //画出变换后图像位置
    Point2f offset((float)img1.cols, 0);
    //Point2f offset(0, 0);

    line(OutImage, scene_corners[0] + offset, scene_corners[1] + offset, Scalar(0, 255, 0), 4);
    line(OutImage, scene_corners[1] + offset, scene_corners[2] + offset, Scalar(0, 255, 0), 4);
    line(OutImage, scene_corners[2] + offset, scene_corners[3] + offset, Scalar(0, 255, 0), 4);
    line(OutImage, scene_corners[3] + offset, scene_corners[0] + offset, Scalar(0, 255, 0), 4);
    imshow("Good Matches & Object detection左图转换后的位置标注", OutImage);
    waitKey(0);
   

    int drift = scene_corners[1].x;                                                        //储存偏移量

    //新建一个矩阵存储配准后四角的位置
    int width = int(max(abs(scene_corners[1].x), abs(scene_corners[2].x)));
    int height = img1.rows;                                                                  //或者：int height = int(max(abs(scene_corners[2].y), abs(scene_corners[3].y)));
    float origin_x = 0,                                                         //左图转换后超出0坐标左侧的图像的宽度大小
        origin_y = 0;

    //计算左图转换之后的宽和高
    if (scene_corners[0].x < 0) 
    {
        if (scene_corners[3].x < 0) 
            origin_x += min(scene_corners[0].x, scene_corners[3].x);
        else origin_x += scene_corners[0].x;
    }
    width -= int(origin_x);

    if (scene_corners[0].y < 0) 
    {
        if (scene_corners[1].y) 
            origin_y += min(scene_corners[0].y, scene_corners[1].y);
        else origin_y += scene_corners[0].y;
    }
    //可选：height-=int(origin_y);
    Mat imageturn = Mat::zeros(width, height, img1.type());     //左图变换后的

    //获取新的变换矩阵，使图像完整显示      
    for (int i = 0;i < 4;i++) 
    { 
        scene_corners[i].x -= origin_x;                             //补偿显示不全的位置，超出0坐标左侧的图像宽度为origin.x
        //可选：scene_corners[i].y -= (float)origin_y; }
    }    
    Mat H1 = getPerspectiveTransform(obj_corners, scene_corners);       //重新计算左图的实际转换矩阵

    //进行左图图像变换，显示效果    左侧图像变换的基准是左右侧对应点，将左侧对应点在图像中的位置转换到该对应点在右图中的位置
    warpPerspective(img1, imageturn, H1, Size(width, height));
    imshow("image_Perspective左图变换后", imageturn);
    waitKey(0);

    //图像融合
    int width_ol = width - int(inlier_minRx - origin_x);                    //重叠区域的宽
    int start_x = int(inlier_minRx - origin_x);                         //imageturn上的重叠区域的起始坐标  右侧内点的最小点-origin_x
    //重叠区域开始的横坐标为右图上的最小横坐标的内点在imageturn上的坐标，右侧为左图的右边界
    cout << "width: " << width << endl;
    cout << "img1.width: " << img1.cols << endl;
    cout << "start_x: " << start_x << endl;
    cout << "width_ol: " << width_ol << endl;

    uchar* ptr = imageturn.data;            //左侧图像转换之后的图像数据
    double alpha = 0, beta = 1;             //定义权重

    //按权重计算重合部分的像素值
    for (int row = 0;row < height;row++) //图像转换之后的高度
    {
        //step可以理解为Mat矩阵中每一行的“步长”，以字节为基本单位，每一行中所有元素的字节总量，是累计了一行中所有元素、所有通道、所有通道的elemSize1之后的值
        //elemSize: 通道数

        ptr = imageturn.data + row * imageturn.step + (start_x)*imageturn.elemSize();       //左图第一通道  从重叠部分开始
        for (int col = 0;col < width_ol;col++)
        {
            uchar* ptr_c1 = ptr + imageturn.elemSize1();       //左图第二通道 
            uchar* ptr_c2 = ptr_c1 + imageturn.elemSize1();     //左图第三通道
            
            uchar* ptr2 = img2.data + row * img2.step + (col + int(inlier_minRx)) * img2.elemSize(); //右图第一通道  从重合部分开始计算
            uchar* ptr2_c1 = ptr2 + img2.elemSize1(); //右图第二通道
            uchar* ptr2_c2 = ptr2_c1 + img2.elemSize1();//右图第三通道

            alpha = double(col) / double(width_ol); 
            beta = 1 - alpha;

            //左图中转换了之后的无像素值的黑点，则完全拷贝右侧图的像素值
            if (*ptr == 0 && *ptr_c1 == 0 && *ptr_c2 == 0) 
            {
                *ptr = (*ptr2);                 //左图该像素第一通道    
                *ptr_c1 = (*ptr2_c1);           //左图该像素第二通道
                *ptr_c2 = (*ptr2_c2);           //左图该像素第三通道
            }

            //不是黑点的按权重计算
            *ptr = (*ptr) * beta + (*ptr2) * alpha;//左图通道1的像素值乘以权重+右侧通道1像素值乘以权重
            *ptr_c1 = (*ptr_c1) * beta + (*ptr2_c1) * alpha;
            *ptr_c2 = (*ptr_c2) * beta + (*ptr2_c2) * alpha;

            //指针后移
            ptr += imageturn.elemSize();
        }
        
    }

   // imshow("image_overlap", imageturn);
   // waitKey(0);

    Mat img_result = Mat::zeros(height, width + img2.cols - drift, img1.type()); //int drift = scene_corners[1].x;
    uchar* ptr_r = imageturn.data;

    for (int row = 0;row < height;row++) 
    {

        //将左侧图像融入结果图像
        ptr_r = img_result.data + row * img_result.step;//指向结果图像的指针  第一通道
        for (int col = 0;col < imageturn.cols;col++)
        {
            uchar* ptr_rc1 = ptr_r + imageturn.elemSize1();//指向结果图像的指针  第二通道    
            uchar* ptr_rc2 = ptr_rc1 + imageturn.elemSize1(); 指向结果图像的指针  第三通道 

            uchar* ptr = imageturn.data + row * imageturn.step + col * imageturn.elemSize();//指向   左侧图像的指针   第一通道
            uchar* ptr_c1 = ptr + imageturn.elemSize1();//第二通道
            uchar* ptr_c2 = ptr_c1 + imageturn.elemSize1();//第三通道
                                                                                
            *ptr_r = *ptr; //全部赋值为左侧图像的像素值
            *ptr_rc1 = *ptr_c1;
            *ptr_rc2 = *ptr_c2;

            ptr_r += img_result.elemSize();
        }

        //将右侧图像融合进结果图像
        ptr_r = img_result.data + row * img_result.step + imageturn.cols * img_result.elemSize();//指向结果图像的指针   从左侧图像的右边界开始计算
        for (int col = imageturn.cols;col < img_result.cols;col++)
        {
            uchar* ptr_rc1 = ptr_r + imageturn.elemSize1();//指向结果图像指针    第二通道
            uchar* ptr_rc2 = ptr_rc1 + imageturn.elemSize1();//指向结果图像       第三通道

            uchar* ptr2 = img2.data + row * img2.step + (col - imageturn.cols + drift) * img2.elemSize();     //指向右侧图像      从重合部分开始计算 重合部分起点为右侧图像中的第一个匹配点对
            uchar* ptr2_c1 = ptr2 + img2.elemSize1();//指向右侧图像   第二通道
            uchar* ptr2_c2 = ptr2_c1 + img2.elemSize1();//指向右侧图像   第三通道

            *ptr_r = *ptr2;
            *ptr_rc1 = *ptr2_c1;
            *ptr_rc2 = *ptr2_c2;

            ptr_r += img_result.elemSize();
        }
    }

    imshow("image_result融合结果", img_result);
    while (1)
    {
		if (waitKey(100) == 19) //cvSaveImage("E:\\final_result.jpg", &IplImage(img_result));
			cv::imwrite("E:\\final_result.jpg", img_result);
		if (waitKey(100) == 27) break; //按esc退出，ctl+s保存图像
    }

    return 0;
}
