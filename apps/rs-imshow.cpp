#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/opencv.hpp>   // Include OpenCV API
#include <vector>

void sift(cv::Mat img1, cv::Mat img2)
{
    using namespace cv;
    using namespace std;
    cout << "start to compare...\n";
    //����Sift�Ļ�������
    int numFeatures = 500;
    //����detector��ŵ�KeyPoints��
    Ptr<SIFT> detector = SIFT::create(numFeatures);
    vector<KeyPoint> m_LeftKey, m_RightKey;
    detector->detect(img1, m_LeftKey);
    detector->detect(img2, m_RightKey);
    //��ӡKeypoints
    cout << "Keypoints1:" << m_LeftKey.size() << endl;
    cout << "Keypoints2:" << m_RightKey.size() << endl;
    
    Mat drawsrc, drawsrc2;
    drawKeypoints(img1, m_LeftKey, drawsrc);
    // imshow("drawsrc", drawsrc);
    drawKeypoints(img2, m_RightKey, drawsrc2);
    // imshow("drawsrc2", drawsrc2);
    
    //����������������,����������ȡ
    Mat dstSIFT, dstSIFT2;
    Ptr<SiftDescriptorExtractor> descriptor = SiftDescriptorExtractor::create();
    descriptor->compute(img1, m_LeftKey, dstSIFT);
    descriptor->compute(img2, m_RightKey, dstSIFT2);
    cout << dstSIFT.cols << endl;
    cout << dstSIFT2.rows << endl;
    
    
    //����BFMatch����ƥ��
    BFMatcher matcher(NORM_L2);
    //����ƥ��������
    vector<DMatch> matches;
    //ʵ��������֮���ƥ��
    matcher.match(dstSIFT, dstSIFT2, matches);

    //����������������ֵ����Сֵ
    double max_dist = 0;
    double min_dist = 1000;
    for (int i = 1; i < dstSIFT.rows; ++i)
    {
        //ͨ��ѭ�����¾��룬����ԽСԽƥ��
        double dist = matches[i].distance;
        if (dist > max_dist)
        max_dist = dist;
        if (dist < min_dist)
        min_dist = dist;
    }
    cout << "min_dist=" << min_dist << endl;
    cout << "max_dist=" << max_dist << endl;
    //ƥ����ɸѡ    
    vector<DMatch> goodMatches;
    for (int i = 0; i < matches.size(); ++i)
    {
        double dist = matches[i].distance;
        if (dist < 0.2 * max_dist)
        goodMatches.push_back(matches[i]);
    }
    cout << "goodMatches:" << goodMatches.size() << endl;
    

    Mat img_matches;
    drawMatches(img1, m_LeftKey, img2, m_RightKey, matches, img_matches);
    // imshow("�?匹配消除�?",img_matches);
    imwrite("result_without_ransanc.jpg", img_matches);
    
    Mat result;
    //ƥ������������ɫ����һ��������ɫ���
    drawMatches(img1, m_LeftKey, img2, m_RightKey, goodMatches, result, 
        Scalar(255, 255, 0), Scalar::all(-1));
    // imshow("after MatchSIFT", result);
    imwrite("result_after_match.jpg", result);

    //RANSACƥ����� ʹ��ɸѡ���siftƥ���
    vector<DMatch> m_Matches = goodMatches;
    // ����ռ�
    int ptCount = (int)m_Matches.size();

    Mat p1(ptCount, 2, CV_32F);         //�洢��ͼƥ�������
    Mat p2(ptCount, 2, CV_32F);         //�洢��ͼƥ�������

    // ��Keypointת��ΪMat
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

    // ��RANSAC��������F
    Mat m_Fundamental;
    vector<uchar> m_RANSACStatus;       // ����������ڴ洢RANSAC��ÿ�����״̬
    findFundamentalMat(p1, p2, m_RANSACStatus, FM_RANSAC);

    // ����Ұ�����  δƥ���

    int OutlinerCount = 0;
    for (int i = 0; i < ptCount; i++)
    {
        if (m_RANSACStatus[i] == 0)    //״̬Ϊ0��ʾҰ��
        {
            OutlinerCount++;
        }
    }
    int InlinerCount = ptCount - OutlinerCount;   // �����ڵ����
    cout << "inner point count: " << InlinerCount << endl;

    // �������������ڱ���ƥ��㣨�ڵ㣩��ƥ���ϵ
    vector<Point2f> m_LeftInlier;           //��ͼƥ���
    vector<Point2f> m_RightInlier;          //��ͼƥ���
    vector<DMatch> m_InlierMatches;         //ƥ���ϵ

    m_InlierMatches.resize(InlinerCount);
    m_LeftInlier.resize(InlinerCount);
    m_RightInlier.resize(InlinerCount);
    InlinerCount = 0;
    float inlier_minRx = img1.cols;        //���ڴ洢�ڵ�����ͼ��С�����꣬�Ա�����ں�

    for (int i = 0; i < ptCount; i++)
    {
        if (m_RANSACStatus[i] != 0)
        {
            //��ͼƥ�������
            m_LeftInlier[InlinerCount].x = p1.at<float>(i, 0);
            m_LeftInlier[InlinerCount].y = p1.at<float>(i, 1);
            //��ͼƥ�������
            m_RightInlier[InlinerCount].x = p2.at<float>(i, 0);
            m_RightInlier[InlinerCount].y = p2.at<float>(i, 1);
            //ƥ���ϵ
            m_InlierMatches[InlinerCount].queryIdx = InlinerCount;
            m_InlierMatches[InlinerCount].trainIdx = InlinerCount;
            
            //�洢ƥ�������ͼ��С������
            if (m_RightInlier[InlinerCount].x < inlier_minRx) 
                inlier_minRx = m_RightInlier[InlinerCount].x;   

            InlinerCount++;
        }
    }

    // ���ڵ�ת��ΪdrawMatches����ʹ�õĸ�ʽ
    vector<KeyPoint> key1(InlinerCount);
    vector<KeyPoint> key2(InlinerCount);
    KeyPoint::convert(m_LeftInlier, key1);
    KeyPoint::convert(m_RightInlier, key2);

    // ��ʾ����F������ڵ�ƥ��
    Mat OutImage;
    drawMatches(img1, key1, img2, key2, m_InlierMatches, OutImage);
    destroyAllWindows();

    //����H���Դ洢RANSAC�õ��ĵ�Ӧ����
    Mat H = findHomography(m_LeftInlier, m_RightInlier, RANSAC);        //��ƽ��֮���ת������

    //�洢��ͼ�Ľǣ�����任����ͼλ��
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point(0, 0); 
    obj_corners[1] = Point(img1.cols, 0);
    obj_corners[2] = Point(img1.cols, img1.rows); 
    obj_corners[3] = Point(0, img1.rows);

    std::vector<Point2f> scene_corners(4);      //��ͼ�ĽǱ任����ͼ��λ��

    //ת��
    perspectiveTransform(obj_corners, scene_corners, H);                //ͶӰ���Ҳ�����ӽ�  ��ƥ���������λ�ý���ת��ͶӰ
    
    //�����任��ͼ��λ��
    Point2f offset((float)img1.cols, 0);
    //Point2f offset(0, 0);

    line(OutImage, scene_corners[0] + offset, scene_corners[1] + offset, Scalar(0, 255, 0), 4);
    line(OutImage, scene_corners[1] + offset, scene_corners[2] + offset, Scalar(0, 255, 0), 4);
    line(OutImage, scene_corners[2] + offset, scene_corners[3] + offset, Scalar(0, 255, 0), 4);
    line(OutImage, scene_corners[3] + offset, scene_corners[0] + offset, Scalar(0, 255, 0), 4);
    // imshow("Good Matches & Object detection", OutImage);
    // waitKey(0);

    int drift = scene_corners[1].x;                                                        //����ƫ����

    //�½�һ������洢��׼���Ľǵ�λ��
    int width = int(max(abs(scene_corners[1].x), abs(scene_corners[2].x)));
    int height = img1.rows;                                                                
    float origin_x = 0,                                                         //��ͼת���󳬳�0��������ͼ��Ŀ�ȴ�С
        origin_y = 0;

    //������ͼת��֮��Ŀ�͸�
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
    //��ѡ��height-=int(origin_y);
    Mat imageturn = Mat::zeros(width, height, img1.type());     //��ͼ�任���

    //��ȡ�µı任����ʹͼ��������ʾ    
    for (int i = 0;i < 4;i++) 
    { 
        scene_corners[i].x -= origin_x;                             //������ʾ��ȫ��λ�ã�����0��������ͼ����Ϊorigin.x
        //��ѡ��scene_corners[i].y -= (float)origin_y; }
    }    
    Mat H1 = getPerspectiveTransform(obj_corners, scene_corners);       //���¼�����ͼ��ʵ��ת������?

    //������ͼͼ��任����ʾЧ��    ���ͼ��任�Ļ�׼�����Ҳ��Ӧ�㣬������Ӧ����ͼ���е�λ��ת�����ö�Ӧ������ͼ�е�λ��
    warpPerspective(img1, imageturn, H1, Size(width, height));
    // imshow("image_Perspective after the left picture change", imageturn);
    // waitKey(0);

    //ͼ���ں�
    int width_ol = width - int(inlier_minRx - origin_x);                    //�ص�����Ŀ�
    int start_x = int(inlier_minRx - origin_x);                         //imageturn�ϵ��ص��������ʼ����  �Ҳ��ڵ����С��-origin_x
    //�ص�����ʼ�ĺ�����Ϊ��ͼ�ϵ���С��������ڵ���imageturn�ϵ����꣬�Ҳ�Ϊ��ͼ���ұ߽�
    cout << "width: " << width << endl;
    cout << "img1.width: " << img1.cols << endl;
    cout << "start_x: " << start_x << endl;
    cout << "width_ol: " << width_ol << endl;

    uchar* ptr = imageturn.data;            //���ͼ��ת��֮���ͼ������
    double alpha = 0, beta = 1;             //����Ȩ��

    //��Ȩ�ؼ����غϲ��ֵ�����ֵ
    for (int row = 0;row < height;row++) //ͼ��ת��֮��ĸ߶�
    {
        //step�������ΪMat������ÿһ�еġ������������ֽ�Ϊ������λ��ÿһ��������Ԫ�ص��ֽ����������ۼ���һ��������Ԫ�ء�����ͨ��������ͨ����elemSize1֮���ֵ
        //elemSize: ͨ����

        ptr = imageturn.data + row * imageturn.step + (start_x)*imageturn.elemSize();      //��ͼ��һͨ��  ���ص����ֿ�ʼ
        for (int col = 0;col < width_ol;col++)
        {
            uchar* ptr_c1 = ptr + imageturn.elemSize1();       //��ͼ�ڶ�ͨ��
            uchar* ptr_c2 = ptr_c1 + imageturn.elemSize1();     //��ͼ����ͨ��
            
            uchar* ptr2 = img2.data + row * img2.step + (col + int(inlier_minRx)) * img2.elemSize(); //��ͼ��һͨ��  ���غϲ��ֿ�ʼ����
            uchar* ptr2_c1 = ptr2 + img2.elemSize1(); //��ͼ�ڶ�ͨ��
            uchar* ptr2_c2 = ptr2_c1 + img2.elemSize1();//��ͼ����ͨ��

            alpha = double(col) / double(width_ol); 
            beta = 1 - alpha;

            //��ͼ��ת����֮���������ֵ�ĺڵ㣬����ȫ�����Ҳ�ͼ������ֵ
            if (*ptr == 0 && *ptr_c1 == 0 && *ptr_c2 == 0) 
            {
                *ptr = (*ptr2);                 //��ͼ�����ص�һͨ��
                *ptr_c1 = (*ptr2_c1);           //��ͼ�����صڶ�ͨ��
                *ptr_c2 = (*ptr2_c2);           //��ͼ�����ص���ͨ��
            }

            //���Ǻڵ�İ�Ȩ�ؼ���
            *ptr = (*ptr) * beta + (*ptr2) * alpha;////��ͼͨ��1������ֵ����Ȩ��+�Ҳ�ͨ��1����ֵ����Ȩ��
            *ptr_c1 = (*ptr_c1) * beta + (*ptr2_c1) * alpha;
            *ptr_c2 = (*ptr_c2) * beta + (*ptr2_c2) * alpha;

            //ָ�����
            ptr += imageturn.elemSize();
        }
        
    }

   // imshow("image_overlap", imageturn);
   // waitKey(0);

    Mat img_result = Mat::zeros(height, width + img2.cols - drift, img1.type()); //int drift = scene_corners[1].x;
    uchar* ptr_r = imageturn.data;


    for (int row = 0;row < height;row++) 
    {

        //�����ͼ��������ͼ��
        ptr_r = img_result.data + row * img_result.step;//ָ����ͼ���ָ��  ��һͨ��
        for (int col = 0;col < imageturn.cols;col++)
        {
            uchar* ptr_rc1 = ptr_r + imageturn.elemSize1();//ָ����ͼ���ָ��  �ڶ�ͨ��      
            uchar* ptr_rc2 = ptr_rc1 + imageturn.elemSize1(); //ָ����ͼ���ָ��  ����ͨ�� 

            uchar* ptr = imageturn.data + row * imageturn.step + col * imageturn.elemSize();//ָ��   ���ͼ���ָ��   ��һͨ��
            uchar* ptr_c1 = ptr + imageturn.elemSize1();//�ڶ�ͨ��
            uchar* ptr_c2 = ptr_c1 + imageturn.elemSize1();//����ͨ��
                                                                                
            *ptr_r = *ptr; //ȫ����ֵΪ���ͼ�������ֵ
            *ptr_rc1 = *ptr_c1;
            *ptr_rc2 = *ptr_c2;

            ptr_r += img_result.elemSize();
        }

        //���Ҳ�ͼ���ںϽ����ͼ��
        ptr_r = img_result.data + row * img_result.step + imageturn.cols * img_result.elemSize();//ָ����ͼ���ָ��   �����ͼ����ұ߽翪ʼ����
        for (int col = imageturn.cols;col < img_result.cols;col++)
        {
            uchar* ptr_rc1 = ptr_r + imageturn.elemSize1();//ָ����ͼ��ָ��    �ڶ�ͨ��
            uchar* ptr_rc2 = ptr_rc1 + imageturn.elemSize1();//ָ����ͼ��       ����ͨ��

            uchar* ptr2 = img2.data + row * img2.step + (col - imageturn.cols + drift) * img2.elemSize();      //ָ���Ҳ�ͼ��      ���غϲ��ֿ�ʼ���� �غϲ������Ϊ�Ҳ�ͼ���еĵ�һ��ƥ����            
            uchar* ptr2_c1 = ptr2 + img2.elemSize1();//ָ���Ҳ�ͼ��   �ڶ�ͨ��
            uchar* ptr2_c2 = ptr2_c1 + img2.elemSize1();//ָ���Ҳ�ͼ��   ����ͨ��

            *ptr_r = *ptr2;
            *ptr_rc1 = *ptr2_c1;
            *ptr_rc2 = *ptr2_c2;

            ptr_r += img_result.elemSize();
        }
    }

    // imshow("image_result", img_result);
    cv::imwrite("final_result.jpg", img_result);
    cout << "successfully compute!\n";
    return;
}


int main(int argc, char * argv[]) try
{
    // printf("argc, %d", argc);
    if(argc > 1){
        cv::Mat img1 = cv::imread("image/a.jpg");
        cv::Mat img2 = cv::imread("image/b.jpg");
        sift(img1, img2);
        return EXIT_SUCCESS;
    }
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
    const auto window_name = "left Image";
    namedWindow(window_name, WINDOW_AUTOSIZE);
    
    while (waitKey(1) < 0 && getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0)
    {
        rs2::frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera

        rs2::video_frame left_frame = data.get_infrared_frame(1);
        rs2::video_frame right_frame = data.get_infrared_frame(2);

        auto pf = left_frame.get_profile().as<rs2::video_stream_profile>();
        auto w = pf.width();
        auto h = pf.height();

        Mat image1(Size(w, h), CV_8UC1, (void*)left_frame.get_data());
        Mat image2(Size(w, h), CV_8UC1, (void*)right_frame.get_data());

        imshow(window_name, image1);
        imshow("right Image", image2);

        static int i = 0;
        if (++i == 100)
        {
            imwrite("a.jpg", image1);
            imwrite("b.jpg", image2);
            sift(image1, image2);
            break;
        }

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



