#include<iostream>
#include<string>
#include<sstream>
#include<opencv2/opencv.hpp>
#include"./lib/akaze.h"

using namespace cv;
using namespace std;

string itos(int i) {
    stringstream s;
    s << i;
    return s.str();
}

void akaze_compute_derivative_kernels(cv::OutputArray kx_, cv::OutputArray ky_, const size_t dx, const size_t dy, const size_t scale);

void akaze_compute_scharr_derivatives(const cv::Mat& src, cv::Mat& dst, const size_t xorder, const size_t yorder, const size_t scale);

bool matIsEqual(const cv::Mat mat1, const cv::Mat mat2);

float matSimilarity(const cv::Mat mat1, Img& mat2);

float matSimilarity(const cv::Mat mat1, const cv::Mat mat2);

BYTE* matToBytes(Mat image);

Mat floatToMat(float** src, int rows, int cols);

Mat byteToMat(BYTE** src, int row, int col);

KeyPoint keypointToKeyPoint(Keypoint kpt);

void matches2points_nndr(const std::vector<cv::KeyPoint>& train,
    const std::vector<cv::KeyPoint>& query,
    const std::vector<std::vector<cv::DMatch> >& matches,
    std::vector<cv::Point2f>& pmatches, float nndr);

void compute_inliers_ransac(const std::vector<cv::Point2f>& matches,
    std::vector<cv::Point2f>& inliers,
    float error, bool use_fund);

void draw_keypoints(cv::Mat& img, const std::vector<cv::KeyPoint>& kpts);

void draw_inliers(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& img_com,
    const std::vector<cv::Point2f>& ptpairs);
static Scalar randomColor(int64 seed);

int main() {
    string file = "C:\\Users\\tota1Noob\\Desktop\\Graffiti.png";
    string file2 = "C:\\Users\\tota1Noob\\Desktop\\testGraffiti.png";

    Mat rgb = imread(file);
    Mat rgb2;
    rgb.copyTo(rgb2);
    Mat src = imread(file, 0);
    if (src.data == NULL) {
        cout << "Cannot openc the file!" << endl;
        return 1;
    }
    Mat image(src.rows, src.cols, CV_32F);
    src.convertTo(image, CV_32F, 1.0 / 255.0, 0);

    int orgRow = image.rows,
        orgCol = image.cols;
    BYTE* imgTmp = matToBytes(src);
    Img tmp(imgTmp, orgRow, orgCol);


    double t1 = getTickCount();
    AKAZEOptions options;
    options.img_height = tmp.rows; options.img_width = tmp.cols;
    libAKAZE::AKAZE akaze(options);
    akaze.Create_Nonlinear_Scale_Space(tmp);
    vector<Keypoint> kpts;
    akaze.Feature_Detection(kpts);
    int t = (6 + 36 + 120) * options.descriptor_channels;
    int col = ceil(t / 8.), size = kpts.size();
    BYTE** featureVector = new BYTE * [size];
    for (int i = 0; i < size; ++i) {
        featureVector[i] = new BYTE[col];
        for (int j = 0; j < col; ++j) {
            featureVector[i][j] = 0;
        }
    }
    akaze.Compute_Descriptors(kpts, featureVector);
    cout << "First done in " << 1000 * (getTickCount() - t1) / getTickFrequency() << "ms" << endl;


    
    Mat match = imread(file2, 0);
    BYTE* byte2 = matToBytes(match);
    Img tmp2(byte2, match.rows, match.cols);
    t1 = getTickCount();
    AKAZEOptions options2;
    options2.img_height = tmp2.rows; options2.img_width = tmp2.cols;
    libAKAZE::AKAZE akaze2(options2);
    akaze2.Create_Nonlinear_Scale_Space(tmp2);
    vector<Keypoint> kpts2;
    akaze2.Feature_Detection(kpts2);
    int size2 = kpts2.size();
    BYTE** featureVector2 = new BYTE * [size2];
    for (int i = 0; i < size2; ++i) {
        featureVector2[i] = new BYTE[col];
        for (int j = 0; j < col; ++j) {
            featureVector2[i][j] = 0;
        }
    }
    akaze2.Compute_Descriptors(kpts2, featureVector2);
    cout << "Second done in " << 1000 * (getTickCount() - t1) / getTickFrequency() << "ms" << endl;

    vector<KeyPoint> cvkpt1;
    for (int i = 0; i < size; ++i) {
        cvkpt1.push_back(keypointToKeyPoint(kpts[i]));
    }
    vector<KeyPoint> cvkpt2;
    for (int i = 0; i < size2; ++i) {
        cvkpt2.push_back(keypointToKeyPoint(kpts2[i]));
    }

    Mat desc1 = byteToMat(featureVector, size, 61);
    Mat desc2 = byteToMat(featureVector2, size2, 61);
    vector<vector<cv::DMatch> > dmatches;
    // Matching Descriptors!!
    vector<cv::Point2f> matches, inliers;
    cv::Ptr<cv::DescriptorMatcher> matcher_l1 = cv::DescriptorMatcher::create("BruteForce-Hamming");

    matcher_l1->knnMatch(desc1, desc2, dmatches, 2);

    // Compute Inliers!!
    matches2points_nndr(cvkpt1, cvkpt2, dmatches, matches, 0.80f);

    compute_inliers_ransac(matches, inliers, 2.50f, false);

    // Compute the inliers statistics
    int nkpts1 = kpts.size();
    int nkpts2 = kpts2.size();
    int nmatches = matches.size() / 2;
    int ninliers = inliers.size() / 2;
    int noutliers = nmatches - ninliers;
    float ratio = 100.0 * ((float)ninliers / (float)nmatches);

    Mat img1_rgb, img2_rgb;
    img1_rgb = cv::Mat(cv::Size(src.cols, src.rows), CV_8UC3);
    img2_rgb = cv::Mat(cv::Size(match.cols, match.rows), CV_8UC3);
    Mat img_com = cv::Mat(cv::Size(src.cols + match.cols, src.rows), CV_8UC3);
    //img1_rgb = imread(file); img2_rgb = imread(file2);

    if (true) {

        // Prepare the visualization
        cvtColor(src, img1_rgb, cv::COLOR_GRAY2BGR);
        cvtColor(match, img2_rgb, cv::COLOR_GRAY2BGR);

        // Show matching statistics
        cout << "Number of Keypoints Image 1: " << nkpts1 << endl;
        cout << "Number of Keypoints Image 2: " << nkpts2 << endl;
        cout << "Number of Matches: " << nmatches << endl;
        cout << "Number of Inliers: " << ninliers << endl;
        cout << "Number of Outliers: " << noutliers << endl;
        cout << "Inliers Ratio: " << ratio << endl << endl;

        //draw_keypoints(img1_rgb, cvkpt1);
        //draw_keypoints(img2_rgb, cvkpt2);
        //imshow("pic1", img1_rgb);
        //imshow("pic2", img2_rgb);
        draw_inliers(img1_rgb, img2_rgb, img_com, inliers);
        //cv::namedWindow("Inliers", cv::WINDOW_AUTO);
        cv::imshow("Inliers", img_com);
        cv::waitKey(0);
    }
    waitKey(0);
    return 0;
}

void akaze_compute_derivative_kernels(cv::OutputArray kx_, cv::OutputArray ky_,
    const size_t dx, const size_t dy, const size_t scale) {

    const int ksize = 3 + 2 * (scale - 1);

    // The usual Scharr kernel
    if (scale == 1) {
        cv::getDerivKernels(kx_, ky_, dx, dy, 0, true, CV_32F);
        return;
    }

    kx_.create(ksize, 1, CV_32F, -1, true);
    ky_.create(ksize, 1, CV_32F, -1, true);
    cv::Mat kx = kx_.getMat();
    cv::Mat ky = ky_.getMat();

    float w = 10.0 / 3.0;
    float norm = 1.0 / (2.0 * scale * (w + 2.0));

    for (int k = 0; k < 2; k++) {
        cv::Mat* kernel = k == 0 ? &kx : &ky;
        int order = k == 0 ? dx : dy;
        float kerI[1000];

        for (int t = 0; t < ksize; t++)
            kerI[t] = 0;

        if (order == 0) {
            kerI[0] = norm;
            kerI[ksize / 2] = w * norm;
            kerI[ksize - 1] = norm;
        }
        else if (order == 1) {
            kerI[0] = -1;
            kerI[ksize / 2] = 0;
            kerI[ksize - 1] = 1;
        }

        cv::Mat temp(kernel->rows, kernel->cols, CV_32F, &kerI[0]);
        temp.copyTo(*kernel);
    }
}

void akaze_compute_scharr_derivatives(const cv::Mat& src, cv::Mat& dst, const size_t xorder,
    const size_t yorder, const size_t scale) {

    cv::Mat kx, ky;
    akaze_compute_derivative_kernels(kx, ky, xorder, yorder, scale);
    cv::sepFilter2D(src, dst, CV_32F, kx, ky);
}

bool matIsEqual(const cv::Mat mat1, const cv::Mat mat2) {
    if (mat1.empty() && mat2.empty()) {
        return true;
    }
    if (mat1.cols != mat2.cols || mat1.rows != mat2.rows || mat1.dims != mat2.dims ||
        mat1.channels() != mat2.channels()) {
        return false;
    }
    if (mat1.size() != mat2.size() || mat1.channels() != mat2.channels() || mat1.type() != mat2.type()) {
        return false;
    }
    int nrOfElements1 = mat1.total() * mat1.elemSize();
    if (nrOfElements1 != mat2.total() * mat2.elemSize()) return false;
    bool lvRet = memcmp(mat1.data, mat2.data, nrOfElements1) == 0;
    return lvRet;
}

float matSimilarity(const cv::Mat mat1, Img& mat2) {
    float sim = 0., tmp = 0.;
    float max = 0.;
    int maxi = 0, maxj = 0;
    float m1 = 0., m2 = 0.;
    for (int i = 0; i < mat1.rows; ++i) {
        tmp = 0.;
        for (int j = 0; j < mat1.cols; ++j) {
            if (mat1.ptr<float>(i)[j] == mat2.pixels[i][j]) continue;
            tmp += fabs(mat1.ptr<float>(i)[j] - mat2.pixels[i][j]);
            if (fabs(mat1.ptr<float>(i)[j] - mat2.pixels[i][j]) > max) {
                max = fabs(mat1.ptr<float>(i)[j] - mat2.pixels[i][j]);
                maxi = i; maxj = j;
                m1 = mat1.ptr<float>(i)[j]; m2 = mat2.pixels[i][j];
            }
        }
        tmp /= (float)mat1.cols;
        sim += tmp;
    }
    cout << "Max difference is " << max << " = " << m1 << " - " << m2 << ", at (" << maxi << ", " << maxj << ")" << endl;
    return sim / (float)mat1.rows;
}

float matSimilarity(const cv::Mat mat1, const cv::Mat mat2) {
    float sim = 0., tmp = 0.;
    float max = 0.;
    int maxi = 0, maxj = 0;
    float m1 = 0., m2 = 0.;
    for (int i = 0; i < mat1.rows; ++i) {
        tmp = 0.;
        for (int j = 0; j < mat1.cols; ++j) {
            if (mat1.ptr<float>(i)[j] == mat2.ptr<float>(i)[j]) continue;
            tmp += fabs(mat1.ptr<float>(i)[j] - mat2.ptr<float>(i)[j]);
            if (fabs(mat1.ptr<float>(i)[j] - mat2.ptr<float>(i)[j]) > max) {
                max = fabs(mat1.ptr<float>(i)[j] - mat2.ptr<float>(i)[j]);
                maxi = i; maxj = j;
                m1 = mat1.ptr<float>(i)[j]; m2 = mat2.ptr<float>(i)[j];
            }
        }
        tmp /= (float)mat1.cols;
        sim += tmp;
    }
    cout << "Max difference is " << max << " = " << m1 << " - " << m2 << ", at (" << maxi << ", " << maxj << ")" << endl;
    return sim / (float)mat1.rows;
}

BYTE* matToBytes(Mat image) {
    int size = image.total() * image.elemSize();
    BYTE* bytes = new BYTE[size];
    // you will have to delete[] that later
    std::memcpy(bytes, image.data, size * sizeof(BYTE));
    return bytes;
}
Mat floatToMat(float** src, int rows, int cols) {
    int size = rows * cols;
    Mat tmp;
    tmp.create(cv::Size(cols, rows), CV_32F);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            tmp.ptr<float>(i)[j] = src[i][j];
        }
    }
    return tmp;
}

Mat byteToMat(BYTE** src, int row, int col) {

    int nType = CV_8UC1;
    Mat outImg = Mat::zeros(row, col, nType);
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            outImg.ptr<uchar>(i)[j] = src[i][j];
        }
    }
    return outImg;
}

KeyPoint keypointToKeyPoint(Keypoint kpt) {
    KeyPoint tmp;
    tmp.angle = kpt.angle;
    tmp.class_id = kpt.class_id;
    tmp.octave = kpt.octave;
    tmp.pt.x = kpt.pt.x;
    tmp.pt.y = kpt.pt.y;
    tmp.response = kpt.response;
    tmp.size = kpt.size;
    return tmp;
}

void matches2points_nndr(const std::vector<cv::KeyPoint>& train,
    const std::vector<cv::KeyPoint>& query,
    const std::vector<std::vector<cv::DMatch> >& matches,
    std::vector<cv::Point2f>& pmatches, float nndr) {

    float dist1 = 0.0, dist2 = 0.0;
    for (size_t i = 0; i < matches.size(); i++) {
        cv::DMatch dmatch = matches[i][0];
        dist1 = matches[i][0].distance;
        dist2 = matches[i][1].distance;

        if (dist1 < nndr * dist2) {
            pmatches.push_back(train[dmatch.queryIdx].pt);
            pmatches.push_back(query[dmatch.trainIdx].pt);
        }
    }
}

void compute_inliers_ransac(const std::vector<cv::Point2f>& matches,
    std::vector<cv::Point2f>& inliers,
    float error, bool use_fund) {

    vector<cv::Point2f> points1, points2;
    cv::Mat H = cv::Mat::zeros(3, 3, CV_32F);
    int npoints = matches.size() / 2;
    cv::Mat status = cv::Mat::zeros(npoints, 1, CV_8UC1);

    for (size_t i = 0; i < matches.size(); i += 2) {
        points1.push_back(matches[i]);
        points2.push_back(matches[i + 1]);
    }

    if (npoints > 8) {
        if (use_fund == true)
            H = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, error, 0.99, status);
        else
            H = cv::findHomography(points1, points2, cv::RANSAC, error, status);

        for (int i = 0; i < npoints; i++) {
            if (status.at<unsigned char>(i) == 1) {
                inliers.push_back(points1[i]);
                inliers.push_back(points2[i]);
            }
        }
    }
}

void draw_keypoints(cv::Mat& img, const std::vector<cv::KeyPoint>& kpts) {

    int x = 0, y = 0;
    float radius = 0.0;
    for (size_t i = 0; i < kpts.size(); i++) {
        x = (int)(kpts[i].pt.x + .5);
        y = (int)(kpts[i].pt.y + .5);
        radius = kpts[i].size / 2.0;
        cv::circle(img, cv::Point(x, y), radius * 2.50, cv::Scalar(0, 255, 0), 1);
        cv::circle(img, cv::Point(x, y), 1.0, cv::Scalar(0, 0, 255), -1);
    }
}

void draw_inliers(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& img_com,
    const std::vector<cv::Point2f>& ptpairs) {

    int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    float rows1 = 0.0, cols1 = 0.0;
    float rows2 = 0.0, cols2 = 0.0;
    float ufactor = 0.0, vfactor = 0.0;

    rows1 = img1.rows;
    cols1 = img1.cols;
    rows2 = img2.rows;
    cols2 = img2.cols;
    ufactor = (float)(cols1) / (float)(cols2);
    vfactor = (float)(rows1) / (float)(rows2);

    // This is in case the input images don't have the same resolution
    //cv::Mat img_aux = cv::Mat(cv::Size(img1.cols, img1.rows), CV_8UC3);
    //cv::resize(img2, img_aux, cv::Size(img1.cols, img1.rows), 0, 0, cv::INTER_LINEAR);

    for (int i = 0; i < img_com.rows; i++) {
        for (int j = 0; j < img_com.cols; j++) {
            if (j < img1.cols) {
                *(img_com.ptr<unsigned char>(i) + 3 * j) = *(img1.ptr<unsigned char>(i) + 3 * j);
                *(img_com.ptr<unsigned char>(i) + 3 * j + 1) = *(img1.ptr<unsigned char>(i) + 3 * j + 1);
                *(img_com.ptr<unsigned char>(i) + 3 * j + 2) = *(img1.ptr<unsigned char>(i) + 3 * j + 2);
            }
            else {
                if (i < img2.rows) {
                    int tmp = j - img1.cols;
                    *(img_com.ptr<unsigned char>(i) + 3 * j) = *(img2.ptr<unsigned char>(i) + 3 * tmp);
                    *(img_com.ptr<unsigned char>(i) + 3 * j + 1) = *(img2.ptr<unsigned char>(i) + 3 * tmp + 1);
                    *(img_com.ptr<unsigned char>(i) + 3 * j + 2) = *(img2.ptr<unsigned char>(i) + 3 * tmp + 2);
                }

            }
        }
    }

    for (size_t i = 0; i < ptpairs.size(); i += 2) {
        x1 = (int)(ptpairs[i].x + .5);
        y1 = (int)(ptpairs[i].y + .5);
        x2 = (int)(ptpairs[i + 1].x + img1.cols + .5);
        y2 = (int)(ptpairs[i + 1].y + .5);
        cv::circle(img_com, cv::Point(x1, y1), 4, randomColor(getTickCount()), 1.5, LINE_AA);
        cv::circle(img_com, cv::Point(x2, y2), 4, randomColor(getTickCount()), 1.5, LINE_AA);
        cv::line(img_com, cv::Point(x1, y1), cv::Point(x2, y2), randomColor(getTickCount()), 1.5, LINE_AA);
    }
}

static Scalar randomColor(int64 seed){
    RNG rng(seed);
    int icolor = (unsigned)rng;
    return Scalar(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
}