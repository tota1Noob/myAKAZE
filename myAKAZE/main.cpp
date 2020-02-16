#include<iostream>
#include<opencv2/opencv.hpp>
#include"./lib/akaze.h"

using namespace cv;
using namespace std;

BYTE* matToBytes(Mat image) {
    int size = image.total() * image.elemSize();
    BYTE* bytes = new BYTE[size];
    // you will have to delete[] that later
    std::memcpy(bytes, image.data, size * sizeof(BYTE));
    return bytes;
}
Mat floatToMat(float** src, int rows, int cols) {
    int size = rows * cols;
    /*float* oneD = new float[size];
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            oneD[i * rows + j] = src[i][j];
        }
    }
    Mat tmp(rows, cols, CV_32FC1, oneD);*/
    Mat tmp;
    tmp.create(cv::Size(cols, rows), CV_32F);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            tmp.ptr<float>(i)[j] = src[i][j];
        }
    }
    return tmp;
}

int main() {
    Mat src = imread("C:\\Users\\tota1Noob\\Desktop\\lena.png", 0);
    if (src.data == NULL) {
        cout << "Cannot openc the file!" << endl;
        return 1;
    }
    Mat image(src.rows, src.cols, CV_32F);
    src.convertTo(image, CV_32F, 1.0 / 255.0, 0);
    //imshow("orginal", image);

    int orgRow = image.rows,
        orgCol = image.cols;
    BYTE* imgTmp = matToBytes(src);
    Img tmp(imgTmp, orgRow, orgCol);
    //tmp.printImg();
    Img tmp2(orgRow, orgCol);
    //gaussian_2D_convolution(tmp, tmp2, 0, 0, 1.0);
    image_derivatives_scharr(tmp, tmp2, 1, 0);
    Mat scharr = floatToMat(tmp2.data(), orgRow, orgCol);
    imshow("mine", scharr);

    Mat opencvScharr(cv::Size(orgCol, orgRow), CV_32F);
    //cv::GaussianBlur(image, opencvScharr, cv::Size(3, 3), 1.0, 1.0, cv::BORDER_REPLICATE);
    cv::Scharr(image, opencvScharr, CV_32F, 1, 0, 1, 0, cv::BORDER_DEFAULT);
    imshow("opencv", opencvScharr);
    //tmp2.printImg();
    //cout << opencvScharr << endl;

    int x = 32, y = 267;
    cout << opencvScharr.ptr<float>(x)[y] << " " << scharr.ptr<float>(x)[y] << " "
        << opencvScharr.ptr<float>(x)[y] / scharr.ptr<float>(x)[y] << endl;


    /*int orgRow = image.rows,
        orgCol = image.cols,
        newRow = orgRow / 2,
        newCol = orgCol / 2;

    Mat half;
    half.create(cv::Size(newCol, newRow), CV_32F);
    cv::resize(image, half, cv::Size(newCol, newRow), 0, 0, INTER_AREA);
    imshow("opencv", half);

    BYTE* imgTmp = matToBytes(src);
    Img org(imgTmp, orgRow, orgCol);
    Img myHalf(newRow, newCol);
    halfsample_image(org, myHalf);
    Mat myHalfMat = floatToMat(myHalf.data(), newRow, newCol);
    imshow("mine", myHalfMat);

    int x = 32, y = 45;
    cout << half.ptr<float>(x)[y] << " " << myHalfMat.ptr<float>(x)[y] << " "
        << half.ptr<float>(x)[y] / myHalfMat.ptr<float>(x)[y] << endl;*/

    waitKey(0);

    return 0;
}