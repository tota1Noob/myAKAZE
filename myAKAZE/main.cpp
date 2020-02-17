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
    //cout << compute_k_percentile(tmp, 0.7, 1.0, 300, 0, 0) << endl;

    /*Mat kx_, ky_, dst;
    cv::getDerivKernels(kx_, ky_, 1, 0, 0, true, CV_32F);
    ky_.ptr<float>(0)[0] = 3.; ky_.ptr<float>(1)[0] = 10; ky_.ptr<float>(2)[0] = 3;
    cout << kx_ << "\n" << ky_ << endl;
    cv::sepFilter2D(image, dst, CV_32F, kx_, ky_);
    imshow("dst", dst);

    Img tmp3(orgRow, orgCol);
    Img kx(3, 1), ky(3, 1);
    kx.ptr(0)[0] = -1; kx.ptr(1)[0] = 0; kx.ptr(2)[0] = 1;
    ky.ptr(0)[0] = 0.09375; ky.ptr(1)[0] = 0.3125; ky.ptr(2)[0] = 0.09375;
    kx.printImg(); cout << endl;  ky.printImg(); cout << endl;
    sepFilter2D(tmp, tmp3, kx, ky);
    cout << tmp3.get(200, 200) << " ";
    Mat sep = floatToMat(tmp3.data(), orgRow, orgCol);
    cout << sep.ptr<float>(200)[200] << " " << dst.ptr<float>(200)[200] << endl;
    imshow("mine", sep);*/



    /*Img a(2, 2);
    Img b(2, 1);
    Img dst(2, 1);
    a.ptr(0)[0] = 2; a.ptr(0)[1] = 1; a.ptr(1)[0] = 0; a.ptr(1)[1] = 1;
    b.ptr(0)[0] = 0; b.ptr(1)[0] = 5;
    solve(a, b, dst);
    dst.printImg();*/


    Mat opencvScharr(cv::Size(orgCol, orgRow), CV_32F);
    //cv::GaussianBlur(image, opencvScharr, cv::Size(3, 3), 1.0, 1.0, cv::BORDER_REPLICATE);
    cv::Scharr(image, opencvScharr, CV_32F, 1, 0, 1, 0, cv::BORDER_DEFAULT);
    imshow("opencv", opencvScharr);
    //tmp2.printImg();
    //cout << opencvScharr << endl;

    Img tmp2(orgRow, orgCol);
    //gaussian_2D_convolution(tmp, tmp2, 0, 0, 1.0);
    image_derivatives_scharr(tmp, tmp2, 1, 0);
    Mat scharr = floatToMat(tmp2.data(), orgRow, orgCol);
    imshow("mine", scharr);

    int x = 123, y = 234;
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