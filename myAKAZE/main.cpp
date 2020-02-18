#include<iostream>
#include<opencv2/opencv.hpp>
#include"./lib/akaze.h"

using namespace cv;
using namespace std;

void akaze_compute_derivative_kernels(cv::OutputArray kx_, cv::OutputArray ky_,  const size_t dx, const size_t dy, const size_t scale);

void akaze_compute_scharr_derivatives(const cv::Mat& src, cv::Mat& dst, const size_t xorder, const size_t yorder, const size_t scale);

bool matIsEqual(const cv::Mat mat1, const cv::Mat mat2);

float matSimilarity(const cv::Mat mat1, Img& mat2);

float matSimilarity(const cv::Mat mat1, const cv::Mat mat2);

BYTE* matToBytes(Mat image);

Mat floatToMat(float** src, int rows, int cols);

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

    /*Mat dst;
    akaze_compute_scharr_derivatives(image, dst, 1, 0, 4);
    imshow("akaze", dst);

    Img tmp2(orgRow, orgCol);
    compute_scharr_derivatives(tmp, tmp2, 1, 0, 4);
    Mat mine = floatToMat(tmp2.data(), orgRow, orgCol);
    imshow("mine", mine);
    cout << matSimilarity(dst, tmp2) << endl;*/

    
    /*Mat kx_, ky_;
    akaze_compute_derivative_kernels(kx_, ky_, 0, 1, 4);
    cout << kx_ << "\n\n" << ky_ << endl;

    Img kx, ky;
    compute_derivative_kernels(kx, ky, 0, 1, 4);
    kx.printImg(); cout << endl; ky.printImg();*/


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
    cv::GaussianBlur(image, opencvScharr, cv::Size(5, 5), 1.0, 1.0, cv::BORDER_REPLICATE);
    //cv::Scharr(image, opencvScharr, CV_32F, 0, 1, 1, 0, cv::BORDER_DEFAULT);
    imshow("opencv", opencvScharr);
    //tmp2.printImg();
    //cout << opencvScharr << endl;

    Img tmp2(orgRow, orgCol);
    gaussian_2D_convolution(tmp, tmp2, 0, 0, 1.0);
    //image_derivatives_scharr(tmp, tmp2, 0, 1);
    Mat scharr = floatToMat(tmp2.data(), orgRow, orgCol);
    imshow("mine", scharr);

    cout << matSimilarity(opencvScharr, scharr) << endl;


    /*int newRow = orgRow / 2,
        newCol = orgCol / 2;

    Mat half;
    half.create(cv::Size(newCol, newRow), CV_32F);
    cv::resize(image, half, cv::Size(newCol, newRow), 0, 0, INTER_AREA);
    imshow("opencv", half);

    imgTmp = matToBytes(src);
    Img org(imgTmp, orgRow, orgCol);
    Img myHalf(newRow, newCol);
    halfsample_image(org, myHalf);
    Mat myHalfMat = floatToMat(myHalf.data(), newRow, newCol);
    imshow("mine", myHalfMat);

    int x = 32, y = 45;
    cout << half.ptr<float>(x)[y] << " " << myHalfMat.ptr<float>(x)[y] << " "
        << half.ptr<float>(x)[y] / myHalfMat.ptr<float>(x)[y] << endl;
    cout << matSimilarity(half, myHalf) << endl;*/

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
            if (mat1.ptr<float>(i)[j] == mat2.ptr(i)[j]) continue;
            tmp += fabs(mat1.ptr<float>(i)[j] - mat2.ptr(i)[j]);
            if (fabs(mat1.ptr<float>(i)[j] - mat2.ptr(i)[j]) > max) {
                max = fabs(mat1.ptr<float>(i)[j] - mat2.ptr(i)[j]);
                maxi = i; maxj = j;
                m1 = mat1.ptr<float>(i)[j]; m2 = mat2.ptr(i)[j];
            }
        }
        tmp /= (float)mat1.cols;
        sim += tmp;
    }
    cout << "Max difference is "  << max << " = " << m1 << " - " << m2 << ", at (" << maxi << ", " << maxj << ")" << endl;
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
