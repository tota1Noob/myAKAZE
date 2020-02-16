#pragma once
#define BYTE unsigned char

/******************************************************************************************/
/*Basic classes & functions*/

//Defined struct "ImgSize" for image size storage
struct ImgSize {
	int height;
	int width;
};

//Defined class "Img" for image data storage,
//where pixel values are kept in a 2D float array "pixels"
//Number of columns and rows are defined as "cols" and "rows", respectively
class Img {
private:
	float** pixels;
public:
	int cols;
	int rows;
	Img();
	Img(BYTE* src, int rows, int cols);
	Img(int rows, int cols);
	Img(ImgSize size);
	~Img();
	ImgSize size();
	float* ptr(int row);
	float get(int row, int col);
	float** data();
	void copyTo(Img& img);
	void printImg();
};

//Defined function "halfsample_image" for image downsampling by a factor of 2
//This is an implementation of the "resize" function in OpenCV
//with the interpolition param set to "INTER_AREA"
//See https://medium.com/@wenrudong/what-is-opencvs-inter-area-actually-doing-282a626a09b3 for reference
void halfsample_image(Img& src, Img& dst);

//Defined function "gaussian_2D_kernel" for the calculation of a gaussian kernel
//of specified size
float** gaussian_2D_kernel(int ksize_x, int ksize_y, float sigma);

//Defined function "gaussian_2D_convolution" for gaussian blurring over a 2D array
void gaussian_2D_convolution(Img& src, Img& dst, int ksize_x, int ksize_y, float sigma);

//Defined function "image_derivatives_scharr" for the calculation of image derivatives
//along the specified axis
void image_derivatives_scharr(Img& src, Img& dst, int xorder, int yorder);