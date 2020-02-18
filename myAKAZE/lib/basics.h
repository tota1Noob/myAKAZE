#pragma once
#define BYTE unsigned char

/******************************************************************************************/
/*Basic classes & functions*/

//Defined struct "ImgSize" for image size storage
struct ImgSize {
	int height;
	int width;
};

//Defined struct "Point" for point coordinates storage
//Notice that x and y are float numbers
struct Pointf {
	float x;
	float y;
};

//Defined class "KeyPoint" for key point storage
//This is just a bad knock-off version of "KeyPoint" in OpenCV
class KeyPoint {
public:
	Pointf pt;
	float size;
	float angle;
	float response;
	int octave;
	int class_id;
	KeyPoint();
	~KeyPoint();
	KeyPoint(float x, float y, float size, float angle, float response, int octave, int class_id);
	KeyPoint(Pointf pt, float size, float angle, float response, int octave, int class_id);
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
	void  create(int rows, int cols);
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

//Defined function "sepFilter2D"
//Works the same way as cv::sepFilter2D, rewritten for the needs of this program
//Only deals with the top, mid and bottom numbers in kx and ky
//(see function "compute_derivative_kernels" for reference)
void sepFilter2D(Img& src, Img& dst, Img& kx, Img& ky);

//Defined function "image_derivatives_scharr" for calculating image derivatives
//along the specified axis
void image_derivatives_scharr(Img& src, Img& dst, int xorder, int yorder);

//Defined function "compute_k_percentile" for calculating the value of contrast factor "k"
float compute_k_percentile(Img& img, float perc, float gscale, int nbins, int ksize_x, int ksize_y);

//Defined function "solve" for solving a 2 * 2 linear system
//Following Cramer's Rule
bool solve(Img& a, Img& b, Img& dst);

//Defined function "compute_derivative_kernels"
//Works as the name indicates
void compute_derivative_kernels(Img& kx, Img& ky, int dx, int dy, int scale);

//Defined function "ompute_scharr_derivatives"
//Works as the name indicates
void compute_scharr_derivatives(Img& src, Img& dst, int xorder, int yorder, int scale);

//Defined four diffusivity functions, "pm_g1", "pm_g2", "weickert_diffusivity" and "charbonnier_diffusivity"
void pm_g1(Img& Lx, Img& Ly, Img& dst, const float k);
void pm_g2(Img& Lx, Img& Ly, Img& dst, const float k);
void weickert_diffusivity(Img& Lx, Img& Ly, Img& dst, const float k);
void charbonnier_diffusivity(Img& Lx, Img& Ly, Img& dst, const float k);

//Defined a series of functions
//"fed_tau_by_process_time" returns the number of time steps per cycle or 0 on failure
//utilizing the 3 functions that follow
int fed_tau_by_process_time(const float T, const int M, const float tau_max, const bool reordering, std::vector<float>& tau);
int fed_tau_by_cycle_time(const float t, const float tau_max, const bool reordering, std::vector<float>& tau);
int fed_tau_internal(const int n, const float scale, const float tau_max, const bool reordering, std::vector<float>& tau);
bool fed_is_prime_internal(const int number);