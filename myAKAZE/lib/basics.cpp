#define _USE_MATH_DEFINES
#include<iostream>
#include< ctime >
#include<vector>
#include<cmath>
#include"basics.h"

/******************************************************************************************/
/*Basic classes & functions*/

Keypoint::Keypoint() {
	this->pt.x = -1.;
	this->pt.y = -1.;
	this->size = 1.;
	this->angle = -1.;
	this->response = 0.;
	this->octave = 0;
	this->class_id = -1;
}

Keypoint::Keypoint(float x, float y, float size, float angle, float response, int octave, int class_id) {
	this->pt.x = x;
	this->pt.y = y;
	this->size = size;
	this->angle = angle;
	this->response = response;
	this->octave = octave;
	this->class_id = class_id;
}

Keypoint::Keypoint(Pointf pt, float size, float angle, float response, int octave, int class_id) {
	this->pt.x = pt.x;
	this->pt.y = pt.y;
	this->size = size;
	this->angle = angle;
	this->response = response;
	this->octave = octave;
	this->class_id = class_id;
}

Img::Img() {
	this->rows = 0;
	this->cols = 0;
	this->pixels = NULL;
}

Img::Img(BYTE* src, int rows, int cols) {
	this->cols = cols;
	this->rows = rows;
	this->pixels = new float* [rows];
	for (int i = 0; i < rows; ++i) {
		this->pixels[i] = new float[cols];
	}
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			this->pixels[i][j] = (float)src[i * cols + j] / 255.0;
		}
	}
}

Img::Img(int rows, int cols) {
	this->cols = cols;
	this->rows = rows;
	this->pixels = new float* [rows];
	for (int i = 0; i < rows; ++i) {
		this->pixels[i] = new float[cols];
		for (int j = 0; j < this->cols; ++j) {
			pixels[i][j] = 0.0;
		}
	}
}

Img::Img(ImgSize size) {
	this->cols = size.width;
	this->rows = size.height;
	this->pixels = new float* [this->rows];
	for (int i = 0; i < this->rows; ++i) {
		this->pixels[i] = new float[this->cols];
		for (int j = 0; j < this->cols; ++j) {
			pixels[i][j] = 0.0;
		}
	}
}

Img::Img(const Img& a) {
	this->cols = a.cols;
	this->rows = a.rows;
	this->pixels = new float* [this->rows];
	for (int i = 0; i < this->rows; ++i) {
		this->pixels[i] = new float[this->cols];
		for (int j = 0; j < this->cols; ++j) {
			float tmp = a.pixels[i][j];
			this->pixels[i][j] = tmp;
		}
	}
}

Img& Img::operator=(const Img& a) {
	this->cols = a.cols;
	this->rows = a.rows;
	this->pixels = new float* [this->rows];
	for (int i = 0; i < this->rows; ++i) {
		this->pixels[i] = new float[this->cols];
		for (int j = 0; j < this->cols; ++j) {
			float tmp = a.pixels[i][j];
			this->pixels[i][j] = tmp;
		}
	}
	return (*this);
}

Img::~Img() {
	for (int i = 0; i < this->rows; i++) {
		delete[] this->pixels[i];
	}
	delete[] pixels;
}

void Img::create(int rows, int cols) {
	this->cols = cols;
	this->rows = rows;
	this->pixels = new float* [rows];
	for (int i = 0; i < rows; ++i) {
		this->pixels[i] = new float[cols];
		for (int j = 0; j < this->cols; ++j) {
			pixels[i][j] = 0.0;
		}
	}
}

void Img::create(ImgSize size) {
	this->cols = size.width;
	this->rows = size.height;
	this->pixels = new float* [this->rows];
	for (int i = 0; i < this->rows; ++i) {
		this->pixels[i] = new float[this->cols];
		for (int j = 0; j < this->cols; ++j) {
			this->pixels[i][j] = 0.0;
		}
	}
}

ImgSize Img::size() {
	ImgSize tmp = { this->rows, this->cols };
	return tmp;
}

float** Img::data() {
	return this->pixels;
}

void Img::copyTo(Img& img) {
	for (int i = 0; i < img.rows; ++i) {
		img.pixels[i] = new float[img.cols];
		for (int j = 0; j < img.cols; ++j) {
			img.pixels[i][j] = this->pixels[i][j];
		}
	}
}

void Img::printImg() {
	printf("[");
	for (int i = 0; i < this->rows; ++i) {
		if (i != 0) printf(" ");
		for (int j = 0; j < this->cols; ++j) {
			printf("%f", this->pixels[i][j]);
			if (j != this->cols - 1) {
				printf(",\t");
			}
		}
		if (i == this->rows - 1) printf("]");
		else printf(";\n");
	}
}

void halfsample_image(Img& src, Img& dst) {
	int srcRow = src.rows,
		srcCol = src.cols,
		dstRow = dst.rows,
		dstCol = dst.cols;
	float tmpSum = 0.0;
	int windowSize;
	//If the row count of src is not an integral multiple of that of dst
	if (srcRow % dstRow != 0) {
		windowSize = (int)(srcRow / dstRow) + 1;
		float realWindowSizeX = (float)srcCol / dstCol,
			realWindowSizeY = (float)srcRow / dstRow,
			weightX = realWindowSizeX - 2.0,
			weightY = realWindowSizeY - 2.0,
			tmpWeight = weightX,
			tmpCounter = 1.0;
		//Because the real window size in this case is not an integer,
		//create a new extended image array for easier handling
		int newRow = srcRow + (int)(srcRow - 2) / 2,
			newCol = srcCol + (int)(srcCol - 2) / 2;
		float** extendedImg;
		extendedImg = new float* [newRow];
		for (int i = 0; i < newRow; ++i) {
			extendedImg[i] = new float[newCol];
			for (int j = 0; j < newCol; ++j) {
				extendedImg[i][j] = 1;
			}
		}
		//Calculate weights along the X axis
		for (int i = 0; i < newRow; ++i) {
			tmpWeight = weightX;
			tmpCounter = 1;
			for (int j = 0; j < newCol; j += windowSize) {
				extendedImg[i][j] *= tmpCounter;
				extendedImg[i][j + 2] *= tmpWeight;
				tmpCounter -= weightX;
				tmpWeight += weightX;
			}
		}
		//Calculate weights along the Y axis
		for (int j = 0; j < newCol; ++j) {
			tmpWeight = weightY;
			tmpCounter = 1;
			for (int i = 0; i < newRow; i += windowSize) {
				extendedImg[i][j] *= tmpCounter;
				extendedImg[i + 2][j] *= tmpWeight;
				tmpCounter -= weightY;
				tmpWeight += weightY;
			}
		}
		//Calculate the final extended image array
		for (int i = 0, m = 0; i < newRow && m < srcRow - 1; i += windowSize, m = m + windowSize - 1) {
			for (int j = 0, n = 0; j < newCol && n < srcCol - 1; j += windowSize, n = n + windowSize - 1) {
				extendedImg[i][j] *= src.pixels[m][n];
				extendedImg[i][j + 1] *= src.pixels[m][n + 1];
				extendedImg[i][j + 2] *= src.pixels[m][n + 2];
				extendedImg[i + 1][j] *= src.pixels[m + 1][n];
				extendedImg[i + 1][j + 1] *= src.pixels[m + 1][n + 1];
				extendedImg[i + 1][j + 2] *= src.pixels[m + 1][n + 2];
				extendedImg[i + 2][j] *= src.pixels[m + 2][n];
				extendedImg[i + 2][j + 1] *= src.pixels[m + 2][n + 1];
				extendedImg[i + 2][j + 2] *= src.pixels[m + 2][n + 2];
			}
		}
		//Slide with a 3 * 3 window. Calculate average
		for (int i = 0, m = 0; i < newRow && m < dstRow; i += windowSize, ++m) {
			for (int j = 0, n = 0; j < newCol && n < dstCol; j += windowSize, ++n) {
				tmpSum = extendedImg[i][j] + extendedImg[i][j + 1] + extendedImg[i][j + 2]
					+ extendedImg[i + 1][j] + extendedImg[i + 1][j + 1] + extendedImg[i + 1][j + 2]
					+ extendedImg[i + 2][j] + extendedImg[i + 2][j + 1] + extendedImg[i + 2][j + 2];
				dst.pixels[m][n] = tmpSum / (realWindowSizeX * realWindowSizeY);
			}
		}

		for (int i = 0; i < newRow; i++) {
			delete[] extendedImg[i];
		}
		delete[] extendedImg;
	}
	//If the row count of src is an integral multiple of that of dst
	else {
		windowSize = srcRow / dstRow;
		//Special case where 3 / 2 = 1 and 3 is an integral multiple of 1
		if (windowSize == 3) {
			tmpSum = src.pixels[0][0] + src.pixels[0][1] + src.pixels[0][2]
				+ src.pixels[1][0] + src.pixels[1][1] + src.pixels[1][2]
				+ src.pixels[2][0] + src.pixels[2][1] + src.pixels[2][2];
			dst.pixels[0][0] = tmpSum / 9.0;
		}
		//Regular cases. Slide with a 2 * 2 window. Calculate average
		else {
			for (int i = 0, m = 0; i < srcRow && m < dstRow; i += windowSize, ++m) {
				for (int j = 0, n = 0; j < srcCol && n < dstCol; j += windowSize, ++n) {
					tmpSum = src.pixels[i][j] + src.pixels[i][j + 1]
						+ src.pixels[i + 1][j] + src.pixels[i + 1][j + 1];
					dst.pixels[m][n] = tmpSum / (windowSize * windowSize);
				}
			}
		}
	}
}

float* gaussian_2D_kernel(int ksize, float sigma) {
	static const float PI = 3.141592654f;
	static const float SQRT2PI = sqrt(2.0 * PI);
	
	float sigma2 = 2.0 * sigma * sigma;
	float sigmap = 1.0 / sigma * SQRT2PI;
	float* kernel = new float[ksize];
	float sum = 0.0;
	int half = ksize / 2;

	for (int n = 0, i = -half; i <= half; ++i, ++n) {
		kernel[n] = exp(-(float)(i * i) / sigma2) * sigmap;
		sum += kernel[n];
	}
	if (sum == 1.0) return kernel;
	sum = 1.0 / sum;
	for (int n = 0; n < ksize; ++n)
		kernel[n] = kernel[n] * sum;
	return kernel;
}

void gaussian_2D_convolution(Img& src, Img& dst, int ksize_x, int ksize_y, float sigma) {

	int row = src.rows, col = src.cols;
	//Compute an appropriate kernel size according to the specified sigma
	if (sigma > ksize_x || sigma > ksize_y || ksize_x == 0 || ksize_y == 0) {
		ksize_x = ceil(2.0 * (1.0 + (sigma - 0.8) / (0.3)));
		ksize_y = ksize_x;
	}
	//The kernel size must be and odd number
	if ((ksize_x % 2) == 0)
		ksize_x += 1;
	if ((ksize_y % 2) == 0)
		ksize_y += 1;
	int half = ksize_x / 2;
	//Calculate the Gaussian kernel
	float sumTmp = 0.;

	float* kernel = gaussian_2D_kernel(ksize_x, sigma);
	Img tmp(row, col);
	//Calculate with kx
	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			sumTmp = 0.;
			for (int k = -half; k <= half; ++k) {
				if (j + k < 0) sumTmp += src.pixels[i][0] * kernel[k + half];
				else if(j + k >= col) sumTmp += src.pixels[i][col - 1] * kernel[k + half];
				else sumTmp += src.pixels[i][j + k] * kernel[k + half];
			}
			tmp.pixels[i][j] = sumTmp;
		}
	}
	//Calculate with ky
	for (int j = 0; j < col; ++j) {
		for (int i = 0; i < row; ++i) {
			sumTmp = 0.;
			for (int k = -half; k <= half; ++k) {
				if (i + k < 0) sumTmp += tmp.pixels[0][j] * kernel[k + half];
				else if (i + k >= row) sumTmp += tmp.pixels[row - 1][j] * kernel[k + half];
				else sumTmp += tmp.pixels[i + k][j] * kernel[k + half];
			}
			dst.pixels[i][j] = sumTmp;
		}
	}
}

void sepFilter2D(Img& src, Img& dst, float* kx, float* ky, int ksize) {
	
	int row = src.rows,
		col = src.cols,
		half = ksize / 2;
	float sumTmp = 0.;
	Img tmp(row, col);
	//Calculate with kx
	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			sumTmp = 0.;
			if (j - half < 0) {
				sumTmp += src.pixels[i][half - j] * kx[0];
			}
			else {
				sumTmp += src.pixels[i][j - half] * kx[0];
			}
			sumTmp += src.pixels[i][j] * kx[half];
			if (j + half >= col) {
				sumTmp += src.pixels[i][2 * col - j - half - 2] * kx[ksize - 1];
			}
			else {
				sumTmp += src.pixels[i][j + half] * kx[ksize - 1];
			}
			tmp.pixels[i][j] = sumTmp;
		}
	}
	//Calculate with ky
	for (int j = 0; j < col; ++j) {
		for (int i = 0; i < row; ++i) {
			sumTmp = 0.;
			if (i - half < 0) {
				sumTmp += tmp.pixels[half - i][j] * ky[0];
			}
			else {
				sumTmp += tmp.pixels[i - half][j] * ky[0];
			}
			sumTmp += tmp.pixels[i][j] * ky[half];
			if (i + half >= row) {
				sumTmp += tmp.pixels[2 * row - i - half - 2][j] * ky[ksize - 1];
			}
			else {
				sumTmp += tmp.pixels[i + half][j] * ky[ksize - 1];
			}
			dst.pixels[i][j] = sumTmp;
		}
	}
}

void image_derivatives_scharr(Img& src, Img& dst, int xorder, int yorder) {

	float kx[3], ky[3];

	if (xorder == 1) {
		kx[0] = -1.; kx[1] = 0.; kx[2] = 1.;
		ky[0] = 3.; ky[1] = 10.; ky[2] = 3.;
	}
	else {
		kx[0] = 3.; kx[1] = 10.; kx[2] = 3.;
		ky[0] = -1.; ky[1] = 0.; ky[2] = 1.;
	}
	sepFilter2D(src, dst, kx, ky, 3);
}

float compute_k_percentile(Img& img, float perc, float gscale, int nbins, int ksize_x, int ksize_y) {

	int nbin = 0, pointNum = 0, threshold = 0;
	float finalK = 0.0, grad = 0.0, pointCount = 0.0, max = 0.0;
	float* hist = new float[nbins];
	for (int i = 0; i < nbins; ++i)
		hist[i] = 0.0;

	Img gaussian(img.rows, img.cols);
	Img dx(img.rows, img.cols);
	Img dy(img.rows, img.cols);
	gaussian_2D_convolution(img, gaussian, ksize_x, ksize_y, gscale);
	image_derivatives_scharr(gaussian, dx, 1, 0);
	image_derivatives_scharr(gaussian, dy, 0, 1);

	// Skip the borders
	for (int i = 1; i < gaussian.rows - 1; ++i) {
		const float* dx_row = dx.pixels[i];
		const float* dy_row = dy.pixels[i];
		for (int j = 1; j < gaussian.cols - 1; ++j) {
			grad = sqrt(dx_row[j] * dx_row[j] + dy_row[j] * dy_row[j]);
			if (grad > max) {
				max = grad;
			}
		}
	}
	for (int i = 1; i < gaussian.rows - 1; ++i) {
		const float* dx_row = dx.pixels[i];
		const float* dy_row = dy.pixels[i];
		for (int j = 1; j < gaussian.cols - 1; ++j) {
			grad = sqrt(dx_row[j] * dx_row[j] + dy_row[j] * dy_row[j]);
			if (grad != 0.0) {
				nbin = floor(nbins * (grad / max));
				if (nbin == nbins) {
					nbin--;
				}
				hist[nbin]++;
				pointCount++;
			}
		}
	}

	threshold = (int)(pointCount * perc);
	int i;
	for (i = 0; pointNum < threshold && i < nbins; ++i) {
		pointNum = pointNum + hist[i];
	}
	if (pointNum < threshold) {
		finalK = 0.03;
	}
	else {
		finalK = max * ((float)i / (float)nbins);
	}

	delete[] hist;
	return finalK;
}

bool solve(Img& a, Img& b, Img& dst) {
	float d = a.pixels[0][0] * a.pixels[1][1] - a.pixels[0][1] * a.pixels[1][0];
	if (d != 0.) {
		dst.pixels[0][0] = (b.pixels[0][0] * a.pixels[1][1] - b.pixels[1][0] * a.pixels[0][1]) / d;
		dst.pixels[1][0] = (b.pixels[1][0] * a.pixels[0][0] - b.pixels[0][0] * a.pixels[1][0]) / d;
		return true;
	}
	else {
		return false;
	}
}

void compute_scharr_derivatives(Img& src, Img& dst, int xorder, int yorder, int scale) {
	const int ksize = 3 + 2 * (scale - 1);
	float* kx = new float[ksize];
	float* ky = new float[ksize];

	if (scale == 1) {
		if (xorder == 1) {
			kx[0] = -1.; kx[1] = 0.; kx[2] = 1.;
			ky[0] = 0.09375; ky[1] = 0.3125; ky[2] = 0.09375;
		}
		else {
			kx[0] = 0.09375; kx[1] = 0.3125; kx[2] = 0.09375;
			ky[0] = -1.; ky[1] = 0.; ky[2] = 1.;
		}
	}
	else {
		float w = 10.0 / 3.0;
		float norm = 1.0 / (2.0 * scale * (w + 2.0));
		if (xorder == 1) {
			kx[0] = -1.; kx[ksize / 2] = 0.; kx[ksize - 1] = 1.;
			ky[0] = norm; ky[ksize / 2] = w * norm; ky[ksize - 1] = norm;
		}
		else {
			kx[0] = norm; kx[ksize / 2] = w * norm; kx[ksize - 1] = norm;
			ky[0] = -1.; ky[ksize / 2]= 0.; ky[ksize - 1]= 1.;
		}
	}
	sepFilter2D(src, dst, kx, ky, ksize);
}

/******************************************************************************************/
/*Diffusivity functions*/

void pm_g1(Img& Lx, Img& Ly, Img& dst, const float k) {
	float inv_k = 1.0 / (k * k);
	for (int i = 0; i < Lx.rows; ++i) {
		const float* Lx_row = Lx.pixels[i];
		const float* Ly_row = Ly.pixels[i];
		float* dst_row = dst.pixels[i];
		for (int j = 0; j < Lx.cols; ++j) {
			dst_row[j] = exp(-inv_k * (Lx_row[j] * Lx_row[j] + Ly_row[j] * Ly_row[j]));
		}
	}
}

void pm_g2(Img& Lx, Img& Ly, Img& dst, const float k) {
	float inv_k = 1.0 / (k * k);
	for (int i = 0; i < Lx.rows; ++i) {
		const float* Lx_row = Lx.pixels[i];
		const float* Ly_row = Ly.pixels[i];
		float* dst_row = dst.pixels[i];
		for (int j = 0; j < Lx.cols; ++j) {
			dst_row[j] = 1.0 / (1.0 + inv_k * (Lx_row[j] * Lx_row[j] + Ly_row[j] * Ly_row[j]));
		}
	}
}

void weickert_diffusivity(Img& Lx, Img& Ly, Img& dst, const float k) {
	float inv_k = 1.0 / (k * k);
	for (int i = 0; i < Lx.rows; ++i) {
		const float* Lx_row = Lx.pixels[i];
		const float* Ly_row = Ly.pixels[i];
		float* dst_row = dst.pixels[i];
		for (int j = 0; j < Lx.cols; ++j) {
			float dl = inv_k * (Lx_row[j] * Lx_row[j] + Ly_row[j] * Ly_row[j]);
			dst_row[j] = 1.0 - exp(-3.315 / (dl * dl * dl * dl));
		}
	}
}

void charbonnier_diffusivity(Img& Lx, Img& Ly, Img& dst, const float k) {
	float inv_k = 1.0 / (k * k);
	for (int i = 0; i < Lx.rows; ++i) {
		const float* Lx_row = Lx.pixels[i];
		const float* Ly_row = Ly.pixels[i];
		float* dst_row = dst.pixels[i];
		for (int j = 0; j < Lx.cols; ++j) {
			float den = sqrt(1.0 + inv_k * (Lx_row[j] * Lx_row[j] + Ly_row[j] * Ly_row[j]));
			dst_row[j] = 1.0 - den;
		}
	}
}

/******************************************************************************************/
/*FED implementation*/

using namespace std;

int fed_tau_by_process_time(const float T, const int M, const float tau_max, const bool reordering, std::vector<float>& tau) {
	return fed_tau_by_cycle_time(T / (float)M, tau_max, reordering, tau);
}

int fed_tau_by_cycle_time(const float t, const float tau_max, const bool reordering, std::vector<float>& tau) {
	int n = 0;          // Number of time steps
	float scale = 0.0;  // Ratio of t we search to maximal t

	// Compute necessary number of time steps
	n = (int)(ceil(sqrt(3.0 * t / tau_max + 0.25f) - 0.5f - 1.0e-8f) + 0.5f);
	scale = 3.0 * t / (tau_max * (float)(n * (n + 1)));
	return fed_tau_internal(n, scale, tau_max, reordering, tau);
}

int fed_tau_internal(const int n, const float scale, const float tau_max, const bool reordering, std::vector<float>& tau) {
	float c = 0.0, d = 0.0;     // Time savers
	vector<float> tauh;    // Helper vector for unsorted taus
	if (n <= 0)
		return 0;
	// Allocate memory for the time step size
	tau = vector<float>(n);
	if (reordering)
		tauh = vector<float>(n);
	// Compute time saver
	c = 1.0f / (4.0f * (float)n + 2.0f);
	d = scale * tau_max / 2.0f;

	// Set up originally ordered tau vector
	for (int k = 0; k < n; ++k) {
		float h = cos(M_PI * (2.0f * (float)k + 1.0f) * c);
		if (reordering)
			tauh[k] = d / (h * h);
		else
			tau[k] = d / (h * h);
	}
	// Permute list of time steps according to chosen reordering function
	int kappa = 0, prime = 0;

	if (reordering == true) {
		// Choose kappa cycle with k = n/2
		// This is a heuristic. We can use Leja ordering instead!!
		kappa = n / 2;
		// Get modulus for permutation
		prime = n + 1;
		while (!fed_is_prime_internal(prime)) {
			prime++;
		}
		// Perform permutation
		for (int k = 0, l = 0; l < n; ++k, ++l) {
			int index = 0;
			while ((index = ((k + 1) * kappa) % prime - 1) >= n) {
				k++;
			}
			tau[l] = tauh[index];
		}
	}
	return n;
}

bool fed_is_prime_internal(const int number) {
	if (number <= 1) {
		return false;
	}
	else if (number == 2 || number == 3 || number == 5 || number == 7) {
		return true;
	}
	else if ((number % 2) == 0 || (number % 3) == 0 || (number % 5) == 0 || (number % 7) == 0) {
		return false;
	}
	else {
		int upperLimit = sqrt(number + 1.0);
		int divisor = 11;
		while (divisor <= upperLimit) {
			if (number % divisor == 0){
				return false;
			}
			divisor += 2;
		}
		return true;
	}
}


void nld_step_scalar(Img& Ld, Img& c, Img& Lstep, const float stepsize) {
	// Diffusion all the image except borders
	for (int y = 1; y < Lstep.rows - 1; y++) {
		const float* c_row = c.pixels[y];
		const float* c_row_p = c.pixels[y + 1];
		const float* c_row_m = c.pixels[y - 1];

		float* Ld_row = Ld.pixels[y];
		float* Ld_row_p = Ld.pixels[y + 1];
		float* Ld_row_m = Ld.pixels[y - 1];
		float* Lstep_row = Lstep.pixels[y];

		for (int x = 1; x < Lstep.cols - 1; x++) {
			float xpos = (c_row[x] + c_row[x + 1]) * (Ld_row[x + 1] - Ld_row[x]);
			float xneg = (c_row[x - 1] + c_row[x]) * (Ld_row[x] - Ld_row[x - 1]);
			float ypos = (c_row[x] + c_row_p[x]) * (Ld_row_p[x] - Ld_row[x]);
			float yneg = (c_row_m[x] + c_row[x]) * (Ld_row[x] - Ld_row_m[x]);
			Lstep_row[x] = 0.5 * stepsize * (xpos - xneg + ypos - yneg);
		}
	}

	// First row
	const float* c_row = c.pixels[0];
	const float* c_row_p = c.pixels[1];
	float* Ld_row = Ld.pixels[0];
	float* Ld_row_p = Ld.pixels[1];
	float* Lstep_row = Lstep.pixels[0];

	for (int x = 1; x < Lstep.cols - 1; x++) {
		float xpos = (c_row[x] + c_row[x + 1]) * (Ld_row[x + 1] - Ld_row[x]);
		float xneg = (c_row[x - 1] + c_row[x]) * (Ld_row[x] - Ld_row[x - 1]);
		float ypos = (c_row[x] + c_row_p[x]) * (Ld_row_p[x] - Ld_row[x]);
		Lstep_row[x] = 0.5 * stepsize * (xpos - xneg + ypos);
	}

	float xpos = (c_row[0] + c_row[1]) * (Ld_row[1] - Ld_row[0]);
	float ypos = (c_row[0] + c_row_p[0]) * (Ld_row_p[0] - Ld_row[0]);
	Lstep_row[0] = 0.5 * stepsize * (xpos + ypos);

	int x = Lstep.cols - 1;
	float xneg = (c_row[x - 1] + c_row[x]) * (Ld_row[x] - Ld_row[x - 1]);
	ypos = (c_row[x] + c_row_p[x]) * (Ld_row_p[x] - Ld_row[x]);
	Lstep_row[x] = 0.5 * stepsize * (-xneg + ypos);

	// Last row
	c_row = c.pixels[Lstep.rows - 1];
	c_row_p = c.pixels[Lstep.rows - 2];
	Ld_row = Ld.pixels[Lstep.rows - 1];
	Ld_row_p = Ld.pixels[Lstep.rows - 2];
	Lstep_row = Lstep.pixels[Lstep.rows - 1];

	for (int x = 1; x < Lstep.cols - 1; x++) {
		float xpos = (c_row[x] + c_row[x + 1]) * (Ld_row[x + 1] - Ld_row[x]);
		float xneg = (c_row[x - 1] + c_row[x]) * (Ld_row[x] - Ld_row[x - 1]);
		float ypos = (c_row[x] + c_row_p[x]) * (Ld_row_p[x] - Ld_row[x]);
		Lstep_row[x] = 0.5 * stepsize * (xpos - xneg + ypos);
	}

	xpos = (c_row[0] + c_row[1]) * (Ld_row[1] - Ld_row[0]);
	ypos = (c_row[0] + c_row_p[0]) * (Ld_row_p[0] - Ld_row[0]);
	Lstep_row[0] = 0.5 * stepsize * (xpos + ypos);

	x = Lstep.cols - 1;
	xneg = (c_row[x - 1] + c_row[x]) * (Ld_row[x] - Ld_row[x - 1]);
	ypos = (c_row[x] + c_row_p[x]) * (Ld_row_p[x] - Ld_row[x]);
	Lstep_row[x] = 0.5 * stepsize * (-xneg + ypos);

	// First and last columns
	for (int i = 1; i < Lstep.rows - 1; i++) {

		const float* c_row = c.pixels[i];
		const float* c_row_m = c.pixels[i - 1];
		const float* c_row_p = c.pixels[i + 1];
		float* Ld_row = Ld.pixels[i];
		float* Ld_row_p = Ld.pixels[i + 1];
		float* Ld_row_m = Ld.pixels[i - 1];
		Lstep_row = Lstep.pixels[i];

		float xpos = (c_row[0] + c_row[1]) * (Ld_row[1] - Ld_row[0]);
		float ypos = (c_row[0] + c_row_p[0]) * (Ld_row_p[0] - Ld_row[0]);
		float yneg = (c_row_m[0] + c_row[0]) * (Ld_row[0] - Ld_row_m[0]);
		Lstep_row[0] = 0.5 * stepsize * (xpos + ypos - yneg);

		float xneg = (c_row[Lstep.cols - 2] + c_row[Lstep.cols - 1]) * (Ld_row[Lstep.cols - 1] - Ld_row[Lstep.cols - 2]);
		ypos = (c_row[Lstep.cols - 1] + c_row_p[Lstep.cols - 1]) * (Ld_row_p[Lstep.cols - 1] - Ld_row[Lstep.cols - 1]);
		yneg = (c_row_m[Lstep.cols - 1] + c_row[Lstep.cols - 1]) * (Ld_row[Lstep.cols - 1] - Ld_row_m[Lstep.cols - 1]);
		Lstep_row[Lstep.cols - 1] = 0.5 * stepsize * (-xneg + ypos - yneg);
	}

	// Ld = Ld + Lstep
	for (int y = 0; y < Lstep.rows; y++) {
		float* Ld_row = Ld.pixels[y];
		float* Lstep_row = Lstep.pixels[y];
		for (int x = 0; x < Lstep.cols; x++) {
			Ld_row[x] = Ld_row[x] + Lstep_row[x];
		}
	}
}