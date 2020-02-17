#include<iostream>
#include<math.h>
#include"akaze.h"

/******************************************************************************************/
/*Basic classes & functions*/

KeyPoint::KeyPoint() {
	this->pt.x = -1.;
	this->pt.y = -1.;
	this->size = -1.;
	this->angle = -1.;
	this->response = 0.;
	this->octave = 0;
	this->class_id = -1;
}

KeyPoint::~KeyPoint() {
	printf("Deleted KeyPoint (%f, %f).", this->pt.x, this->pt.y);
}

KeyPoint::KeyPoint(float x, float y, float size, float angle, float response, int octave, int class_id) {
	this->pt.x = x;
	this->pt.y = y;
	this->size = size;
	this->angle = angle;
	this->response = response;
	this->octave = octave;
	this->class_id = class_id;
}

KeyPoint::KeyPoint(Pointf pt, float size, float angle, float response, int octave, int class_id) {
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

Img::~Img() {
	for (int i = 0; i < this->rows; i++) {
		delete[] this->pixels[i];
	}
	delete[] pixels;
}

ImgSize Img::size() {
	ImgSize tmp = { this->rows, this->cols };
	return tmp;
}

float* Img::ptr(int row) {
	return this->pixels[row];
}

float Img::get(int row, int col) {
	return this->pixels[row][col];
}

float** Img::data() {
	return this->pixels;
}

void Img::copyTo(Img& img) {
	img.cols = this->cols;
	img.rows = this->rows;
	img.pixels = new float* [img.rows];
	for (int i = 0; i < img.rows; ++i) {
		img.pixels[i] = new float[img.cols];
		for (int j = 0; j < img.cols; ++j) {
			img.pixels[i][j] = this->pixels[i][j];
		}
	}
}

void Img::printImg() {
	for (int i = 0; i < this->rows; ++i) {
		for (int j = 0; j < this->cols; ++j) {
			printf("%f", this->pixels[i][j]);
			if (j == this->cols - 1) {
				printf("\n");
			}
			else {
				printf("\t");
			}
		}
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
				extendedImg[i][j] *= src.get(m, n);
				extendedImg[i][j + 1] *= src.get(m, n + 1);
				extendedImg[i][j + 2] *= src.get(m, n + 2);
				extendedImg[i + 1][j] *= src.get(m + 1, n);
				extendedImg[i + 1][j + 1] *= src.get(m + 1, n + 1);
				extendedImg[i + 1][j + 2] *= src.get(m + 1, n + 2);
				extendedImg[i + 2][j] *= src.get(m + 2, n);
				extendedImg[i + 2][j + 1] *= src.get(m + 2, n + 1);
				extendedImg[i + 2][j + 2] *= src.get(m + 2, n + 2);
			}
		}
		//Slide with a 3 * 3 window. Calculate average
		for (int i = 0, m = 0; i < newRow && m < dstRow; i += windowSize, ++m) {
			for (int j = 0, n = 0; j < newCol && n < dstCol; j += windowSize, ++n) {
				tmpSum = extendedImg[i][j] + extendedImg[i][j + 1] + extendedImg[i][j + 2]
					+ extendedImg[i + 1][j] + extendedImg[i + 1][j + 1] + extendedImg[i + 1][j + 2]
					+ extendedImg[i + 2][j] + extendedImg[i + 2][j + 1] + extendedImg[i + 2][j + 2];
				dst.ptr(m)[n] = tmpSum / (realWindowSizeX * realWindowSizeY);
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
			tmpSum = src.get(0, 0) + src.get(0, 1) + src.get(0, 2)
				+ src.get(1, 0) + src.get(1, 1) + src.get(1, 2)
				+ src.get(2, 0) + src.get(2, 1) + src.get(2, 2);
			dst.ptr(0)[0] = tmpSum / 9.0;
		}
		//Regular cases. Slide with a 2 * 2 window. Calculate average
		else {
			for (int i = 0, m = 0; i < srcRow && m < dstRow; i += windowSize, ++m) {
				for (int j = 0, n = 0; j < srcCol && n < dstCol; j += windowSize, ++n) {
					tmpSum = src.get(i, j) + src.get(i, j + 1)
						+ src.get(i + 1, j) + src.get(i + 1, j + 1);
					dst.ptr(m)[n] = tmpSum / (windowSize * windowSize);
				}
			}
		}
	}
}

float** gaussian_2D_kernel(int ksize_x, int ksize_y, float sigma) {
	int centerX = ksize_x / 2,
		centerY = ksize_y / 2,
		offsetX = 0, offsetY = 0;
	float** kernel = new float* [ksize_y];
	float pi = (float)3.141592654, sum = 0.0;
	for (int i = 0; i < ksize_y; ++i) {
		kernel[i] = new float[ksize_x];
		for (int j = 0; j < ksize_x; ++j) {
			offsetX = j - centerX;
			offsetY = i - centerY;
			kernel[i][j] = exp(-1.0 * (float)(offsetX * offsetX + offsetY * offsetY) / (2.0 * sigma * sigma));
			kernel[i][j] = kernel[i][j] / (2.0 * pi * sigma * sigma);
			sum += kernel[i][j];
		}
	}
	sum = 1.0 / sum;
	for (int i = 0; i < ksize_y; ++i) {
		for (int j = 0; j < ksize_x; ++j) {
			kernel[i][j] = kernel[i][j] * sum;
		}
	}
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
	//Calculate the Gaussian kernel
	float** kernel = gaussian_2D_kernel(ksize_x, ksize_y, sigma);
	//Generate larger array for border replication
	int newRow = ksize_y + row - 1,
		newCol = ksize_x + col - 1,
		paddingX = ksize_x / 2,
		paddingY = ksize_y / 2;
	float** extended = new float* [newRow];
	for (int i = 0; i < newRow; ++i) {
		extended[i] = new float[newCol];
	}
	//Copy src data
	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			extended[i + paddingY][j + paddingX] = src.get(i, j);
		}
	}
	//Replicate top and bottom borders
	for (int i = paddingY - 1; i >= 0; --i) {
		for (int j = paddingX; j < paddingX + col; ++j) {
			extended[i][j] = extended[i + 1][j];
		}
	}
	for (int i = row + paddingY; i < newRow; ++i) {
		for (int j = paddingX; j < paddingX + col; ++j) {
			extended[i][j] = extended[i - 1][j];
		}
	}
	//Replicate left and right borders
	for (int i = 0; i < newRow; ++i) {
		for (int j = paddingX - 1; j >= 0; --j) {
			extended[i][j] = extended[i][j + 1];
		}
	}
	for (int i = 0; i < newRow; ++i) {
		for (int j = col + paddingX; j < newCol; ++j) {
			extended[i][j] = extended[i][j - 1];
		}
	}
	//Perform gaussian blur
	float sumTmp = 0.0;
	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			sumTmp = 0.0;
			for (int m = 0; m < ksize_y; ++m) {
				for (int n = 0; n < ksize_x; ++n) {
					sumTmp += extended[i + m][j + n] * kernel[m][n];
				}
			}
			dst.ptr(i)[j] = sumTmp;
		}
	}
	for (int i = 0; i < newRow; i++) {
		delete[] extended[i];
	}
	delete[] extended;
}

void sepFilter2D(Img& src, Img& dst, Img& kx, Img& ky) {
	int row = src.rows,
		col = src.cols,
		ksize = kx.rows,
		half = ksize / 2;
	float sumTmp = 0.;
	//Calculate with kx
	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			sumTmp = 0.;
			if (j - half < 0) {
				sumTmp += src.get(i, half - j) * kx.get(0, 0);
			}
			else {
				sumTmp += src.get(i, j - half) * kx.get(0, 0);
			}
			sumTmp += src.get(i, j) * kx.get(half, 0);
			if (j + half >= col) {
				sumTmp += src.get(i, col - half) * kx.get(ksize - 1, 0);
			}
			else {
				sumTmp += src.get(i, j + half) * kx.get(ksize - 1, 0);
			}
			dst.ptr(i)[j] = sumTmp;
		}
	}
	//Calculate with ky
	float aux = 0.;
	for (int j = 0; j < col; ++j) {
		for (int i = 0; i < row; ++i) {
			sumTmp = 0.;
			if (i - half < 0) {
				sumTmp += dst.get(half - i, j) * ky.get(0, 0);
			}
			else {
				sumTmp += aux * ky.get(0, 0);
			}
			sumTmp += dst.get(i, j) * ky.get(half, 0);
			if (i + half >= row) {
				sumTmp += dst.get(row - half, j) * ky.get(ksize - 1, 0);
			}
			else {
				sumTmp += dst.get(i + half, j) * ky.get(ksize - 1, 0);
			}
			aux = dst.ptr(i)[j];
			dst.ptr(i)[j] = sumTmp;
		}
	}
}

void image_derivatives_scharr(Img& src, Img& dst, int xorder, int yorder) {
	Img kx(3, 1), ky(3, 1);
	
	if (xorder == 1) {
		kx.ptr(0)[0] = -1; kx.ptr(1)[0] = 0; kx.ptr(2)[0] = 1;
		ky.ptr(0)[0] = 3; ky.ptr(1)[0] = 10; ky.ptr(2)[0] = 3;
	}
	else {
		kx.ptr(0)[0] = 3; kx.ptr(1)[0] = 10; kx.ptr(2)[0] = 3;
		ky.ptr(0)[0] = -1; ky.ptr(1)[0] = 0; ky.ptr(2)[0] = 1;
	}
	sepFilter2D(src, dst, kx, ky);
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
		const float* dx_row = dx.ptr(i);
		const float* dy_row = dy.ptr(i);
		for (int j = 1; j < gaussian.cols - 1; ++j) {
			grad = sqrt(dx_row[j] * dx_row[j] + dy_row[j] * dy_row[j]);
			if (grad > max) {
				max = grad;
			}
		}
	}
	for (int i = 1; i < gaussian.rows - 1; ++i) {
		const float* dx_row = dx.ptr(i);
		const float* dy_row = dy.ptr(i);
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
	float d = a.get(0, 0) * a.get(1, 1) - a.get(0, 1) * a.get(1, 0);
	if (d != 0.) {
		dst.ptr(0)[0] = (b.get(0, 0) * a.get(1, 1) - b.get(1, 0) * a.get(0, 1)) / d;
		dst.ptr(1)[0] = (b.get(1, 0) * a.get(0, 0) - b.get(0, 0) * a.get(1, 0)) / d;
		return true;
	}
	else {
		return false;
	}
}

/*void compute_derivative_kernels(Img kx, Img ky, int dx, int dy, int scale) {
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
		Img* kernel = k == 0 ? &kx : &ky;
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
}*/