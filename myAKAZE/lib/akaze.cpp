#include<iostream>
#include<math.h>
#include"akaze.h"

/******************************************************************************************/
/*Basic classes & functions*/
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
	}
}

Img::Img(ImgSize size) {
	this->cols = size.width;
	this->rows = size.height;
	this->pixels = new float* [this->rows];
	for (int i = 0; i < this->rows; ++i) {
		this->pixels[i] = new float[this->cols];
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

void image_derivatives_scharr(Img& src, Img& dst, int xorder, int yorder) {
	float weights[3][3];
	if (xorder == 1) {
		weights[0][0] = -3; weights[0][1] = 0; weights[0][2] = 3;
		weights[1][0] = -10; weights[1][1] = 0; weights[1][2] = 10;
		weights[2][0] = -3; weights[2][1] = 0; weights[2][2] = 3;
	}
	else {
		weights[0][0] = -3; weights[0][1] = -10; weights[0][2] = -3;
		weights[1][0] = 0; weights[1][1] = 0; weights[1][2] = 0;
		weights[2][0] = 3; weights[2][1] = 10; weights[2][2] = 3;
	}
	int row = src.rows,
		col = src.cols,
		newRow = row + 2,
		newCol = col + 2;
	float** extended = new float* [newRow];
	for (int i = 0; i < newRow; ++i) {
		extended[i] = new float[newCol];
	}
	//Copy src data
	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			extended[i + 1][j + 1] = src.get(i, j);
		}
	}
	//Replicate borders
	for (int i = 1; i <= col; ++i) {
		extended[0][i] = extended[1][i];
		extended[newRow - 1][i] = extended[newRow - 2][i];
	}
	for (int i = 0; i < newRow; ++i) {
		extended[i][0] = extended[i][1];
		extended[i][newCol - 1] = extended[i][newCol - 2];
	}
	//Perform Scharr
	float sumTmp = 0.0;
	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			sumTmp = extended[i][j] * weights[0][0] + extended[i][j + 1] * weights[0][1] + extended[i][j + 2] * weights[0][2]
				+ extended[i + 1][j] * weights[1][0] + extended[i + 1][j + 1] * weights[1][1] + extended[i + 1][j + 2] * weights[1][2]
				+ extended[i + 2][j] * weights[2][0] + extended[i + 2][j + 1] * weights[2][1] + extended[i + 2][j + 2] * weights[2][2];
			dst.ptr(i)[j] = sumTmp;
		}
	}
	for (int i = 0; i < newRow; i++) {
		delete[] extended[i];
	}
	delete[] extended;
}