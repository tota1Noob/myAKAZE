#include<iostream>
#include"akaze.h"
#include <cassert>
using namespace std;
using namespace libAKAZE;

int Fea_3(BYTE* pImage, int iWid, int iHei, BYTE** fFea, int& iFea_num, int& iFea_dim) {
    Img tmp(pImage, iHei, iWid);
    AKAZEOptions options;
    options.img_height = tmp.rows; options.img_width = tmp.cols;
    libAKAZE::AKAZE akaze(options);
    akaze.Create_Nonlinear_Scale_Space(tmp);
    vector<Keypoint> kpts;
    akaze.Feature_Detection(kpts);
    int size = kpts.size();
    BYTE** featureVector = NULL;
    
    akaze.Compute_Descriptors(kpts, featureVector);
    akaze.Show_Computation_Times();
    iFea_num = size;
    return 0;
}


/* ************************************************************************* */
AKAZE::AKAZE(const AKAZEOptions& options) : options(options) {

    ncycles = 0;
    reordering = true;

    /*if (options.descriptor_size > 0 && options.descriptor >= MLDB_UPRIGHT) {
        generateDescriptorSubsample(descriptorSamples_, descriptorBits_, options.descriptor_size,
            options.descriptor_pattern_size, options.descriptor_channels);
    }*/
    
    Allocate_Memory_Evolution();
}

/* ************************************************************************* */
AKAZE::~AKAZE() {
    evolution.clear();
}

/* ************************************************************************* */
void AKAZE::Allocate_Memory_Evolution() {
    clock_t t1 = clock();
    float rfactor = 0.0;
    int level_height = 0, level_width = 0;

    // Allocate the dimension of the matrices for the evolution
    
    evolution.reserve(options.omax * options.nsublevels);
    int sizeCount = 0;
    
    for (int i = 0, power = 1; i <= options.omax - 1; ++i, power *= 2) {
        rfactor = 1.0f / power;
        level_height = (int)(options.img_height * rfactor);
        level_width = (int)(options.img_width * rfactor);
        // Smallest possible octave and allow one scale if the image is small
        if ((level_width < 80 || level_height < 40) && i != 0) {
            options.omax = i;
            break;
        }
        for (int j = 0; j < options.nsublevels; ++j) {
            Evolution step;
            ImgSize size(level_height, level_width);
            step.Lx.create(size);
            step.Ly.create(size);
            step.Lxx.create(size);
            step.Lxy.create(size);
            step.Lyy.create(size);
            step.Lt.create(size);
            step.Ldet.create(size);
            step.Lflow.create(size);
            step.Lstep.create(size);
            step.Lsmooth.create(size);

            step.esigma = options.soffset * pow(2.0f, (float)(j) / (float)(options.nsublevels) + i);
            step.sigma_size = fRound(step.esigma);
            step.etime = 0.5 * (step.esigma * step.esigma);
            step.octave = i;
            step.sublevel = j;
            evolution.push_back(step);
        }
        ++sizeCount;
    }
    evolution.reserve(sizeCount);

    // Allocate memory for the number of cycles and time steps
    for (size_t i = 1; i < evolution.size(); i++) {
        int naux = 0;
        vector<float> tau;
        float ttime = 0.0;
        ttime = evolution[i].etime - evolution[i - 1].etime;
        naux = fed_tau_by_process_time(ttime, 1, 0.25, reordering, tau);
        nsteps.push_back(naux);
        tsteps.push_back(tau);
        ncycles++;
    }
    clock_t t2 = clock();
    timing.initialize = 1000 * (t2 - t1) / (double)CLOCKS_PER_SEC;
}

/* ************************************************************************* */
int AKAZE::Create_Nonlinear_Scale_Space(Img& img) {


    if (evolution.size() == 0) {
        cerr << "Error generating the nonlinear scale space!!" << endl;
        cerr << "Firstly you need to call AKAZE::Allocate_Memory_Evolution()" << endl;
        return -1;
    }

    clock_t t1 = clock();

    // Copy the original image to the first level of the evolution
    img.copyTo(evolution[0].Lt);
    gaussian_2D_convolution(evolution[0].Lt, evolution[0].Lt, 0, 0, options.soffset);
    evolution[0].Lt.copyTo(evolution[0].Lsmooth);

    // First compute the kcontrast factor
    options.kcontrast = compute_k_percentile(img, options.kcontrast_percentile,
        1.0, options.kcontrast_nbins, 0, 0);

    clock_t t2 = clock();
    timing.kcontrast = 1000 * (t2 - t1) / (double)CLOCKS_PER_SEC;
                              
    // Now generate the rest of evolution levels
    for (size_t i = 1; i < evolution.size(); i++) {

        if (evolution[i].octave > evolution[i - 1].octave) {
            halfsample_image(evolution[i - 1].Lt, evolution[i].Lt);
            options.kcontrast = options.kcontrast * 0.75;
        }
        else {
            evolution[i - 1].Lt.copyTo(evolution[i].Lt);
        }

        gaussian_2D_convolution(evolution[i].Lt, evolution[i].Lsmooth, 0, 0, 1.0);

        // Compute the Gaussian derivatives Lx and Ly
        image_derivatives_scharr(evolution[i].Lsmooth, evolution[i].Lx, 1, 0);
        image_derivatives_scharr(evolution[i].Lsmooth, evolution[i].Ly, 0, 1);

        // Compute the conductivity equation
        switch (options.diffusivity) {
        case PM_G1:
            pm_g1(evolution[i].Lx, evolution[i].Ly, evolution[i].Lflow, options.kcontrast);
            break;
        case PM_G2:
            pm_g2(evolution[i].Lx, evolution[i].Ly, evolution[i].Lflow, options.kcontrast);
            break;
        case WEICKERT:
            weickert_diffusivity(evolution[i].Lx, evolution[i].Ly, evolution[i].Lflow, options.kcontrast);
            break;
        case CHARBONNIER:
            charbonnier_diffusivity(evolution[i].Lx, evolution[i].Ly, evolution[i].Lflow, options.kcontrast);
            break;
        default:
            cerr << "Diffusivity: " << options.diffusivity << " is not supported" << endl;
        }

        // Perform FED n inner steps
        for (int j = 0; j < nsteps[i - 1]; j++)
            nld_step_scalar(evolution[i].Lt, evolution[i].Lflow, evolution[i].Lstep, tsteps[i - 1][j]);
    }

    t2 = clock();
    timing.scale = 1000 * (t2 - t1) / (double)CLOCKS_PER_SEC;

    return 0;
}

void AKAZE::Feature_Detection(std::vector<Keypoint>& kpts) {

    clock_t t1 = clock();

    vector<Keypoint>().swap(kpts);
    Compute_Determinant_Hessian_Response();
    Find_Scale_Space_Extrema(kpts);
    Do_Subpixel_Refinement(kpts);
    clock_t t2 = clock();
    timing.detector = 1000 * (t2 - t1) / (double)CLOCKS_PER_SEC;
}

void AKAZE::Compute_Multiscale_Derivatives() {

    for (int i = 0; i < (int)evolution.size(); i++) {
        float ratio = pow(2.0f, (float)evolution[i].octave);
        int sigma_size_ = fRound(evolution[i].esigma * options.derivative_factor / ratio);

        compute_scharr_derivatives(evolution[i].Lsmooth, evolution[i].Lx, 1, 0, sigma_size_);
        compute_scharr_derivatives(evolution[i].Lsmooth, evolution[i].Ly, 0, 1, sigma_size_);
        compute_scharr_derivatives(evolution[i].Lx, evolution[i].Lxx, 1, 0, sigma_size_);
        compute_scharr_derivatives(evolution[i].Ly, evolution[i].Lyy, 0, 1, sigma_size_);
        compute_scharr_derivatives(evolution[i].Lx, evolution[i].Lxy, 0, 1, sigma_size_);
    }
}

void AKAZE::Compute_Determinant_Hessian_Response() {
    clock_t t1 = clock();
    // Firstly compute the multiscale derivatives
    Compute_Multiscale_Derivatives();
    for (size_t i = 0; i < evolution.size(); i++) {
        if (options.verbosity == true)
            cout << "Computing detector response. Determinant of Hessian. Evolution time: " << evolution[i].etime << endl;

        float ratio = pow(2.0f, (float)evolution[i].octave);
        int sigma_size = fRound(evolution[i].esigma * options.derivative_factor / ratio);
        int sigma_size_quat = sigma_size * sigma_size * sigma_size * sigma_size;

        for (int ix = 0; ix < evolution[i].Ldet.rows; ix++) {
            const float* lxx = evolution[i].Lxx.pixels[ix];
            const float* lxy = evolution[i].Lxy.pixels[ix];
            const float* lyy = evolution[i].Lyy.pixels[ix];
            float* ldet = evolution[i].Ldet.pixels[ix];
            for (int jx = 0; jx < evolution[i].Ldet.cols; jx++)
                ldet[jx] = (lxx[jx] * lyy[jx] - lxy[jx] * lxy[jx]) * sigma_size_quat;
        }
    }
    clock_t t2 = clock();
    timing.derivatives = 1000 * (t2 - t1) / (double)CLOCKS_PER_SEC;
}

void AKAZE::Find_Scale_Space_Extrema(std::vector<Keypoint>& kpts) {

    float value = 0.0;
    float dist = 0.0, ratio = 0.0, smax = 0.0;
    int npoints = 0, id_repeated = 0;
    int sigma_size_ = 0, left_x = 0, right_x = 0, up_y = 0, down_y = 0;
    bool is_extremum = false, is_repeated = false, is_out = false;
    Keypoint point;
    vector<Keypoint> kpts_aux;
    kpts_aux.reserve(1000);
    kpts.reserve(1000);
    smax = 10.0 * sqrtf(2.0f);

    clock_t t1 = clock();

    for (size_t i = 0; i < evolution.size(); i++) {
        for (int ix = 1; ix < evolution[i].Ldet.rows - 1; ix++) {

            float* ldet_m = evolution[i].Ldet.pixels[ix - 1];
            float* ldet = evolution[i].Ldet.pixels[ix];
            float* ldet_p = evolution[i].Ldet.pixels[ix + 1];

            for (int jx = 1; jx < evolution[i].Ldet.cols - 1; jx++) {

                is_extremum = false;
                is_repeated = false;
                is_out = false;
                value = ldet[jx];

                // Filter the points with the detector threshold
                if (value > options.dthreshold&& value >= options.min_dthreshold &&
                    value > ldet[jx - 1] && value > ldet[jx + 1] &&
                    value > ldet_m[jx - 1] && value > ldet_m[jx] && value > ldet_m[jx + 1] &&
                    value > ldet_p[jx - 1] && value > ldet_p[jx] && value > ldet_p[jx + 1]) {

                    is_extremum = true;
                    point.response = fabs(value);
                    point.size = evolution[i].esigma * options.derivative_factor;
                    point.octave = evolution[i].octave;
                    point.class_id = i;
                    ratio = pow(2.0f, point.octave);
                    sigma_size_ = fRound(point.size / ratio);
                    point.pt.x = jx;
                    point.pt.y = ix;

                    // Compare response with the same and lower scale
                    for (size_t ik = 0; ik < npoints; ik++) {

                        if ((point.class_id - 1) == kpts_aux[ik].class_id ||
                            point.class_id == kpts_aux[ik].class_id) {

                            dist = (point.pt.x * ratio - kpts_aux[ik].pt.x) * (point.pt.x * ratio - kpts_aux[ik].pt.x) +
                                (point.pt.y * ratio - kpts_aux[ik].pt.y) * (point.pt.y * ratio - kpts_aux[ik].pt.y);

                            if (dist <= point.size * point.size) {
                                if (point.response > kpts_aux[ik].response) {
                                    id_repeated = ik;
                                    is_repeated = true;
                                }
                                else {
                                    is_extremum = false;
                                }
                                break;
                            }
                        }
                    }

                    // Check out of bounds
                    if (is_extremum == true) {

                        // Check that the point is under the image limits for the descriptor computation
                        left_x = fRound(point.pt.x - smax * sigma_size_) - 1;
                        right_x = fRound(point.pt.x + smax * sigma_size_) + 1;
                        up_y = fRound(point.pt.y - smax * sigma_size_) - 1;
                        down_y = fRound(point.pt.y + smax * sigma_size_) + 1;

                        if (left_x < 0 || right_x >= evolution[i].Ldet.cols ||
                            up_y < 0 || down_y >= evolution[i].Ldet.rows) {
                            is_out = true;
                        }

                        if (is_out == false) {
                            if (is_repeated == false) {
                                point.pt.x = point.pt.x * ratio + .5 * (ratio - 1.0);
                                point.pt.y = point.pt.y * ratio + .5 * (ratio - 1.0);
                                kpts_aux.push_back(point);
                                npoints++;
                            }
                            else {
                                point.pt.x = point.pt.x * ratio + .5 * (ratio - 1.0);
                                point.pt.y = point.pt.y * ratio + .5 * (ratio - 1.0);
                                kpts_aux[id_repeated] = point;
                            }
                        } // if is_out
                    } //if is_extremum
                }
            } // for jx
        } // for ix
    } // for i
    kpts_aux.resize(npoints);
    npoints = 0;

    // Now filter points with the upper scale level
    for (size_t i = 0; i < kpts_aux.size(); i++) {

        is_repeated = false;
        const Keypoint& point = kpts_aux[i];
        for (size_t j = i + 1; j < kpts_aux.size(); j++) {

            // Compare response with the upper scale
            if ((point.class_id + 1) == kpts_aux[j].class_id) {

                dist = (point.pt.x - kpts_aux[j].pt.x) * (point.pt.x - kpts_aux[j].pt.x) +
                    (point.pt.y - kpts_aux[j].pt.y) * (point.pt.y - kpts_aux[j].pt.y);

                if (dist <= point.size * point.size) {
                    if (point.response < kpts_aux[j].response) {
                        is_repeated = true;
                        break;
                    }
                }
            }
        }

        if (is_repeated == false) {
            kpts.push_back(point);
            ++npoints;
        }
    }
    kpts.resize(npoints);
    clock_t t2 = clock();
    timing.extrema = 1000 * (t2 - t1) / (double)CLOCKS_PER_SEC;
}

/* ************************************************************************* */
void AKAZE::Do_Subpixel_Refinement(std::vector<Keypoint>& kpts) {

    float Dx = 0.0, Dy = 0.0, ratio = 0.0;
    float Dxx = 0.0, Dyy = 0.0, Dxy = 0.0;
    int x = 0, y = 0;
    Img A(2, 2);
    Img b(2, 1);
    Img dst(2, 1);

    clock_t t1 = clock();

    for (size_t i = 0; i < kpts.size(); i++) {
        ratio = pow(2.f, kpts[i].octave);
        x = fRound(kpts[i].pt.x / ratio);
        y = fRound(kpts[i].pt.y / ratio);

        // Compute the gradient
        Dx = (0.5) * (*(evolution[kpts[i].class_id].Ldet.pixels[y] + x + 1)
            - *(evolution[kpts[i].class_id].Ldet.pixels[y] + x - 1));
        Dy = (0.5) * (*(evolution[kpts[i].class_id].Ldet.pixels[y + 1]+ x)
            - *(evolution[kpts[i].class_id].Ldet.pixels[y - 1] + x));

        // Compute the Hessian
        Dxx = (*(evolution[kpts[i].class_id].Ldet.pixels[y] + x + 1)
            + *(evolution[kpts[i].class_id].Ldet.pixels[y] + x - 1)
            - 2.0 * (*(evolution[kpts[i].class_id].Ldet.pixels[y] + x)));

        Dyy = (*(evolution[kpts[i].class_id].Ldet.pixels[y + 1] + x)
            + *(evolution[kpts[i].class_id].Ldet.pixels[y - 1] + x)
            - 2.0 * (*(evolution[kpts[i].class_id].Ldet.pixels[y] + x)));

        Dxy = (0.25) * (*(evolution[kpts[i].class_id].Ldet.pixels[y + 1] + x + 1)
            + (*(evolution[kpts[i].class_id].Ldet.pixels[y - 1] + x - 1)))
            - (0.25) * (*(evolution[kpts[i].class_id].Ldet.pixels[y - 1] + x + 1)
                + (*(evolution[kpts[i].class_id].Ldet.pixels[y + 1] + x - 1)));

        // Solve the linear system
        A.pixels[0][0] = Dxx;
        A.pixels[1][1] = Dyy;
        A.pixels[1][0] = Dxy;
        A.pixels[0][1] = Dxy;
        b.pixels[0][0] = -Dx;
        b.pixels[1][0] = -Dy;

        solve(A, b, dst);

        if (fabs(dst.pixels[0][0]) <= 1.0 && fabs(dst.pixels[1][0]) <= 1.0) {
            kpts[i].pt.x = x + dst.pixels[0][0];
            kpts[i].pt.y = y + dst.pixels[1][0];
            int power = powf(2, evolution[kpts[i].class_id].octave);
            kpts[i].pt.x = kpts[i].pt.x * power + .5 * (power - 1.f);
            kpts[i].pt.y = kpts[i].pt.y * power + .5 * (power - 1.f);
            kpts[i].angle = 0.0;

            // In OpenCV the size of a keypoint its the diameter
            kpts[i].size *= 2.0;
        }
        // Delete the point since its not stable
        else {
            kpts.erase(kpts.begin() + i);
            i--;
        }
    }

    clock_t t2 = clock();
    timing.subpixel = 1000 * (t2 - t1) / (double)CLOCKS_PER_SEC;
}


void AKAZE::Compute_Main_Orientation(Keypoint& kpt) const {

    int ix = 0, iy = 0, idx = 0, s = 0, level = 0;
    float xf = 0.0, yf = 0.0, gweight = 0.0, ratio = 0.0, PI = 3.14159f;
    float resX[109], resY[109], Ang[109];
    const int id[] = { 6,5,4,3,2,1,0,1,2,3,4,5,6 };

    // Variables for computing the dominant direction
    float sumX = 0.0, sumY = 0.0, max = 0.0, ang1 = 0.0, ang2 = 0.0;

    // Get the information from the keypoint
    level = kpt.class_id;
    ratio = (float)(1 << evolution[level].octave);
    s = fRound(0.5 * kpt.size / ratio);
    xf = kpt.pt.x / ratio;
    yf = kpt.pt.y / ratio;

    // Calculate derivatives responses for points within radius of 6*scale
    for (int i = -6; i <= 6; ++i) {
        for (int j = -6; j <= 6; ++j) {
            if (i * i + j * j < 36) {
                iy = fRound(yf + j * s);
                ix = fRound(xf + i * s);

                gweight = gauss25[id[i + 6]][id[j + 6]];
                resX[idx] = gweight * (*(evolution[level].Lx.pixels[iy] + ix));
                resY[idx] = gweight * (*(evolution[level].Ly.pixels[iy] + ix));
                Ang[idx] = atan2(resY[idx], resX[idx]);
                if (Ang[idx] < 0) Ang[idx] += 2 * PI;
                ++idx;
            }
        }
    }

    // Loop slides pi/3 window around feature point
    for (ang1 = 0; ang1 < 2.0 * PI; ang1 += 0.15f) {
        ang2 = (ang1 + PI / 3.0f > 2.0 * PI ? ang1 - 5.0f * PI / 3.0f : ang1 + PI / 3.0f);
        sumX = sumY = 0.f;

        for (size_t k = 0; k < 109; ++k) {
            // Get angle from the x-axis of the sample point
            const float& ang = Ang[k];

            // Determine whether the point is within the window
            if (ang1 < ang2 && ang1 < ang && ang < ang2) {
                sumX += resX[k];
                sumY += resY[k];
            }
            else if (ang2 < ang1 && ((ang > 0 && ang < ang2) || (ang > ang1&& ang < 2.0 * PI))) {
                sumX += resX[k];
                sumY += resY[k];
            }
        }

        // if the vector produced from this window is longer than all
        // previous vectors then this forms the new dominant direction
        if (sumX * sumX + sumY * sumY > max) {
            // store largest orientation
            max = sumX * sumX + sumY * sumY;
            kpt.angle = atan2(sumY, sumX);
            if (kpt.angle < 0) kpt.angle += 2 * PI;
        }
    }
}

void AKAZE::Get_MLDB_Full_Descriptor(const Keypoint& kpt, BYTE* desc) const {

    const int max_channels = 3;
    assert(options.descriptor_channels <= max_channels);
    float values[16 * max_channels];
    const double size_mult[3] = { 1, 2.0 / 3.0, 1.0 / 2.0 };

    float ratio = (float)(1 << kpt.octave);
    float scale = (float)fRound(0.5f * kpt.size / ratio);
    float xf = kpt.pt.x / ratio;
    float yf = kpt.pt.y / ratio;
    float co = cos(kpt.angle);
    float si = sin(kpt.angle);
    int pattern_size = options.descriptor_pattern_size;

    int dpos = 0;
    for (int lvl = 0; lvl < 3; lvl++) {
        int val_count = (lvl + 2) * (lvl + 2);
        int sample_step = static_cast<int>(ceil(pattern_size * size_mult[lvl]));
        MLDB_Fill_Values(values, sample_step, kpt.class_id, xf, yf, co, si, scale);
        MLDB_Binary_Comparisons(values, desc, val_count, dpos);
    }
}

void AKAZE::MLDB_Fill_Values(float* values, int sample_step, int level,
    float xf, float yf, float co, float si, float scale) const {

    int pattern_size = options.descriptor_pattern_size;
    int nr_channels = options.descriptor_channels;
    int valpos = 0;

    for (int i = -pattern_size; i < pattern_size; i += sample_step) {
        for (int j = -pattern_size; j < pattern_size; j += sample_step) {

            float di = 0.0, dx = 0.0, dy = 0.0;
            int nsamples = 0;

            for (int k = i; k < i + sample_step; k++) {
                for (int l = j; l < j + sample_step; l++) {

                    float sample_y = yf + (l * co * scale + k * si * scale);
                    float sample_x = xf + (-l * si * scale + k * co * scale);

                    int y1 = fRound(sample_y);
                    int x1 = fRound(sample_x);

                    float ri = *(evolution[level].Lt.pixels[y1] + x1);
                    di += ri;

                    if (nr_channels > 1) {
                        float rx = *(evolution[level].Lx.pixels[y1] + x1);
                        float ry = *(evolution[level].Ly.pixels[y1] + x1);
                        if (nr_channels == 2) {
                            dx += sqrtf(rx * rx + ry * ry);
                        }
                        else {
                            float rry = rx * co + ry * si;
                            float rrx = -rx * si + ry * co;
                            dx += rrx;
                            dy += rry;
                        }
                    }
                    nsamples++;
                }
            }
            nsamples /= 1.0f;
            di *= nsamples;
            dx *= nsamples;
            dy *= nsamples;

            values[valpos] = di;

            if (nr_channels > 1)
                values[valpos + 1] = dx;

            if (nr_channels > 2)
                values[valpos + 2] = dy;

            valpos += nr_channels;
        }
    }
}

void AKAZE::MLDB_Binary_Comparisons(float* values, BYTE* desc,
    int count, int& dpos) const {

    int nr_channels = options.descriptor_channels;

    for (int pos = 0; pos < nr_channels; pos++) {
        for (int i = 0; i < count; i++) {
            float ival = values[nr_channels * i + pos];
            for (int j = i + 1; j < count; j++) {
                int res = ival > values[nr_channels * j + pos];
                desc[dpos >> 3] |= (res << (dpos & 7));
                dpos++;
            }
        }
    }
}

int AKAZE::Compute_Descriptors(std::vector<Keypoint>& kpts, BYTE** desc) {

    //double t1 = 0.0, t2 = 0.0;

    //t1 = cv::getTickCount();

    clock_t t1 = clock();

    for (int i = 0; i < (int)(kpts.size()); i++) {
        Compute_Main_Orientation(kpts[i]);
        Get_MLDB_Full_Descriptor(kpts[i], desc[i]);
    }
    clock_t t2 = clock();
    timing.descriptor = 1000 * (t2 - t1) / (double)CLOCKS_PER_SEC;
    return kpts.size();
    //t2 = cv::getTickCount();
    //timing_.descriptor = 1000.0 * (t2 - t1) / cv::getTickFrequency();
}