#pragma once
#include"config.h"

int Fea_3(BYTE* pImage, int iWid, int iHei, BYTE** fFea, int& iFea_num, int& iFea_dim);

namespace libAKAZE {

    class AKAZE {

    private:

        AKAZEOptions options;                      ///< Configuration options for AKAZE
        //std::vector<TEvolution> evolution_;         ///< Vector of nonlinear diffusion evolution

        /// FED parameters
        int ncycles;                               ///< Number of cycles
        bool reordering;                           ///< Flag for reordering time steps
        std::vector<std::vector<float > > tsteps;  ///< Vector of FED dynamic time steps
        std::vector<int> nsteps;                   ///< Vector of number of steps per cycle

        /// Matrices for the M-LDB descriptor computation
        Img descriptorSamples;
        Img descriptorBits;
        Img bitMask;

        /// Computation times variables in ms
        AKAZETiming timing;

    public:
        std::vector<Evolution> evolution;
        /// AKAZE constructor with input options
        /// @param options AKAZE configuration options
        /// @note This constructor allocates memory for the nonlinear scale space
        AKAZE(const AKAZEOptions& options);

        /// Destructor
        ~AKAZE();

        /// Allocate the memory for the nonlinear scale space
        void Allocate_Memory_Evolution();

        /// This method creates the nonlinear scale space for a given image
        /// @param img Input image for which the nonlinear scale space needs to be created
        /// @return 0 if the nonlinear scale space was created successfully, -1 otherwise
        int Create_Nonlinear_Scale_Space(Img& img);

        /// @brief This method selects interesting keypoints through the nonlinear scale space
        /// @param kpts Vector of detected keypoints
        void Feature_Detection(std::vector<Keypoint>& kpts);

        /// This method computes the feature detector response for the nonlinear scale space
        /// @note We use the Hessian determinant as the feature detector response
        void Compute_Determinant_Hessian_Response();

        /// This method computes the multiscale derivatives for the nonlinear scale space
        void Compute_Multiscale_Derivatives();

        /// This method finds extrema in the nonlinear scale space
        void Find_Scale_Space_Extrema(std::vector<Keypoint>& kpts);

        /// This method performs subpixel refinement of the detected keypoints fitting a quadratic
        void Do_Subpixel_Refinement(std::vector<Keypoint>& kpts);

        /// Feature description methods
        int Compute_Descriptors(std::vector<Keypoint>& kpt, BYTE** desc);

        /// This method computes the main orientation for a given keypoint
        /// @param kpt Input keypoint
        /// @note The orientation is computed using a similar approach as described in the original SURF method.
        /// See Bay et al., Speeded Up Robust Features, ECCV 2006.
        /// A-KAZE uses first order derivatives computed from the nonlinear scale space in contrast to Haar wavelets
        void Compute_Main_Orientation(Keypoint& kpt) const;

        /// Computes the rotation invariant M-LDB binary descriptor (maximum descriptor length)
        /// @param kpt Input keypoint
        /// @param desc Binary-based descriptor
        void Get_MLDB_Full_Descriptor(const Keypoint& kpt, BYTE* desc) const;

        /// Fill the comparison values for the MLDB rotation invariant descriptor
        void MLDB_Fill_Values(float* values, int sample_step, int level,
            float xf, float yf, float co, float si, float scale) const;

        /// Do the binary comparisons to obtain the descriptor
        void MLDB_Binary_Comparisons(float* values, BYTE* desc, int count, int& dpos) const;

        /// Display timing information
        void Show_Computation_Times() const {
            std::cout << "Total: " << timing.initialize + timing.scale + timing.detector + timing.descriptor << std::endl;
            std::cout << "(*) Time Initializing: " << timing.initialize << std::endl;
            std::cout << "(*) Time Scale Space: " << timing.scale << std::endl;
            std::cout << "   - Time Kcontrast: " << timing.kcontrast << std::endl;
            std::cout << "(*) Time Detector: " << timing.detector << std::endl;
            std::cout << "   - Time Derivatives: " << timing.derivatives << std::endl;
            std::cout << "   - Time Extrema: " << timing.extrema << std::endl;
            std::cout << "   - Time Subpixel: " << timing.subpixel << std::endl;
            std::cout << "(*) Time Descriptor: " << timing.descriptor << std::endl;
            std::cout << std::endl;
        }

        /// Return the computation times
        AKAZETiming Get_Computation_Times() const {
            return timing;
        }
    };


    /// This function computes the value of a 2D Gaussian function
    inline float gaussian(float x, float y, float sigma) {
        return expf(-(x * x + y * y) / (2.0f * sigma * sigma));
    }

    /// This funtion rounds float to nearest integer
    inline int fRound(float flt) {
        return (int)(flt + 0.5f);
    }
}