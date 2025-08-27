#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <array>

class MatchingMaskBuilder {
public:    
    struct Params {
        float Zmin = 0.5f; // min distance from camera
        float Zmax = 10.0f;
        float dEuler = 0.0f; // error in euler angles measurement
        float dT = 0.0f; // error in translation mesurement
        int   gridCell = 16; // size of a grig (prefilter)
        bool  useParallel = true; // perform parallel calculations
    };

    MatchingMaskBuilder() = default;

    // Builds a matching mask (rows = #left, cols = #right)
    cv::Mat build(const std::vector<cv::KeyPoint>& keypointsL,
        const std::vector<cv::KeyPoint>& keypointsR,
        const cv::Mat& projL, const cv::Mat& projR,
        const cv::Mat& Kleft, const cv::Mat& Kright,
        const Params& P);

private:
    // --- geometry / algebra ---

    // Calculates rotation matrix based on Euler angles (first three elements of eutt - other three params are translation, not used)
    static void calcRotMat(const std::array<float, 6>& eutt, cv::Mat& R);

    // Calculates Euler angles from rotation matrix 
    static cv::Vec3f rotationMatrixToEulerAngles(const cv::Mat& R);

    // Calculates transformation matrix from "left" camera to "right"
    static void calcRTl2r(const std::vector<cv::Mat>& R,
        const std::vector<cv::Mat>& T,
        cv::Mat& RTl2r);

    // Calculates Euler angles and translation from projection matrix
    static void projMatToEulerTranslation(const cv::Mat& projL, const cv::Mat& projR,
        cv::Vec3f& eulerl2r, cv::Vec3f& Tl2r);

    // Generates the 2^6 corner combinations of the uncertainty hypercube (+-dEuler, +-dT) around the nominal pose
    static void distortParams(const cv::Vec3f& eulerl2r, const cv::Vec3f& Tl2r,
        float dEuler, float dT,
        std::vector<std::array<float, 6>>& seqs);

    // Creates transformation matrixes for all combinations of distorted parameters
    static void buildAllRT(const std::vector<std::array<float, 6>>& params,
        std::vector<cv::Mat>& allRT);

    // Projects a left-image pixel Pl at depth Z through K_l^{-1}, transforms by [R|t] (L->R), and reprojects with K_r to right-image pixel (u,v)
    static cv::Vec2d calcPr(const cv::Point2f& Pl, float Z,
        const cv::Mat& RTl2r,
        const cv::Mat& Kl, const cv::Mat& Kr);

    // From a set of possible point positions, finds the most extreme
    static void findPminMax(const cv::Point2f& Pl, float Z,
        const std::vector<cv::Mat>& allRTl2r,
        const cv::Mat& Kl, const cv::Mat& Kr,
        cv::Point2f& PuMin, cv::Point2f& PuMax);

    // Float Axis-Aligned Bounding Box
    static inline void quadAABBf(const std::vector<cv::Point2f>& hull, cv::Rect2f& aabb);
    
    // Helper function - returns intersection of two cv::RECT
    static inline cv::Rect2f intersect(const cv::Rect2f& a, const cv::Rect2f& b);

    // Grid for prefiltering
    struct Grid {
        float cell = 16.f, ox = 0.f, oy = 0.f;
        int cols = 0, rows = 0;
        std::vector<std::vector<int>> bins;
        Grid() = default;
        Grid(float minx, float miny, float maxx, float maxy, int cellPx,
            const std::vector<cv::KeyPoint>& kps);
        
        // Queries the grid and appends indices of keypoints whose coordinates fall inside the given rectangle
        // Does not clear the output vector — values are appended
        void queryRect(const cv::Rect2f& r, std::vector<int>& out) const;

        // Returns the full rectangular region covered by the current grid
        // This may be slightly larger than the actual keypoint area due to grid quantization
        cv::Rect2f bounds() const { return cv::Rect2f(ox, oy, cols * cell, rows * cell); }
    };
};
