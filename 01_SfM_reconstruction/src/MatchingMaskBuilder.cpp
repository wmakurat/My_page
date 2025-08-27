#include "MatchingMaskBuilder.h"
#include <limits>
#include <algorithm>
#include <cmath>
#include <omp.h>


void MatchingMaskBuilder::calcRotMat(const std::array<float, 6>& eutt, cv::Mat& R)
{
    double sx = std::sin(eutt[0]), cx = std::cos(eutt[0]);
    double sy = std::sin(eutt[1]), cy = std::cos(eutt[1]);
    double sz = std::sin(eutt[2]), cz = std::cos(eutt[2]);

    cv::Mat Rx = (cv::Mat_<double>(3, 3) <<
        1, 0, 0,
        0, cx, sx,
        0, -sx, cx);
    cv::Mat Ry = (cv::Mat_<double>(3, 3) <<
        cy, 0, -sy,
        0, 1, 0,
        sy, 0, cy);
    cv::Mat Rz = (cv::Mat_<double>(3, 3) <<
        cz, sz, 0,
        -sz, cz, 0,
        0, 0, 1);

    R = (Rx * Ry * Rz).t();
}

cv::Vec3f MatchingMaskBuilder::rotationMatrixToEulerAngles(const cv::Mat& R)
{
    double sy = std::sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) +
        R.at<double>(1, 0) * R.at<double>(1, 0));
    bool singular = sy < 1e-6;
    double x, y, z;
    if (!singular) {
        x = std::atan2(R.at<double>(2, 1), R.at<double>(2, 2));
        y = std::atan2(-R.at<double>(2, 0), sy);
        z = std::atan2(R.at<double>(1, 0), R.at<double>(0, 0));
    }
    else {
        x = std::atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
        y = std::atan2(-R.at<double>(2, 0), sy);
        z = 0;
    }
    return cv::Vec3f((float)x, (float)y, (float)z);
}

void MatchingMaskBuilder::calcRTl2r(const std::vector<cv::Mat>& R,
    const std::vector<cv::Mat>& T,
    cv::Mat& RTl2r)
{
    std::vector<cv::Mat> RT(2);
    cv::Mat homorow = (cv::Mat_<double>(1, 4) << 0, 0, 0, 1);
    for (int i = 0; i < 2; ++i) {
        cv::hconcat(R[i], T[i], RT[i]);
        cv::vconcat(RT[i], homorow, RT[i]);
    }
    cv::Mat H = RT[1] * RT[0].inv();
    RTl2r = H(cv::Rect(0, 0, 4, 3)).clone();
}

void MatchingMaskBuilder::projMatToEulerTranslation(const cv::Mat& projL, const cv::Mat& projR,
    cv::Vec3f& eulerl2r, cv::Vec3f& Tl2r)
{
    cv::Mat Rx, Ry, Rz;
    std::vector<cv::Mat> euler(2), T(2), R(2), K(2);

    for (int i = 0; i < 2; ++i) {
        const cv::Mat& P = (i == 0 ? projL : projR);
        cv::Mat Ti;
        cv::decomposeProjectionMatrix(P, K[i], R[i], Ti, Rx, Ry, Rz, euler[i]);
        for (int j = 0; j < Ti.rows; ++j) Ti.at<double>(j, 0) /= Ti.at<double>(3, 0);
        Ti = Ti(cv::Rect(0, 0, 1, 3)).clone();
        T[i] = -R[i] * Ti;
    }

    cv::Mat RTl2r, Rl2r, Tl2rMat;
    calcRTl2r(R, T, RTl2r);
    RTl2r(cv::Rect(0, 0, 3, 3)).copyTo(Rl2r);

    Tl2rMat = -RTl2r(cv::Rect(0, 0, 3, 3)).inv() * RTl2r(cv::Rect(3, 0, 1, 3));

    eulerl2r = rotationMatrixToEulerAngles(Rl2r);
    Tl2r = cv::Vec3f((float)Tl2rMat.at<double>(0, 0),
        (float)Tl2rMat.at<double>(1, 0),
        (float)Tl2rMat.at<double>(2, 0));
}

void MatchingMaskBuilder::distortParams(const cv::Vec3f& eulerl2r, const cv::Vec3f& Tl2r,
    float dEuler, float dT,
    std::vector<std::array<float, 6>>& seqs)
{
    float ex[2] = { eulerl2r[0] - dEuler, eulerl2r[0] + dEuler };
    float ey[2] = { eulerl2r[1] - dEuler, eulerl2r[1] + dEuler };
    float ez[2] = { eulerl2r[2] - dEuler, eulerl2r[2] + dEuler };
    float tx[2] = { Tl2r[0] - dT, Tl2r[0] + dT };
    float ty[2] = { Tl2r[1] - dT, Tl2r[1] + dT };
    float tz[2] = { Tl2r[2] - dT, Tl2r[2] + dT };

    seqs.clear(); seqs.reserve(64);
    for (int i0 = 0; i0 < 2; ++i0)
        for (int i1 = 0; i1 < 2; ++i1)
            for (int i2 = 0; i2 < 2; ++i2)
                for (int i3 = 0; i3 < 2; ++i3)
                    for (int i4 = 0; i4 < 2; ++i4)
                        for (int i5 = 0; i5 < 2; ++i5) {
                            seqs.push_back({ ex[i0], ey[i1], ez[i2], tx[i3], ty[i4], tz[i5] });
                        }
}

void MatchingMaskBuilder::buildAllRT(const std::vector<std::array<float, 6>>& params,
    std::vector<cv::Mat>& allRT)
{
    allRT.resize(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        const auto& p = params[i];
        cv::Mat R; calcRotMat(p, R);
        cv::Mat T = -R * (cv::Mat_<double>(3, 1) << p[3], p[4], p[5]);
        allRT[i].create(3, 4, CV_64F);
        R.copyTo(allRT[i](cv::Rect(0, 0, 3, 3)));
        T.copyTo(allRT[i](cv::Rect(3, 0, 1, 3)));
    }
}

cv::Vec2d MatchingMaskBuilder::calcPr(const cv::Point2f& Pl, float Z,
    const cv::Mat& RTl2r,
    const cv::Mat& Kl, const cv::Mat& Kr)
{
    cv::Mat Plh = (cv::Mat_<double>(3, 1) << Pl.x, Pl.y, 1.0);
    cv::Mat Pcl = Z * Kl.inv() * Plh;
    cv::Mat Plch; cv::vconcat(Pcl, cv::Mat::ones(1, 1, CV_64F), Plch);
    cv::Mat Prh = Kr * RTl2r * Plch;
    double w = Prh.at<double>(2, 0);
    cv::Mat Pr = Prh / w;
    return { Pr.at<double>(0, 0), Pr.at<double>(1, 0) };
}

void MatchingMaskBuilder::findPminMax(const cv::Point2f& Pl, float Z,
    const std::vector<cv::Mat>& allRTl2r,
    const cv::Mat& Kl, const cv::Mat& Kr,
    cv::Point2f& PuMin, cv::Point2f& PuMax)
{
    double uMin = std::numeric_limits<double>::infinity();
    double uMax = -std::numeric_limits<double>::infinity();
    cv::Point2f minPt, maxPt;

    for (const auto& RT : allRTl2r) {
        cv::Vec2d Pr = calcPr(Pl, Z, RT, Kl, Kr);
        if (Pr[0] < uMin) { uMin = Pr[0]; minPt = cv::Point2f((float)Pr[0], (float)Pr[1]); }
        if (Pr[0] > uMax) { uMax = Pr[0]; maxPt = cv::Point2f((float)Pr[0], (float)Pr[1]); }
    }
    PuMin = minPt;
    PuMax = maxPt;
}


void MatchingMaskBuilder::quadAABBf(const std::vector<cv::Point2f>& hull, cv::Rect2f& aabb)
{
    float xmin = hull[0].x, xmax = hull[0].x;
    float ymin = hull[0].y, ymax = hull[0].y;
    for (const auto& h : hull) {
        xmin = std::min(xmin, h.x); xmax = std::max(xmax, h.x);
        ymin = std::min(ymin, h.y); ymax = std::max(ymax, h.y);
    }
    aabb = cv::Rect2f(xmin, ymin, xmax - xmin, ymax - ymin);
}

cv::Rect2f MatchingMaskBuilder::intersect(const cv::Rect2f& a, const cv::Rect2f& b)
{
    float x0 = std::max(a.x, b.x);
    float y0 = std::max(a.y, b.y);
    float x1 = std::min(a.x + a.width, b.x + b.width);
    float y1 = std::min(a.y + a.height, b.y + b.height);
    if (x1 <= x0 || y1 <= y0) return cv::Rect2f();
    return cv::Rect2f(x0, y0, x1 - x0, y1 - y0);
}


MatchingMaskBuilder::Grid::Grid(float minx, float miny, float maxx, float maxy,
    int cellPx, const std::vector<cv::KeyPoint>& kps)
{
    cell = (float)std::max(4, cellPx);
    const float pad = cell; // edges margin
    ox = minx - pad; oy = miny - pad;
    float wx = (maxx - minx) + 2 * pad;
    float wy = (maxy - miny) + 2 * pad;
    cols = std::max(1, (int)std::ceil(wx / cell));
    rows = std::max(1, (int)std::ceil(wy / cell));

    bins.resize(cols * rows);
    for (int i = 0; i < (int)kps.size(); ++i) {
        const auto& p = kps[i].pt;
        int cx = (int)std::floor((p.x - ox) / cell);
        int cy = (int)std::floor((p.y - oy) / cell);
        if (cx >= 0 && cy >= 0 && cx < cols && cy < rows)
            bins[cy * cols + cx].push_back(i);
    }
}

void MatchingMaskBuilder::Grid::queryRect(const cv::Rect2f& r, std::vector<int>& out) const
{
    if (r.width <= 0 || r.height <= 0) return;
    int cx0 = std::max(0, (int)std::floor((r.x - ox) / cell));
    int cy0 = std::max(0, (int)std::floor((r.y - oy) / cell));
    int cx1 = std::min(cols - 1, (int)std::floor(((r.x + r.width) - ox) / cell));
    int cy1 = std::min(rows - 1, (int)std::floor(((r.y + r.height) - oy) / cell));
    for (int cy = cy0; cy <= cy1; ++cy)
        for (int cx = cx0; cx <= cx1; ++cx) {
            const auto& b = bins[cy * cols + cx];
            out.insert(out.end(), b.begin(), b.end());
        }
}

cv::Mat MatchingMaskBuilder::build(const std::vector<cv::KeyPoint>& keypointsL,
    const std::vector<cv::KeyPoint>& keypointsR,
    const cv::Mat& projL, const cv::Mat& projR,
    const cv::Mat& Kleft, const cv::Mat& Kright,
    const Params& P)
{
    cv::Mat mask = cv::Mat::zeros((int)keypointsL.size(), (int)keypointsR.size(), CV_8U);
    if (keypointsL.empty() || keypointsR.empty()) return mask;

    cv::Vec3f euler, T;
    projMatToEulerTranslation(projL, projR, euler, T);

    std::vector<std::array<float, 6>> paramSeq;
    distortParams(euler, T, P.dEuler, P.dT, paramSeq);

    std::vector<cv::Mat> allRT; allRT.reserve(paramSeq.size());
    buildAllRT(paramSeq, allRT);

    float minx = keypointsR[0].pt.x, maxx = minx;
    float miny = keypointsR[0].pt.y, maxy = miny;
    for (const auto& k : keypointsR) {
        minx = std::min(minx, k.pt.x); maxx = std::max(maxx, k.pt.x);
        miny = std::min(miny, k.pt.y); maxy = std::max(maxy, k.pt.y);
    }
    Grid grid(minx, miny, maxx, maxy, P.gridCell, keypointsR);
    const cv::Rect2f worldBB = grid.bounds();

    #pragma omp parallel if(P.useParallel)
    {
        std::vector<cv::Point2f> poly, hull;
        std::vector<int> candidates;
        poly.reserve(8); hull.reserve(8); candidates.reserve(64);

        #pragma omp for schedule(dynamic,32)
        for (int i = 0; i < (int)keypointsL.size(); ++i) {
            poly.clear(); hull.clear(); candidates.clear();

            cv::Point2f PuMin[2], PuMax[2];
            findPminMax(keypointsL[i].pt, P.Zmin, allRT, Kleft, Kright, PuMin[0], PuMax[0]);
            findPminMax(keypointsL[i].pt, P.Zmax, allRT, Kleft, Kright, PuMin[1], PuMax[1]);

            poly = { PuMin[0], PuMax[0], PuMax[1], PuMin[1] };
            cv::convexHull(poly, hull, true, true);
            if (hull.size() < 3) continue;

            cv::Rect2f aabb; quadAABBf(hull, aabb);
            aabb = intersect(aabb, worldBB);
            if (aabb.width <= 0 || aabb.height <= 0) continue;

            grid.queryRect(aabb, candidates);

            for (int idx : candidates) {
                const auto& p = keypointsR[idx].pt;
                if (p.x < aabb.x || p.x >= aabb.x + aabb.width ||
                    p.y < aabb.y || p.y >= aabb.y + aabb.height)
                    continue;

                if (cv::pointPolygonTest(hull, p, false) >= 0.0) {
                    mask.at<uchar>(i, idx) = 255;
                }
            }
        }
    }

    return mask;
}
