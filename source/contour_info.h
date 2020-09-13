#pragma once
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

void blob_mean_and_tangent(const vector<Point>& contour, Point2d& center, Point2d& tangent);

class ContourInfo
{
public:
	vector<Point> contour;
	Rect rect;
	Mat mask;
	Point2d center;
	Point2d tangent;
	double angle;
	vector<double> local_xrng;
	Point2d point0;					// (left endpoint, center)
	Point2d point1;					// (right endpoint, center)
	ContourInfo* pred;
	ContourInfo* succ;
	bool assembled;
public:
	ContourInfo(const vector<Point>& contour, const Rect& rect, const Mat& mask);
	double project_x(const Point2d& point) const;
	double local_overlap(const ContourInfo& other);
	bool operator==(const ContourInfo& other);
};

struct Edge
{
	double score;
	ContourInfo* left;
	ContourInfo* right;
};

/*----------------------------------------- function definition -----------------------------------------*/

void blob_mean_and_tangent(const vector<Point>& contour, Point2d& center, Point2d& tangent)
{
	Moments moments = cv::moments(contour);

	double area = moments.m00;

	double mean_x = moments.m10 / area;
	double mean_y = moments.m01 / area;

	center = Point2d(mean_x, mean_y);					// initialize center

	Mat moments_matrix = (Mat_<double>(2, 2) << moments.mu20, moments.mu11,
		moments.mu11, moments.mu02);
	Mat W, U, Vt;
	cv::SVDecomp(moments_matrix, W, U, Vt);

	double x = U.at<double>(0, 0);
	double y = U.at<double>(1, 0);
	tangent = Point2d(x, y);							// initialize tangent
}

ContourInfo::ContourInfo(const vector<Point>& contour, const Rect& rect, const Mat& mask) :
	contour(contour), rect(rect), mask(mask)
{
	blob_mean_and_tangent(contour, center, tangent);
	angle = std::atan2(tangent.y, tangent.x);

	vector<double> clx;
	for (const auto& point : contour) {
		clx.push_back(project_x(point));
	}

	double lxmin = *std::min_element(clx.begin(), clx.end());
	double lxmax = *std::max_element(clx.begin(), clx.end());

	local_xrng.push_back(lxmin);
	local_xrng.push_back(lxmax);

	point0 = center + tangent * lxmin;
	point1 = center + tangent * lxmax;

	pred = nullptr;
	succ = nullptr;
	assembled = false;
}

double ContourInfo::project_x(const Point2d& point) const
{
	return tangent.dot(point - center);
}

double ContourInfo::local_overlap(const ContourInfo& other)
{
	double xmin = project_x(other.point0);
	double xmax = project_x(other.point1);

	return std::min(local_xrng[1], xmax) - std::max(local_xrng[0], xmin);
}

bool ContourInfo::operator==(const ContourInfo& other)
{
	return (point0 == other.point0) && (point1 == other.point1);
}