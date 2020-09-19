#pragma once
#define _USE_MATH_DEFINES
#include <iostream>
#include <algorithm>
#include <numeric>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>
#include "contour_info.h"
#include "params.h"
#include "visualize.h"
#include "praxis.h"
using namespace std;
using namespace cv;

int PAGE_MARGIN_X = 50;			// reduced px to ignore near L/R edge
int PAGE_MARGIN_Y = 20;			// reduced px to ignore near T/B edge

double OUTPUT_ZOOM = 1.0;		// how much to zoom output relative to *original* image
int OUTPUT_DPI = 300;			// just affects stated DPI of PNG, not appearance
int REMAP_DECIMATE = 16;		// downscaling factor for remapping image

int ADAPTIVE_WINSZ = 55;		// window size for adaptive threshold inreduced px

int TEXT_MIN_WIDTH = 15;		// min reduced px width of detected text contour
int TEXT_MIN_HEIGHT = 2;		// min reduced px height of detected text contour
double TEXT_MIN_ASPECT = 1.5;	// filter out text contours velow this w/h ratio
int TEXT_MAX_THICKNESS = 10;	// max reduced px thickness of detected text contour

double EDGE_MAX_OVERLAP = 1.0;	// max reduced px horiz. overlap of contours in span
double EDGE_MAX_LENGTH = 100.0;	// max reduced px length of edge connecting contours
double EDGE_ANGLE_COST = 10.0;	// cost of angles in edges (tradeoff vs. length)
double EDGE_MAX_ANGLE = 7.5;	// maximum change in angle allowed between contours

int SPAN_MIN_WIDTH = 30;		// minimum reduced px width for span
int SPAN_PX_PER_STEP = 20;		// reduced px spacing for sampling along spans
double FOCAL_LENGTH = 1.2;		// normalized focal length of camera

int RVEC_BEGIN = 0;
int TVEC_BEGIN = 3;
int SLOPES_BEGIN = 6;
int Y_BEGIN = 8;

int DEBUG_LEVEL = 0;			// 0=none, 1=some, 2=lots, 3=all

Mat K = (cv::Mat_<double>(3, 3) << FOCAL_LENGTH, 0, 0,
									0, FOCAL_LENGTH, 0,
									0, 0, 1);

/*----------------------------------------- function declaration -----------------------------------------*/

void resize_to_screen(const Mat& src, Mat& resized, int maxw = 1280, int maxh = 700);
void get_page_extents(const Mat& small, Mat& page, vector<Point>& outline);
void get_contours(string name, const Mat& small, const Mat& pagemask, string masktype, vector<ContourInfo>& contours_out);
Mat get_mask(string name, const Mat& small, const Mat& pagemask, string masktype);
Mat box(int width, int height);
Mat make_tight_mask(const vector<Point>& contour, int xmin, int ymin, int width, int height);
void assemble_span(string name, const Mat& small, const Mat& pagemask, vector<ContourInfo>& cinfo_list, vector<vector<ContourInfo*>>& spans);
void generate_candidate_edge(Edge& edge);
double angle_dist(double angle_right, double angle_left);
void sample_spans(const Size& shape, vector<vector<ContourInfo*>>& spans, vector<vector<Point2d>>& span_pts);
void pix2norm(const Size& shape, vector<Point2d>& points);
void keypoints_from_samples(string name, const Mat& small, const Mat& pagemask, const vector<Point>& page_outline,
	const vector<vector<Point2d>>& span_points, vector<Point2d>& corners, vector<double>& y_coords, vector<vector<double>>& x_coords);
void get_default_params(const vector<Point2d>& corners, vector<Point2d>& rough_dims, const vector<double>& y_coords,
	const vector<vector<double>>& x_coords, Params& p);
void stack_points(const vector<Point2d>& corners, const vector<vector<Point2d>>& span_pts, int npts, vector<Point2d>& dst_pts);
void optimize_params(string name, const Mat& small, const vector<Point2d>& dst_pts, Params& p);
double objective(double* dta, int size, const vector<Point2d>& dst_pts, Params& p);
void project_xy(const vector<double>& x, const vector<double>& y, double* params, vector<Point2d>& img_pts);
void polyval(const vector<double>& coeffs, const vector<double>& values, vector<double>& results);
void concat_xyz(const vector<double>& x, const vector<double>& y, const vector<double>& z, vector<Point3d>& xyz);
double sum_squared_error(const vector<Point2d>& dst_pts, const vector<Point2d>& img_pts);
void get_page_dims(const vector<Point2d>& corners, Params& p, double* page_dims);
double objective2(double* dims, int size, const vector<Point2d>& dst_br, Params& p);
void remap_image(string name, const Mat& img, const Mat& small, const double* page_dims, const Params& p);
int round_nearest_multiple(double num, int factor);
vector<double> linspace(double a, double b, int N);
void meshgrid(const vector<double>& xs, const vector<double>& ys, vector<double>& x_coords, vector<double>& y_coords);
void reshape(const vector<Point2d>& img_pts, int height, int width, Mat& img_x_coords, Mat& img_y_coords);

/*----------------------------------------- function definition -----------------------------------------*/

void resize_to_screen(const Mat& src, Mat& resized, int maxw, int maxh)
{
	double scl_x = (double)src.cols / maxw;
	double scl_y = (double)src.rows / maxh;

	double scl = std::ceil(std::max(scl_x, scl_y));

	if (scl > 1.0) {
		double inv_scl = 1.0 / scl;
		cv::resize(src, resized, Size(0, 0), inv_scl, inv_scl, cv::INTER_AREA);
	}
	else {
		src.copyTo(resized);
	}
}

void get_page_extents(const Mat& small, Mat& page, vector<Point>& outline)
{
	int height = small.rows;
	int width = small.cols;

	int xmin = PAGE_MARGIN_X;
	int ymin = PAGE_MARGIN_Y;
	int xmax = width - PAGE_MARGIN_X;
	int ymax = height - PAGE_MARGIN_Y;

	page = Mat::zeros(height, width, CV_8UC1);
	cv::rectangle(page, Point(xmin, ymin), Point(xmax, ymax), (255, 255, 255), -1);

	outline.push_back(Point(xmin, ymin));
	outline.push_back(Point(xmin, ymax));
	outline.push_back(Point(xmax, ymax));
	outline.push_back(Point(xmax, ymin));
}

void get_contours(string name,
	const Mat& small,
	const Mat& pagemask,
	string masktype,
	vector<ContourInfo>& contours_out)
{
	Mat mask = get_mask(name, small, pagemask, masktype);

	vector<vector<Point>> contours;
	cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	for (const auto& contour : contours) {
		Rect rect = cv::boundingRect(contour);

		int xmin = rect.x;
		int ymin = rect.y;
		int width = rect.width;
		int height = rect.height;

		if (width < TEXT_MIN_WIDTH || height < TEXT_MIN_HEIGHT ||
			width < TEXT_MIN_ASPECT * height) {
			continue;
		}

		Mat tight_mask = make_tight_mask(contour, xmin, ymin, width, height);

		Mat col_sum;
		cv::reduce(tight_mask, col_sum, 0, cv::REDUCE_SUM, CV_64F);

		double min, max;
		cv::minMaxLoc(col_sum, &min, &max);
		if (max > TEXT_MAX_THICKNESS) {
			continue;
		}

		contours_out.push_back(ContourInfo(contour, rect, tight_mask));

		if (DEBUG_LEVEL >= 2) {
			visualize_contours(name, small, contours_out);
		}
	}
}

Mat get_mask(string name, const Mat& small, const Mat& pagemask, string masktype)
{
	Mat sgray;
	cv::cvtColor(small, sgray, cv::COLOR_RGB2GRAY);

	Mat mask;
	if (masktype == "text") {
		cv::adaptiveThreshold(sgray, mask, 255,
			cv::ADAPTIVE_THRESH_MEAN_C,
			cv::THRESH_BINARY_INV,
			ADAPTIVE_WINSZ,
			25);

		if (DEBUG_LEVEL >= 3) {
			debug_show(name, 0.1, "threshholded", mask);
		}

		cv::dilate(mask, mask, box(9, 1));

		if (DEBUG_LEVEL >= 3) {
			debug_show(name, 0.2, "dilated", mask);
		}

		cv::erode(mask, mask, box(1, 3));

		if (DEBUG_LEVEL >= 3) {
			debug_show(name, 0.3, "eroded", mask);
		}
	}
	else {
		cv::adaptiveThreshold(sgray, mask, 255,
			cv::ADAPTIVE_THRESH_MEAN_C,
			cv::THRESH_BINARY_INV,
			ADAPTIVE_WINSZ,
			7);

		if (DEBUG_LEVEL >= 3) {
			debug_show(name, 0.4, "threshholded", mask);
		}

		cv::erode(mask, mask, box(1, 3), Point(-1, -1), 3);

		if (DEBUG_LEVEL >= 3) {
			debug_show(name, 0.5, "eroded", mask);
		}

		cv::dilate(mask, mask, box(8, 2));

		if (DEBUG_LEVEL >= 3) {
			debug_show(name, 0.6, "dilated", mask);
		}
	}

	return cv::min(mask, pagemask);
}

Mat box(int width, int height) { return Mat::ones(height, width, CV_8UC1); }

Mat make_tight_mask(const vector<Point>& contour, int xmin, int ymin, int width, int height)
{
	Point p(xmin, ymin);
	Mat tight_mask = Mat::zeros(height, width, CV_8UC1);

	vector<vector<Point>> tight_contour(1);
	for (const auto& point : contour) {
		tight_contour[0].push_back(point - p);
	}

	cv::drawContours(tight_mask, tight_contour, 0, (1, 1, 1), -1);
	return tight_mask;
}

void assemble_span(string name,
	const Mat& small,
	const Mat& pagemask,
	vector<ContourInfo>& cinfo_list,
	vector<vector<ContourInfo*>>& spans)
{
	// sort list
	std::stable_sort(begin(cinfo_list), end(cinfo_list),
		[](const ContourInfo& a, const ContourInfo& b) {
			return a.rect.y < b.rect.y;
		});

	// generate all candidate edges
	vector<Edge> candidate_edges;
	for (int i = 0; i < cinfo_list.size(); i++) {
		for (int j = 0; j < i; j++) {
			Edge edge{ -1.0, &cinfo_list[i], &cinfo_list[j] };
			generate_candidate_edge(edge);
			if (edge.score != -1.0) {
				candidate_edges.push_back(edge);
			}
		}
	}

	// sort candidate edges by score(lower is better)
	std::stable_sort(begin(candidate_edges), end(candidate_edges),
		[](const Edge& a, const Edge& b) {
			return a.score < b.score;
		});

	// for each candidate edge
	for (Edge& edge : candidate_edges) {
		if (edge.left->succ == nullptr && edge.right->pred == nullptr) {
			edge.left->succ = edge.right;
			edge.right->pred = edge.left;
		}
	}

	// assemble span
	for (ContourInfo& cinfo : cinfo_list) {
		ContourInfo* p = &cinfo;

		while (p->pred != nullptr) {
			p = p->pred;
		}

		vector<ContourInfo*> cur_span;
		double width = 0.0;

		while (p != nullptr && p->assembled == false) {
			cur_span.push_back(p);
			width += p->local_xrng[1] - p->local_xrng[0];
			p->assembled = true;
			p = p->succ;
		}

		if (width > SPAN_MIN_WIDTH) {
			spans.push_back(cur_span);
		}
	}

	if (DEBUG_LEVEL >= 2) {
		visualize_spans(name, small, pagemask, spans);
	}
}

void generate_candidate_edge(Edge& edge)
{
	if (edge.left->point0.x > edge.right->point1.x) {
		ContourInfo* tmp = edge.left;
		edge.left = edge.right;
		edge.right = tmp;
	}

	double x_overlap_left = edge.left->local_overlap(*edge.right);
	double x_overlap_right = edge.right->local_overlap(*edge.left);

	Point2f overall_tangent = edge.right->center - edge.left->center;
	double overall_angle = std::atan2(overall_tangent.y, overall_tangent.x);

	double delta_angle = std::max(angle_dist(edge.left->angle, overall_angle),
		angle_dist(edge.right->angle, overall_angle)) * 180 / M_PI;

	double x_overlap = std::max(x_overlap_left, x_overlap_right);

	double dist = cv::norm(edge.right->point0 - edge.left->point1);

	if (dist > EDGE_MAX_LENGTH ||
		x_overlap > EDGE_MAX_OVERLAP ||
		delta_angle > EDGE_MAX_ANGLE) {
		edge.score = -1;
	}
	else {
		edge.score = dist + delta_angle * EDGE_ANGLE_COST;
	}
}

double angle_dist(double angle_right, double angle_left)
{
	double diff = angle_right - angle_left;

	while (diff > M_PI) {
		diff -= 2 * M_PI;
	}

	while (diff < -M_PI) {
		diff += 2 * M_PI;
	}

	return std::abs(diff);
}

void sample_spans(const Size& shape,
	vector<vector<ContourInfo*>>& spans,
	vector<vector<Point2d>>& span_pts)
{
	for (const vector<ContourInfo*>& span : spans) {

		vector<Point2d> contour_pts;

		for (const ContourInfo* cinfo : span) {
			Mat totals;
			cinfo->mask.copyTo(totals);
			for (int i = 0; i < cinfo->mask.rows; i++) {
				totals.row(i) *= i;
			}

			cv::reduce(totals, totals, 0, cv::REDUCE_SUM, CV_64F);

			Mat col_sum;
			cv::reduce(cinfo->mask, col_sum, 0, cv::REDUCE_SUM, CV_64F);

			Mat means = totals / col_sum;

			int xmin = cinfo->rect.x;
			int ymin = cinfo->rect.y;
			int step = SPAN_PX_PER_STEP;
			int start = ((means.cols - 1) % step) / 2;

			for (int x = start; x < means.cols; x += step) {
				contour_pts.push_back(Point2d(x + xmin, means.at<double>(0, x) + ymin));
			}
		}
		pix2norm(shape, contour_pts);
		span_pts.push_back(contour_pts);
	}
}

void pix2norm(const Size& shape, vector<Point2d>& points)
{
	int height = shape.height;
	int width = shape.width;
	double scl = 2.0 / (std::max(height, width));

	Point2d offset(width * 0.5, height * 0.5);

	for (Point2d& point : points) {
		point = (point - offset) * scl;
	}
}

void keypoints_from_samples(string name,
	const Mat& small,
	const Mat& pagemask,
	const vector<Point>& page_outline,
	const vector<vector<Point2d>>& span_pts,
	vector<Point2d>& corners,
	vector<double>& y_coords,
	vector<vector<double>>& x_coords)
{
	Point2d all_evecs(0, 0);
	double all_weights = 0;

	for (const vector<Point2d>& pts : span_pts) {
		Mat pts_stack = Mat(pts).reshape(1);
		PCA pca_analysis(pts_stack, Mat(), PCA::DATA_AS_ROW, 1);

		double weight = cv::norm(pts.back() - pts[0]);
		all_evecs += Point2d(pca_analysis.eigenvectors) * weight;
		all_weights += weight;
	}

	Point2d x_dir = all_evecs / all_weights;

	if (x_dir.x < 0) {
		x_dir = -x_dir;
	}

	Point2d y_dir(-x_dir.y, x_dir.x);

	vector<Point> hull;
	cv::convexHull(page_outline, hull);

	vector<Point2d> page_coords;
	for (const Point& point : hull) {
		page_coords.push_back(Point2d(point));
	}

	pix2norm(pagemask.size(), page_coords);

	vector<double> px_coords, py_coords;
	for (int i = 0; i < page_coords.size(); i++) {
		px_coords.push_back(page_coords[i].dot(x_dir));
		py_coords.push_back(page_coords[i].dot(y_dir));
	}

	double px0 = *std::min_element(px_coords.begin(), px_coords.end());
	double px1 = *std::max_element(px_coords.begin(), px_coords.end());
	double py0 = *std::min_element(py_coords.begin(), py_coords.end());
	double py1 = *std::max_element(py_coords.begin(), py_coords.end());

	corners.push_back(px0 * x_dir + py0 * y_dir);	// p00
	corners.push_back(px1 * x_dir + py0 * y_dir);	// p10
	corners.push_back(px1 * x_dir + py1 * y_dir);	// p11
	corners.push_back(px0 * x_dir + py1 * y_dir);	// p01

	for (const vector<Point2d>& pts : span_pts) {
		vector<double> py_coords, px_coords;
		for (const Point2d& pt : pts) {
			py_coords.push_back(pt.dot(y_dir));
			px_coords.push_back(pt.dot(x_dir) - px0);
		}
		double mean = std::accumulate(py_coords.begin(), py_coords.end(), 0.0) / py_coords.size();
		y_coords.push_back(mean - py0);
		x_coords.push_back(px_coords);
	}

	if (DEBUG_LEVEL >= 2) {
		visualize_span_points(name, small, span_pts, corners);
	}
}

void get_default_params(const vector<Point2d>& corners,
	vector<Point2d>& rough_dims,
	const vector<double>& y_coords,
	const vector<vector<double>>& x_coords,
	Params& p)
{
	double page_width = cv::norm(corners[1] - corners[0]);
	double page_height = cv::norm(corners.back() - corners[0]);
	rough_dims.push_back(Point2d(page_width, page_height));

	vector<Point3d> corners_object3d{
		Point3d(0, 0, 0),
		Point3d(page_width, 0, 0),
		Point3d(page_width, page_height, 0),
		Point3d(0, page_height, 0)
	};

	// rvec 값이 부정확한 것 같음..
	Mat rvec, tvec;
	cv::solvePnP(corners_object3d, corners, K, Mat(), rvec, tvec);

	p.set(rvec, tvec, { 0, 0 }, y_coords, x_coords);
}

void stack_points(const vector<Point2d>& corners,
	const vector<vector<Point2d>>& span_pts,
	int npts,
	vector<Point2d>& dst_pts)
{
	dst_pts.resize(1 + npts);
	dst_pts[0] = corners[0];

	int i = 1;
	for (const auto& span : span_pts) {
		for (const auto& pt : span) {
			dst_pts[i++] = pt;
		}
	}
}

void optimize_params(string name,
	const Mat& small,
	const vector<Point2d>& dst_pts,
	Params& p)
{
	using namespace chrono;

	system_clock::time_point start = system_clock::now();
	praxis(0.001, 0.001, p.size, 0, p.data, dst_pts, p, objective);
	system_clock::time_point end = system_clock::now();
	duration<double> sec = end - start;
	cout << "optimization(params) took " << sec.count() << "sec." << endl;
}

double objective(double* dta, int s, const vector<Point2d>& dst_pts, Params& p)
{
	int size = p.npts + 1;

	vector<double> x(size, 0);
	vector<double> y(size, 0);

	std::copy(dta + Y_BEGIN + p.nspans, dta + Y_BEGIN + p.nspans + p.npts, begin(x) + 1);

	int k = 1;
	for (int i = 0; i < p.span_cnts.size(); i++) {
		double temp = (dta + Y_BEGIN)[i];
		for (int j = 0; j < p.span_cnts[i]; j++) {
			y[k++] = temp;
		}
	}

	vector<Point2d> img_pts;
	project_xy(x, y, dta, img_pts);

	return sum_squared_error(dst_pts, img_pts);
}

void project_xy(const vector<double>& x,
	const vector<double>& y,
	double* params,
	vector<Point2d>& img_pts)
{
	double alpha = params[SLOPES_BEGIN];
	double beta = params[SLOPES_BEGIN + 1];

	vector<double> poly{ alpha + beta, -2 * alpha - beta, alpha, 0 };

	vector<double> z(x.size());
	polyval(poly, x, z);

	vector<Point3d> xyz(x.size());
	concat_xyz(x, y, z, xyz);

	cv::Mat rvec(1, 3, CV_64F, params + RVEC_BEGIN);
	cv::Mat tvec(1, 3, CV_64F, params + TVEC_BEGIN);

	cv::projectPoints(xyz, rvec, tvec, K, cv::Mat(), img_pts);
}

void polyval(const vector<double>& coeffs,
	const vector<double>& values,
	vector<double>& results)
{
	for (int i = 0; i < values.size(); i++) {
		double val = values[i];
		double res = coeffs[0];
		for (int j = 1; j < coeffs.size(); j++) {
			res *= val;
			res += coeffs[j];
		}
		results[i] = res;
	}
}

void concat_xyz(const vector<double>& x,
	const vector<double>& y,
	const vector<double>& z,
	vector<Point3d>& xyz)
{
	for (int i = 0; i < x.size(); i++) {
		xyz[i] = Point3d{ x[i], y[i], z[i] };
	}
}

double sum_squared_error(const vector<Point2d>& dst_pts,
	const vector<Point2d>& img_pts)
{
	double sum = 0.0;
	for (int i = 0; i < img_pts.size(); i++) {
		double x_diff = dst_pts[i].x - img_pts[i].x;
		double y_diff = dst_pts[i].y - img_pts[i].y;
		sum += x_diff * x_diff + y_diff * y_diff;
	}
	return sum;
}

void get_page_dims(const vector<Point2d>& corners, Params& p, double* page_dims)
{
	using namespace chrono;

	vector<Point2d> dst_br(1, corners[2]);

	system_clock::time_point start = system_clock::now();
	praxis(0.001, 0.001, 2, 0, page_dims, dst_br, p, objective2);
	system_clock::time_point end = system_clock::now();
	duration<double> sec = end - start;
	cout << "optimization(dims) took " << sec.count() << "sec." << endl;
	cout << "got page dims " << page_dims[0] << " x " << page_dims[1] << endl;
}

double objective2(double* dims, int size, const vector<Point2d>& dst_br, Params& p)
{
	vector<double> x{ dims[0] };
	vector<double> y{ dims[1] };

	vector<Point2d> proj_br;
	project_xy(x, y, p.data, proj_br);

	return sum_squared_error(dst_br, proj_br);
}

void remap_image(string name, const Mat& img, const Mat& small, const double* page_dims, const Params& p)
{
	double temp = 0.5 * page_dims[1] * OUTPUT_ZOOM * img.rows;
	int height = round_nearest_multiple(temp, REMAP_DECIMATE);
	int width = round_nearest_multiple(height * page_dims[0] / page_dims[1], REMAP_DECIMATE);

	cout << " output will be " << width << " x " << height << endl;

	int height_small = height / REMAP_DECIMATE;
	int width_small = width / REMAP_DECIMATE;

	vector<double> page_x_range = linspace(0, page_dims[0], width_small);
	vector<double> page_y_range = linspace(0, page_dims[1], height_small);

	vector<double> x_coords;
	vector<double> y_coords;

	meshgrid(page_x_range, page_y_range, x_coords, y_coords);

	vector<Point2d> img_pts;
	project_xy(x_coords, y_coords, p.data, img_pts);

	norm2pix(img.size(), img_pts);

	Mat img_x_coords(height_small, width_small, CV_64F);
	Mat img_y_coords(height_small, width_small, CV_64F);

	reshape(img_pts, height_small, width_small, img_x_coords, img_y_coords);

	cv::resize(img_x_coords, img_x_coords, Size(width, height), 0.0, 0.0, INTER_CUBIC);
	cv::resize(img_y_coords, img_y_coords, Size(width, height), 0.0, 0.0, INTER_CUBIC);

	Mat img_gray;
	cv::cvtColor(img, img_gray, COLOR_RGB2GRAY);

	img_x_coords.convertTo(img_x_coords, CV_32FC1);
	img_y_coords.convertTo(img_y_coords, CV_32FC1);

	Mat remapped;
	cv::remap(img_gray, remapped, img_x_coords, img_y_coords, INTER_CUBIC, BORDER_REPLICATE);

	Mat thresh;
	cv::adaptiveThreshold(remapped, thresh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, ADAPTIVE_WINSZ, 25);

	cv::imwrite(name + "_thresh.png", thresh);
}

int round_nearest_multiple(double num, int factor)
{
	int i = (int)num;
	int remain = i % factor;
	if (remain == 0) {
		return i;
	}
	else {
		return i + factor - remain;
	}
}

vector<double> linspace(double a, double b, int N)
{
	vector<double> xs(N);
	double h = (b - a) / (N - 1);

	vector<double>::iterator x;
	double val;
	for (x = xs.begin(), val = a; x != xs.end(); x++, val += h) {
		*x = val;
	}
	return xs;
}

void meshgrid(const vector<double>& xs,
	const vector<double>& ys,
	vector<double>& x_coords,
	vector<double>& y_coords)
{
	int nx = xs.size();
	int ny = ys.size();
	int size = nx * ny;

	x_coords.resize(size);
	y_coords.resize(size);

	for (int i = 0; i < ny; i++) {
		auto dst = begin(x_coords) + nx * i;
		std::copy(begin(xs), end(xs), dst);
		double y = ys[i];
		for (int j = 0; j < nx; j++) {
			y_coords[j + nx * i] = y;
		}
	}
}

void reshape(const vector<Point2d>& img_pts, int height, int width,
	Mat& img_x_coords, Mat& img_y_coords)
{
	int k = 0;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			img_x_coords.at<double>(i, j) = img_pts[k].x;
			img_y_coords.at<double>(i, j) = img_pts[k].y;
			k++;
		}
	}
}