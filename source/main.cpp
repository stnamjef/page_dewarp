#include "page_dewarp.h"

int main()
{
	string name = "boston_cooking_a.jpg";
	Mat img = cv::imread(name);
	Mat small;
	resize_to_screen(img, small);

	Mat pagemask;
	vector<Point> page_outline;
	get_page_extents(small, pagemask, page_outline);

	vector<ContourInfo> cinfo_list;
	get_contours(name, small, pagemask, "text", cinfo_list);

	vector<vector<ContourInfo*>> spans;
	assemble_span(name, small, pagemask, cinfo_list, spans);

	vector<vector<Point2d>> span_points;
	sample_spans(small.size(), spans, span_points);

	vector<Point2d> corners;
	vector<double> y_coords;
	vector<vector<double>> x_coords;
	keypoints_from_samples(name, small, pagemask, page_outline,
		span_points, corners, y_coords, x_coords);

	Params params;
	vector<Point2d> rough_dims;
	get_default_params(corners, rough_dims, y_coords, x_coords, params);

	vector<Point2d> dst_points;
	stack_points(corners, span_points, params.npts, dst_points);

	optimize_params(name, small, dst_points, params);

	double page_dims[2]{ rough_dims[0].x, rough_dims[0].y };
	get_page_dims(corners, params, page_dims);

	remap_image(name, img, small, page_dims, params);

	return 0;
}