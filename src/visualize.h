#pragma once
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "contourinfo.h"
using namespace std;
using namespace cv;

string DEBUG_OUTPUT = "screen";        // file, screen, both
string WINDOW_NAME = "Dewarp";        // Window name for visualization

vector<Scalar> CCOLORS
{
    {255, 0, 0},
    {255, 63, 0},
    {255, 127, 0},
    {255, 191, 0},
    {255, 255, 0},
    {191, 255, 0},
    {127, 255, 0},
    {63, 255, 0},
    {0, 255, 0},
    {0, 255, 63},
    {0, 255, 127},
    {0, 255, 191},
    {0, 255, 255},
    {0, 191, 255},
    {0, 127, 255},
    {0, 63, 255},
    {0, 0, 255},
    {63, 0, 255},
    {127, 0, 255},
    {191, 0, 255},
    {255, 0, 255},
    {255, 0, 191},
    {255, 0, 127},
    {255, 0, 63}
};

/*----------------------------------------- function declaration -----------------------------------------*/

void norm2pix(const Size& shape, vector<Point2d>& points);
void debug_show(string name, double step, string text, const Mat& display);
void visualize_contours(string name, const Mat& small, const vector<ContourInfo>& cinfo_list);
void visualize_spans(string name, const Mat& small, const Mat& pagemask, const vector<vector<ContourInfo*>>& spans);
void visualize_span_points(string name, const Mat& small, const vector<vector<Point2d>>& span_points, const vector<Point2d>& corners);

/*----------------------------------------- function definition -----------------------------------------*/

void norm2pix(const Size& shape, vector<Point2d>& points)
{
    int height = shape.height;
    int width = shape.width;
    double scl = std::max(height, width) * 0.5;

    Point2d offset(width * 0.5, height * 0.5);

for (Point2d& point : points)
    {
        point = point * scl + offset;
    }
}

void debug_show(string name, double step, string text, const Mat& display)
{
    if (DEBUG_OUTPUT != "screen")
    {
        string filetext = text;
        std::replace(text.begin(), text.end(), ' ', '_');
        string outfile = name + "_debug_" + std::to_string(step) + '_' + filetext + ".png";
        cv::imwrite(outfile, display);
    }

    if (DEBUG_OUTPUT != "file")
    {
        Mat image;
        display.copyTo(image);

        Point location(16, image.rows - 16);

        cv::putText(image, text, location,
                    cv::FONT_HERSHEY_SIMPLEX, 1.0,
                    Scalar(0, 0, 0), 3, cv::LINE_AA);

        cv::putText(image, text, location,
                    cv::FONT_HERSHEY_SIMPLEX, 1.0,
                    Scalar(255, 255, 255), 1, cv::LINE_AA);

        cv::imshow(WINDOW_NAME, image);
        cv::waitKey();
    }
}

void visualize_contours(string name, const Mat& small, const vector<ContourInfo>& cinfo_list)
{
    Mat regions = Mat::zeros(small.size(), CV_8UC3);

    for (int i = 0; i < cinfo_list.size(); i++)
    {
        vector<vector<Point>> contour(1, cinfo_list[i].contour);
        cv::drawContours(regions, contour, 0,
                         CCOLORS[i % CCOLORS.size()], -1);
    }

    Mat mask;
    cv::reduce(regions.reshape(1, regions.total()), mask, 2, cv::REDUCE_MAX, CV_8UC1);
    mask = mask.reshape(0, regions.rows);
    //cv::threshold(mask, mask, 0, 255, cv::THRESH_BINARY);

    Mat inv_mask;
    cv::bitwise_not(mask, inv_mask);

    Mat display, black, colored;

    small.copyTo(display, inv_mask);
    small.copyTo(black, mask);
    regions.copyTo(colored, mask);

    display = display + black / 2 + colored / 2;

for (const auto& cinfo : cinfo_list)
    {
        cv::circle(display, cinfo.center, 3,
                   Scalar(255, 255, 255), 1, cv::LINE_AA);

        cv::line(display, cinfo.point0, cinfo.point1,
                 Scalar(255, 255, 255), 1, cv::LINE_AA);
    }

    debug_show(name, 3, "contours", display);
}

void visualize_spans(string name, const Mat& small, const Mat& pagemask, const vector<vector<ContourInfo*>>& spans)
{
    Mat regions = Mat::zeros(small.size(), CV_8UC3);

    for (int i = 0; i < spans.size(); i++)
    {
        vector<vector<Point>> contours;
for (const auto& cinfo : spans[i])
        {
            contours.push_back(cinfo->contour);
        }
        cv::drawContours(regions, contours, -1, CCOLORS[i * 3 % CCOLORS.size()], -1);
    }

    Mat mask;
    cv::reduce(regions.reshape(1, regions.total()), mask, 2, cv::REDUCE_MAX, CV_8UC1);
    mask = mask.reshape(0, regions.rows);

    Mat inv_mask;
    cv::bitwise_not(mask, inv_mask);

    Mat display, black, colored;

    small.copyTo(display, inv_mask);
    small.copyTo(black, mask);
    regions.copyTo(colored, mask);

    display = display + black / 2 + colored / 2;

    debug_show(name, 2, "spans", display);
}

void visualize_span_points(string name, const Mat& small, const vector<vector<Point2d>>& span_points, const vector<Point2d>& corners)
{
    Mat display;
    small.copyTo(display);

    for (int i = 0; i < span_points.size(); i++)
    {
        vector<Point2d> points = span_points[i];

        norm2pix(small.size(), points);

        Mat pts_stack = Mat(points).reshape(1);
        PCA pca_analysis(pts_stack, Mat(), PCA::DATA_AS_ROW, 1);

        Point2d mean = Point2d(pca_analysis.mean);
        Point2d evec = Point2d(pca_analysis.eigenvectors);

        vector<double> dps;
for (Point2d& point : points)
        {
            dps.push_back(point.dot(evec));
        }

        double dpm = mean.dot(evec);

        Point2d point0 = mean + evec * (*std::min_element(dps.begin(), dps.end()) - dpm);
        Point2d point1 = mean + evec * (*std::max_element(dps.begin(), dps.end()) - dpm);

for (const Point2d& point : points)
        {
            cv::circle(display, Point(point), 3,
                       CCOLORS[i % CCOLORS.size()], -1, cv::LINE_AA);
        }

        cv::line(display, Point(point0), Point(point1),
                 Scalar(255, 255, 255), 1, cv::LINE_AA);
    }

    debug_show(name, 3, "span points", display);
}