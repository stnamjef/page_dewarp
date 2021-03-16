#include "pagedewarp.h"

int main(int argc, char** argv)
{
    int optfind;
    if (argc < 2)
    {
        cout << "usage: " << argv[0] << " [options]";
        cout << " IMAGE1 [IMAGE2 ...]" << endl;
        cout << endl;
        cout << "[options]:" << endl;
        cout << "  -mx N   - margin X (default=" << PAGE_MARGIN_X << ")" << endl;
        cout << "  -my N   - margin Y (default=" << PAGE_MARGIN_Y << ")" << endl;
        cout << "  -m  N   - margin X and Y" << endl;
        cout << "  -z  N.N - zoom out (default=" << OUTPUT_ZOOM << ")" << endl;
        cout << "  -d  N   - DPI out (default=" << OUTPUT_DPI << ")" << endl;
        cout << "  -r  N   - remap decimate (default=" << REMAP_DECIMATE << ")" << endl;
        cout << "  -tw N   - text min width (default=" << TEXT_MIN_WIDTH << ")" << endl;
        cout << "  -th N   - text min height (default=" << TEXT_MIN_HEIGHT << ")" << endl;
        cout << "  -t  N   - text min width and height" << endl;
        cout << "  -ta N.N - text min aspect (default=" << TEXT_MIN_ASPECT << ")" << endl;
        cout << "  -tt N   - text max thickness (default=" << TEXT_MAX_THICKNESS << ")" << endl;
        cout << "  -eo N.N - edge max overlap (default=" << EDGE_MAX_OVERLAP << ")" << endl;
        cout << "  -el N.N - edge max length (default=" << EDGE_MAX_LENGTH << ")" << endl;
        cout << "  -ea N.N - edge angle cost (default=" << EDGE_ANGLE_COST << ")" << endl;
        cout << "  -em N.N - edge max angle (default=" << EDGE_MAX_ANGLE << ")" << endl;
        cout << "  -sw N   - span min width (default=" << SPAN_MIN_WIDTH << ")" << endl;
        cout << "  -sp N   - span px per step (default=" << SPAN_PX_PER_STEP << ")" << endl;
        cout << "  -fl N.N - focal length (default=" << FOCAL_LENGTH << ")" << endl;
        cout << "  -ro     - remap only (no threshold)" << endl;
        exit(1);
    }

    for (int i = 1; i < argc; i++)
    {
        char* targ = argv[i];
        optfind = 0;
        if (i + 1 < argc)
        {
            if (strcmp(targ, "-mx") == 0)
            {
                i++;
                PAGE_MARGIN_X = atoi(argv[i]);
                optfind = 1;
                cout << "PAGE_MARGIN_X = " << PAGE_MARGIN_X << endl;
            }
            if (strcmp(targ, "-my") == 0)
            {
                i++;
                PAGE_MARGIN_Y = atoi(argv[i]);
                optfind = 1;
                cout << "PAGE_MARGIN_Y = " << PAGE_MARGIN_Y << endl;
            }
            if (strcmp(targ, "-m") == 0)
            {
                i++;
                PAGE_MARGIN_X = atoi(argv[i]);
                PAGE_MARGIN_Y = PAGE_MARGIN_X;
                optfind = 1;
                cout << "PAGE_MARGIN_X = " << PAGE_MARGIN_X << endl;
                cout << "PAGE_MARGIN_Y = " << PAGE_MARGIN_Y << endl;
            }
            if (strcmp(targ, "-z") == 0)
            {
                i++;
                OUTPUT_ZOOM = atof(argv[i]);
                optfind = 1;
                cout << "OUTPUT_ZOOM = " << OUTPUT_ZOOM << endl;
            }
            if (strcmp(targ, "-d") == 0)
            {
                i++;
                OUTPUT_DPI = atoi(argv[i]);
                optfind = 1;
                cout << "OUTPUT_DPI = " << OUTPUT_DPI << endl;
            }
            if (strcmp(targ, "-r") == 0)
            {
                i++;
                REMAP_DECIMATE = atoi(argv[i]);
                optfind = 1;
                cout << "REMAP_DECIMATE = " << REMAP_DECIMATE << endl;
            }
            if (strcmp(targ, "-tw") == 0)
            {
                i++;
                TEXT_MIN_WIDTH = atoi(argv[i]);
                optfind = 1;
                cout << "TEXT_MIN_WIDTH = " << TEXT_MIN_WIDTH << endl;
            }
            if (strcmp(targ, "-th") == 0)
            {
                i++;
                TEXT_MIN_HEIGHT = atoi(argv[i]);
                optfind = 1;
                cout << "TEXT_MIN_HEIGHT = " << TEXT_MIN_HEIGHT << endl;
            }
            if (strcmp(targ, "-t") == 0)
            {
                i++;
                TEXT_MIN_WIDTH = atoi(argv[i]);
                TEXT_MIN_HEIGHT = TEXT_MIN_WIDTH;
                optfind = 1;
                cout << "TEXT_MIN_WIDTH = " << TEXT_MIN_WIDTH << endl;
                cout << "TEXT_MIN_HEIGHT = " << TEXT_MIN_HEIGHT << endl;
            }
            if (strcmp(targ, "-ta") == 0)
            {
                i++;
                TEXT_MIN_ASPECT = atof(argv[i]);
                optfind = 1;
                cout << "TEXT_MIN_ASPECT = " << TEXT_MIN_ASPECT << endl;
            }
            if (strcmp(targ, "-tt") == 0)
            {
                i++;
                TEXT_MAX_THICKNESS = atoi(argv[i]);
                optfind = 1;
                cout << "TEXT_MAX_THICKNESS = " << TEXT_MAX_THICKNESS << endl;
            }
            if (strcmp(targ, "-eo") == 0)
            {
                i++;
                EDGE_MAX_OVERLAP = atof(argv[i]);
                optfind = 1;
                cout << "EDGE_MAX_OVERLAP = " << EDGE_MAX_OVERLAP << endl;
            }
            if (strcmp(targ, "-el") == 0)
            {
                i++;
                EDGE_MAX_LENGTH = atof(argv[i]);
                optfind = 1;
                cout << "EDGE_MAX_LENGTH = " << EDGE_MAX_LENGTH << endl;
            }
            if (strcmp(targ, "-ea") == 0)
            {
                i++;
                EDGE_ANGLE_COST = atof(argv[i]);
                optfind = 1;
                cout << "EDGE_ANGLE_COST = " << EDGE_ANGLE_COST << endl;
            }
            if (strcmp(targ, "-em") == 0)
            {
                i++;
                EDGE_MAX_ANGLE = atof(argv[i]);
                optfind = 1;
                cout << "EDGE_MAX_ANGLE = " << EDGE_MAX_ANGLE << endl;
            }
            if (strcmp(targ, "-sw") == 0)
            {
                i++;
                SPAN_MIN_WIDTH = atoi(argv[i]);
                optfind = 1;
                cout << "SPAN_MIN_WIDTH = " << SPAN_MIN_WIDTH << endl;
            }
            if (strcmp(targ, "-sp") == 0)
            {
                i++;
                SPAN_PX_PER_STEP = atoi(argv[i]);
                optfind = 1;
                cout << "SPAN_PX_PER_STEP = " << SPAN_PX_PER_STEP << endl;
            }
            if (strcmp(targ, "-fl") == 0)
            {
                i++;
                FOCAL_LENGTH = atof(argv[i]);
                optfind = 1;
                cout << "FOCAL_LENGTH = " << FOCAL_LENGTH << endl;
            }
        }
        if (strcmp(targ, "-ro") == 0)
        {
            THRES_FLG = !(THRES_FLG);
            optfind = 1;
            cout << "THRES_FLG = " << THRES_FLG << endl;
        }
        if (!optfind)
        {
            string name = argv[i];
            cout << name << endl;
            Mat img = cv::imread(name);
            Mat small;
            resize_to_screen(img, small);

            cout << "loaded " << name << " with size " << img.size;
            cout << " and resized to " << small.size << endl;

            Mat pagemask;
            vector<Point> page_outline;
            get_page_extents(small, pagemask, page_outline);

            vector<ContourInfo> cinfo_list;
            get_contours(name, small, pagemask, "text", cinfo_list);

            vector<vector<ContourInfo*>> spans;
            assemble_span(name, small, pagemask, cinfo_list, spans);

            if (spans.size() < 1)
            {
                cout << "skipping " << name << " because only ";
                cout << spans.size() << " spans" << endl;
            }

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

            double page_dims[2] { rough_dims[0].x, rough_dims[0].y };
            get_page_dims(corners, params, page_dims);

            remap_image(name, img, small, page_dims, params);
            if (THRES_FLG == 0)
            {
                cout << "wrote " << name + "_remap.png" << endl;
            } else {
                cout << "wrote " << name + "_thresh.png" << endl;
            }
        }
    }

    return 0;
}
