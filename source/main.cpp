#include <regex>
#include <iostream>
#include <filesystem>
#include "page_dewarp.h"
#include "thread_pool.h"
namespace fs = std::filesystem;

/*----------------------------------------- function declaration -----------------------------------------*/

void parse_arguments(int argc, char** argv, string& input_dir, string& output_dir, int& num_workers);
void check_and_create_directories(const string& input_dir, const string& output_dir);
void print_help();
void print_config(const string& input_dir, const string& output_dir, int num_workers);
int count_total_images(string image_dir);
void dewarp_image(const string& img_name, const string& in_path, const string& out_path);

/*----------------------------------------- function definition -----------------------------------------*/

int main(int argc, char** argv)
{
	// default arguments
	string input_dir = "./";
	string output_dir = "./dewarped";
	int num_workers = 1;

	// parse arguments
	parse_arguments(argc, argv, input_dir, output_dir, num_workers);

    // check if directories exist
    check_and_create_directories(input_dir, output_dir);

    // print configurations
    print_config(input_dir, output_dir, num_workers);

	// count the total number of images
	int num_images = count_total_images(input_dir);

	// prepare threads to work
	ThreadPool pool(num_images, num_workers);

	fs::directory_iterator iter(input_dir);
	while (iter != fs::end(iter)) {
		// get in_path & out_path
		string image_name = (*iter).path().filename().string();
		string in_path = (*iter).path().string();
		string out_path = output_dir + "/" + image_name;
		// start dewarping
		pool.enqueue_job(std::bind(dewarp_image, image_name, in_path, out_path));
		iter++;
	}
}

void parse_arguments(int argc, char** argv, string& input_dir, string& output_dir, int& num_workers)
{
	std::regex pattern("-(.*)=(.*)");

	for (int i = 1; i < argc; i++) {
		std::string arg = argv[i];
		std::smatch matches;
		if (std::regex_match(arg, matches, pattern)) {
			auto it = matches.begin();
			it++;
			if ((*it) == "idir") {
				it++;
                input_dir = *it;
			}
			else if ((*it) == "odir") {
				it++;
                output_dir = *it;
			}
			else if ((*it) == "nw") {
				it++;
				num_workers = std::stoi(*it);
				if (num_workers < 1) {
					cout << "--num_workers should be greater than 0" << endl;
					exit(1);
				}
			}
            else if ((*it) == "mx"){
                it++;
                PAGE_MARGIN_X = std::stoi(*it);
            }
            else if ((*it) == "my"){
                it++;
                PAGE_MARGIN_Y = std::stoi(*it);
            }
            else if ((*it) == "m"){
                it++;
                PAGE_MARGIN_X = std::stoi(*it);
                PAGE_MARGIN_Y = PAGE_MARGIN_X;
            }
            else if ((*it) == "z"){
                it++;
                OUTPUT_ZOOM = std::stof(*it);
            }
            else if ((*it) == "d"){
                it++;
                OUTPUT_DPI = std::stoi(*it);
            }
            else if ((*it) == "r"){
                it++;
                REMAP_DECIMATE = std::stoi(*it);
            }
            else if ((*it) == "tw"){
                it++;
                TEXT_MIN_WIDTH = std::stoi(*it);
            }
            else if ((*it) == "th"){
                it++;
                TEXT_MIN_HEIGHT = std::stoi(*it);
            }
            else if ((*it) == "t"){
                it++;
                TEXT_MIN_WIDTH = std::stoi(*it);
                TEXT_MIN_HEIGHT = TEXT_MIN_WIDTH;
            }
            else if ((*it) == "ta"){
                it++;
                TEXT_MIN_ASPECT = std::stof(*it);
            }
            else if ((*it) == "tt"){
                it++;
                TEXT_MAX_THICKNESS = std::stoi(*it);
            }
            else if ((*it) == "eo"){
                it++;
                EDGE_MAX_OVERLAP = std::stof(*it);
            }
            else if ((*it) == "el"){
                it++;
                EDGE_MAX_LENGTH = std::stof(*it);
            }
            else if ((*it) == "ea"){
                it++;
                EDGE_ANGLE_COST = std::stof(*it);
            }
            else if ((*it) == "em"){
                it++;
                EDGE_MAX_ANGLE = std::stof(*it);
            }
            else if ((*it) == "sw"){
                it++;
                SPAN_MIN_WIDTH = std::stoi(*it);
            }
            else if ((*it) == "sp"){
                it++;
                SPAN_PX_PER_STEP = std::stoi(*it);
            }
            else if ((*it) == "fl"){
                it++;
                FOCAL_LENGTH = std::stof(*it);
            }
            else if ((*it) == "ro") {
                it++;
                THRES_FLG = !(THRES_FLG);
            }
            else if ((*it) == "db") {
                it++;
                DEBUG_LEVEL = !(DEBUG_LEVEL);
                cout << "Warning: Parallel processing does not supported in debug mode." << endl;
            }
			else {
				cout << "Invalid arguments." << endl;
				print_help();
				exit(1);
			}
		}
		else {
			cout << "Argument expression does not match." << endl;
			print_help();
			exit(1);
		}
	}
}

void print_help()
{
    cout << "  -idir S   - input directory (default=NONE)" << endl;
    cout << "  -odir S   - output directory (default=./dewarped)" << endl;
    cout << "  -nw   N   - number of workers for parallel processing (default=1)" << endl;
    cout << "  -mx   N   - margin X (default=" << PAGE_MARGIN_X << ")" << endl;
    cout << "  -my   N   - margin Y (default=" << PAGE_MARGIN_Y << ")" << endl;
    cout << "  -m    N   - margin X and Y" << endl;
    cout << "  -z    N.N - zoom out (default=" << OUTPUT_ZOOM << ")" << endl;
    cout << "  -d    N   - DPI out (default=" << OUTPUT_DPI << ")" << endl;
    cout << "  -r    N   - remap decimate (default=" << REMAP_DECIMATE << ")" << endl;
    cout << "  -tw   N   - text min width (default=" << TEXT_MIN_WIDTH << ")" << endl;
    cout << "  -th   N   - text min height (default=" << TEXT_MIN_HEIGHT << ")" << endl;
    cout << "  -t    N   - text min width and height" << endl;
    cout << "  -ta   N.N - text min aspect (default=" << TEXT_MIN_ASPECT << ")" << endl;
    cout << "  -tt   N   - text max thickness (default=" << TEXT_MAX_THICKNESS << ")" << endl;
    cout << "  -eo   N.N - edge max overlap (default=" << EDGE_MAX_OVERLAP << ")" << endl;
    cout << "  -el   N.N - edge max length (default=" << EDGE_MAX_LENGTH << ")" << endl;
    cout << "  -ea   N.N - edge angle cost (default=" << EDGE_ANGLE_COST << ")" << endl;
    cout << "  -em   N.N - edge max angle (default=" << EDGE_MAX_ANGLE << ")" << endl;
    cout << "  -sw   N   - span min width (default=" << SPAN_MIN_WIDTH << ")" << endl;
    cout << "  -sp   N   - span px per step (default=" << SPAN_PX_PER_STEP << ")" << endl;
    cout << "  -fl   N.N - focal length (default=" << FOCAL_LENGTH << ")" << endl;
    cout << "  -ro       - remap only (no threshold)" << endl;
    cout << "  -db       - debug mode; an integer value of 0 or 1 (default=0)" << endl;
}

void check_and_create_directories(const string& input_dir, const string& output_dir)
{
    if (input_dir == "./") {
        cout << "Input directory must be entered." << endl;
        print_help();
        exit(1);
    }
    else if(!fs::exists(input_dir)) {
        cout << "Input directory (" << input_dir << ") ";
        cout << "does not exist." << endl;
        exit(1);
    }
    else if (!fs::exists(output_dir)) {
        cout << "Creating output directory (" << output_dir << ")" << endl;
        fs::create_directories(output_dir);
    }
    else {
        return;
    }
}

void print_config(const string& input_dir, const string& output_dir, int num_workers)
{
    cout << "User Configurations:" << endl;
    cout << "  -INPUT_DIR          = " << input_dir << endl;
    cout << "  -OUTPUT_DIR         = " << output_dir << endl;
    cout << "  -NUM_WORKERS        = " << num_workers << endl;
    cout << "  -PAGE_MARGIN_X      = " << PAGE_MARGIN_X << endl;
    cout << "  -PAGE_MARGIN_Y      = " << PAGE_MARGIN_Y << endl;
    cout << "  -OUTPUT_ZOOM        = " << OUTPUT_ZOOM << endl;
    cout << "  -OUTPUT_DPI         = " << OUTPUT_DPI << endl;
    cout << "  -REMAP_DECIMATE     = " << REMAP_DECIMATE << endl;
    cout << "  -TEXT_MIN_WIDTH     = " << TEXT_MIN_WIDTH << endl;
    cout << "  -TEXT_MIN_HEIGHT    = " << TEXT_MIN_HEIGHT << endl;
    cout << "  -TEXT_MIN_ASPECT    = " << TEXT_MIN_ASPECT << endl;
    cout << "  -TEXT_MAX_THICKNESS = " << TEXT_MAX_THICKNESS << endl;
    cout << "  -EDGE_MAX_OVERLAP   = " << EDGE_MAX_OVERLAP << endl;
    cout << "  -EDGE_MAX_LENGTH    = " << EDGE_MAX_LENGTH << endl;
    cout << "  -EDGE_ANGLE_COST    = " << EDGE_ANGLE_COST << endl;
    cout << "  -EDGE_MAX_ANGLE     = " << EDGE_MAX_ANGLE << endl;
    cout << "  -SPAN_MIN_WIDTH     = " << SPAN_MIN_WIDTH <<  endl;
    cout << "  -SPAN_PX_PER_STEP   = " << SPAN_PX_PER_STEP << endl;
    cout << "  -FOCAL_LENGTH       = " << FOCAL_LENGTH << endl;
    cout << "  -THRES_FLG          = " << THRES_FLG << endl;
    cout << "  -DEBUG_LEVEL        = " << DEBUG_LEVEL << endl;
}

int count_total_images(string image_dir)
{
	fs::directory_iterator iter(image_dir);
	return (int)std::distance(iter, fs::end(iter));
}

void dewarp_image(const string& img_name, const string& in_path, const string& out_path)
{
	Mat img = cv::imread(in_path);
	Mat small;
	resize_to_screen(img, small);

	// std::cout << "loaded " << name << " with size " << img.size;
	// std::cout << " and resized to " << small.size << endl;

	Mat pagemask;
	vector<Point> page_outline;
	get_page_extents(small, pagemask, page_outline);

	vector<ContourInfo> cinfo_list;
	get_contours(img_name, small, pagemask, "text", cinfo_list);

	vector<vector<ContourInfo*>> spans;
	assemble_span(img_name, small, pagemask, cinfo_list, spans);

	/*if (spans.size() < 1) {
		std::cout << "skipping " << name << " because only ";
		std::cout << spans.size() << " spans" << endl;
	}*/

	vector<vector<Point2d>> span_points;
	sample_spans(small.size(), spans, span_points);

	vector<Point2d> corners;
	vector<double> y_coords;
	vector<vector<double>> x_coords;
	keypoints_from_samples(img_name, small, pagemask, page_outline,
		span_points, corners, y_coords, x_coords);

	Params params;
	vector<Point2d> rough_dims;
	get_default_params(corners, rough_dims, y_coords, x_coords, params);

	vector<Point2d> dst_points;
	stack_points(corners, span_points, params.npts, dst_points);

	optimize_params(small, dst_points, params);

	double page_dims[2]{ rough_dims[0].x, rough_dims[0].y };
	get_page_dims(corners, params, page_dims);

	remap_image(out_path, img, small, page_dims, params);
}