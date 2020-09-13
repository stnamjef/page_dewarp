#pragma once
#include <iostream>
#include <vector>
#include <numeric>
using namespace std;

class Params
{
public:
	double* data;
	int nrvec;
	int ntvec;
	int nslps;
	int nspans;
	int npts;
	int size;
	vector<int> span_cnts;
public:
	Params();
	~Params();
	void set(const Mat& rvec,
		const Mat& tvec,
		const vector<double>& cubic_slopes,
		const vector<double>& ycoords,
		const vector<vector<double>>& xcoords);
};

Params::Params() :
	data(nullptr),
	nrvec(3),
	ntvec(3),
	nslps(2),
	nspans(0),
	npts(0),
	size(0) {}

Params::~Params() { delete[] data; }

void Params::set(const Mat& rvec,
	const Mat& tvec,
	const vector<double>& cubic_slopes,
	const vector<double>& ycoords,
	const vector<vector<double>>& xcoords)
{
	for (const auto x : xcoords) {
		span_cnts.push_back((int)x.size());
	}

	nspans = span_cnts.size();
	npts = std::accumulate(begin(span_cnts), end(span_cnts), 0);
	size = nrvec + ntvec + nslps + nspans + npts;

	data = new double[size];

	std::copy(rvec.begin<double>(), rvec.end<double>(), data);
	std::copy(tvec.begin<double>(), tvec.end<double>(), data + nrvec);
	std::copy(begin(cubic_slopes), end(cubic_slopes), data + nrvec + ntvec);
	std::copy(begin(ycoords), end(ycoords), data + nrvec + ntvec + nslps);

	int offset = nrvec + ntvec + nslps + nspans;
	for (int i = 0; i < span_cnts.size(); i++) {
		std::copy(begin(xcoords[i]), end(xcoords[i]), data + offset);
		offset += span_cnts[i];
	}
}