#pragma once
#include <functional>
#include <string>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

class ThreadPool
{
private:
	int count;
	int n_jobs;
	int n_threads;
	std::vector<std::thread> worker_threads;
	std::queue<std::function<void()>> jobs;
	std::condition_variable cv_job_q;
	std::mutex m_job_q;
	bool stop_all;
public:
	ThreadPool(int n_jobs, int n_threads);
	~ThreadPool();
	void enqueue_job(std::function<void()> job);
private:
	void worker_thread();
};

ThreadPool::ThreadPool(int n_jobs, int n_threads) :
	count(0), n_jobs(n_jobs), n_threads(n_threads), stop_all(false)
{
	worker_threads.reserve(n_threads);
	for (int i = 0; i < n_threads; i++) {
		worker_threads.emplace_back([this]() { this->worker_thread(); });
	}
}

void ThreadPool::worker_thread()
{
	while (true) {
		std::unique_lock<std::mutex> lock(m_job_q);
		cv_job_q.wait(lock, [this]() { return !this->jobs.empty() || stop_all; });
		if (stop_all && this->jobs.empty()) {
			return;
		}

		std::function<void()> job = std::move(jobs.front());
		jobs.pop();
		count++;
		cout << "Working: [" << count << '/' << n_jobs << "]\r";
		cout.flush();
		lock.unlock();

		job();
	}
}

ThreadPool::~ThreadPool()
{
	stop_all = true;
	cv_job_q.notify_all();
	for (auto& t : worker_threads) {
		t.join();
	}
	cout << endl << "Done." << endl;
}

void ThreadPool::enqueue_job(std::function<void()> job)
{
	if (stop_all) {
		throw std::runtime_error("ThreadPool stopped.");
	}

	{
		std::lock_guard<std::mutex> lock(m_job_q);
		jobs.push(std::move(job));
	}
	cv_job_q.notify_one();
}