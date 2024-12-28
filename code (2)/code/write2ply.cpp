#include <iostream>
#include <pcl/visualization/pcl_visualizer.h>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include "../cv_helper.hpp"
#include <thread>
#include <condition_variable>
#include <mutex>
#include <queue>

using namespace std;
using namespace pcl;


const string from("E:\\宇哥\\666.bag");
const string to("D:\\我的\\研途\\毕业论文研\\数据\\ply\\");


struct fst_st {
	frameset fst;
	int index;
};


static int index = 0;
queue<struct fst_st> m_que;
mutex m_lock;
condition_variable my_cond;


void thread_run(void)
{
	points ps;
	pointcloud pc;
	rs2::align align_to_color(RS2_STREAM_COLOR);

	PointCloud<PointXYZ>::Ptr baseCloudPtr(new PointCloud<PointXYZ>);

	while (true)
	{
		unique_lock<mutex> u_lock(m_lock);
		my_cond.wait(u_lock, [] {
			if (!m_que.empty())
				return true;
			return false;
			});

		struct fst_st fst = m_que.front();
		m_que.pop();
		u_lock.unlock();

		fst.fst = align_to_color.process(fst.fst);
		depth_frame depth = fst.fst.get_depth_frame();

		ps = pc.calculate(depth);

		const vertex* vtx = ps.get_vertices();
		baseCloudPtr->clear();
		for (int i = 0; i < ps.size(); i++)
		{
			if (!isnan(vtx[i].z))
			{
				PointXYZ point;
				point.x = vtx[i].x;
				point.y = vtx[i].y;
				point.z = vtx[i].z;
				baseCloudPtr->points.push_back(point);
			}
		}
		baseCloudPtr->is_dense = true;
		baseCloudPtr->width = baseCloudPtr->size();
		baseCloudPtr->height = 1;

		writeCloud2PlyFile(baseCloudPtr, to + std::to_string(fst.index) + ".ply");
	}
}

int main()
{
	config cfg;

	cfg.enable_device_from_file(from);

	pipeline pipe;
	pipe.start(cfg);

	rates_printer printer;

	thread t_1(thread_run);
	thread t_2(thread_run);
	thread t_3(thread_run);

	cout << "t_1 id : " << t_1.get_id() << endl;
	cout << "t_2 id : " << t_2.get_id() << endl;
	cout << "t_3 id : " << t_3.get_id() << endl;

	t_1.detach();
	t_2.detach();
	t_3.detach();

	device device = pipe.get_active_profile().get_device();
	playback playback = device.as<rs2::playback>();
	while (playback.current_status() != RS2_PLAYBACK_STATUS_STOPPED)
	{
		struct fst_st fst;
		fst.fst = pipe.wait_for_frames();

		fst.index = index++;

		unique_lock<mutex> u_lock(m_lock);
		m_que.push(fst);
		u_lock.unlock();
		my_cond.notify_one();

		frame color = fst.fst.get_color_frame();
		imshow("color", frame_to_mat(color));
		waitKey(1);
	}
}




int readCloudFromPlyFile(PointCloud<PointXYZ>::Ptr cloud, string fileName)
{
	if (io::loadPLYFile<PointXYZ>(fileName, *cloud) == -1)
	{
		cout << "read file failed" << endl;
		system("pause");
		return -1;
	}

	return 0;
}

int writeCloud2PlyFile(PointCloud<PointXYZ>::Ptr cloud, string fileName)
{
	if (io::savePLYFile<PointXYZ>(fileName, *cloud,true) == -1)
	{
		cout << "write file failed" << endl;
		system("pause");
		return -1;
	}
	return 0;
}