#include <iostream>
#include <pcl/visualization/pcl_visualizer.h>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include <pcl/filters/voxel_grid.h>
#include "../cv_helper.hpp"
#include <pcl/filters/passthrough.h>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <queue>
#include <fstream>
#include <Windows.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/radius_outlier_removal.h>
#include  <opencv2/imgcodecs.hpp>


using namespace std;
using namespace rs2;
using namespace cv;
using namespace pcl;
using pc_ptr = PointCloud<PointXYZ>::Ptr;
using pc = PointCloud<PointXYZ>;

const string bagName("555");
const string from("D:\\IntelRealSenseBag\\" + bagName + ".bag");
const string toPly("D:\\我的\\研途\\毕业论文研\\我\\数据\\" + bagName + "\\ply\\");
const string toCsv("D:\\我的\\研途\\毕业论文研\\我\\数据\\" + bagName + "\\csv\\");
const string toImg("D:\\我的\\研途\\毕业论文研\\我\\数据\\" + bagName + "\\img\\");

typedef struct {
	frameset fst;
	int index;
}Fst;

condition_variable mCond;
mutex mLock;
queue<Fst *> que;
bool needLock = false;
bool play = true;

int writePc2PlyFile(const pc_ptr &cloud, string fileName)
{
	if (io::savePLYFile<PointXYZ>(fileName, *cloud, true) == -1)
	{
		cout << "write file failed" << endl;
		system("pause");
		return -1;
	}
	return 0;
}

void writePc2CSVFile(const pc_ptr &cloud, string fileName)
{
	ofstream ofs;
	ofs.open(fileName, ios::out);
	ofs << "x" << "," << "y" << "," << "z"<< endl;
	for (int index = 0; index < cloud->points.size(); index++)
	{
		ofs << cloud->points[index].x << ","
			<< cloud->points[index].y << ","
			<< cloud->points[index].z;
		ofs << endl;
	}
	ofs.close();
}

void initCamera(shared_ptr<pcl::visualization::PCLVisualizer> viewer)
{
	viewer->setCameraPosition(0, 0, 0, 0, 0, 0, 0, 0, 1);
	viewer->setBackgroundColor(255, 0, 0);
	viewer->addCoordinateSystem(0.1);
	viewer->initCameraParameters();

}


void thread_run(void)
{
	rs2::align align_to_color(RS2_STREAM_COLOR);

	pointcloud pcd;
	points ps;

	pc_ptr passThroughXCloudPtr(new pc);
	pc_ptr passThroughYCloudPtr(new pc);
	pc_ptr passThroughZCloudPtr(new pc);
	pc_ptr voxelGridCloudPtr(new pc);
	pc_ptr removeOutlierCloudPtr(new pc);
	pc_ptr showCloudPtr(new pc);


	while (true)
	{
		if (needLock)
			continue;
		passThroughXCloudPtr->clear();
		passThroughYCloudPtr->clear();
		passThroughZCloudPtr->clear();
		voxelGridCloudPtr->clear();
		removeOutlierCloudPtr->clear();
		showCloudPtr->clear();

		unique_lock<mutex> lock(mLock);

		mCond.wait(lock, [=] {
			if (que.empty())
				return false;
			if (needLock)
				return false;
			return true;
			});

		Fst *fst = que.front();
		que.pop();
		lock.unlock();

		depth_frame depth = fst->fst.get_depth_frame();
		frame color = fst->fst.get_color_frame();
		imwrite(toImg + to_string(fst->index) + ".jpg", frame_to_mat(color));

		ps = pcd.calculate(depth);

		auto vertices = ps.get_vertices();
		for (int i = 0; i < ps.size(); i++)
		{
			if (!isnan(vertices[i].z))
			{
				if (vertices[i].x == 0 && vertices[i].y == 0 && vertices[i].z == 0)
					continue;
				PointXYZ point;
				point.x = vertices[i].x;
				point.y = -vertices[i].y;
				point.z = -vertices[i].z;
				passThroughXCloudPtr->points.emplace_back(point);
			}
		}

		passThroughXCloudPtr->width = passThroughXCloudPtr->points.size();
		passThroughXCloudPtr->height = 1;
		passThroughXCloudPtr->points.resize(passThroughXCloudPtr->width);
		passThroughXCloudPtr->is_dense = true;

		pcl::PassThrough<pcl::PointXYZ> ps;
		ps.setInputCloud(passThroughXCloudPtr);
		ps.setFilterFieldName("x");
		ps.setFilterLimits(-0.3f, 0.3f);
		ps.filter(*passThroughZCloudPtr);
		;
		ps.setInputCloud(passThroughZCloudPtr);
		ps.setFilterFieldName("z");
		ps.setFilterLimits(-1.3f, 0.0f);
		//ps.setFilterLimits(0.0f, 1.2f);
		ps.filter(*passThroughYCloudPtr);

		ps.setInputCloud(passThroughYCloudPtr);
		ps.setFilterFieldName("y");
		ps.setFilterLimits(-0.19f, 0.5f);
		ps.filter(*voxelGridCloudPtr);

		if (voxelGridCloudPtr->points.size() != 0)
		{
			VoxelGrid<PointXYZ> sor;
			sor.setInputCloud(voxelGridCloudPtr);
			sor.setLeafSize(0.05f, 0.02f, 0.01f);
			sor.filter(*removeOutlierCloudPtr);

			RadiusOutlierRemoval<pcl::PointXYZ> outrem;
			outrem.setInputCloud(removeOutlierCloudPtr);
			outrem.setRadiusSearch(0.08);
			outrem.setMinNeighborsInRadius(30);
			outrem.setKeepOrganized(true);
			outrem.filter(*showCloudPtr);

			writePc2CSVFile(showCloudPtr, toCsv + to_string(fst->index) + ".csv");
			writePc2PlyFile(showCloudPtr, toPly + to_string(fst->index) + ".ply");
		}
		else
			writePc2CSVFile(voxelGridCloudPtr, toCsv + to_string(fst->index) + ".csv");	
		
		delete fst;
	}

}

void stausChanged(rs2_playback_status status)
{
	if (status == RS2_PLAYBACK_STATUS_STOPPED)
		play = false;
}


int main()
{
	config cfg;
	cfg.enable_device_from_file(from);
	pipeline pipe;
	pipe.start(cfg);

	for(int i=0;i<10;i++)
		thread(thread_run).detach();

	device device = pipe.get_active_profile().get_device();
	playback pb = device.as<playback>();
	pb.set_playback_speed(0.1);
	pb.set_status_changed_callback(stausChanged);
	int index = 0;
	
	colorizer colorizer;

	while (play)
	{
		frameset _fst = pipe.wait_for_frames();
		Fst *fst = new Fst;
		fst->fst = _fst;
		fst->index = index++;
		cout << index << endl;
		needLock = true;
		unique_lock<mutex> lock(mLock);
		que.push(fst);
		lock.unlock();
		needLock = false;
		mCond.notify_all();

		//frame depth_color = _fst.get_depth_frame().apply_filter(colorizer);
		//cv::imshow("depth", frame_to_mat(depth_color));

		cv::waitKey(1);
	}

	while (true)
	{
		unique_lock<mutex> lock(mLock);
		if (que.empty())
		{
			lock.unlock();
			break;
		}
		lock.unlock();
		Sleep(500);
	}
	cout << "等待任务全部完成" << endl;
	Sleep(8000);
}




