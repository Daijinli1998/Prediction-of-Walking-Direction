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
#include <pcl/common/centroid.h>
#include <pcl/filters/radius_outlier_removal.h>

using namespace std;
using namespace rs2;
using namespace cv;
using namespace pcl;
using pc_ptr = PointCloud<PointXYZ>::Ptr;
using pc = PointCloud<PointXYZ>;

const string bagName("20230311_230129");
const string from("D:\\IntelRealSenseBag\\" + bagName + ".bag");
const string toPly("D:\\我的\\研途\\毕业论文研\\数据\\ply\\");
const string toCsv("D:\\我的\\研途\\毕业论文研\\数据\\csv\\");

typedef struct {
	frameset fst;
	int index;
}Fst;

condition_variable mCond;
mutex mLock;
queue<Fst*> que;
bool needLock = false;
bool play = true;

int writePc2PlyFile(const pc_ptr& cloud, string fileName)
{
	if (io::savePLYFile<PointXYZ>(fileName, *cloud, true) == -1)
	{
		cout << "write file failed" << endl;
		system("pause");
		return -1;
	}
	return 0;
}

void writePc2CSVFile(const pc_ptr& cloud, string fileName)
{
	ofstream ofs;
	ofs.open(fileName, ios::out);
	for (int index = 0; index < cloud->points.size(); index++)
	{
		ofs << cloud->points[index].x << ","
			<< cloud->points[index].y << ","
			<< cloud->points[index].z << ",";
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
	pc_ptr centroidCloudPtr(new pc);

	visualization::PCLVisualizer viewer;
	viewer.addPointCloud(showCloudPtr, "cloud");
	visualization::PointCloudColorHandlerCustom<PointXYZ> rgb(centroidCloudPtr, 255, 0, 0);
	//viewer.addPointCloud(centroidCloudPtr,rgb, "centroid");

	while (true)
	{
		passThroughXCloudPtr->clear();
		passThroughYCloudPtr->clear();
		passThroughZCloudPtr->clear();
		voxelGridCloudPtr->clear();
		showCloudPtr->clear();

		unique_lock<mutex> lock(mLock);

		mCond.wait(lock, [=] {
			if (que.empty())
				return false;
			return true;
			});

		Fst* fst = que.front();
		que.pop();
		lock.unlock();

		depth_frame depth = fst->fst.get_depth_frame();
		delete fst;
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
				point.y = vertices[i].y;
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
		ps.setFilterLimits(-1.0f, 0.0f);
		ps.filter(*passThroughYCloudPtr);

		ps.setInputCloud(passThroughYCloudPtr);
		ps.setFilterFieldName("y");
		ps.setFilterLimits(-0.15f,0.3f);
		ps.filter(*voxelGridCloudPtr);

		VoxelGrid<PointXYZ> sor;
		sor.setInputCloud(voxelGridCloudPtr);
		sor.setLeafSize(0.01f, 0.01f, 0.01f);
		sor.filter(*removeOutlierCloudPtr);

		RadiusOutlierRemoval<pcl::PointXYZ> outrem;
		outrem.setInputCloud(removeOutlierCloudPtr);
		outrem.setRadiusSearch(0.08);
		outrem.setMinNeighborsInRadius(35);
		outrem.setKeepOrganized(false);
		outrem.filter(*showCloudPtr);
		
		/*Eigen::Vector4f centroid;
		compute3DCentroid(*showCloudPtr, centroid);
		centroidCloudPtr->points.emplace_back(PointXYZ(centroid[0], centroid[1], centroid[2]));*/

		viewer.updatePointCloud(showCloudPtr,"cloud");
		//viewer.updatePointCloud(centroidCloudPtr, rgb,"centroid");
		viewer.spinOnce(100);
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

	thread(thread_run).detach();

	device device = pipe.get_active_profile().get_device();
	playback pb = device.as<playback>();
	pb.set_playback_speed(0.5);
	pb.set_status_changed_callback(stausChanged);
	int index = 0;

	colorizer colorizer;

	while (play)
	{
		frameset _fst = pipe.wait_for_frames();
		Fst* fst = new Fst;
		fst->fst = _fst;
		fst->index = index++;
		cout << index << endl;
		unique_lock<mutex> lock(mLock);
		que.push(fst);
		lock.unlock();
		mCond.notify_one();
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
	Sleep(1000);
}




