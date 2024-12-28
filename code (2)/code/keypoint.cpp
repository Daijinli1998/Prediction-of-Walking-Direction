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
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>

using namespace std;
using namespace rs2;
using namespace cv;
using namespace pcl;
using pc_ptr = PointCloud<PointXYZ>::Ptr;
using pc = PointCloud<PointXYZ>;

const string bagName("555");
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

	device device = pipe.get_active_profile().get_device();
	playback pb = device.as<playback>();
	pb.set_playback_speed(0.1);
	pb.set_status_changed_callback(stausChanged);

	colorizer colorizer;

	pc_ptr passThroughXCloudPtr(new pc);
	pc_ptr passThroughYCloudPtr(new pc);
	pc_ptr passThroughZCloudPtr(new pc);
	pc_ptr voxelGridCloudPtr(new pc);
	pc_ptr removeOutlierCloudPtr(new pc);
	pc_ptr showCloudPtr(new pc);
	pc_ptr centroidCloudPtr(new pc);
	pc_ptr thighCloudPtr(new pc);
	pc_ptr kneeCloudPtr(new pc);
	pc_ptr ankleCloudPtr(new pc);

	rs2::align align_to_depth(RS2_STREAM_DEPTH);

	pointcloud pcd;
	points ps;

	visualization::PCLVisualizer viewer;
	viewer.addPointCloud(showCloudPtr, "cloud");
	visualization::PointCloudColorHandlerCustom<PointXYZ> rgb_r(thighCloudPtr, 255, 0, 0);
	visualization::PointCloudColorHandlerCustom<PointXYZ> rgb_g(kneeCloudPtr, 0, 255, 0);
	visualization::PointCloudColorHandlerCustom<PointXYZ> rgb_b(ankleCloudPtr, 0, 0, 255);
	//viewer.addPointCloud(centroidCloudPtr,rgb, "centroid");
	viewer.addPointCloud(thighCloudPtr, rgb_r, "thigh");
	viewer.addPointCloud(kneeCloudPtr, rgb_g, "knee");
	viewer.addPointCloud(ankleCloudPtr, rgb_b, "ankle");
	while (play)
	{
		frameset _fst = pipe.wait_for_frames();

		passThroughXCloudPtr->clear();
		passThroughYCloudPtr->clear();
		passThroughZCloudPtr->clear();
		voxelGridCloudPtr->clear();
		showCloudPtr->clear();

		depth_frame depth = _fst.get_depth_frame();

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
		ps.filter(*passThroughYCloudPtr);

		ps.setInputCloud(passThroughYCloudPtr);
		ps.setFilterFieldName("y");
		ps.setFilterLimits(-0.19f, 0.5f);
		ps.filter(*voxelGridCloudPtr);

		VoxelGrid<PointXYZ> sor;
		sor.setInputCloud(voxelGridCloudPtr);
		sor.setLeafSize(0.05f, 0.02f, 0.005f);
		sor.filter(*removeOutlierCloudPtr);

		RadiusOutlierRemoval<pcl::PointXYZ> outrem;
		outrem.setInputCloud(removeOutlierCloudPtr);
		outrem.setRadiusSearch(0.08);
		outrem.setMinNeighborsInRadius(30);
		outrem.setKeepOrganized(false);
		outrem.filter(*showCloudPtr);

		/*Eigen::Vector4f centroid;
		compute3DCentroid(*showCloudPtr, centroid);
		centroidCloudPtr->points.emplace_back(PointXYZ(centroid[0], centroid[1], centroid[2]));*/
		/*PointXYZ min_p, max_p;
		getMinMax3D<PointXYZ>(*showCloudPtr, min_p, max_p);
		
		vector<int> ankle_index_list;
		vector<int> knee_index_list;
		vector<int> thigh_index_list;
		auto points = showCloudPtr->points;
		for (int i = 0; i < points.size(); i++)
		{
			if (points[i].y < (max_p.y - min_p.y) / 2 - 0.025)
				ankle_index_list.push_back(i);
			else if (points[i].y > (max_p.y - min_p.y) / 2 + 0.025)
				thigh_index_list.push_back(i);
			else
				knee_index_list.push_back(i);
		}

		copyPointCloud<PointXYZ>(*showCloudPtr, ankle_index_list, *ankleCloudPtr);
		copyPointCloud<PointXYZ>(*showCloudPtr, thigh_index_list, *thighCloudPtr);
		copyPointCloud<PointXYZ>(*showCloudPtr, knee_index_list, *kneeCloudPtr);*/

		viewer.updatePointCloud(showCloudPtr, "cloud");
		/*viewer.updatePointCloud(ankleCloudPtr, rgb_b,"ankle");
		viewer.updatePointCloud(kneeCloudPtr, rgb_g, "knee");
		viewer.updatePointCloud(thighCloudPtr, rgb_r, "thigh");*/
		//viewer.updatePointCloud(centroidCloudPtr, rgb,"centroid");
		viewer.spinOnce(1);

	}


}




