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
#include <Windows.h>

using namespace std;
using namespace rs2;
using namespace cv;
using namespace pcl;
using pc_ptr = PointCloud<PointXYZ>::Ptr;
using pc = PointCloud<PointXYZ>;

const string bagName("20221203_171606");
const string from("D:\\IntelRealSenseBag\\" + bagName + ".bag");
const string outputDir("D:\\我的\\研途\\毕业论文研\\我\\数据\\misc\\" + bagName + "\\");


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

	CreateDirectory(outputDir.c_str(), NULL);

	device device = pipe.get_active_profile().get_device();
	playback pb = device.as<playback>();
	pb.set_playback_speed(0.1);
	pb.set_status_changed_callback(stausChanged);
	int index = 0;
	
	colorizer colorizer;

	pc_ptr passThroughXCloudPtr(new pc);
	pc_ptr passThroughYCloudPtr(new pc);
	pc_ptr passThroughZCloudPtr(new pc);
	pc_ptr voxelGridCloudPtr(new pc);
	pc_ptr removeOutlierCloudPtr(new pc);
	pc_ptr showCloudPtr(new pc);

	while (play)
	{
		bool output = false;
		String outDir = outputDir + to_string(index) + "\\";

		frameset _fst = pipe.wait_for_frames();
		cout << index << endl;

		rs2::align align_to_color(RS2_STREAM_COLOR);

		pointcloud pcd;
		points ps;

		passThroughXCloudPtr->clear();
		passThroughYCloudPtr->clear();
		passThroughZCloudPtr->clear();
		voxelGridCloudPtr->clear();
		removeOutlierCloudPtr->clear();
		showCloudPtr->clear();

		depth_frame depth = _fst.get_depth_frame();
		frame color = _fst.get_color_frame();

		frame depth_color = _fst.get_depth_frame().apply_filter(colorizer);
		cv::imshow("depth", frame_to_mat(depth_color));
		cv::imshow("color", frame_to_mat(color));

		cout << "输入'a'将输出ply，img，csv到文件夹" << endl;
		char c = cv::waitKey(-1);
		if (c == 'a')
		{
			output = true;
			CreateDirectory(outDir.c_str(), NULL);
			imwrite(outDir + "color.jpg", frame_to_mat(color));
			imwrite(outDir + "depth.jpg", frame_to_mat(depth_color));
		}
		
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

		cout << passThroughXCloudPtr->points.size() << "size" << endl;

		if (output)
		{
			writePc2PlyFile(passThroughXCloudPtr, outDir + "original.ply");
			writePc2CSVFile(passThroughXCloudPtr, outDir + "original.csv");
		}

		pcl::PassThrough<pcl::PointXYZ> tmp;
		tmp.setInputCloud(passThroughXCloudPtr);
		tmp.setFilterFieldName("x");
		tmp.setFilterLimits(-0.3f, 0.3f);
		tmp.filter(*passThroughZCloudPtr);

		tmp.setInputCloud(passThroughZCloudPtr);
		tmp.setFilterFieldName("z");
		tmp.setFilterLimits(-1.2f, 0.0f);
		tmp.filter(*passThroughYCloudPtr);
		
		tmp.setInputCloud(passThroughYCloudPtr);
		tmp.setFilterFieldName("y");
		tmp.setFilterLimits(-0.26f, 0.263f);
		tmp.filter(*voxelGridCloudPtr);
		
		if (output) {
			writePc2PlyFile(voxelGridCloudPtr, outDir + "passThrough.ply");
			writePc2CSVFile(voxelGridCloudPtr, outDir + "passThrough.csv");
		}

		if (voxelGridCloudPtr->points.size() != 0)
		{
			VoxelGrid<PointXYZ> sor;
			sor.setInputCloud(voxelGridCloudPtr);
			sor.setLeafSize(0.05f, 0.02f, 0.01f);
			sor.filter(*removeOutlierCloudPtr);
		
			if (output) {
				writePc2PlyFile(removeOutlierCloudPtr, outDir + "voxelGrid.ply");
				writePc2CSVFile(removeOutlierCloudPtr, outDir + "voxelGrid.csv");
			}

			RadiusOutlierRemoval<pcl::PointXYZ> outrem;
			outrem.setInputCloud(removeOutlierCloudPtr);
			outrem.setRadiusSearch(0.08);
			outrem.setMinNeighborsInRadius(30);
			outrem.setKeepOrganized(true);
			outrem.filter(*showCloudPtr);

			if (output) {
				writePc2PlyFile(showCloudPtr, outDir + "removeOutlier.ply");
				writePc2CSVFile(showCloudPtr, outDir + "removeOutlier.csv");
			}
		}

		if (output) {
			cout << "已输出到" << outDir << endl;
			cout << "Done" << endl;
		}
		

		index++;
	}
}




