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

const string inputPlyFile("D:\\我的\\研途\\毕业论文研\\我\\数据\\论文点云\\直立2.ply");
const string outputDir("D:\\我的\\研途\\毕业论文研\\我\\数据\\论文点云\\直立2\\" );

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
	ofs << "x" << "," << "y" << "," << "z" << endl;
	for (int index = 0; index < cloud->points.size(); index++)
	{
		ofs << cloud->points[index].x << ","
			<< cloud->points[index].y << ","
			<< cloud->points[index].z;
		ofs << endl;
	}
	ofs.close();
}





visualization::PCLVisualizer viewer;

int main()
{

	CreateDirectory(outputDir.c_str(), NULL);

	pc_ptr passThroughXCloudPtr(new pc);
	pc_ptr passThroughYCloudPtr(new pc);
	pc_ptr passThroughZCloudPtr(new pc);
	pc_ptr voxelGridCloudPtr(new pc);
	pc_ptr removeOutlierCloudPtr(new pc);
	pc_ptr showCloudPtr(new pc);

	viewer.addPointCloud(showCloudPtr, "cloud");

	readCloudFromPlyFile(passThroughXCloudPtr, inputPlyFile);

	writePc2PlyFile(passThroughXCloudPtr, outputDir + "original.ply");
	writePc2CSVFile(passThroughXCloudPtr, outputDir + "original.csv");

	pcl::PassThrough<pcl::PointXYZ> tmp;
	tmp.setInputCloud(passThroughXCloudPtr);
	tmp.setFilterFieldName("x");
	tmp.setFilterLimits(-0.35f, 0.35f);
	tmp.filter(*passThroughZCloudPtr);

	tmp.setInputCloud(passThroughZCloudPtr);
	tmp.setFilterFieldName("z");
	tmp.setFilterLimits(-1.2f,0.0f);
	tmp.filter(*passThroughYCloudPtr);

	tmp.setInputCloud(passThroughYCloudPtr);
	tmp.setFilterFieldName("y");
	tmp.setFilterLimits(-0.30f, 0.5f);
	tmp.filter(*voxelGridCloudPtr);

	writePc2PlyFile(voxelGridCloudPtr, outputDir + "passThrough.ply");
	writePc2CSVFile(voxelGridCloudPtr, outputDir + "passThrough.csv");


	if (voxelGridCloudPtr->points.size() != 0)
	{
		VoxelGrid<PointXYZ> sor;
		sor.setInputCloud(voxelGridCloudPtr);
		sor.setLeafSize(0.05f, 0.02f, 0.01f);
		sor.filter(*removeOutlierCloudPtr);


		writePc2PlyFile(removeOutlierCloudPtr, outputDir + "voxelGrid.ply");
		writePc2CSVFile(removeOutlierCloudPtr, outputDir + "voxelGrid.csv");


		RadiusOutlierRemoval<pcl::PointXYZ> outrem;
		outrem.setInputCloud(removeOutlierCloudPtr);
		outrem.setRadiusSearch(0.08);
		outrem.setMinNeighborsInRadius(25);
		outrem.setKeepOrganized(true);
		outrem.filter(*showCloudPtr);

		writePc2PlyFile(showCloudPtr, outputDir + "removeOutlier.ply");
		writePc2CSVFile(showCloudPtr, outputDir + "removeOutlier.csv");

		viewer.updatePointCloud(showCloudPtr, "cloud");
		viewer.spin();
		
	}

}




