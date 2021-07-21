#pragma once

#include <stdio.h>
#include <time.h>
#include <iostream>
#include <thread>
#include <queue>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define m_threshold 125
#define outOfRange(x, max) (x < 0 || x >= max)

struct Coor
{
	int x_resolution, y_resolution, z_resolution;
	double x_min, dx, y_min, dy, z_min, dz;

	Coor(int X_resolution = 10, double X_min = 0.0, double X_max = 0.0,
		int Y_resolution = 10, double Y_min = 0.0, double Y_max = 0.0,
		int Z_resolution = 10, double Z_min = 0.0, double Z_max = 0.0) :
		x_resolution(X_resolution), y_resolution(Y_resolution), z_resolution(Z_resolution),
		x_min(X_min), dx((X_max - X_min) / (double)X_resolution),
		y_min(Y_min), dy((Y_max - Y_min) / (double)Y_resolution),
		z_min(Z_min), dz((Z_max - Z_min) / (double)Z_resolution) {}
	__device__ int at(int x, int y, int z) { return z * x_resolution * y_resolution + y * x_resolution + x; }
};

struct XYZ
{
	int x, y, z;
};

class Model
{
private:
	int CameraNum, neiborSize, point_count;
	Eigen::Matrix<float, 3, 4>* host_projection, * dev_projection;
	cv::Mat* host_image, *coloured_image;
	cv::cuda::GpuMat* dev_image;
	cv::cuda::PtrStepSz<uchar>* host_ptr, * dev_ptr;
	bool* host_voxel, * host_surface;
	bool* dev_voxel, * dev_surface;
	float* host_normalx, * host_normaly, * host_normalz;
	float* dev_normalx, * dev_normaly, * dev_normalz;
	Coor coor;
	void getNormal();
	cv::Vec3f getColour(double coorX, double coorY, double coorZ);
	void findfisrstPoint(int& x, int& y, int& z);

public:
	Model(int cm, int cx, int cy, int cz);
	~Model();
	void loadMatrix(const char* pFileName);
	void loadImage(const char* pDir, const char* pPrefix, const char* pSuffix);
	void getModel();
	void saveModel(const char* pFileName);
	void saveModelWithNormal(const char* pFileName);
	void saveModelWithNormal_CUDA(const char* pFileName);
	Eigen::Vector3f Model::getNormal(int indX, int indY, int indZ);
	void saveVoxel(const char* pFileName);
	friend __global__ void kernel_getModel(int CameraNum, bool* dev_voxel, Eigen::Matrix<float, 3, 4>* dev_projection, cv::cuda::PtrStepSz<uchar>* image, Coor coor);
	friend __device__ bool checkRange(double coorX, double coorY, double coorZ, Eigen::Matrix<float, 3, 4> projection, cv::cuda::PtrStepSz<uchar> image);
	friend __global__ void kernel_getSurface(bool* dev_voxel, bool* dev_surface, Coor coor);
	friend __global__ void kernel_getNormal(float* dev_normalx, float* dev_normaly, float* dev_normalz, bool* dev_voxel, bool* dev_surface, int neiborSize, Coor coor);
	void loadColourImage(const char* pDir, const char* pPrefix, const char* pSuffix);
	void Model::saveColouredPly(const char* pFileName);
	void Model::surface();
};
