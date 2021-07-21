#include "kernel.h"

void Model::getModel()
{
	size_t size = coor.x_resolution * coor.y_resolution * coor.z_resolution * sizeof(bool);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Malloc
	cudaEventRecord(start);
	cudaMalloc((void**)&dev_voxel, size); //0.0023 ~ 0.0030 ms
	cudaMemset(dev_voxel, true, size);  //0.23ms
	cudaMalloc((void**)&dev_surface, size); //0.0023 ~ 0.0030 ms

	//Compute Config
	dim3 numBlocks(coor.x_resolution / 16 + 1, coor.y_resolution / 16 + 1);
	dim3 threadsPerBlock(16, 16);

	//Compute
	cudaEventRecord(start);
	kernel_getModel <<<numBlocks, threadsPerBlock>>> (CameraNum, dev_voxel, dev_projection, dev_ptr, coor); //4.70 ms
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("kernel_getModel() Time(Recorded by CUDA API): %f ms\n", milliseconds);
	cudaEventRecord(start);
	kernel_getSurface <<<numBlocks, threadsPerBlock>>> (dev_voxel, dev_surface, coor); //1.88 ms
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("kernel_getSurface() Time(Recorded by CUDA API): %f ms\n", milliseconds);

	//Copy results to CPU memory
	host_voxel = new bool[size / sizeof(bool)];
	host_surface = new bool[size / sizeof(bool)];
	cudaEventRecord(start);
	cudaMemcpy(host_surface, dev_surface, size, cudaMemcpyDeviceToHost); //7.75 ms ~ 3.48 GB/s
	cudaMemcpy(host_voxel, dev_voxel, size, cudaMemcpyDeviceToHost); //7.75 ms ~ 3.48 GB/s
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("cudaMemcpy() Time(Recorded by CUDA API): %f ms\n", milliseconds);

	//printf("Effective Bandwidth: %f GB/s\n", size * xxxx / milliseconds / 1e6);
}

void Model::getNormal()
{
	clock_t t = clock();
	size_t size = coor.x_resolution * coor.y_resolution * coor.z_resolution * sizeof(float);

	cudaMalloc((void**)&dev_normalx, size);
	cudaMalloc((void**)&dev_normaly, size);
	cudaMalloc((void**)&dev_normalz, size);

	dim3 numBlocks(coor.x_resolution / 16 + 1, coor.y_resolution / 16 + 1);
	dim3 threadsPerBlock(16, 16);
	kernel_getNormal << <numBlocks, threadsPerBlock >> > (dev_normalx, dev_normaly, dev_normalz, dev_voxel, dev_surface, neiborSize, coor);

	host_normalx = new float[size / sizeof(float)];
	host_normaly = new float[size / sizeof(float)];
	host_normalz = new float[size / sizeof(float)];
	cudaDeviceSynchronize();
	cudaMemcpy(host_normalx, dev_normalx, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_normaly, dev_normaly, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_normalz, dev_normalz, size, cudaMemcpyDeviceToHost);
	printf("getNormal(): %d ms\n", (clock() - t) * 1000 / CLOCKS_PER_SEC);
}
__device__ bool checkRange(double coorX, double coorY, double coorZ, Eigen::Matrix<float, 3, 4> projection, cv::cuda::PtrStepSz<uchar> image)
{
	Eigen::Vector3f vec3 = projection * Eigen::Vector4f(coorX, coorY, coorZ, 1);
	int indX = vec3[1] / vec3[2];
	int indY = vec3[0] / vec3[2];
	if (outOfRange(indX, image.rows) || outOfRange(indY, image.cols))
		return false;
	return image(indX, indY) > m_threshold;
}

__global__ void kernel_getModel(int CameraNum, bool* dev_voxel, Eigen::Matrix<float, 3, 4>* dev_projection, cv::cuda::PtrStepSz<uchar>* dev_ptr, Coor coor)
{
	//300 points per thread
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= coor.x_resolution || y >= coor.y_resolution) return;
	double coorX = coor.x_min + x * coor.dx;
	double coorY = coor.y_min + y * coor.dy;
	for (int z = 0; z < coor.z_resolution; ++z)
	{
		double coorZ = coor.z_min + z * coor.dz;
		for (int i = 0; i < CameraNum; ++i)
		{
			if (!checkRange(coorX, coorY, coorZ, dev_projection[i], dev_ptr[i]))
			{
				*(dev_voxel + coor.at(x, y, z)) = false;
				break;
			}
		}
	}
}

__global__ void kernel_getSurface(bool* dev_voxel, bool* dev_surface, Coor coor)
{
	//300 points per thread
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= coor.x_resolution || y >= coor.y_resolution) return;

	const int dx[6] = { -1, 0, 0, 0, 0, 1 };
	const int dy[6] = { 0, 1, -1, 0, 0, 0 };
	const int dz[6] = { 0, 0, 0, 1, -1, 0 };

	auto voutOfRange = [&](int indexX, int indexY, int indexZ) {
		return indexX < 0 || indexY < 0 || indexZ < 0
			|| indexX >= coor.x_resolution
			|| indexY >= coor.y_resolution
			|| indexZ >= coor.z_resolution;
	};

	for (int z = 0; z < coor.z_resolution; ++z)
	{
		if (!*(dev_voxel + coor.at(x, y, z)))
		{
			*(dev_surface + coor.at(x, y, z)) = false;
			continue;
		}
		/*
		//By adjusting the parameters,
		//we can change the number of surface points, thus optimising the speed and quality.
		int ans = 0;
		for (int i = 0; i < 6; i++)
			if(voutOfRange(x + dx[i], y + dy[i], z + dz[i]) || !*(dev_voxel + coor.at(x + dx[i], y + dy[i], z + dz[i]))) ans++;
		*(dev_surface + coor.at(x, y, z)) = (ans >= 2 && ans <= 5);
		*/
		bool ans = false;
		for (int i = 0; i < 6; i++)
			ans = ans || voutOfRange(x + dx[i], y + dy[i], z + dz[i])
			|| !*(dev_voxel + coor.at(x + dx[i], y + dy[i], z + dz[i]));
		*(dev_surface + coor.at(x, y, z)) = ans;
		
	}
}

__global__ void kernel_getNormal(float* dev_normalx, float *dev_normaly, float *dev_normalz, bool* dev_voxel, bool* dev_surface, int neiborSize, Coor coor)
{
	int indX = blockIdx.x * blockDim.x + threadIdx.x;
	int indY = blockIdx.y * blockDim.y + threadIdx.y;
	if (indX >= coor.x_resolution || indY >= coor.y_resolution) return;

	auto voutOfRange = [&](int indexX, int indexY, int indexZ) {
		return indexX < 0 || indexY < 0 || indexZ < 0
			|| indexX >= coor.x_resolution
			|| indexY >= coor.y_resolution
			|| indexZ >= coor.z_resolution;
	};
	
	Eigen::Vector3f neiborList[343] = { Eigen::Vector3f::Zero() };

	for (int indZ = 0; indZ < coor.z_resolution; ++indZ)
	{
		if (!*(dev_surface + coor.at(indX, indY, indZ))) continue;
		//Point Normal
		Eigen::Vector3f innerCenter = Eigen::Vector3f::Zero();
		int ncount = 0, icount = 0;
		for (int dX = -neiborSize; dX <= neiborSize; dX++)
			for (int dY = -neiborSize; dY <= neiborSize; dY++)
				for (int dZ = -neiborSize; dZ <= neiborSize; dZ++)
				{
					if (!dX && !dY && !dZ)
						continue;
					int neiborX = indX + dX;
					int neiborY = indY + dY;
					int neiborZ = indZ + dZ;
					if (!voutOfRange(neiborX, neiborY, neiborZ))
					{
						double coorX = coor.x_min + neiborX * coor.dx;
						double coorY = coor.y_min + neiborY * coor.dy;
						double coorZ = coor.z_min + neiborZ * coor.dz;
						if (*(dev_surface + coor.at(neiborX, neiborY, neiborZ)))
							neiborList[ncount++] = Eigen::Vector3f(coorX, coorY, coorZ);
						else if (*(dev_voxel + coor.at(neiborX, neiborY, neiborZ)))
						{
							innerCenter += Eigen::Vector3f(coorX, coorY, coorZ);
							icount++;
						}
					}
				}
		float coorX = coor.x_min + indX * coor.dx;
		float coorY = coor.y_min + indY * coor.dy;
		float coorZ = coor.z_min + indZ * coor.dz;
		Eigen::Vector3f point(coorX, coorY, coorZ);

		Eigen::MatrixXf matA(3, ncount);
		for (int i = 0; i < ncount; i++)
			matA.col(i) = neiborList[i] - point;
		Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigenSolver;
		eigenSolver.computeDirect(matA * matA.transpose());
		//Only computeDirect for 2x2 and 3x3 matrix will work with CUDA.
		//Reference: https://stackoverflow.com/questions/43820009/eigen-jacobisvd-cuda-compile-error
		Eigen::Vector3f eigenValues = eigenSolver.eigenvalues();
		int indexEigen = 0;
		if (abs(eigenValues[1]) < abs(eigenValues[indexEigen]))
			indexEigen = 1;
		if (abs(eigenValues[2]) < abs(eigenValues[indexEigen]))
			indexEigen = 2;
		Eigen::Vector3f normalVector = eigenSolver.eigenvectors().col(indexEigen);

		innerCenter /= icount;
		//Eigen::Vector3f innerCenter = Eigen::Vector3f::Zero();
		//for (auto const& vec : innerList)
		//	innerCenter += vec;
		//innerCenter /= innerList.size();

		if (normalVector.dot(point - innerCenter) < 0)
			normalVector *= -1;
		int offset = coor.at(indX, indY, indZ);
		*(dev_normalx + offset) = normalVector(0);
		*(dev_normaly + offset) = normalVector(1);
		*(dev_normalz + offset) = normalVector(2);
	}
}
