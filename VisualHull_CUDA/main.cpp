#pragma warning(disable:4819)
#pragma warning(disable:4244)
#pragma warning(disable:4267)

#include "kernel.h"

int main()
{
	//Test OpenCV with CUDA
	//std::cout << cv::getBuildInformation() << std::endl;
	cv::cuda::GpuMat test;
	test.create(1, 1, CV_8U);
	//The first call of any gpu function is slow due to CUDA context initialization.
	//All next calls will be faster. So I call some gpu function before time measurement.
	//Reference: https://stackoverflow.com/questions/19454373/too-slow-gpumat-uploading-of-an-small-image

	clock_t t0 = clock(), t;
	Model model(20, 300, 300, 300);

	//Load coloured images in the background
	auto getcolourimage = [&]() {
		model.loadColourImage("../wd_data", "WD2_", "_00020.png");
	};
	std::thread gcl(getcolourimage);

	t = clock();
	model.loadMatrix("../calibParamsI.txt");
	printf("Load matrices and copy them to GPU memory: %d ms\n", (clock() - t) * 1000 / CLOCKS_PER_SEC);

	t = clock();
	model.loadImage("../wd_segmented", "WD2_", "_00020_segmented.png");
	printf("Load images and copy them to GPU memory: %d ms\n", (clock() - t) * 1000 / CLOCKS_PER_SEC);

	//t = clock();
	model.getModel();
	//printf("Get model and surface, then copy them to CPU memory: %d ms\n", (clock() - t) * 1000 / CLOCKS_PER_SEC);

	t = clock();
	//Multithreading
	auto save_model = [&]() {
		clock_t t = clock();
		model.saveModel("../WithoutNormal.xyz");
		printf("Save model without normal: %d ms\n", (clock() - t) * 1000 / CLOCKS_PER_SEC);
	};
	auto poissonrecon = [&]() {
		clock_t t = clock();
		model.saveModelWithNormal_CUDA("../WithNormal.xyz");
		printf("Save model with normal: %d ms\n", (clock() - t) * 1000 / CLOCKS_PER_SEC);
		t = clock();
		system("PoissonRecon --in ../WithNormal.xyz --out ../mesh.ply");
		printf("Poisson Surface Reconstruction: %d ms\n", (clock() - t) * 1000 / CLOCKS_PER_SEC);
	};
	std::thread with_nm(poissonrecon);
	std::thread wout_nm(save_model);
	with_nm.join();
	wout_nm.join();
	printf("Save model and ply: %d ms\n", (clock() - t) * 1000 / CLOCKS_PER_SEC);

	t = clock() - t0;
	std::cout << "Total time: " << (float(t) / CLOCKS_PER_SEC) << "seconds\n";

	gcl.join();
	t = clock();
	model.saveColouredPly("../Coloured.ply");
	printf("\nmodel.saveColouredPly(): %d ms\n", (clock() - t) * 1000 / CLOCKS_PER_SEC);
	t = clock();
	system("PoissonRecon --in ../Coloured.ply --out ../ColouredMesh.ply --colors");
	printf("PoissonRecon (Coloured): %d ms\n", (clock() - t) * 1000 / CLOCKS_PER_SEC);
	printf("ColouredMesh.ply saved.\n");

	//model.saveVoxel("../Voxel.xyz");
	//model.surface();
	system("pause");
	return (0);
}

Model::Model(int cm, int resX, int resY, int resZ)
	:CameraNum(cm), coor(resX, -5, 5, resY, -10, 10, resZ, 15, 30), neiborSize((resX <= 100) ? 1 : resX / 100) {}

Model::~Model()
{
	cudaFree(dev_normalx);
	cudaFree(dev_normaly);
	cudaFree(dev_normalz);
	cudaFree(dev_voxel);
	cudaFree(dev_surface);
	cudaFree(dev_ptr);
	cudaFree(dev_projection);
	delete[] host_normalx;
	delete[] host_normaly;
	delete[] host_normalz;
	delete[] host_surface;
	delete[] host_voxel;
	delete[] host_ptr;
	delete[] dev_image;
	delete[] host_image;
	delete[] host_projection;
	//cudaFree(host_projection);
}

void Model::loadMatrix(const char* pFileName)
{
	//pageable memory
	host_projection = new Eigen::Matrix<float, 3, 4>[CameraNum];
	
	//Using cudaHostAllocMapped to read camera parameter.(It will copy the data automatically.)
	//cudaHostAlloc((void**)&host_projection, CameraNum * sizeof(Eigen::Matrix<float, 3, 4>), cudaHostAllocMapped);

	//pinned memory.The same with pageable memory...Maybe the data is too small...
	//cudaMallocHost((void**)&host_projection, CameraNum * sizeof(Eigen::Matrix<float, 3, 4>));

	//cudaHostAllocWriteCombined
	//cudaHostAlloc((void**)&host_projection, CameraNum * sizeof(Eigen::Matrix<float, 3, 4>), cudaHostAllocWriteCombined);
	

	FILE* fp = fopen(pFileName, "r");
	//std::ifstream fin(pFileName);
	int num = 0, k = 0;
	Eigen::Matrix<float, 3, 3> matInt;
	Eigen::Matrix<float, 3, 4> matExt;
	while (fscanf(fp, "%d", &num) != EOF)
		//while (fin >> num)
	{
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				//fin >> matInt(i, j);
				fscanf(fp, "%f", &matInt(i, j));

		float temp;
		//fin >> temp;
		//fin >> temp;
		fscanf(fp, "%f", &temp);
		fscanf(fp, "%f", &temp);

		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 4; j++)
				//fin >> matExt(i, j);
				fscanf(fp, "%f", &matExt(i, j));

		host_projection[k++] = matInt * matExt;
	}
	fclose(fp);
	cudaMalloc((void**)&dev_projection, CameraNum * sizeof(Eigen::Matrix<float, 3, 4>));
	cudaMemcpy(dev_projection, host_projection, CameraNum * sizeof(Eigen::Matrix<float, 3, 4>), cudaMemcpyHostToDevice);
}

void Model::loadImage(const char* pDir, const char* pPrefix, const char* pSuffix)
{
	//Array of GpuMats: https://answers.opencv.org/question/89050/passing-an-array-of-cvgpumat-to-a-cuda-kernel/
	host_image = new cv::Mat[CameraNum];
	dev_image = new cv::cuda::GpuMat[CameraNum];

	std::string fileName(pDir);
	fileName += '/';
	fileName += pPrefix;
	auto imr = [&](int start, int end) {
		for (int i = start; i < end; ++i)
		{
			host_image[i] = cv::imread(fileName + std::to_string(i) + pSuffix, CV_8UC1);
			dev_image[i].upload(host_image[i]);
		}
	};
	int thrn = 4;
	std::thread* imrt = new std::thread[thrn];
	for (int i = 0; i < thrn; ++i)
		imrt[i] = std::thread(imr, CameraNum * i / thrn, CameraNum * (i + 1) / thrn);
	for (int i = 0; i < thrn; ++i)
		imrt[i].join();
	delete[] imrt;

	host_ptr = new cv::cuda::PtrStepSz<uchar>[CameraNum];
	for (int i = 0; i < CameraNum; ++i)
		host_ptr[i] = dev_image[i];
	cudaMalloc(&dev_ptr, CameraNum * sizeof(cv::cuda::PtrStepSz<uchar>));
	cudaMemcpy(dev_ptr, host_ptr, CameraNum * sizeof(cv::cuda::PtrStepSz<uchar>), cudaMemcpyHostToDevice);
}
void Model::findfisrstPoint(int& x, int& y, int& z)
{
	for (; x < coor.x_resolution && y < coor.y_resolution && z < coor.z_resolution; ++x, ++y, ++z)
		if (*(host_surface + coor.at(x, y, z)))
			return;
	//If cannot find a point on the surface along the diagonal:
	if (x == coor.x_resolution || y == coor.y_resolution || z == coor.z_resolution)
	{
		for (int x = 0; x < coor.x_resolution; x++)
			for (int y = 0; y < coor.y_resolution; y++)
				for (int z = 0; z < coor.z_resolution; z++)
					if (*(host_surface + coor.at(x, y, z)))
						return;
	}
}
void Model::saveModel(const char* pFileName)
{
	const int dx[18] = { -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1 };
	const int dy[18] = { -1, 0, 0, 0, 1, -1, -1, -1, 0, 0, 1, 1, 1, -1, 0, 0, 0, 1 };
	const int dz[18] = { 0, -1, 0, 1, 0, -1, 0, 1, -1, 1, -1, 0, 1, 0 - 1, 0, 1, 0 };

	auto voutOfRange = [&](int indexX, int indexY, int indexZ) {
		return indexX < 0 || indexY < 0 || indexZ < 0
			|| indexX >= coor.x_resolution
			|| indexY >= coor.y_resolution
			|| indexZ >= coor.z_resolution;
	};

	FILE* fp = fopen(pFileName, "w");
	int x = 0, y = 0, z = 0;
	//Find a point on the surface:
	findfisrstPoint(x, y, z);
	//BFS:
	XYZ xyz{ x, y, z };
	std::queue<XYZ> que;
	que.push(xyz);
	point_count = 1;
	bool* vis = new bool[coor.x_resolution * coor.y_resolution * coor.y_resolution];
	memset(vis, true, coor.x_resolution * coor.y_resolution * coor.y_resolution * sizeof(bool));
	*(vis + coor.at(x, y, z)) = false;
	while (!que.empty())
	{
		xyz = que.front();
		que.pop();
		double coorX = coor.x_min + xyz.x * coor.dx;
		double coorY = coor.y_min + xyz.y * coor.dy;
		double coorZ = coor.z_min + xyz.z * coor.dz;
		fprintf(fp, "%lf %lf %lf\n", coorX, coorY, coorZ);
		for (int i = 0; i < 18; i++)
		{
			int tx = xyz.x + dx[i], ty = xyz.y + dy[i], tz = xyz.z + dz[i];
			if (!voutOfRange(tx, ty, tz) && *(vis + coor.at(tx, ty, tz))
				&& *(host_surface + coor.at(tx, ty, tz)))
			{
				que.push(XYZ{ tx, ty, tz });
				*(vis + coor.at(tx, ty, tz)) = false;
				++point_count;
			}
		}
	}
	fclose(fp);
	delete[] vis;
}

void Model::saveModelWithNormal(const char* pFileName)
{
	//Not used
	const int dx[18] = { -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1 };
	const int dy[18] = { -1, 0, 0, 0, 1, -1, -1, -1, 0, 0, 1, 1, 1, -1, 0, 0, 0, 1 };
	const int dz[18] = { 0, -1, 0, 1, 0, -1, 0, 1, -1, 1, -1, 0, 1, 0 - 1, 0, 1, 0 };

	auto voutOfRange = [&](int indexX, int indexY, int indexZ) {
		return indexX < 0 || indexY < 0 || indexZ < 0
			|| indexX >= coor.x_resolution
			|| indexY >= coor.y_resolution
			|| indexZ >= coor.z_resolution;
	};

	FILE* fp = fopen(pFileName, "w");
	int x = 0, y = 0, z = 0;
	//Find a point on the surface:
	findfisrstPoint(x, y, z);
	//BFS:
	XYZ xyz{ x, y, z };
	std::queue<XYZ> que;
	que.push(xyz);
	bool* vis = new bool[coor.x_resolution * coor.y_resolution * coor.y_resolution];
	memset(vis, true, coor.x_resolution * coor.y_resolution * coor.y_resolution * sizeof(bool));
	*(vis + coor.at(x, y, z)) = false;
	while (!que.empty())
	{
		xyz = que.front();
		que.pop();
		double coorX = coor.x_min + xyz.x * coor.dx;
		double coorY = coor.y_min + xyz.y * coor.dy;
		double coorZ = coor.z_min + xyz.z * coor.dz;
		fprintf(fp, "%lf %lf %lf ", coorX, coorY, coorZ);
		Eigen::Vector3f nor = getNormal(xyz.x, xyz.y, xyz.z);
		fprintf(fp, "%f %f %f\n", nor(0), nor(1), nor(2));
		for (int i = 0; i < 18; i++)
		{
			int tx = xyz.x + dx[i], ty = xyz.y + dy[i], tz = xyz.z + dz[i];
			if (!voutOfRange(tx, ty, tz) && *(vis + coor.at(tx, ty, tz))
				&& *(host_surface + coor.at(tx, ty, tz)))
			{
				que.push(XYZ{ tx, ty, tz });
				*(vis + coor.at(tx, ty, tz)) = false;
			}
		}
	}
	fclose(fp);
	delete[] vis;
}

void Model::saveModelWithNormal_CUDA(const char* pFileName)
{
	//Using GPU to accelerate the process of getting PointCloud Normal.
	getNormal();

	const int dx[18] = { -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1 };
	const int dy[18] = { -1, 0, 0, 0, 1, -1, -1, -1, 0, 0, 1, 1, 1, -1, 0, 0, 0, 1 };
	const int dz[18] = { 0, -1, 0, 1, 0, -1, 0, 1, -1, 1, -1, 0, 1, 0 - 1, 0, 1, 0 };

	auto voutOfRange = [&](int indexX, int indexY, int indexZ) {
		return indexX < 0 || indexY < 0 || indexZ < 0
			|| indexX >= coor.x_resolution
			|| indexY >= coor.y_resolution
			|| indexZ >= coor.z_resolution;
	};

	FILE* fp = fopen(pFileName, "w");
	int x = 0, y = 0, z = 0;
	//Find a point on the surface:
	findfisrstPoint(x, y, z);
	//BFS:
	XYZ xyz{ x, y, z };
	std::queue<XYZ> que;
	que.push(xyz);
	bool* vis = new bool[coor.x_resolution * coor.y_resolution * coor.y_resolution];
	memset(vis, true, coor.x_resolution * coor.y_resolution * coor.y_resolution * sizeof(bool));
	*(vis + coor.at(x, y, z)) = false;
	while (!que.empty())
	{
		xyz = que.front();
		que.pop();
		double coorX = coor.x_min + xyz.x * coor.dx;
		double coorY = coor.y_min + xyz.y * coor.dy;
		double coorZ = coor.z_min + xyz.z * coor.dz;
		int offset = coor.at(xyz.x, xyz.y, xyz.z);
		fprintf(fp, "%lf %lf %lf %f %f %f\n", coorX, coorY, coorZ, *(host_normalx + offset), *(host_normaly + offset), *(host_normalz + offset));
		for (int i = 0; i < 18; i++)
		{
			int tx = xyz.x + dx[i], ty = xyz.y + dy[i], tz = xyz.z + dz[i];
			if (!voutOfRange(tx, ty, tz) && *(vis + coor.at(tx, ty, tz))
				&& *(host_surface + coor.at(tx, ty, tz)))
			{
				que.push(XYZ{ tx, ty, tz });
				*(vis + coor.at(tx, ty, tz)) = false;
			}
		}
	}
	fclose(fp);
	delete[] vis;
}

Eigen::Vector3f Model::getNormal(int indX, int indY, int indZ)
{
	//Not used
	//PointCloud Normal
	auto voutOfRange = [&](int indexX, int indexY, int indexZ) {
		return indexX < 0 || indexY < 0 || indexZ < 0
			|| indexX >= coor.x_resolution
			|| indexY >= coor.y_resolution
			|| indexZ >= coor.z_resolution;
	};

	std::vector<Eigen::Vector3f> neiborList;
	//std::vector<Eigen::Vector3f> innerList;

	Eigen::Vector3f innerCenter = Eigen::Vector3f::Zero();
	int count = 0;

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
					if (*(host_surface + coor.at(neiborX, neiborY, neiborZ)))
						neiborList.push_back(Eigen::Vector3f(coorX, coorY, coorZ));
					else if (*(host_voxel + coor.at(neiborX, neiborY, neiborZ)))
					{
						innerCenter += Eigen::Vector3f(coorX, coorY, coorZ);
						count++;
					}
					//innerList.push_back(Eigen::Vector3f(coorX, coorY, coorZ));
				}
			}

	float coorX = coor.x_min + indX * coor.dx;
	float coorY = coor.y_min + indY * coor.dy;
	float coorZ = coor.z_min + indZ * coor.dz;
	Eigen::Vector3f point(coorX, coorY, coorZ);

	Eigen::MatrixXf matA(3, neiborList.size());
	for (int i = 0; i < neiborList.size(); i++)
		matA.col(i) = neiborList[i] - point;
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigenSolver(matA * matA.transpose());
	Eigen::Vector3f eigenValues = eigenSolver.eigenvalues();
	int indexEigen = 0;
	if (abs(eigenValues[1]) < abs(eigenValues[indexEigen]))
		indexEigen = 1;
	if (abs(eigenValues[2]) < abs(eigenValues[indexEigen]))
		indexEigen = 2;
	Eigen::Vector3f normalVector = eigenSolver.eigenvectors().col(indexEigen);

	innerCenter /= count;
	//Eigen::Vector3f innerCenter = Eigen::Vector3f::Zero();
	//for (auto const& vec : innerList)
	//	innerCenter += vec;
	//innerCenter /= innerList.size();

	if (normalVector.dot(point - innerCenter) < 0)
		normalVector *= -1;
	return normalVector;
}

void Model::surface()
{
	//Just a test: A point on the surface must have at least a neibor in the nearest 18 points.
	const int dx[18] = { -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1 };
	const int dy[18] = { -1, 0, 0, 0, 1, -1, -1, -1, 0, 0, 1, 1, 1, -1, 0, 0, 0, 1 };
	const int dz[18] = { 0, -1, 0, 1, 0, -1, 0, 1, -1, 1, -1, 0, 1, 0 - 1, 0, 1, 0 };

	auto voutOfRange = [&](int indexX, int indexY, int indexZ) {
		return indexX < 0 || indexY < 0 || indexZ < 0
			|| indexX >= coor.x_resolution
			|| indexY >= coor.y_resolution
			|| indexZ >= coor.z_resolution;
	};
	int ans = 0;
	for (int x = 0; x < coor.z_resolution; ++x)
		for (int y = 0; y < coor.z_resolution; ++y)
			for (int z = 0; z < coor.z_resolution; ++z)
				if (*(host_surface + coor.at(x, y, z)))
				{
					ans = 0;
					for (int i = 0; i < 18; i++)
						if (!voutOfRange(x + dx[i], y + dy[i], z + dz[i])
							&& *(host_surface + coor.at(x + dx[i], y + dy[i], z + dz[i])))
							ans++;
					if (ans == 0)
					{
						printf("Find it!\n");
						return;
					}
				}
	printf("Could not find it!\n");
}

void Model::saveVoxel(const char* pFileName)
{
	//For Debug
	FILE* fp = fopen(pFileName, "w");
	for (int x = 0; x < coor.x_resolution; x++)
		for (int y = 0; y < coor.y_resolution; y++)
			for (int z = 0; z < coor.z_resolution; z++)
				if (*(host_voxel + coor.at(x, y, z)))
					fprintf(fp, "%d %d %d\n", x, y, z);
}

void Model::loadColourImage(const char* pDir, const char* pPrefix, const char* pSuffix)
{
	coloured_image = new cv::Mat[CameraNum];
	std::string fileName(pDir);

	fileName += '/';
	fileName += pPrefix;
	for (int i = 0; i < CameraNum; ++i)
	{
		//std::cout << fileName + std::to_string(i) + pSuffix << std::endl;
		coloured_image[i] = cv::imread(fileName + std::to_string(i) + pSuffix);
	}
}

void Model::saveColouredPly(const char* pFileName)
{
	const int dx[18] = { -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1 };
	const int dy[18] = { -1, 0, 0, 0, 1, -1, -1, -1, 0, 0, 1, 1, 1, -1, 0, 0, 0, 1 };
	const int dz[18] = { 0, -1, 0, 1, 0, -1, 0, 1, -1, 1, -1, 0, 1, 0 - 1, 0, 1, 0 };

	auto voutOfRange = [&](int indexX, int indexY, int indexZ) {
		return indexX < 0 || indexY < 0 || indexZ < 0
			|| indexX >= coor.x_resolution
			|| indexY >= coor.y_resolution
			|| indexZ >= coor.z_resolution;
	};

	FILE* fp = fopen(pFileName, "w");
	fprintf(fp, "ply\n");
	fprintf(fp, "format ascii 1.0\n");
	fprintf(fp, "element vertex %d\n", point_count);
	fprintf(fp, "property float x\n");
	fprintf(fp, "property float y\n");
	fprintf(fp, "property float z\n");
	fprintf(fp, "property float nx\n");
	fprintf(fp, "property float ny\n");
	fprintf(fp, "property float nz\n");
	fprintf(fp, "property uchar red\n");
	fprintf(fp, "property uchar green\n");
	fprintf(fp, "property uchar blue\n");
	fprintf(fp, "end_header\n");
	int x = 0, y = 0, z = 0;
	//Find a point on the surface:
	findfisrstPoint(x, y, z);
	//BFS:
	XYZ xyz{ x, y, z };
	std::queue<XYZ> que;
	que.push(xyz);
	bool* vis = new bool[coor.x_resolution * coor.y_resolution * coor.y_resolution];
	memset(vis, true, coor.x_resolution * coor.y_resolution * coor.y_resolution * sizeof(bool));
	*(vis + coor.at(x, y, z)) = false;
	while (!que.empty())
	{
		xyz = que.front();
		que.pop();
		double coorX = coor.x_min + xyz.x * coor.dx;
		double coorY = coor.y_min + xyz.y * coor.dy;
		double coorZ = coor.z_min + xyz.z * coor.dz;
		int offset = coor.at(xyz.x, xyz.y, xyz.z);
		cv::Vec3f colour = getColour(coorX, coorY, coorZ);
		fprintf(fp, "%lf %lf %lf %f %f %f %f %f %f\n", coorX, coorY, coorZ, *(host_normalx + offset), *(host_normaly + offset), *(host_normalz + offset), colour(2), colour(1), colour(0));
		for (int i = 0; i < 18; i++)
		{
			int tx = xyz.x + dx[i], ty = xyz.y + dy[i], tz = xyz.z + dz[i];
			if (!voutOfRange(tx, ty, tz) && *(vis + coor.at(tx, ty, tz))
				&& *(host_surface + coor.at(tx, ty, tz)))
			{
				que.push(XYZ{ tx, ty, tz });
				*(vis + coor.at(tx, ty, tz)) = false;
			}
		}
	}
	fclose(fp);
	delete[] vis;
}

cv::Vec3f Model::getColour(double coorX, double coorY, double coorZ)
{
	cv::Vec3f colour = cv::Vec3f(0, 0, 0);
	for (int i = 0; i < CameraNum; i++)
	{
		Eigen::Vector3f vec3 = host_projection[i] * Eigen::Vector4f(coorX, coorY, coorZ, 1);
		int indX = vec3[1] / vec3[2];
		int indY = vec3[0] / vec3[2];
		colour += coloured_image[i].at<cv::Vec3b>((uint)(indX), (uint)(indY));
	}
	colour /= CameraNum;
	return colour;
}
