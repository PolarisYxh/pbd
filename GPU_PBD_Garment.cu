#include "Common.h"
#include <iostream>
#include "cuda_runtime.h"
#include "ConfigurationLoader.h"
#include "OBJLoader.h"
#include "ParticleData.h"
#include "IndexedFaceMesh.h"
#include "SimulationModel.h"
#include "CollisionDetection.h"
#include "DistanceFieldCollisionDetection.h"
#include "Timing.h"
#include "TimeStepController.h"
#include "IDFactory.h"

#include "My_GLSL.h"
#include "BMP_IO.h"
#include "MyVector.h"
#include "device_launch_parameters.h"

using namespace std;
using namespace Utilities;
using namespace PBD;

INIT_TIMING
INIT_LOGGING
int PBD::IDFactory::id = 0;

Configuration conf;
SimulationModel model;
DistanceFieldCollisionDetection cd;
TimeStepController sim;
std::vector<std::vector<OBJLoader::Vec3f>> globalPosition;
float seamTolerance = 0.01;

void buildModel();
void createPatternMesh(string str, const vector<float> translate, const vector<float> scale, bool findSpecialSeam);
void initClothConstraints();
void initBetweenPatternSeamConstraints();
void initInnerPatterSeamConstraints();
void initDistanceConstraints();
void initBendingConstraints();
void initSpecialSeamConstraints();
void createAvaterOne(string filename, int currentFrame);
void loadObj(const std::string& filename, VertexData& vd, IndexedFaceMesh& mesh, const Vector3r& scale, bool isCloth, bool findSpecialSeam);
void updateAvater(string filename, int currentFrame);
void moveAvaterToRight();
void moveAvaterToLeft();

void timeStep();
void Init_GLSL();
void display();
void keyPress(unsigned char, int, int);
void mouseClick(int, int, int, int);
void mouseMotion(int, int);

unsigned int vOffset = 0;
unsigned int vtOffset = 0;
unsigned int vnOffset = 0;

short simulationMethod = 1;
short bendingMethod = 1;

int screenWidth = 1024;
int screenHeight = 768;
bool doPause = true;
vector<unsigned int> selectedParticles;
Vector3r oldMousePos;
float	zoom = 62;
float	swing_angle = 0 + 3 * 20;
float	elevate_angle = 0;
float	center[3] = { 0, 0.0, -0.5 };
bool polygonMode = false;

//鼠标响应
bool mouseLeftDown;
bool mouseRightDown;
bool mouseMiddleDown;

///////////////////////////////////////////////////////////////////////////////////////////
//  Shader functions
///////////////////////////////////////////////////////////////////////////////////////////
GLuint depth_FBO = 0;
GLuint depth_texture = 0;

GLuint shadow_program = 0;
GLuint phong_program = 0;
GLuint phong_program_body = 0;
GLuint seam_program = 0;

GLuint vertex_handle = 0;
GLuint normal_handle = 0;
GLuint triangle_handle = 0;

GLuint vertex_handle_body = 0;
GLuint normal_handle_body = 0;
GLuint triangle_handle_body = 0;

GLuint seam_VEO = 0;
GLuint seam_VBO = 0;

GLuint collision_sphere = 0;
GLuint normal_collision_sphere = 0;
GLuint vertex_collision_sphere = 0;

float light_position[3] = { -2, 2, 4 };
IndexedFaceMesh mesh;
IndexedFaceMesh::Faces m_indices;
int faceNum;
RigidBody* body;
IndexedFaceMesh mesh_body;
IndexedFaceMesh::Faces body_indices;
int body_faceNum;
ParticleData pd;
int point;
VertexData body_VD;
int pointNum;

int stepCount = 0;
__global__ void printtr()
{
	printf("form Here \n");
}
int main(int argc, char* argv[])
{
	printtr << <1, 5 >> > ();

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	int dev;
	for (dev = 0; dev < deviceCount; dev++)
	{
		int driver_version(0), runtime_version(0);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);
		if (dev == 0)
			if (deviceProp.minor = 9999 && deviceProp.major == 9999)
				printf("\n");
		printf("\nDevice%d:\"%s\"\n", dev, deviceProp.name);
		cudaDriverGetVersion(&driver_version);
		printf("CUDA驱动版本:                                   %d.%d\n", driver_version / 1000, (driver_version % 1000) / 10);
		cudaRuntimeGetVersion(&runtime_version);
		printf("CUDA运行时版本:                                 %d.%d\n", runtime_version / 1000, (runtime_version % 1000) / 10);
		printf("设备计算能力:                                   %d.%d\n", deviceProp.major, deviceProp.minor);
		printf("Total amount of Global Memory:                  %u bytes\n", deviceProp.totalGlobalMem);
		printf("Number of SMs:                                  %d\n", deviceProp.multiProcessorCount);
		printf("Total amount of Constant Memory:                %u bytes\n", deviceProp.totalConstMem);
		printf("Total amount of Shared Memory per block:        %u bytes\n", deviceProp.sharedMemPerBlock);
		printf("Total number of registers available per block:  %d\n", deviceProp.regsPerBlock);
		printf("Warp size:                                      %d\n", deviceProp.warpSize);
		printf("Maximum number of threads per SM:               %d\n", deviceProp.maxThreadsPerMultiProcessor);
		printf("Maximum number of threads per block:            %d\n", deviceProp.maxThreadsPerBlock);
		printf("Maximum size of each dimension of a block:      %d x %d x %d\n", deviceProp.maxThreadsDim[0],
			deviceProp.maxThreadsDim[1],
			deviceProp.maxThreadsDim[2]);
		printf("Maximum size of each dimension of a grid:       %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
		printf("Maximum memory pitch:                           %u bytes\n", deviceProp.memPitch);
		printf("Texture alignmemt:                              %u bytes\n", deviceProp.texturePitchAlignment);
		printf("Clock rate:                                     %.2f GHz\n", deviceProp.clockRate * 1e-6f);
		printf("Memory Clock rate:                              %.0f MHz\n", deviceProp.memoryClockRate * 1e-3f);
		printf("Memory Bus Width:                               %d-bit\n", deviceProp.memoryBusWidth);
	}
//	conf = conf.getConfiguration("D:\\Program\\GPU_PBD_Garment\\data\\walk\\plane-1681.configFile");
//	conf = conf.getConfiguration("D:\\Program\\GPU_PBD_Garment\\data\\walk\\dress-333-Female Walk-9021.configFile");
//	conf = conf.getConfiguration("D:\\Program\\GPU_PBD_Garment\\data\\walk\\dress-333-Female Walk-complex.configFile");
//	conf = conf.getConfiguration("D:\\Program\\GPU_PBD_Garment\\data\\walk\\dress-335-98440-Female Walk-9021.configFile");
	// tolerance=0.012
//	conf = conf.getConfiguration("D:\\Program\\GPU_PBD_Garment\\data\\walk\\dress-335-13294-Female Walk-9021.configFile");
	// tolerance=0.015
//	conf = conf.getConfiguration("D:\\Program\\GPU_PBD_Garment\\data\\walk\\dress-311-11796-Female Walk-9021.configFile");
//	conf = conf.getConfiguration("D:\\Program\\GPU_PBD_Garment\\data\\walk\\dress-258-12816-Female Walk-9021.configFile");
	conf = conf.getConfiguration("D:\\Program\\GPU_PBD_Garment\\data\\multiMaterial\\40-40-1681-48-48-8765.configFile");

	/*if (conf.getBodyPath() != "")
	{
		int endFrame = conf.getEndFrame();
		globalPosition.reserve(endFrame);
		for (int i = 1; i <= endFrame; i++)
		{
			std::vector<OBJLoader::Vec3f> x;
			string fileName = "D:\\Program\\GPU_PBD_Garment\\data\\walk\\sequence_vertex\\walk" + to_string(i) + ".obj";
			OBJLoader::loadObjVertex(fileName, &x);
			globalPosition.push_back(x);
		}
	}*/

	buildModel();

	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowPosition(300, 100);
	glutInitWindowSize(screenWidth, screenHeight);
	glutCreateWindow("PBD_Garment_GPU");
	glutDisplayFunc(display);
	glutKeyboardFunc(keyPress); 
	glutMouseFunc(mouseClick);
	glutMotionFunc(mouseMotion);
	glutIdleFunc(timeStep);
	Init_GLSL();
	glutMainLoop();
	
	cudaDeviceReset();
	return 0;
}

void timeStep()
{
	if (doPause)
		return;
	START_TIMING("====================总用时");
 	if (conf.getBodyPath() != "")
	{
		unsigned int stepSize = conf.getStepSize();
		unsigned int startFrame = conf.getStartFrame();
		unsigned int endFrame = conf.getEndFrame();
		string bvhName = conf.getBvhName();
		if ((++stepCount) % stepSize == 0)
		{
			int currentFrame = startFrame + (stepCount / stepSize);
			if (currentFrame > endFrame)
			{

			}
			else
			{
				//清除上一人体模型所占用的内存
			/*	cd.popCollisionObject();
				model.cleanRigidBodies();
				createAvaterOne(bvhName, currentFrame);*/
			//	直接更新人体模型的点、法线, 由220ms减少到30ms
			//	updateAvater(bvhName, currentFrame);
			}
		}
	}

	sim.step(model, conf);

	for (unsigned int i = 0; i < model.getTriangleModels().size(); i++)
		model.getTriangleModels()[i]->updateMeshNormals(model.getParticles());
	STOP_TIMING_AVG_PRINT;
	glutPostRedisplay();
}

void buildModel()
{
	unsigned int clothNum = conf.getClothNum();
	for (int i = 0; i < clothNum; i++)
	{
		string clothPath = conf.getClothPath().at(i);
		vector<float> translate = conf.getTranslate().at(i);
		vector<float> scale = conf.getScale().at(i);
		bool findSpecialSeam = conf.getFindSpecialSeam().at(i);
		createPatternMesh(clothPath, translate, scale, findSpecialSeam);
	}
	//为mesh粒子设置mass属性
	ParticleData& pd = model.getParticles();
	for (unsigned int i = 0; i < model.getTriangleModels().size(); i++)
	{
		float massCoeff = conf.getClothCoeff().at(i)[3];
		unsigned int indexoffset = model.getTriangleModels().at(i)->getIndexOffset();
		unsigned int pdNum = model.getTriangleModels().at(i)->getParticleMesh().numVertices();
		for (unsigned int j = 0; j < pdNum; j++)
		{
			pd.setMass(indexoffset + j, massCoeff);
		}
	}
	for (int i = 0; i < conf.getZeroMassPointsVector().size(); i++)
	{
		pd.setMass(conf.getZeroMassPointsVector()[i], 0.0);
	}
	//初始化服装缝合线等约束
	initClothConstraints();

	//设置各pattern的拉伸、弯曲等属性，初始化碰撞约束
	SimulationModel::TriangleModelVector& tm = model.getTriangleModels();
	for (unsigned int i = 0; i < tm.size(); i++)
	{
		TriangleModel* tri = model.getTriangleModels().at(i);
		tri->setFrictionCoeff(conf.getClothCoeff().at(i)[0]);
		tri->setRestitutionCoeff(conf.getClothCoeff().at(i)[1]);
		tri->setBendingCoeff(conf.getClothCoeff().at(i)[2]);
		tri->setDampingCoeff(conf.getClothCoeff().at(i)[4]);

		vector<vector<unsigned int>> faces;
		IndexedFaceMesh::Faces indices = tri->getParticleMesh().getFaces();
		vector<unsigned int> face;
		for (int i = 0; i < indices.size(); i++)
		{
			face.push_back(indices[i]);
			if ((i + 1) % 3 == 0)
			{
				faces.push_back(face);
				face.clear();
			}
		}
		const unsigned int nVert = tm[i]->getParticleMesh().numVertices();
		unsigned int offset = tm[i]->getIndexOffset();
		cd.addCollisionSphereOnFaces(i, CollisionDetection::CollisionObject::TriangleModelCollisionObjectType, faces, faces.size(), &pd.getPosition(offset), nVert, 0.5 * 0.2 * 0.25 * 1.3);
	}

	//创建人体模型
	if (conf.getBvhPath() != "")
	{
		string bvhName = conf.getBvhName();
		unsigned int startFrame = conf.getStartFrame();
		createAvaterOne(bvhName, startFrame);
	}
}

void createPatternMesh(string str, const vector<float> translate, const vector<float> scale, bool findSpecialSeam)
{
	std::vector<std::vector<OBJLoader::Vec3f>> xVec;
	std::vector<std::vector<MeshFaceIndices>>facesVec;
	std::vector<std::vector<OBJLoader::Vec3f>>normalsVec;
	std::vector<std::vector<OBJLoader::Vec2f>>texcoordsVec;
	OBJLoader::Vec3f s = { (float)scale[0], (float)scale[1], (float)scale[2] };

	vOffset = 0;
	vtOffset = 0;
	vnOffset = 0;
	OBJLoader::loadPatternObj(str, &xVec, &facesVec, &normalsVec, &texcoordsVec, s, vOffset, vtOffset, vnOffset);

	std::vector<OBJLoader::Vec3f> x;
	std::vector<OBJLoader::Vec3f> normals;
	std::vector<OBJLoader::Vec2f> texCoords;
	std::vector<MeshFaceIndices> faces;

	for (int i = 0; i < xVec.size(); i++)
	{
		x.clear();
		normals.clear();
		texCoords.clear();
		faces.clear();

		x = xVec.at(i);
		normals = normalsVec.at(i);                                  // 没有使用
		texCoords = texcoordsVec.at(i);
		faces = facesVec.at(i);

		VertexData vd;
		IndexedFaceMesh mesh;

		mesh.release();
		vd.release();
		const unsigned int nPoints = (unsigned int)x.size();
		const unsigned int nFaces = (unsigned int)faces.size();
		const unsigned int nTexCoords = (unsigned int)texCoords.size();
		mesh.initMesh(nPoints, nFaces * 2, nFaces, true);                    // 点数，边数，面数
		vd.reserve(nPoints);
		for (unsigned int i = 0; i < nPoints; i++)                     //存储点坐标
		{
			vd.addVertex(Vector3r(x[i][0], x[i][1], x[i][2]));
		}
		for (unsigned int i = 0; i < nTexCoords; i++)                  //存储纹理坐标
		{
			mesh.addUV(texCoords[i][0], texCoords[i][1]);
		}
		for (unsigned int i = 0; i < nFaces; i++)                      //存储面信息
		{
			// Reduce the indices by one
			int posIndices[3];
			int texIndices[3];
			for (int j = 0; j < 3; j++)
			{
				posIndices[j] = faces[i].posIndices[j] - 1;
				if (nTexCoords > 0)
				{
					texIndices[j] = faces[i].texIndices[j] - 1;
					mesh.addUVIndex(texIndices[j]);
				}
			}

			mesh.addFace(&posIndices[0]);
		}
		mesh.buildNeighbors(true, findSpecialSeam);

		mesh.updateNormals(vd, 0);
		mesh.updateVertexNormals(vd);

		cout << "Number of triangles: " << nFaces << endl;
		cout << "Number of vertices: " << nPoints << endl;

		//加入model
		IndexedFaceMesh::UVs uvs = mesh.getUVs();
		IndexedFaceMesh::UVIndices uvIndices = mesh.getUVIndices();
		Vector3r *points =new Vector3r[vd.size()];
		const vector<Vector3r>* m_x;
		m_x = vd.getVertices();
		for (int i = 0; i < vd.size(); i++)
		{
			points[i] = (*m_x).at(i);
		}
		cout << "点数：" << vd.size() << endl;
		IndexedFaceMesh::Faces m_indices = mesh.getFaces();
		unsigned int* indices = new unsigned int[m_indices.size()];
		for (int i = 0; i < m_indices.size(); i++)
		{
			indices[i] = m_indices.at(i);
		}
		cout << "面索引数：" << m_indices.size() << endl;
	
		model.addTriangleModel(vd.size(), (m_indices.size()) / 3, &points[0], &indices[0], uvIndices, uvs, findSpecialSeam);
	}
	x.clear();
	normals.clear();
	texCoords.clear();
	faces.clear();
	xVec.clear();
	normalsVec.clear();
	texcoordsVec.clear();
	facesVec.clear();
}

void updateAvater(string filename, int currentFrame)
{
	//START_TIMING("---1");
	std::vector<OBJLoader::Vec3f> x = globalPosition[currentFrame - 1];
	//STOP_TIMING_AVG_PRINT;
	//START_TIMING("---2");
	cout << "Loading motion........................walk." << currentFrame << "obj" << endl;
	VertexData& vd = model.getRigidBodies()[0]->getVertexData();
	VertexData& vd_local = model.getRigidBodies()[0]->getVertexDataLocal();
	for (int i = 0; i < vd.size(); i++)
	{
		vd.setPosition(i, Vector3r(x[i][0], x[i][1], x[i][2]));
		vd_local.setPosition(i, Vector3r(x[i][0], x[i][1], x[i][2]));
	}
	//STOP_TIMING_AVG_PRINT;
	//START_TIMING("---3");
	model.getRigidBodies()[0]->updateMeshNormals(vd);
	model.getRigidBodies()[0]->updateMeshNormals(vd_local);
	//STOP_TIMING_AVG_PRINT;
}

void createAvaterOne(string filename, int currentFrame)
{
	SimulationModel::RigidBodyVector& rb = model.getRigidBodies();
	sim.setCollisionDetection(model, &cd);

	string bvhPath = conf.getBvhPath();
	
	//string fileName = bvhPath + filename + to_string(currentFrame) + ".obj";
	string fileName = conf.getBodyPath();

	IndexedFaceMesh mesh;
	VertexData vd;
	loadObj(fileName, vd, mesh, Vector3r(1.0,1.0,1.0), false, false);
	RigidBody* rigidBody = new RigidBody();

	rigidBody->initBody(vd,	mesh, Vector3r(1.0, 1.0, 1.0));
	rb.push_back(rigidBody);
	const std::vector<Vector3r>* vertices = rigidBody->getVertexDataLocal().getVertices();
	const unsigned int nVert = static_cast<unsigned int>(vertices->size());
	vector<vector<unsigned int>> faces;
	IndexedFaceMesh::Faces indices = mesh.getFaces();
	const  unsigned int nFace = static_cast<unsigned int>(indices.size() / 3);
	vector<unsigned int> face;
	for (int i = 0; i < indices.size(); i++)
	{
		face.push_back(indices[i]);
		if ((i + 1) % 3 == 0)
		{
			faces.push_back(face);
			face.clear();
		}
	}
	cd.addCollisionSphereOnFaces(rb.size() - 1, CollisionDetection::CollisionObject::RigidBodyCollisionObjectType, faces, nFace, &(*vertices)[0], nVert, 0);
	mesh.release();
	vd.release();
	faces.clear();
	indices.clear();
	face.clear();
}

void loadObj(const std::string& filename, VertexData& vd, IndexedFaceMesh& mesh, const Vector3r& scale, bool isCloth, bool findSpecialSeam)
{
	std::vector<OBJLoader::Vec3f> x;
	std::vector<OBJLoader::Vec3f> normals;
	std::vector<OBJLoader::Vec2f> texCoords;
	std::vector<MeshFaceIndices> faces;
	OBJLoader::Vec3f s = { (float)scale[0], (float)scale[1], (float)scale[2] };
	OBJLoader::loadObj(filename, &x, &faces, &normals, &texCoords, s);

	mesh.release();
	const unsigned int nPoints = (unsigned int)x.size();
	const unsigned int nFaces = (unsigned int)faces.size();
	const unsigned int nTexCoords = (unsigned int)texCoords.size();
	mesh.initMesh(nPoints, nFaces * 2, nFaces, isCloth);                    // 点数，边数，面数
	vd.reserve(nPoints);
	for (unsigned int i = 0; i < nPoints; i++)                     //存储点坐标
	{
		vd.addVertex(Vector3r(x[i][0], x[i][1], x[i][2]));
	}
	for (unsigned int i = 0; i < nTexCoords; i++)                  //存储纹理坐标
	{
		mesh.addUV(texCoords[i][0], texCoords[i][1]);
	}
	for (unsigned int i = 0; i < nFaces; i++)                      //存储面信息
	{
		// Reduce the indices by one
		int posIndices[3];
		int texIndices[3];
		for (int j = 0; j < 3; j++)
		{
			posIndices[j] = faces[i].posIndices[j] - 1;
			if (nTexCoords > 0)
			{
				texIndices[j] = faces[i].texIndices[j] - 1;
				mesh.addUVIndex(texIndices[j]);
			}
		}

		mesh.addFace(&posIndices[0]);
	}
	mesh.buildNeighbors(isCloth, findSpecialSeam);

	mesh.updateNormals(vd, 0);
	mesh.updateVertexNormals(vd);

	x.clear();
	faces.clear();
	normals.clear();
	texCoords.clear();

	LOG_INFO << "Number of vertices: " << nPoints;
}

void initClothConstraints()
{
	//最后调整缝合线位置
	initDistanceConstraints();
	initBendingConstraints();
	initSpecialSeamConstraints();
	initInnerPatterSeamConstraints();
	initBetweenPatternSeamConstraints();
}

void initBetweenPatternSeamConstraints()
{
	ParticleData pd = model.getParticles();
	for (unsigned int i = 0; i < model.getTriangleModels().size(); i++)
	{
		IndexedFaceMesh mesh1 = model.getTriangleModels().at(i)->getParticleMesh();
		IndexedFaceMesh::BorderEdges borderEdge1 = mesh1.getBorderEdges();
		IndexedFaceMesh::Edges edges1 = mesh1.getEdges();
		IndexedFaceMesh::SeamEdges& seamEdge1 = mesh1.getSeamEdges();
		unsigned int indexOffset1 = model.getTriangleModels().at(i)->getIndexOffset();
		for (int j = 0; j < borderEdge1.size(); j++)
		{
			IndexedFaceMesh::Edge e1 = edges1.at(borderEdge1.at(j));
			Vector3r point11 = pd.getPosition(e1.m_vert[0] + indexOffset1);
			Vector3r point12 = pd.getPosition(e1.m_vert[1] + indexOffset1);
			for (unsigned int k = i + 1; k < model.getTriangleModels().size(); k++)
			{
				IndexedFaceMesh& mesh2 = model.getTriangleModels().at(k)->getParticleMesh();
				IndexedFaceMesh::BorderEdges borderEdge2 = mesh2.getBorderEdges();
				IndexedFaceMesh::Edges edges2 = mesh2.getEdges();
				IndexedFaceMesh::SeamEdges& seamEdge2 = mesh2.getSeamEdges();
				unsigned int indexOffset2 = model.getTriangleModels().at(k)->getIndexOffset();
				for (int l = 0; l < borderEdge2.size(); l++)
				{
					IndexedFaceMesh::Edge e2 = edges2.at(borderEdge2.at(l));
					Vector3r point21 = pd.getPosition(e2.m_vert[0] + indexOffset2);
					Vector3r point22 = pd.getPosition(e2.m_vert[1] + indexOffset2);
					if ((point11 - point21).norm() < seamTolerance && (point12 - point22).norm() < seamTolerance)                  //将非完全重合的点对缝合
					{
						model.addSeamConstraint(e1.m_vert[0] + indexOffset1, e2.m_vert[0] + indexOffset2);
						model.addSeamConstraint(e1.m_vert[1] + indexOffset1, e2.m_vert[1] + indexOffset2);
						model.addSeamConstraint(e1.m_vert[0] + indexOffset1, e2.m_vert[1] + indexOffset2);
						model.addSeamConstraint(e1.m_vert[1] + indexOffset1, e2.m_vert[0] + indexOffset2);
						seamEdge1.push_back(borderEdge1.at(j));
						seamEdge2.push_back(borderEdge2.at(l));
					}
					if ((point11 - point22).norm() < seamTolerance && (point12 - point21).norm() < seamTolerance)
					{
						model.addSeamConstraint(e1.m_vert[0] + indexOffset1, e2.m_vert[0] + indexOffset2);
						model.addSeamConstraint(e1.m_vert[1] + indexOffset1, e2.m_vert[1] + indexOffset2);
						model.addSeamConstraint(e1.m_vert[0] + indexOffset1, e2.m_vert[1] + indexOffset2);
						model.addSeamConstraint(e1.m_vert[1] + indexOffset1, e2.m_vert[0] + indexOffset2);
						seamEdge1.push_back(borderEdge1.at(j));
						seamEdge2.push_back(borderEdge2.at(l));
					}
				}
				//	mesh2.release();
				borderEdge2.clear();
				edges2.clear();
			}
		}
		mesh1.release();
		borderEdge1.clear();
		edges1.clear();
	}
	pd.release();
}

void initInnerPatterSeamConstraints()
{
	ParticleData pd = model.getParticles();
	for (unsigned int i = 0; i < model.getTriangleModels().size(); i++)
	{
		IndexedFaceMesh mesh = model.getTriangleModels().at(i)->getParticleMesh();
		IndexedFaceMesh::BorderEdges borderEdge = mesh.getBorderEdges();
		IndexedFaceMesh::Edges edges = mesh.getEdges();
		IndexedFaceMesh::SeamEdges& seamEdge = mesh.getSeamEdges();
		unsigned int indexOffset = model.getTriangleModels().at(i)->getIndexOffset();
		for (int j = 0; j < borderEdge.size(); j++)
		{
			IndexedFaceMesh::Edge e1 = edges.at(borderEdge.at(j));
			Vector3r point11 = pd.getPosition(e1.m_vert[0] + indexOffset);
			Vector3r point12 = pd.getPosition(e1.m_vert[1] + indexOffset);
			for (int k = j + 1; k < borderEdge.size(); k++)
			{
				IndexedFaceMesh::Edge e2 = edges.at(borderEdge.at(k));
				Vector3r point21 = pd.getPosition(e2.m_vert[0] + indexOffset);
				Vector3r point22 = pd.getPosition(e2.m_vert[1] + indexOffset);
				if ((point11 - point21).norm() < seamTolerance && (point12 - point22).norm() < seamTolerance)
				{
					model.addSeamConstraint(e1.m_vert[0] + indexOffset, e2.m_vert[0] + indexOffset);
					model.addSeamConstraint(e1.m_vert[1] + indexOffset, e2.m_vert[1] + indexOffset);
					model.addSeamConstraint(e1.m_vert[0] + indexOffset, e2.m_vert[1] + indexOffset);
					model.addSeamConstraint(e1.m_vert[1] + indexOffset, e2.m_vert[0] + indexOffset);
					seamEdge.push_back(borderEdge.at(j));
					seamEdge.push_back(borderEdge.at(k));
				}
				else if ((point11 - point22).norm() < seamTolerance && (point12 - point21).norm() < seamTolerance)
				{
					model.addSeamConstraint(e1.m_vert[0] + indexOffset, e2.m_vert[1] + indexOffset);
					model.addSeamConstraint(e1.m_vert[1] + indexOffset, e2.m_vert[0] + indexOffset);
					model.addSeamConstraint(e1.m_vert[0] + indexOffset, e2.m_vert[0] + indexOffset);
					model.addSeamConstraint(e1.m_vert[1] + indexOffset, e2.m_vert[1] + indexOffset);
					seamEdge.push_back(borderEdge.at(j));
					seamEdge.push_back(borderEdge.at(k));
				}
			}
		}
		mesh.release();
		borderEdge.clear();
		edges.clear();
	}
	pd.release();
}

void initDistanceConstraints()
{
	for (unsigned int cm = 0; cm < model.getTriangleModels().size(); cm++)
	{
		if (simulationMethod == 1)  //distance constrians
		{
			const unsigned int offset = model.getTriangleModels()[cm]->getIndexOffset();
			const unsigned int nEdges = model.getTriangleModels()[cm]->getParticleMesh().numEdges();
			const IndexedFaceMesh::Edge* edges = model.getTriangleModels()[cm]->getParticleMesh().getEdges().data();
			for (unsigned int i = 0; i < nEdges; i++)
			{
				const unsigned int v1 = edges[i].m_vert[0] + offset;
				const unsigned int v2 = edges[i].m_vert[1] + offset;

				model.addDistanceConstraint(v1, v2, cm);                             //为每条边添加约束
			}
		}
	}
}

void initBendingConstraints()
{
	for (unsigned int cm = 0; cm < model.getTriangleModels().size(); cm++)
	{
		if (bendingMethod != 0)
		{
			const unsigned int offset = model.getTriangleModels()[cm]->getIndexOffset();
			IndexedFaceMesh mesh = model.getTriangleModels()[cm]->getParticleMesh();
			unsigned int nEdges = mesh.numEdges();
			const IndexedFaceMesh::Edge* edges = mesh.getEdges().data();        //获取particle Mesh的所有边
			const unsigned int* tris = mesh.getFaces().data();
			for (unsigned int i = 0; i < nEdges; i++)
			{
				const int tri1 = edges[i].m_face[0];
				const int tri2 = edges[i].m_face[1];
				if ((tri1 != 0xffffffff) && (tri2 != 0xffffffff))
				{
					// Find the triangle points which do not lie on the axis
					const int axisPoint1 = edges[i].m_vert[0];         // 获得两个面的公共点
					const int axisPoint2 = edges[i].m_vert[1];
					int point1 = -1;
					int point2 = -1;
					for (int j = 0; j < 3; j++)                        // 获得两个面的非公共点
					{
						if ((tris[3 * tri1 + j] != axisPoint1) && (tris[3 * tri1 + j] != axisPoint2))
						{
							point1 = tris[3 * tri1 + j];
							break;
						}
					}
					for (int j = 0; j < 3; j++)
					{
						if ((tris[3 * tri2 + j] != axisPoint1) && (tris[3 * tri2 + j] != axisPoint2))
						{
							point2 = tris[3 * tri2 + j];
							break;
						}
					}
					if ((point1 != -1) && (point2 != -1))
					{
						const unsigned int vertex1 = point1 + offset;
						const unsigned int vertex2 = point2 + offset;
						const unsigned int vertex3 = edges[i].m_vert[0] + offset;
						const unsigned int vertex4 = edges[i].m_vert[1] + offset;
						if (bendingMethod == 1)
							model.addDihedralConstraint(vertex1, vertex2, vertex3, vertex4);
						//增加距离为2的距离约束
						model.addDistanceConstraint(vertex1, vertex2, cm);
						//model.addDistanceConstraint(vertex1, vertex2, cm);
						//model.addDistanceConstraint(vertex1, vertex2, cm);
						//model.addDistanceConstraint(vertex1, vertex2, cm);
					}
				}
			}
			mesh.release();
		}
	}
}

void initSpecialSeamConstraints()
{
	ParticleData& pd = model.getParticles();
	//存储“开放边”（非缝合线部分的边界）
	vector<vector<unsigned int>> openBorders;
	for (unsigned int i = 0; i < model.getTriangleModels().size(); i++)
	{
		if (!model.getTriangleModels().at(i)->getSpecialSeam())
		{
			vector<unsigned int> openBorder;
			openBorders.push_back(openBorder);
			continue;
		}
		IndexedFaceMesh mesh = model.getTriangleModels().at(i)->getParticleMesh();
		IndexedFaceMesh::BorderEdges borderEdge = mesh.getBorderEdges();
		IndexedFaceMesh::Edges edges = mesh.getEdges();
		IndexedFaceMesh::SeamEdges seamEdges = mesh.getSeamEdges();
		vector<unsigned int> openBorder;
		unsigned int indexOffset = model.getTriangleModels().at(i)->getIndexOffset();
		for (int j = 0; j < borderEdge.size(); j++)
		{
			unsigned int e1 = borderEdge.at(j);
			int flag = 0;
			for (int k = 0; k < seamEdges.size(); k++)
			{
				unsigned int e2 = borderEdge.at(k);
				if (e1 == e2)
				{
					flag = 1;
				}
			}
			if (flag == 0)openBorder.push_back(j);
		}
		openBorders.push_back(openBorder);
	}

	//寻找最高点,及其所属pattern
	Vector3r highestPoint = Vector3r(-100.0, -100.0, -100.0);
	int triNum = -1;
	for (unsigned int i = 0; i < model.getTriangleModels().size(); i++)
	{
		if (!model.getTriangleModels().at(i)->getSpecialSeam()) continue;
		IndexedFaceMesh mesh = model.getTriangleModels().at(i)->getParticleMesh();
		IndexedFaceMesh::BorderEdges borderEdge = mesh.getBorderEdges();
		IndexedFaceMesh::Edges edges = mesh.getEdges();
		unsigned int indexOffset = model.getTriangleModels().at(i)->getIndexOffset();
		for (int j = 0; j < borderEdge.size(); j++)
		{
			IndexedFaceMesh::Edge e = edges.at(borderEdge.at(j));
			Vector3r point1 = pd.getPosition(e.m_vert[0] + indexOffset);
			Vector3r point2 = pd.getPosition(e.m_vert[1] + indexOffset);
			if (point1[1] > highestPoint[1])
			{
				highestPoint = point1;
				triNum = i;
			}
			if (point2[1] > highestPoint[1])
			{
				highestPoint = point2;
				triNum = i;
			}
		}
		mesh.release();
		borderEdge.clear();
		edges.clear();
	}

	if (triNum == -1)return;

	//寻找包含“special seam”的所有pattern
	IndexedFaceMesh mesh = model.getTriangleModels().at(triNum)->getParticleMesh();
	IndexedFaceMesh::BorderEdges borderEdge = mesh.getBorderEdges();
	IndexedFaceMesh::Edges edges = mesh.getEdges();
	IndexedFaceMesh::SeamEdges seamEdges = mesh.getSeamEdges();
	unsigned int indexOffset = model.getTriangleModels().at(triNum)->getIndexOffset();
	vector<unsigned int> patternNums;
	bool hasPush;
	for (unsigned int i = 0; i < model.getTriangleModels().size(); i++)
	{
		if (!model.getTriangleModels().at(i)->getSpecialSeam()) continue;
		hasPush = false;
		IndexedFaceMesh mesh1 = model.getTriangleModels().at(i)->getParticleMesh();
		IndexedFaceMesh::Edges edges1 = mesh1.getEdges();
		IndexedFaceMesh::BorderEdges borderEdge1 = mesh1.getBorderEdges();
		vector<unsigned int> openBorder1 = openBorders.at(i);
		unsigned int indexOffset1 = model.getTriangleModels().at(i)->getIndexOffset();
		for (int j = 0; j < openBorder1.size(); j++)
		{
			if (hasPush) break;
			IndexedFaceMesh::Edge e1 = edges1.at(borderEdge1.at(openBorder1.at(j)));
			Vector3r point11 = pd.getPosition(e1.m_vert[0] + indexOffset1);
			Vector3r point12 = pd.getPosition(e1.m_vert[1] + indexOffset1);
			for (int k = 0; k < openBorders.at(triNum).size(); k++)
			{
				IndexedFaceMesh::Edge e2 = edges.at(borderEdge.at(openBorders.at(triNum).at(k)));
				Vector3r point21 = pd.getPosition(e2.m_vert[0] + indexOffset);
				Vector3r point22 = pd.getPosition(e2.m_vert[1] + indexOffset);
				if ((point11 == point21) || (point11 == point22) || (point12 == point21) || (point12 == point22))
				{
					patternNums.push_back(i);
					hasPush = true;
					break;
				}
			}
		}
	}

	//固定“special seam”
	for (int i = 0; i < patternNums.size(); i++)
	{
		vector<unsigned int> openBorder = openBorders.at(patternNums.at(i));
		IndexedFaceMesh mesh1 = model.getTriangleModels().at(patternNums.at(i))->getParticleMesh();
		IndexedFaceMesh::Edges edges1 = mesh1.getEdges();
		IndexedFaceMesh::BorderEdges borderEdge1 = mesh1.getBorderEdges();
		unsigned int indexOffset1 = model.getTriangleModels().at(patternNums.at(i))->getIndexOffset();
		for (int j = 0; j < openBorder.size(); j++)
		{
			IndexedFaceMesh::Edge e = edges1.at(borderEdge1.at(openBorder.at(j)));
			pd.setMass(e.m_vert[0] + indexOffset1, 0.0);
			pd.setMass(e.m_vert[1] + indexOffset1, 0.0);
		}
	}
}

void Init_GLSL()
{
	zoom = conf.getZoom();
	swing_angle = conf.getSwingAngle();
	elevate_angle = conf.getElevateAngle();
	center[0] = conf.getCenter()[0];
	center[1] = conf.getCenter()[1];
	center[2] = conf.getCenter()[2];

	//Init GLEW
	GLenum err = glewInit();
	if (err != GLEW_OK)  printf(" Error initializing GLEW! \n");
	else printf("Initializing GLEW succeeded!\n");

	//Init depth texture and FBO
	glGenFramebuffers(1, &depth_FBO);
	glBindFramebuffer(GL_FRAMEBUFFER, depth_FBO);
	// Depth texture. Slower than a depth buffer, but you can sample it later in your shader
	glGenTextures(1, &depth_texture);
	glBindTexture(GL_TEXTURE_2D, depth_texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 1024, 1024, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depth_texture, 0);
	glDrawBuffer(GL_NONE);
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) printf("Init_Shadow_Map failed.\n");

	//Load shader program
	shadow_program = Setup_GLSL("shadow");
	phong_program = Setup_GLSL("phong");
	phong_program_body = Setup_GLSL("phong_body");
	seam_program = Setup_GLSL("seam");

	//Create EBO
	glGenBuffers(1, &vertex_handle);
	glGenBuffers(1, &normal_handle);
	glGenBuffers(1, &triangle_handle);

	glGenBuffers(1, &vertex_handle_body);
	glGenBuffers(1, &normal_handle_body);
	glGenBuffers(1, &triangle_handle_body);

	glGenBuffers(1, &seam_VEO);
	glGenBuffers(1, &seam_VBO);

	glGenBuffers(1, &collision_sphere);
	glGenBuffers(1, &normal_collision_sphere);
	glGenBuffers(1, &vertex_collision_sphere);
}

void display()
{
	GLuint uniloc;

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, screenWidth, screenHeight);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(4, (float)screenWidth / (float)screenHeight, 1, 100);
	glMatrixMode(GL_MODELVIEW);
	glShadeModel(GL_SMOOTH);

	glLoadIdentity();
	glClearColor(223 / 255.0, 223 / 255.0, 207 / 255.0, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	gluLookAt(0, 0, zoom, 0, 1.5, 5, 0, 1, 0);     //三个坐标，依次设置脑袋的位置，眼睛看的位置和头顶的方向

	glDisable(GL_LIGHTING);
	glEnable(GL_DEPTH_TEST);
	glUseProgram(shadow_program);
	uniloc = glGetUniformLocation(shadow_program, "shadow_texture");
	glUniform1i(uniloc, 0);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, depth_texture);
	uniloc = glGetUniformLocation(shadow_program, "light_position");
	glUniform3fv(uniloc, 1, light_position);

	glRotated(elevate_angle, 1, 0, 0);
	glRotated(swing_angle, 0, 1, 0);
	glTranslatef(center[0], center[1], center[2]);
	glScaled(1.8, 1.8, 1.8);

	//=========================================服装

	for (int k = 0; k < model.getTriangleModels().size(); k++)
	{
		glUseProgram(phong_program);
		uniloc = glGetUniformLocation(phong_program, "light_position");
		glUniform3fv(uniloc, 1, light_position);
		GLuint color = glGetUniformLocation(phong_program, "adjustNum");
		glUniform1i(color, k);
		GLuint c0 = glGetAttribLocation(phong_program, "position");
		GLuint c1 = glGetAttribLocation(phong_program, "normal");
		glEnableVertexAttribArray(c0);
		glEnableVertexAttribArray(c1);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangle_handle);
		mesh = model.getTriangleModels()[k]->getParticleMesh();
		m_indices = mesh.getFaces();
		faceNum = m_indices.size();
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * faceNum, &m_indices[0], GL_STATIC_DRAW);                    //存储 面 数据 

		glBindBuffer(GL_ARRAY_BUFFER, vertex_handle);
		pd = model.getParticles();
		point = mesh.numVertices();
		unsigned int indexOffset = model.getTriangleModels()[k]->getIndexOffset();
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * point * 3, &(pd.getPosition(indexOffset)), GL_DYNAMIC_DRAW);                   //存储 点 数据
		glVertexAttribPointer(c0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, (void*)0);

		glBindBuffer(GL_ARRAY_BUFFER, normal_handle);
		//修改数量计算方式
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * mesh.getVertexNormals().size() * 3, &(mesh.getVertexNormals().at(0)), GL_DYNAMIC_DRAW);               //存储 法线数据
		glVertexAttribPointer(c1, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, (void*)0);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangle_handle);
		//线框模式
		if (polygonMode)
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		else
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glDrawElements(GL_TRIANGLES, faceNum, GL_UNSIGNED_INT, (void*)0);

		mesh.release();
		m_indices.clear();
	}

	//渲染缝合线  ===========================

	glUseProgram(seam_program);
	GLuint c4 = glGetUniformLocation(seam_program, "pos");
	glEnableVertexAttribArray(c4);
	//	glLineWidth(2.0);
	for (int k = 0; k < model.getTriangleModels().size(); k++)
	{
		IndexedFaceMesh m = model.getTriangleModels()[k]->getParticleMesh();
		for (int i = 0; i < m.getBorderEdgesNum(); i++)
		{
			IndexedFaceMesh::Edge e = m.getEdges().at(m.getBorderEdges().at(i));
			unsigned int indexOffset = model.getTriangleModels()[k]->getIndexOffset();
			unsigned int  indices[] = { e.m_vert[0] ,e.m_vert[1] };

			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, seam_VEO);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

			point = m.numVertices();
			glBindBuffer(GL_ARRAY_BUFFER, seam_VBO);
			glBufferData(GL_ARRAY_BUFFER, sizeof(float) * point * 3, &(pd.getPosition(indexOffset)), GL_DYNAMIC_DRAW);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, (void*)0);

			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, seam_VEO);
			glDrawElements(GL_LINES, 2, GL_UNSIGNED_INT, 0);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			glBindVertexArray(0);
		}
		m.release();
	}
	pd.release();

	//====================================人体

	if (conf.getBodyPath() != "")
	{
		glUseProgram(phong_program_body);
		uniloc = glGetUniformLocation(phong_program_body, "light_position");
		glUniform3fv(uniloc, 1, light_position);
		glLineWidth(1.0);
		GLuint c2 = glGetAttribLocation(phong_program_body, "position");
		GLuint c3 = glGetAttribLocation(phong_program_body, "normal");
		glEnableVertexAttribArray(c2);
		glEnableVertexAttribArray(c3);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangle_handle_body);
		body = model.getRigidBodies().at(0);
		mesh_body = body->getMesh();
		body_indices = mesh_body.getFaces();
		body_faceNum = body_indices.size();
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * body_faceNum, &body_indices[0], GL_STATIC_DRAW);    //记录面的索引，int型

		body_VD = body->getVertexData();
		pointNum = body_VD.size();
		glBindBuffer(GL_ARRAY_BUFFER, vertex_handle_body);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * pointNum * 3, &(body_VD.getPosition(0)), GL_DYNAMIC_DRAW); //记录点的位置，每个点3个float值
		glVertexAttribPointer(c2, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, (void*)0);

		glBindBuffer(GL_ARRAY_BUFFER, normal_handle_body);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * mesh_body.getVertexNormals().size() * 3, &(mesh_body.getVertexNormals().at(0)), GL_DYNAMIC_DRAW);
		glVertexAttribPointer(c3, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, (void*)0);               //记录法线信息，每个面一个法线，3个float

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangle_handle_body);
		//消除线框模式
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glDrawElements(GL_TRIANGLES, body_faceNum, GL_UNSIGNED_INT, (void*)0);

		glDisableVertexAttribArray(c2);
		glDisableVertexAttribArray(c3);


		mesh_body.release();
		body_indices.clear();
		body_VD.release();
	}

	glutSwapBuffers();
}

void keyPress(unsigned char key, int mousex, int mousey)
{
	switch (key)
	{
		case 27: exit(0);
		case 'a':
		{
			zoom -= 2;
			if (zoom < 0.3) zoom = 0.3;
			break;
		}
		case 'z':
		{
			zoom += 2;
			break;
		}
		case 'j':
		{
			swing_angle += 3;
			break;
		}
		case 'l':
		{
			swing_angle -= 3;
			break;
		}
		case 'i':
		{
			elevate_angle += 3;
			break;
		}
		case 'k':
		{
			elevate_angle -= 3;
			break;
		}
		case ' ':
		{
			if (doPause)doPause = false;
			else doPause = true;
			break;
		}
		// 依次 左、上、右、下键
		case 's':
		{
			center[0] -= 0.3;
			break;
		}
		case 'e':
		{
			center[1] += 0.3;
			break;
		}
		case 'f':
		{
			center[0] += 0.3;
			break;
		}
		case 'd':
		{
			center[1] -= 0.3;
			break;
		}
		case 't':
		{
			center[2] += 0.3;
			break;
		}
		case 'y':
		{
			center[2] -= 0.3;
			break;
		}
		case 'p':
		{
			if (polygonMode)polygonMode = false;
			else
			{
				polygonMode = true;
			}
			break;
		}
		case 'm':
		{
	//		saveCurrentPatternOBJ(model);
			break;
		}
		case '=':
		{
			moveAvaterToRight();
			break;
		}
		case '-':
		{
			moveAvaterToLeft();
			break;
		}
	}
	glutPostRedisplay();
}

void mouseClick(int, int, int, int)
{

}

void mouseMotion(int, int)
{

}

void moveAvaterToLeft()
{
	cout << "move to left ........................." << endl;
	VertexData& vd = model.getRigidBodies()[0]->getVertexData();
	VertexData& vd_local = model.getRigidBodies()[0]->getVertexDataLocal();
	for (int i = 0; i < vd.size(); i++)
	{
		vd.setPosition(i, vd.getPosition(i) + Vector3r(0.002, 0, 0));
		vd_local.setPosition(i, vd_local.getPosition(i) + Vector3r(0.002, 0, 0));
	}
	model.getRigidBodies()[0]->updateMeshNormals(vd);
	model.getRigidBodies()[0]->updateMeshNormals(vd_local);
}

void moveAvaterToRight()
{
	cout << "move to right ........................." << endl;
	VertexData& vd = model.getRigidBodies()[0]->getVertexData();
	VertexData& vd_local = model.getRigidBodies()[0]->getVertexDataLocal();
	for (int i = 0; i < vd.size(); i++)
	{
		vd.setPosition(i, vd.getPosition(i) - Vector3r(0.002, 0, 0));
		vd_local.setPosition(i, vd_local.getPosition(i) - Vector3r(0.002, 0, 0));
	}
	model.getRigidBodies()[0]->updateMeshNormals(vd);
	model.getRigidBodies()[0]->updateMeshNormals(vd_local);
}
