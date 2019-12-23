#include "Timing.h"
#include "DistanceFieldCollisionDetection.h"
#include "IDFactory.h"
#include "omp.h"
#include <iostream>

#include "ConfigurationLoader.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <device_atomic_functions.h>
#include <thrust/host_vector.h> 
#include <thrust/device_vector.h>


using namespace PBD;
using namespace std;
using namespace Utilities;

int DistanceFieldCollisionDetection::DistanceFieldCollisionSphereOnFaces::TYPE_ID = 5;


DistanceFieldCollisionDetection::DistanceFieldCollisionDetection() :
	CollisionDetection()
{
}

DistanceFieldCollisionDetection::~DistanceFieldCollisionDetection()
{
}

void DistanceFieldCollisionDetection::updateBVH(SimulationModel& model)
{
	#pragma omp parallel default(shared)
	{
		// Update BVHs
#pragma omp for schedule(static)  

		for (int i = 0; i < (int)m_collisionObjects.size(); i++)
		{
			CollisionDetection::CollisionObject* co = m_collisionObjects[i];
			if (isDistanceFieldCollisionObject(co))                                   // 全为true
			{
				if ((co->m_bodyType == CollisionDetection::CollisionObject::TriangleModelCollisionObjectType))
				{
					DistanceFieldCollisionObjectOnFaces* sco = (DistanceFieldCollisionObjectOnFaces*)co;
					sco->m_bvhf.update();
				}
			}
		}
	}
}

void PBD::DistanceFieldCollisionDetection::collisionDetection(SimulationModel& mode)
{
}

void PBD::DistanceFieldCollisionDetection::collisionDetection(SimulationModel& model, Configuration& conf)
{
	model.resetContacts();
	const SimulationModel::RigidBodyVector& rigidBodies = model.getRigidBodies();
	const SimulationModel::TriangleModelVector& triModels = model.getTriangleModels();
	const ParticleData& pd = model.getParticles();

	//omp_set_num_threads(1);
	std::vector<std::vector<ContactData> > contacts_mt;	                                // 需要碰撞处理的数据
#ifdef _DEBUG
	const unsigned int maxThreads = 1;
#else
	const unsigned int maxThreads = omp_get_max_threads();

#endif
	contacts_mt.resize(maxThreads);
	//START_TIMING("-----碰撞");

	// Update BVHs
	//START_TIMING("-----bvhf更新");
	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static)  
		for (int i = 0; i < (int)m_collisionObjects.size(); i++)
		{
			CollisionDetection::CollisionObject* co = m_collisionObjects[i];
			if (isDistanceFieldCollisionObject(co))                                   // 全为true
			{
				DistanceFieldCollisionObjectOnFaces* sco = (DistanceFieldCollisionObjectOnFaces*)co;
				sco->m_bvhf.update();
				sco->m_bvhf.copyHullsMemory();
			}
		}
	}
	//STOP_TIMING_AVG_PRINT;

	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static)
		for (int i = 0; i < conf.getCollisionPairs().size(); i++)
		{
			CollisionDetection::CollisionObject* co1 = m_collisionObjects[conf.getCollisionPairs().at(i)[0]];
			CollisionDetection::CollisionObject* co2 = m_collisionObjects[conf.getCollisionPairs().at(i)[1]];
			if ((co1->m_bodyType == CollisionDetection::CollisionObject::TriangleModelCollisionObjectType)  //当一个为particle模型，一个为刚体模型时
				&& (co2->m_bodyType == CollisionDetection::CollisionObject::RigidBodyCollisionObjectType))
			{
				RigidBody* rb2 = rigidBodies[co2->m_bodyIndex];
				TriangleModel* tm = triModels[co1->m_bodyIndex];
				const unsigned int offset = tm->getIndexOffset();
				const IndexedFaceMesh& mesh = tm->getParticleMesh();
				const unsigned int numVert = mesh.numVertices();
				const Real restitutionCoeff = tm->getRestitutionCoeff() + rb2->getRestitutionCoeff();
				const Real frictionCoeff = tm->getFrictionCoeff() + rb2->getFrictionCoeff();
				float tolerance = tm->getCollisionCoeff();
				collisionDetectionRBSolidOnFaces(conf.getCollisionPairs().at(i)[0], conf.getCollisionPairs().at(i)[1], pd, offset, numVert, (DistanceFieldCollisionSphereOnFaces*)co1, rb2, (DistanceFieldCollisionSphereOnFaces*)co2,
					restitutionCoeff, frictionCoeff, contacts_mt,tolerance);
			}

			if ((co1->m_bodyType == CollisionDetection::CollisionObject::TriangleModelCollisionObjectType)  //两个皆为particle模型时
				&& (co2->m_bodyType == CollisionDetection::CollisionObject::TriangleModelCollisionObjectType))
			{
				TriangleModel* tm1 = triModels[co1->m_bodyIndex];
				const unsigned int offset1 = tm1->getIndexOffset();
				const IndexedFaceMesh& mesh1 = tm1->getParticleMesh();
				const unsigned int numVert1 = mesh1.numVertices();
				TriangleModel* tm2 = triModels[co2->m_bodyIndex];
				const unsigned int offset2 = tm2->getIndexOffset();
				const IndexedFaceMesh& mesh2 = tm2->getParticleMesh();
				const unsigned int numVert2 = mesh2.numVertices();
				const Real restitutionCoeff = tm1->getRestitutionCoeff() + tm2->getRestitutionCoeff();
				const Real frictionCoeff = tm1->getFrictionCoeff() + tm2->getFrictionCoeff();
				collisionDetectionParticlesOnFaces(conf.getCollisionPairs().at(i)[0], conf.getCollisionPairs().at(i)[1], pd, offset1, numVert1, (DistanceFieldCollisionSphereOnFaces*)co1, offset2, numVert2,
					(DistanceFieldCollisionSphereOnFaces*)co2, restitutionCoeff, frictionCoeff, contacts_mt);
			}
		}
	}
	//STOP_TIMING_AVG_PRINT;

	//START_TIMING("-----生成约束");
	
	for (unsigned int i = 0; i < contacts_mt.size(); i++)
	{
		for (unsigned int j = 0; j < contacts_mt[i].size(); j++)               //多个线程
		{
			if (contacts_mt[i][j].m_type == 1)
				addParticleRigidBodyContact(contacts_mt[i][j].m_first, contacts_mt[i][j].m_second, contacts_mt[i][j].m_index1, contacts_mt[i][j].m_index2,
					contacts_mt[i][j].m_cp1, contacts_mt[i][j].m_cp2, contacts_mt[i][j].m_rbCenter, contacts_mt[i][j].m_normal,
					contacts_mt[i][j].m_dist, contacts_mt[i][j].m_restitution, contacts_mt[i][j].m_friction);
			else if (contacts_mt[i][j].m_type == 2)
			{
				addParticlesContact(contacts_mt[i][j].m_first, contacts_mt[i][j].m_second, contacts_mt[i][j].m_index1, contacts_mt[i][j].m_index2, contacts_mt[i][j].m_index3, contacts_mt[i][j].m_index4,
					contacts_mt[i][j].m_cp1, contacts_mt[i][j].m_cp2, contacts_mt[i][j].m_normal,
					contacts_mt[i][j].m_dist, contacts_mt[i][j].m_restitution, contacts_mt[i][j].m_friction);
			}
		}
	}
	//STOP_TIMING_AVG_PRINT;
}

__device__ bool checkOverlap(float r1, float* x1, float r2, float* x2)
{
	return (r1 + r2) * (r1 + r2) > ((x1[0] - x2[0]) * (x1[0] - x2[0]) + (x1[1] - x2[1]) * (x1[1] - x2[1]) + (x1[2] - x2[2]) * (x1[2] - x2[2]));
}

__device__ bool isLeaf(BVHOnFaces::Node node)
{
	return (node.children[0] < 0 && node.children[1] < 0);
}

__global__ void traverseIterative16Road(BVHOnFaces::Node * d_nodes, BoundingSphere * d_cloBS, BVHOnFaces::Node * d_bodyNodes, BoundingSphere * d_bodyBS,
	int N, int* d_contact, int numPerContact, unsigned int* d_bodyLeafNode, unsigned int* d_cloIndex16)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < N)
	{
		int child = index % 16;
		index = (int)(index / 16);

		__shared__ unsigned int* bodyLeafNode;
		bodyLeafNode = d_bodyLeafNode;
		__shared__ BVHOnFaces::Node * nodes;
		nodes = d_nodes;
		__shared__ unsigned int* cloIndex16;
		cloIndex16 = d_cloIndex16;
		__shared__ BoundingSphere * cloBS;
		cloBS = d_cloBS;

		BoundingSphere bss = d_bodyBS[bodyLeafNode[index]];
		float r = bss.m_r;
		float* x = bss.m_x;

		unsigned int stack[64];
		unsigned int* stackPtr = stack;
		*stackPtr++ = NULL;
		unsigned int node = cloIndex16[child];

		__shared__ int startIndex[2];
		int count = threadIdx.x / 16;
		//startIndex[count] = index * numPerContact;
		atomicExch(&startIndex[count], index * numPerContact);
		int a = 0;
		for (int i = 0; i < numPerContact; i++)
		{
			atomicExch(&d_contact[startIndex[count] + i], 0);
			//d_contact[startIndex[count] + i] = 0;
		}

		unsigned int childL;
		unsigned int childR;
		bool overlapL;
		bool overlapR;
		bool isLeafL;
		bool isLeafR;
		bool traverseL;
		bool traverseR;
		do
		{
			childL = nodes[node].children[0];
			childR = nodes[node].children[1];
			overlapL = (checkOverlap(r, x, cloBS[childL].m_r, cloBS[childL].m_x));
			overlapR = (checkOverlap(r, x, cloBS[childR].m_r, cloBS[childR].m_x));
			isLeafL = isLeaf(nodes[childL]);
			isLeafR = isLeaf(nodes[childR]);
			traverseL = (overlapL && !isLeafL);
			traverseR = (overlapR && !isLeafR);
			int start;
			if (overlapL && isLeafL)
			{
				//先存的body , 后 clo
				// 问题：不同的线程读startIndex[count],同时加一，存在覆盖

				
				//atomicExch(&d_contact[atomicAdd(&startIndex[count], 1)], bodyLeafNode[index]);
				
				//atomicExch(&d_contact[atomicAdd(&startIndex[count], 1)], childL);
				d_contact[startIndex[count]++] = bodyLeafNode[index];
				d_contact[startIndex[count]++] = childL;
			}
			if (overlapR && isLeafR)
			{
				/*start = atomicAdd(&startIndex[count], 1);
				atomicExch(&d_contact[start], bodyLeafNode[index]);
				start = atomicAdd(&startIndex[count], 1);
				atomicExch(&d_contact[start], childR);*/
				d_contact[startIndex[count]++] = bodyLeafNode[index];
				d_contact[startIndex[count]++] = childR;
			}

			if (!traverseL && !traverseR)
			{
				*stackPtr = 0;
				node = *(--stackPtr);
			}
			else
			{
				node = (traverseL) ? childL : childR;
				if (traverseL && traverseR)
					* (stackPtr++) = childR;
			}
		} while (node != NULL);
	}
}

__global__ void traverseIterative(BVHOnFaces::Node * d_nodes, BoundingSphere * bs, BVHOnFaces::Node * d_bodyNodes, BoundingSphere * d_bodyBS, int N,
	int* d_contact, int numPerContact, unsigned int* d_bodyLeafNode)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < N)
	{
		BoundingSphere bss = d_bodyBS[d_bodyLeafNode[index]];
		float r = bss.m_r;
		float* x = bss.m_x;

		unsigned int stack[32];
		unsigned int* stackPtr = stack;
		*stackPtr++ = NULL;
		unsigned int node = 0;
		unsigned int startIndex = index * numPerContact;
		for (int i = 0; i < numPerContact; i++)
		{
			d_contact[startIndex + i] = 0;
		}
		unsigned int childL;
		unsigned int childR;
		bool overlapL;
		bool overlapR;
		bool isLeafL;
		bool isLeafR;
		bool traverseL;
		bool traverseR;
		do
		{
			childL = d_nodes[node].children[0];
			childR = d_nodes[node].children[1];
			overlapL = (checkOverlap(r, x, bs[childL].m_r, bs[childL].m_x));
			overlapR = (checkOverlap(r, x, bs[childR].m_r, bs[childR].m_x));
			isLeafL = isLeaf(d_nodes[childL]);
			isLeafR = isLeaf(d_nodes[childR]);
			traverseL = (overlapL && !isLeafL);
			traverseR = (overlapR && !isLeafR);

			if (overlapL && isLeafL)
			{
				//先存的body , 后 clo
				d_contact[startIndex++] = d_bodyLeafNode[index];
				d_contact[startIndex++] = childL;
			}
			if (overlapR && isLeafR)
			{
				d_contact[startIndex++] = d_bodyLeafNode[index];
				d_contact[startIndex++] = childR;
			}

			if (!traverseL && !traverseR)
				node = *--stackPtr;
			else
			{
				node = (traverseL) ? childL : childR;
				if (traverseL && traverseR)
					* stackPtr++ = childR;
			}
		} while (node != NULL);
	}
}

__global__ void contactCallback(int* d_contactRes, int N, float* d_collisionInfo,
	unsigned int* d_cloFacesIndex, Vector3r * d_cloVertices, unsigned int* d_bodyFacesIndex, Vector3r * d_bodyVertices,
	unsigned int* d_cloList, unsigned int* d_bodyList, BVHOnFaces::Node * d_cloNodes, BVHOnFaces::Node * d_bodyNodes,float tolerance)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < N)
	{
		int bodyIndex = d_contactRes[index * 2];
		int cloIndex = d_contactRes[index * 2 + 1];

		unsigned int cloFaceIndex = d_cloList[d_cloNodes[cloIndex].begin] * 3;
		unsigned int bodyFaceIndex = d_bodyList[d_bodyNodes[bodyIndex].begin] * 3;
		Vector3r bodyFace0 = d_bodyVertices[d_bodyFacesIndex[bodyFaceIndex]];
		Vector3r bodyFace1 = d_bodyVertices[d_bodyFacesIndex[bodyFaceIndex + 1]];
		Vector3r bodyFace2 = d_bodyVertices[d_bodyFacesIndex[bodyFaceIndex + 2]];

		for (int i = 0; i < 3; i++)
		{
			Vector3r x_w = d_cloVertices[d_cloFacesIndex[cloFaceIndex + i]];

			Vector3r n = (bodyFace1 - bodyFace0).cross(bodyFace2 - bodyFace0);
			n = n.normalize();

			Vector3r pa = x_w - bodyFace0 - n * tolerance;
			Vector3r pb = x_w - bodyFace1 - n * tolerance;
			Vector3r pc = x_w - bodyFace2 - n * tolerance;

			Vector3r cp_w;
			bool coll = (pb.cross(pc).dot(n) > 0) && (pb.dot(n) < 0) && (pc.cross(pa).dot(n) > 0) && (pa.cross(pb).dot(n) > 0);
			if (coll)
			{
				float t = (n.dot(bodyFace0) - n.dot(x_w)) / (n.dot(n));
				cp_w = x_w + n * (tolerance + t);
				Real dist = (x_w - cp_w).norm();
				int startIndex = index * 39 + i * 13;
				d_collisionInfo[startIndex + 0] = (cloIndex * 1.0f);
				d_collisionInfo[startIndex + 1] = (bodyIndex * 1.0f);
				d_collisionInfo[startIndex + 2] = (x_w[0]);
				d_collisionInfo[startIndex + 3] = (x_w[1]);
				d_collisionInfo[startIndex + 4] = (x_w[2]);
				d_collisionInfo[startIndex + 5] = (cp_w[0]);
				d_collisionInfo[startIndex + 6] = (cp_w[1]);
				d_collisionInfo[startIndex + 7] = (cp_w[2]);
				d_collisionInfo[startIndex + 8] = (n[0]);
				d_collisionInfo[startIndex + 9] = (n[1]);
				d_collisionInfo[startIndex + 10] = (n[2]);
				d_collisionInfo[startIndex + 11] = (dist);
				d_collisionInfo[startIndex + 12] = (i * 1.0f);
			}
		}
	}
}

__global__ void traverseIterative16RoadOnParticles(BVHOnFaces::Node* d_clo1Nodes, BoundingSphere* d_clo1BS, BVHOnFaces::Node* d_clo2Nodes, BoundingSphere* d_clo2BS,
	int N, int* d_contact, int numPerContact, unsigned int* d_clo2LeafNode, unsigned int* d_clo1Index16,
	Vector3r * d_clo1FaceCenter, Vector3r* d_clo2FaceCenter, unsigned int* d_clo1List, unsigned int* d_clo2List)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < N)
	{
		int child = index % 16;
		index = (int)(index / 16);

		__shared__ unsigned int* clo2LeafNode;
		clo2LeafNode = d_clo2LeafNode;
		__shared__ BVHOnFaces::Node * nodes;
		nodes = d_clo1Nodes;
		__shared__ unsigned int* cloIndex16;
		cloIndex16 = d_clo1Index16;
		__shared__ BoundingSphere * cloBS;
		cloBS = d_clo1BS;

		BoundingSphere bss = d_clo2BS[clo2LeafNode[index]];
		float r = bss.m_r;
		float* x = bss.m_x;

		unsigned int stack[64];
		unsigned int* stackPtr = stack;
		*stackPtr++ = NULL;
		unsigned int node = cloIndex16[child];

		__shared__ int startIndex[2];
		int count = threadIdx.x / 16;
		startIndex[count] = index * numPerContact;

		unsigned int childL;
		unsigned int childR;
		bool overlapL;
		bool overlapR;
		bool isLeafL;
		bool isLeafR;
		bool traverseL;
		bool traverseR;
		do
		{
			childL = nodes[node].children[0];
			childR = nodes[node].children[1];
			Vector3r clo1LFacecCenter = d_clo1FaceCenter[d_clo1List[d_clo1Nodes[childL].begin] * 3];
			Vector3r clo1RFacecCenter = d_clo1FaceCenter[d_clo1List[d_clo1Nodes[childR].begin] * 3];
			Vector3r clo2FacecCenter = d_clo2FaceCenter[d_clo2List[d_clo2Nodes[clo2LeafNode[index]].begin] * 3];
			overlapL = ((clo1LFacecCenter - clo2FacecCenter).norm() > 0.001) && (checkOverlap(r, x, cloBS[childL].m_r, cloBS[childL].m_x));
			overlapR = ((clo1RFacecCenter - clo2FacecCenter).norm() > 0.001) && (checkOverlap(r, x, cloBS[childR].m_r, cloBS[childR].m_x));
			isLeafL = isLeaf(nodes[childL]);
			isLeafR = isLeaf(nodes[childR]);
			traverseL = (overlapL && !isLeafL);
			traverseR = (overlapR && !isLeafR);

			if (overlapL && isLeafL)
			{
				//先存的body , 后 clo
				d_contact[startIndex[count]++] = clo2LeafNode[index];
				d_contact[startIndex[count]++] = childL;
			}
			if (overlapR && isLeafR)
			{
				d_contact[startIndex[count]++] = clo2LeafNode[index];
				d_contact[startIndex[count]++] = childR;
			}

			if (!traverseL && !traverseR)
			{
				*stackPtr = 0;
				node = *(--stackPtr);
			}
			else
			{
				node = (traverseL) ? childL : childR;
				if (traverseL && traverseR)
					* (stackPtr++) = childR;
			}
		} while (node != NULL);
	}
}

__global__ void contactCallbackOnParticles(int* d_contactRes, int N, float* d_collisionInfo,
	unsigned int* d_clo1FacesIndex, Vector3r* d_clo1Vertices, unsigned int* d_clo2FacesIndex, Vector3r* d_clo2Vertices,
	unsigned int* d_clo1List, unsigned int* d_clo2List, BVHOnFaces::Node* d_clo1Nodes, BVHOnFaces::Node* d_clo2Nodes)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < N)
	{
		float tolerance = 0.012;
		int clo2Index = d_contactRes[index * 2];
		int clo1Index = d_contactRes[index * 2 + 1];
		int clo1FaceIndex = d_clo1List[d_clo1Nodes[clo1Index].begin] * 3;
		int clo2FaceIndex = d_clo2List[d_clo2Nodes[clo2Index].begin] * 3;
		//printf("\n %d %d %d %d %d %d %d %d", clo2Index, clo1Index, d_clo1Nodes[clo1Index].begin, d_clo2Nodes[clo2Index].begin,
		//	d_clo1List[d_clo1Nodes[clo1Index].begin], d_clo2List[d_clo2Nodes[clo2Index].begin],
		//	clo1FaceIndex, clo2FaceIndex);
		Vector3r clo2Face0 = d_clo2Vertices[d_clo2FacesIndex[clo2FaceIndex]];
		Vector3r clo2Face1 = d_clo2Vertices[d_clo2FacesIndex[clo2FaceIndex + 1]];
		Vector3r clo2Face2 = d_clo2Vertices[d_clo2FacesIndex[clo2FaceIndex + 2]];

		for (int i = 0; i < 3; i++)
		{
			Vector3r x_w = d_clo1Vertices[d_clo1FacesIndex[clo1FaceIndex + i]];

			Vector3r n = (clo2Face1 - clo2Face0).cross(clo2Face2 - clo2Face0);
			n = n.normalize();

			Vector3r pa = x_w - clo2Face0;
			if (pa.dot(n) < 0) n = -n;

			pa = pa - n * tolerance;
			Vector3r pb = x_w - clo2Face1 - n * tolerance;
			Vector3r pc = x_w - clo2Face2 - n * tolerance;

			for (int j = 0; j < 13; j++)
				d_collisionInfo[index * 39 + i * 13 + j] = 0.0;
			Vector3r cp_w;
			bool coll = (pb.cross(pc).dot(n) > 0) && (pb.dot(n) < 0) && (pc.cross(pa).dot(n) > 0) && (pa.cross(pb).dot(n) > 0);
			coll = coll || (pb.cross(pc).dot(n) < 0) && (pb.dot(n) < 0) && (pc.cross(pa).dot(n) < 0) && (pa.cross(pb).dot(n) < 0);
			if (coll)
			{
				float t = (n.dot(clo2Face0) - n.dot(x_w)) / (n.dot(n));
				cp_w = x_w + n * (tolerance + t);
				Real dist = (x_w - cp_w).norm();
				int startIndex = index * 39 + i * 13;
				d_collisionInfo[startIndex + 0] = (clo1Index * 1.0f);
				d_collisionInfo[startIndex + 1] = (clo2Index * 1.0f);
				d_collisionInfo[startIndex + 2] = (x_w[0]);
				d_collisionInfo[startIndex + 3] = (x_w[1]);
				d_collisionInfo[startIndex + 4] = (x_w[2]);
				d_collisionInfo[startIndex + 5] = (cp_w[0]);
				d_collisionInfo[startIndex + 6] = (cp_w[1]);
				d_collisionInfo[startIndex + 7] = (cp_w[2]);
				d_collisionInfo[startIndex + 8] = (n[0]);
				d_collisionInfo[startIndex + 9] = (n[1]);
				d_collisionInfo[startIndex + 10] = (n[2]);
				d_collisionInfo[startIndex + 11] = (dist);
				d_collisionInfo[startIndex + 12] = (i * 1.0f);
			}
		}
	}
}

/////////////////////////////////////////////////////////
/////  OnFaces
/////////////////////////////////////////////////////////
//服装pattern之间的碰撞
void DistanceFieldCollisionDetection::collisionDetectionParticlesOnFaces(unsigned int ii, unsigned int kk, const ParticleData& pd, const unsigned int offset1, const unsigned int numVert1,
	DistanceFieldCollisionSphereOnFaces* co1, const unsigned int offset2, const unsigned int numVert2, DistanceFieldCollisionSphereOnFaces* co2,
	const Real restitutionCoeff, const Real frictionCoeff,
	std::vector<std::vector<ContactData>>& contacts_mt)
{
	BVHOnFaces clo1BVHF = ((DistanceFieldCollisionDetection::DistanceFieldCollisionSphereOnFaces*)co1)->m_bvhf;
	BVHOnFaces clo2BVHF = ((DistanceFieldCollisionDetection::DistanceFieldCollisionSphereOnFaces*)co2)->m_bvhf;

	cudaError_t err = cudaSuccess;

	//======= cloth1 nodes ===========
	BVHOnFaces::Node* d_clo1Nodes = clo1BVHF.d_nodes;

	//======= cloth1 hulls ===========
	BoundingSphere* d_clo1BS = clo1BVHF.d_hulls;

	//======= cloth2 nodes ===========
	BVHOnFaces::Node* d_clo2Nodes = clo2BVHF.d_nodes;

	//======== cloth2 leaf node index
	unsigned int* d_clo2LeafNode = clo2BVHF.d_leaf;

	//======= cloth2 hulls ===========
	BoundingSphere* d_clo2BS = clo2BVHF.d_hulls;

	//======== contact info ===========
	int* h_contact;
	int numPerContact = 40;
	int clo2LeafNodeNum = clo2BVHF.getLeaf().size();
	h_contact = (int*)malloc(sizeof(int) * clo2LeafNodeNum * numPerContact);
	int* d_contact;
	err = cudaMalloc((int**)& d_contact, sizeof(int) * clo2LeafNodeNum * numPerContact);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "failed to allocate d_contact (error code: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaMemset(d_contact, 0, sizeof(int) * clo2LeafNodeNum * numPerContact);

	Vector3r* d_clo1FaceCenter = clo1BVHF.d_faceCenter;
	Vector3r* d_clo2FaceCenter = clo2BVHF.d_faceCenter;
	unsigned int* d_clo1List = clo1BVHF.d_lst;
	unsigned int* d_clo2List = clo2BVHF.d_lst;
	unsigned int* d_clo1Index16 = clo1BVHF.d_index32;
	//START_TIMING("-----CUDA碰撞test");
	dim3 block(32);
	dim3 grid((clo2LeafNodeNum * 16 + block.x - 1) / block.x);
	//traverseIterative16RoadOnParticles << <grid, block >> > (d_clo1Nodes, d_clo1BS, d_clo2Nodes, d_clo2BS, clo2LeafNodeNum * 16, d_contact, numPerContact,
	//														d_clo2LeafNode, d_clo1Index16, d_clo1FaceCenter, d_clo2FaceCenter,d_clo1List,d_clo2List);
	traverseIterative16Road << <grid, block >> > (d_clo1Nodes, d_clo1BS, d_clo2Nodes, d_clo2BS, clo2LeafNodeNum * 16, d_contact, numPerContact, d_clo2LeafNode, d_clo1Index16);
	cudaThreadSynchronize();
	//STOP_TIMING_AVG_PRINT;

	//START_TIMING("----111----");
	//======== copy result ============
	err = cudaMemcpy(h_contact, d_contact, sizeof(int) * clo2LeafNodeNum * numPerContact, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy d_contact (error code: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	vector<int> res;
	int count = clo2LeafNodeNum * numPerContact;
	for (int i = 0; i < count; i = i + 2)
	{
		if (h_contact[i] != 0)
		{
			res.push_back(h_contact[i]);
			res.push_back(h_contact[i + 1]);
		}
	}
	//cout << res.size() << " ";

	int* h_contactRes = res.data();
	int* d_contactRes;
	err = cudaMalloc((int**)& d_contactRes, sizeof(int) * res.size());
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate d_contactRes (error code: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(d_contactRes, h_contactRes, sizeof(int) * (res.size()), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy d_contactRes (error code: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	//STOP_TIMING_AVG_PRINT;

	//START_TIMING("----222----");
	// collisonInfo
	int collisionNum = (res.size()) / 2;
	float* h_collisionInfo;
	h_collisionInfo = (float*)malloc(sizeof(float) * collisionNum * 39);
	float* d_collisionInfo;
	err = cudaMalloc((float**)& d_collisionInfo, sizeof(float) * collisionNum * 39);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate d_collisionInfo (error code: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	//STOP_TIMING_AVG_PRINT;

	unsigned int* d_clo1FacesIndex = clo1BVHF.d_faceIndex;
	Vector3r* d_clo1Vertices = clo1BVHF.d_vertices;
	unsigned int* d_clo2FacesIndex = clo2BVHF.d_faceIndex;
	Vector3r* d_clo2Vertices = clo2BVHF.d_vertices;

	//START_TIMING("-----CUDA碰撞callback");
	dim3 block2(512);
	dim3 grid2((collisionNum + block2.x - 1) / block2.x);
	contactCallbackOnParticles << <grid2, block2 >> > (d_contactRes, collisionNum, d_collisionInfo,
		d_clo1FacesIndex, d_clo1Vertices, d_clo2FacesIndex, d_clo2Vertices, d_clo1List, d_clo2List,
		d_clo1Nodes, d_clo2Nodes);
	cudaThreadSynchronize();
	//STOP_TIMING_AVG_PRINT;

	//START_TIMING("----555----");
	err = cudaMemcpy(h_collisionInfo, d_collisionInfo, sizeof(float) * 39 * collisionNum, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "failed to copy d_collisionInfo (error code: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	int tid = omp_get_thread_num();
	for (int i = 0; i < collisionNum * 3; i++)
	{
		int count = i * 13;
		if (h_collisionInfo[count] >= 0.1)
		{
			float* clo2Hull = clo2BVHF.hull((int)h_collisionInfo[count + 1]).x();
			contacts_mt[tid].push_back({ ii,kk,2,clo2BVHF.getIndice((int)h_collisionInfo[count+1],(int)h_collisionInfo[count + 12]) + offset2,
				clo1BVHF.getIndice((int)h_collisionInfo[count],0)+offset1,clo1BVHF.getIndice((int)h_collisionInfo[count],1) + offset1,
				clo1BVHF.getIndice((int)h_collisionInfo[count],2) + offset1,
				Vector3r(h_collisionInfo[count + 2], h_collisionInfo[count + 3] ,h_collisionInfo[count + 4]),
				Vector3r(h_collisionInfo[count + 5], h_collisionInfo[count + 6] ,h_collisionInfo[count + 7]),
				Vector3r(clo2Hull[0],clo2Hull[1],clo2Hull[2]),
				Vector3r(h_collisionInfo[count + 8], h_collisionInfo[count + 9] ,h_collisionInfo[count + 10]),
				h_collisionInfo[count + 11],restitutionCoeff,frictionCoeff });
		}
	}
	err = cudaFree(d_contact);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "failed to free d_contact (error code: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaFree(d_contactRes);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "failed to free d_contactRes (error code: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaFree(d_collisionInfo);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "failed to free d_collisionInfo (error code: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	free(h_contact);
	free(h_collisionInfo);
	res.clear();
	//STOP_TIMING_AVG_PRINT;
}




void DistanceFieldCollisionDetection::collisionDetectionRBSolidOnFaces(unsigned int ii, unsigned int kk, const ParticleData & pd, const unsigned int offset, const unsigned int numVert,
	DistanceFieldCollisionSphereOnFaces * co1, RigidBody * rb2, DistanceFieldCollisionSphereOnFaces * co2,
	const Real restitutionCoeff, const Real frictionCoeff, std::vector<std::vector<ContactData> > & contacts_mt,float tolerance)
{
	//	START_TIMING("collisionDetectionRBSolidOnFaces");
	//START_TIMING("-------cuda test ready");
	BVHOnFaces cloBVHF = ((DistanceFieldCollisionDetection::DistanceFieldCollisionSphereOnFaces*)co1)->m_bvhf;
	BVHOnFaces bodyBVHF = ((DistanceFieldCollisionDetection::DistanceFieldCollisionSphereOnFaces*)co2)->m_bvhf;

	cudaError_t err = cudaSuccess;

	//======= cloth nodes ===========
	BVHOnFaces::Node* d_cloNodes = cloBVHF.d_nodes;

	//======= cloth hulls ===========
	BoundingSphere* d_cloBS = cloBVHF.d_hulls;

	//======= body nodes ===========
	BVHOnFaces::Node* d_bodyNodes = bodyBVHF.d_nodes;

	//======== body leaf node index
	unsigned int* d_bodyLeafNode = bodyBVHF.d_leaf;

	//======= body hulls ===========
	BoundingSphere* d_bodyBS = bodyBVHF.d_hulls;

	//======== contact info ===========
	int* h_contact;
	int numPerContact = 80;
	int bodyLeafNodeNum = bodyBVHF.getLeaf().size();
	h_contact = (int*)malloc(sizeof(int) * bodyLeafNodeNum * numPerContact);
	int* d_contact;
	err = cudaMalloc((int**)& d_contact, sizeof(int) * bodyLeafNodeNum * numPerContact);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "failed to allocate d_contact (error code: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	//STOP_TIMING_AVG_PRINT;

	//if (bodyLeafNodeNum > 2000)
	//{
		unsigned int* d_cloIndex16 = cloBVHF.d_index32;
		//START_TIMING("-----CUDA碰撞test");
		dim3 block(32);
		dim3 grid((bodyLeafNodeNum * 16 + block.x - 1) / block.x);
		traverseIterative16Road << <grid, block >> > (d_cloNodes, d_cloBS, d_bodyNodes, d_bodyBS, bodyLeafNodeNum * 16, d_contact, numPerContact, d_bodyLeafNode, d_cloIndex16);
		cudaThreadSynchronize();
		//STOP_TIMING_AVG_PRINT;
	//}
	//START_TIMING("----111----");
	//======== copy result ============
	err = cudaMemcpy(h_contact, d_contact, sizeof(int) * bodyLeafNodeNum * numPerContact, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy d_contact (error code: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	vector<int> res;
	int count = bodyLeafNodeNum * numPerContact;
	for (int i = 0; i < count; i = i + 2)
	{
		if (h_contact[i] != 0)
		{
			res.push_back(h_contact[i]);
			res.push_back(h_contact[i + 1]);
		}
	}
	//cout << res.size() << " ";
	
	int* h_contactRes = res.data();
	int* d_contactRes;
	err = cudaMalloc((int**)& d_contactRes, sizeof(int) * res.size());
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate d_contactRes (error code: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(d_contactRes, h_contactRes, sizeof(int) * (res.size()), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy d_contactRes (error code: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	//STOP_TIMING_AVG_PRINT;

	//START_TIMING("----444----");
		// collisonInfo
	int collisionNum = (res.size()) / 2;
	float* h_collisionInfo;
	h_collisionInfo = (float*)malloc(sizeof(float) * collisionNum * 39);
	float* d_collisionInfo;
	err = cudaMalloc((float**)& d_collisionInfo, sizeof(float) * collisionNum * 39);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate d_collisionInfo (error code: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaMemset(d_collisionInfo, 0.0, sizeof(float)* collisionNum * 39);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to memset d_collisionInfo (error code: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	//STOP_TIMING_AVG_PRINT;

	unsigned int* d_cloFacesIndex = cloBVHF.d_faceIndex;
	Vector3r* d_cloVertices = cloBVHF.d_vertices;
	unsigned int* d_bodyFacesIndex = bodyBVHF.d_faceIndex;
	Vector3r* d_bodyVertices = bodyBVHF.d_vertices;
	unsigned int* d_cloList = cloBVHF.d_lst;
	unsigned int* d_bodyList = bodyBVHF.d_lst;

	//START_TIMING("-----CUDA碰撞callback");
	dim3 block2(1024);
	dim3 grid2((collisionNum + block2.x - 1) / block2.x);
	contactCallback << <grid2, block2 >> > (d_contactRes, collisionNum, d_collisionInfo,
											d_cloFacesIndex,d_cloVertices,d_bodyFacesIndex,d_bodyVertices,d_cloList,d_bodyList,
											d_cloNodes,d_bodyNodes,tolerance);
	cudaThreadSynchronize();

	//STOP_TIMING_AVG_PRINT;
	//START_TIMING("----555----");
	err = cudaMemcpy(h_collisionInfo, d_collisionInfo, sizeof(float) * 39 * collisionNum, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "failed to  copy d_collisionInfo (error code: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	int tid = omp_get_thread_num();
	vector<float> t;
	for (int i = 0; i < collisionNum * 3; i++)
	{
		int count = i * 13;
		if (h_collisionInfo[count] != 0.0)
		{
			float* bodyHull = bodyBVHF.hull((int)h_collisionInfo[count + 1]).x();
			contacts_mt[tid].push_back({ ii,kk,1,cloBVHF.getIndice((int)h_collisionInfo[count],(int)h_collisionInfo[count + 12]) + offset,0,NULL,NULL,
				Vector3r(h_collisionInfo[count + 2], h_collisionInfo[count + 3] ,h_collisionInfo[count + 4]),
				Vector3r(h_collisionInfo[count + 5], h_collisionInfo[count + 6] ,h_collisionInfo[count + 7]),
				Vector3r(bodyHull[0],bodyHull[1],bodyHull[2]),
				Vector3r(h_collisionInfo[count + 8], h_collisionInfo[count + 9] ,h_collisionInfo[count + 10]),
				h_collisionInfo[count + 11],restitutionCoeff,frictionCoeff });
		}
	}
	err = cudaFree(d_contact);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "failed to free d_contact (error code: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaFree(d_contactRes);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "failed to free d_contactRes (error code: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaFree(d_collisionInfo);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "failed to free d_collisionInfo (error code: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	free(h_contact);
	free(h_collisionInfo);
	res.clear();
	//STOP_TIMING_AVG_PRINT;
	//	STOP_TIMING_AVG_PRINT;
}


bool DistanceFieldCollisionDetection::isDistanceFieldCollisionObject(CollisionObject * co) const
{
	return (co->getTypeId() == DistanceFieldCollisionDetection::DistanceFieldCollisionSphereOnFaces::TYPE_ID);
}

//====================以三角面片为基本元素============================
//====================以三角面片为基本元素============================

void PBD::DistanceFieldCollisionDetection::addCollisionSphereOnFaces(const unsigned int bodyIndex, const unsigned int bodyType, vector<vector<unsigned int>> faces, const unsigned int numFaces, const Vector3r * vertices, const unsigned int numVertices, const Real radius, const bool testMesh, const bool invertSDF)
{
	DistanceFieldCollisionDetection::DistanceFieldCollisionSphereOnFaces* csf = new DistanceFieldCollisionDetection::DistanceFieldCollisionSphereOnFaces();
	csf->m_bodyIndex = bodyIndex;
	csf->m_bodyType = bodyType;
	csf->m_radius = radius;
	csf->m_bvhf.init(faces, numFaces, vertices, numVertices);
	csf->m_bvhf.construct();
	csf->m_bvhf.copyMemory();
	csf->m_testMesh = testMesh;
	if (invertSDF)                              // SDF：Signed Distance Field
		csf->m_invertSDF = -1.0;
	m_collisionObjects.push_back(csf);
}
void PBD::DistanceFieldCollisionDetection::popCollisionObject()
{
	m_collisionObjects.pop_back();
}


void DistanceFieldCollisionDetection::DistanceFieldCollisionObjectOnFaces::approximateNormal(const Vector3r & x, const Real tolerance, Vector3r & n)
{

}


bool DistanceFieldCollisionDetection::DistanceFieldCollisionObjectOnFaces::collisionTest(const Vector3r & x, const Real tolerance, Vector3r & cp, Vector3r & n, Real & dist, const Real maxDist)
{
	return true;
}

Real DistanceFieldCollisionDetection::DistanceFieldCollisionSphereOnFaces::distance(const Vector3r & x, const Real tolerance)
{
	return 0.0;
}

bool DistanceFieldCollisionDetection::DistanceFieldCollisionSphereOnFaces::collisionTest(const Vector3r & x, const Real tolerance, Vector3r & cp, Vector3r & n, Real & dist, const Real maxDist)
{
	return true;
}