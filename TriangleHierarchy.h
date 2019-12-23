#ifndef __TRIANGLEHIERARCHY_H__
#define __TRIANGLEHIERARCHY_H__

#include "Common.h"
#include "BoundingSphere.h"
#include "BVHTree.h"
#include <vector>
#include <MyVector.h>
#include <cuda_runtime.h>
using namespace std;

namespace PBD
{
	class BVHOnFaces :public BVHTree<BoundingSphere>
	{
	public:
		using super = BVHTree<BoundingSphere>;

		BVHOnFaces();

		void init(const vector<vector<unsigned int >>faces, const unsigned int numFaces, const Vector3r* vertices, const unsigned int numVertices);
		Vector3r const& entity_position(unsigned int i) const final;
		void compute_hull_approx(unsigned int b, unsigned int n, BoundingSphere& hull)
			const final;
		Vector3r getFaceCenter(unsigned int i);
		vector<Vector3r> getFace(unsigned int node_index);
		unsigned int getIndice(unsigned int node_index, unsigned int i);
		unsigned int getFaceNum() { return m_faces.size(); };
		vector<vector<unsigned int>> getFaces() { return m_faces; };

		void copyMemory();
		void copyHullsMemory();

	public:
		vector<vector<unsigned int>> m_faces;     //面
		unsigned int m_numFaces;
		vector<Vector3r> m_faceCenter;                    //面的质点
		const Vector3r* m_vertices;              //各点
		unsigned int m_numVertices;
		vector<unsigned int> m_faceIndex;

	public:
		unsigned int* d_lst;
		Node* d_nodes;
		BoundingSphere* d_hulls;
		unsigned int* d_index32;
		unsigned int* d_leaf;
		Vector3r* d_faceCenter;

		unsigned int* d_faceIndex;
		Vector3r* d_vertices;

	};
}

#endif 
