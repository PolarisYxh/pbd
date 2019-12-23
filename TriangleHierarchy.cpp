#include "TriangleHierarchy.h"
#include <iostream>
#include <unordered_set>
#include <set>


using pool_set = std::set<unsigned int>;
using namespace PBD;
using namespace std;

BVHOnFaces::BVHOnFaces()
	:super(0)
{

}

Vector3r const& BVHOnFaces::entity_position(unsigned int i)const
{
	return m_faceCenter[i];
}
void BVHOnFaces::init(vector<vector<unsigned int >> faces, const unsigned int numFaces, const Vector3r* vertices, const unsigned int numVertices)
{
	m_lst.resize(numFaces);
	m_faceCenter.resize(numFaces);
	m_faces = faces;
	m_numFaces = numFaces;
	m_vertices = vertices;
	m_numVertices = numVertices;
	//初始化各个面的中心
	for (unsigned int i = 0; i < numFaces; i++)
	{
		for (unsigned int j = 0; j < 3; j++)
		{
			m_faceCenter[i] += m_vertices[m_faces[i][j]];
			m_faceIndex.push_back(faces[i][j]);
		}
		m_faceCenter[i] /= 3.0;
	}
}

void BVHOnFaces::compute_hull_approx(unsigned int b, unsigned int n, BoundingSphere & hull) const
{
	// compute center
	Vector3r x;
	x.setZero();
	for (unsigned int i = b; i < b + n; i++)
		for (unsigned int j = 0; j < 3; j++)
		{
			Vector3r y = m_vertices[m_faces[m_lst[i]][j]];
			x += y;
		}

	x /= (float)(n * 3);

	float radius2 = 0.0;
	for (unsigned int i = b; i < b + n; i++)
	{
		for (unsigned int j = 0; j < 3; j++)
		{
			radius2 = std::max(radius2, (x - m_vertices[m_faces[m_lst[i]][j]]).squaredNorm());
		}
	}
//	cout << hull.x()[0] << " " << hull.x()[1] << " " << hull.x()[2] << endl;
	hull.x()[0] = x[0];
	hull.x()[1] = x[1];
	hull.x()[2] = x[2];
//	cout << hull.x()[0] << " " << hull.x()[1] << " " << hull.x()[2] << endl;
//	cout << hull.r();
	hull.r() = sqrt(radius2);
//	cout << " " << hull.r() << sqrt(radius2) << endl;
}

Vector3r PBD::BVHOnFaces::getFaceCenter(unsigned int node_index)
{

	return m_faceCenter[m_lst[m_nodes[node_index].begin]];
}

vector<Vector3r> BVHOnFaces::getFace(unsigned int node_index)
{
	unsigned int index = getFaceIndex(node_index);
	vector<unsigned int> indice = m_faces[index];
	vector<Vector3r> face;
	for (int i = 0; i < 3; i++)
	{
		face.push_back(m_vertices[indice[i]]);
	}
	return face;
}

unsigned int PBD::BVHOnFaces::getIndice(unsigned int node_index, unsigned int i)
{
	unsigned int index = getFaceIndex(node_index);
	return m_faces[index][i];
}

void PBD::BVHOnFaces::copyHullsMemory()
{
	cudaFree(d_hulls);
	cudaError_t err = cudaMalloc((BoundingSphere * *)& d_hulls, sizeof(BoundingSphere) * m_hulls.size());
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate bvhtree.inl : d_hulls  (error code:: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(d_hulls, m_hulls.data(), sizeof(BoundingSphere) * m_hulls.size(), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy bvhtree.inl : d_hulls  (error code:: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	//d_vertices
	cudaFree(d_vertices);
	err = cudaMalloc((Vector3r * *)& d_vertices, sizeof(Vector3r) * m_numVertices);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate bvhtree.inl : d_vertices  (error code:: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(d_vertices, m_vertices, sizeof(Vector3r) * m_numVertices, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy bvhtree.inl : d_vertices  (error code:: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void PBD::BVHOnFaces::copyMemory()
{
	//d_lst
	cudaError_t err = cudaMalloc((unsigned int**)& d_lst, sizeof(unsigned int) * m_lst.size());
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate bvhtree.inl : d_lst  (error code:: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(d_lst, m_lst.data(), sizeof(unsigned int) * m_lst.size(), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy bvhtree.inl : d_lst  (error code:: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	//d_nodes
	err = cudaMalloc((Node**)& d_nodes, sizeof(Node) * m_nodes.size());
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate bvhtree.inl : d_nodes  (error code:: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(d_nodes, m_nodes.data(), sizeof(Node) * m_nodes.size(), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy bvhtree.inl : d_nodes  (error code:: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	//d_hulls
	err = cudaMalloc((BoundingSphere * *)& d_hulls, sizeof(BoundingSphere) * m_hulls.size());
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate bvhtree.inl : d_hulls  (error code:: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(d_hulls, m_hulls.data(), sizeof(BoundingSphere) * m_hulls.size(), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy bvhtree.inl : d_hulls  (error code:: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	//d_index32
	err = cudaMalloc((unsigned int * *)& d_index32, sizeof(unsigned int) * m_index32.size());
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate bvhtree.inl : d_index32  (error code:: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(d_index32, m_index32.data(), sizeof(unsigned int) * m_index32.size(), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy bvhtree.inl : d_index32  (error code:: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	//d_leaf
	err = cudaMalloc((unsigned int**)& d_leaf, sizeof(unsigned int) * m_leaf.size());
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate bvhtree.inl : d_leaf  (error code:: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(d_leaf, m_leaf.data(), sizeof(unsigned int) * m_leaf.size(), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy bvhtree.inl : d_leaf  (error code:: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	//d_faceIndex
	err = cudaMalloc((unsigned int**)& d_faceIndex, sizeof(unsigned int) * m_faceIndex.size());
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate bvhtree.inl : d_faceIndex  (error code:: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(d_faceIndex, m_faceIndex.data(), sizeof(unsigned int) * m_faceIndex.size(), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy bvhtree.inl : d_faceIndex  (error code:: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	//d_vertices
	err = cudaMalloc((Vector3r **)& d_vertices, sizeof(Vector3r) * m_numVertices);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate bvhtree.inl : d_vertices  (error code:: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(d_vertices, m_vertices, sizeof(Vector3r) * m_numVertices, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy bvhtree.inl : d_vertices  (error code:: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	//d_faceCenter
	err = cudaMalloc((Vector3r * *)& d_faceCenter, sizeof(Vector3r) * m_faceCenter.size());
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate bvhtree.inl : d_faceCenter  (error code:: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(d_faceCenter, m_faceCenter.data(), sizeof(Vector3r) * m_faceCenter.size(), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy bvhtree.inl : d_faceCenter  (error code:: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	printf("success\n");
}


