#include "TriangleModel.h"
using namespace PBD;
TriangleModel& TriangleModel::operator=(TriangleModel const& other)
{
	unsigned int m_indexOffset = other.m_indexOffset;
	IndexedFaceMesh m_particleMesh = other.m_particleMesh;
	Real m_restitutionCoeff = other.m_restitutionCoeff;
	Real m_frictionCoeff = other.m_frictionCoeff;
	Real m_bendingCoeff = other.m_bendingCoeff;
	bool m_findSpecialSeam = other.m_findSpecialSeam;
	Real m_slideFrictionCoeff = other.m_slideFrictionCoeff;
	Real m_collisionCoeff = other.m_collisionCoeff;
	return *this;
}

TriangleModel::TriangleModel() :
	m_particleMesh()
{
	m_restitutionCoeff = 0.5;
	m_frictionCoeff = 0.5;           //系数越大，拉伸性能越好
	m_bendingCoeff = 0.5;
	m_slideFrictionCoeff = 0.0;
	m_collisionCoeff = 0.01;
}

TriangleModel::~TriangleModel(void)
{
	cleanupModel();
}

void TriangleModel::cleanupModel()
{
	m_particleMesh.release();
}

void TriangleModel::updateMeshNormals(const ParticleData& pd)
{
	m_particleMesh.updateNormals(pd, m_indexOffset);
	m_particleMesh.updateVertexNormals(pd);
}

void PBD::TriangleModel::updateConstraints()
{
}

IndexedFaceMesh & TriangleModel::getParticleMesh() 
{
	return m_particleMesh;
}

void TriangleModel::initMesh(const unsigned int nPoints, const unsigned int nFaces, const unsigned int indexOffset, unsigned int* indices, const IndexedFaceMesh::UVIndices& uvIndices,const IndexedFaceMesh::NormalIndices& normalIndices, const IndexedFaceMesh::UVs& uvs)
{
	m_indexOffset = indexOffset;
	m_particleMesh.release();

	m_particleMesh.initMesh(nPoints, nFaces * 2, nFaces, true);

	for (unsigned int i = 0; i < nFaces; i++)
	{
		m_particleMesh.addFace(&indices[3 * i]);
	}
	m_particleMesh.copyUVs(uvIndices,normalIndices, uvs);
	m_particleMesh.buildNeighbors(true, false);
}

unsigned int TriangleModel::getIndexOffset() const
{
	return m_indexOffset;
}
