#include "RigidBody.h"
using namespace PBD;
#include<MyVector.h>

void RigidBody::initBody(const VertexData& vertice, const PBD::IndexedFaceMesh& mesh, const Vector3r& scale)
{
	const unsigned int nVertices = vertice.size();
	const unsigned int nFaces = mesh.numFaces();
	const Vector3r* vertices = &vertice.getPosition(0);
	const unsigned int* indices = mesh.getFaces().data();
	const IndexedFaceMesh::UVIndices uvIndices = mesh.getUVIndices();
	const IndexedFaceMesh::NormalIndices normalIndices = mesh.getNormalIndices();
	const IndexedFaceMesh::UVs& uvs = mesh.getUVs();

	m_mesh.release();
	m_mesh.initMesh(nVertices, nFaces * 2, nFaces, false);
	m_vertexData_local.resize(nVertices);
	m_vertexData.resize(nVertices);
	for (unsigned int i = 0; i < nVertices; i++)
	{
		m_vertexData_local.getPosition(i) = vertices[i];
		m_vertexData.getPosition(i) = m_vertexData_local.getPosition(i);
	}

	for (unsigned int i = 0; i < nFaces; i++)
	{
		m_mesh.addFace(&indices[3 * i]);
	}
	m_mesh.copyUVs(uvIndices,normalIndices, uvs);
	m_mesh.buildNeighbors(false, false);

	m_frictionCoeff = 0.5;
	m_restitutionCoeff = 0.5;

	updateMeshNormals(m_vertexData);
}

void RigidBody::release()
{
	m_mesh.release();
	m_vertexData.release();
	m_vertexData_local.release();
	m_restitutionCoeff = 0.0;
	m_frictionCoeff = 0.0;
}

VertexData& PBD::RigidBody::getVertexData()
{
	return m_vertexData;
}

const VertexData& PBD::RigidBody::getVertexData() const
{
	return m_vertexData;
}

VertexData& PBD::RigidBody::getVertexDataLocal()
{
	return m_vertexData_local;
}

const VertexData& PBD::RigidBody::getVertexDataLocal() const
{
	return m_vertexData_local;
}

IndexedFaceMesh& PBD::RigidBody::getMesh()
{
	return m_mesh;
}

const IndexedFaceMesh& PBD::RigidBody::getMesh() const
{
	return m_mesh;
}



void PBD::RigidBody::updateMeshNormals(const VertexData& vd)
{
	m_mesh.updateNormals(vd, 0);
	m_mesh.updateVertexNormals(vd);
}
