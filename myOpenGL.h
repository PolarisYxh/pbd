#pragma once

//#include <glad/glad.h>
//#include <GLFW/glfw3.h>
#include <stb_image.h>
#include <iostream>

#include<QOpenGLWidget>
#include <QOpenGLFunctions>

#include<ConfigurationLoader.h>
#include<SimulationModel.h>
#include<IndexedFaceMesh.h>

using namespace Utilities;
using namespace PBD;

class myOpenGL :
	public QOpenGLWidget,protected QOpenGLFunctions
{
	Q_OBJECT
public:
	myOpenGL(QWidget* parent = 0);
	~myOpenGL();
public:
	void initializeGL() ;
	void resizeGL(int w, int h) ;
	void paintGL() ;

public:
	Configuration* conf;
	SimulationModel* model;
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
	IndexedFaceMesh::UVs uvs;
	bool polygonMode;

signals:
public slots:
};

