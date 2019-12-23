//#include<My_GLSL.h>
//#include<myOpenGL.h>
//#include<qtimer.h>
//
//
////opengl32.lib
////glfw3.lib
////cudart.lib
////qtmain.lib
////Qt53DCore.lib
////Qt53DAnimation.lib
////Qt53DExtras.lib
////Qt53DInput.lib
////Qt53DLogic.lib
////Qt53DRender.lib
////Qt5Core.lib
////Qt5Gui.lib
////Qt5OpenGL.lib
////Qt5OpenGLExtensions.lib
////Qt5Widgets.lib
////lib\freeglut_rd.lib
////glu32.lib
////lib\AntTweakBar_rd.lib
////lib\glew_rd.lib
//
//// settings
//const unsigned int SCR_WIDTH = 800;
//const unsigned int SCR_HEIGHT = 600;
//
//unsigned int texture;
//unsigned int VBO1,VBO2, VAO, EBO;
//GLuint ourShader = 0;
//GLuint depth_FBO = 0;
//GLuint shadow_program = 0;
//GLuint depth_texture = 0;
//
//float	zoom = 62;
//float	swing_angle = 0 + 3 * 20;
//float	elevate_angle = 0;
//float	center[3] = { 0, 0.0, -0.5 };
//bool	polygonMode = false;
//
//float light_position[3] = { -2, 2, 4 };
//
//myOpenGL::myOpenGL(QWidget* parent)
//	: QOpenGLWidget(parent)
//{
//}
//
//myOpenGL::~myOpenGL()
//{
//}
//
//
//
//void myOpenGL::initializeGL()
//{
//	
//	initializeOpenGLFunctions();
//	
//	glClearColor(1, 1, 1, 0);
//
//	GLenum err = glewInit();
//	if (err != GLEW_OK)  printf(" Error initializing GLEW! \n");
//	else printf("Initializing GLEW succeeded!\n");
//
//	//Init depth texture and FBO
//	initializeOpenGLFunctions();
//	glGenFramebuffers(1, &depth_FBO);
//	glBindFramebuffer(GL_FRAMEBUFFER, depth_FBO);
//	// Depth texture. Slower than a depth buffer, but you can sample it later in your shader
//	glGenTextures(1, &depth_texture);
//	glBindTexture(GL_TEXTURE_2D, depth_texture);
//	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 1024, 1024, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
//	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
//	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
//	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depth_texture, 0);
//	glDrawBuffer(GL_NONE);
//	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) printf("Init_Shadow_Map failed.\n");
//
//	//Shader ourShader("4.1.texture.vs", "4.1.texture.fs");
//	ourShader = Setup_GLSL("4.1.texture");
//	shadow_program = Setup_GLSL("shadow");
//
//	// load and create a texture 
//	// -------------------------
//	glGenTextures(1, &texture);
//	glBindTexture(GL_TEXTURE_2D, texture); // all upcoming GL_TEXTURE_2D operations now have effect on this texture object
//	// set the texture wrapping parameters
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
//	// set texture filtering parameters
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//
//	int width, height, nrChannels;
//	// The FileSystem::getPath(...) is part of the GitHub repository so we can find files on any IDE/platform; replace it with your own image path.
//	unsigned char* data = stbi_load("QtSource/White_Dots_on_Blk.jpg", &width, &height, &nrChannels, 0);
//	if (data)
//	{
//		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
//		glGenerateMipmap(GL_TEXTURE_2D);
//	}
//	else
//	{
//		std::cout << "Failed to load texture" << std::endl;
//	}
//	stbi_image_free(data);
//}
//
//void myOpenGL::resizeGL(int width, int height)
//{
//	//未使用
//	Q_UNUSED(width);
//	Q_UNUSED(height);
//}
//
//void myOpenGL::paintGL()
//{
//	if (!conf)
//		return;
//		
//	zoom = conf->getZoom();
//	swing_angle = conf->getSwingAngle();
//	elevate_angle = conf->getElevateAngle();
//	center[0] = conf->getCenter()[0];
//	center[1] = conf->getCenter()[1];
//	center[2] = conf->getCenter()[2];
//
//	glBindFramebuffer(GL_FRAMEBUFFER, 0);
//	glMatrixMode(GL_PROJECTION);
//	glLoadIdentity();
//	gluPerspective(6, (float)622 / (float)621, 1, 200);
//	glMatrixMode(GL_MODELVIEW);
//	glShadeModel(GL_SMOOTH);
//			
//	glLoadIdentity();
//	glClearColor(240 / 255.0, 240 / 255.0, 240 / 255.0,0);
//	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//	gluLookAt(0, 0, zoom, 0, -0.5, 0, 0, 1, 0);     //三个坐标，依次设置脑袋的位置，眼睛看的位置和头顶的方向
//
//	glDisable(GL_LIGHTING);
//	glEnable(GL_DEPTH_TEST);
//	glUseProgram(shadow_program);
//	GLint uniloc = glGetUniformLocation(shadow_program, "shadow_texture");
//	glUniform1i(uniloc, 0);
//	glActiveTexture(GL_TEXTURE0);
//	glBindTexture(GL_TEXTURE_2D, depth_texture);
//	uniloc = glGetUniformLocation(shadow_program, "light_position");
//	glUniform3fv(uniloc, 1, light_position);
//
//
//	glRotated(elevate_angle, 1, 0, 0);
//	glRotated(swing_angle, 0, 1, 0);
//	glTranslatef(center[0], center[1], center[2]);
//	glScaled(1.8, 1.8, 1.8);
//
//	//=========================================服装
//	glGenVertexArrays(1, &VAO);
//	glGenBuffers(1, &VBO1);
//	glGenBuffers(1, &VBO2);
//	glGenBuffers(1, &EBO);
//	for (int k = 0; k < model->getTriangleModels().size(); k++)
//	{
//		glBindVertexArray(VAO);
//
//		//indice
//		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
//		mesh = model->getTriangleModels()[k]->getParticleMesh();
//		m_indices = mesh.getFaces();
//		faceNum = m_indices.size();
//		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * faceNum, &m_indices[0], GL_STATIC_DRAW);                    //存储 面 数据 
//
//		//pd
//		glBindBuffer(GL_ARRAY_BUFFER, VBO1);
//		pd = model->getParticles();
//		point = mesh.numVertices();
//		unsigned int indexOffset = model->getTriangleModels()[k]->getIndexOffset();
//		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * point * 3, &(pd.getPosition(indexOffset)), GL_DYNAMIC_DRAW);                   //存储 点 数据
//		//position attribute
//		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, (void*)0);
//		glEnableVertexAttribArray(0);
//
//		//uv
//		glBindBuffer(GL_ARRAY_BUFFER, VBO2);
//		uvs = mesh.getUVs();
//		glBufferData(GL_ARRAY_BUFFER, sizeof(Vector2r) * point, &(uvs.at(0)), GL_DYNAMIC_DRAW);
//		// texture coord attribute
//		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vector2r), (void*)0);
//		glEnableVertexAttribArray(1);
//
//		
//		glUseProgram(ourShader);
//		glBindTexture(GL_TEXTURE_2D, texture);
//
//		glBindVertexArray(VAO);
//		if (polygonMode)
//			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
//		else
//			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
//		glDrawElements(GL_TRIANGLES, faceNum, GL_UNSIGNED_INT, (void*)0);
//		
//		mesh.release();
//		m_indices.clear();
//
//	}
//}






#include<My_GLSL.h>
#include<myOpenGL.h>
#include<qtimer.h>

myOpenGL::myOpenGL(QWidget* parent)
	: QOpenGLWidget(parent)
{
	polygonMode = false;
}

myOpenGL::~myOpenGL()
{
}


float	zoom = 62;
float	swing_angle = 0 + 3 * 20;
float	elevate_angle = 0;
float	center[3] = { 0, 0.0, -0.5 };


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


void myOpenGL::initializeGL()
{
	initializeOpenGLFunctions();

	glClearColor(225.0/255, 220.0/255, 205.0/255, 0);

	//Init GLEW
	GLenum err = glewInit();
	if (err != GLEW_OK)  printf(" Error initializing GLEW! \n");
	else printf("Initializing GLEW succeeded!\n");

	//Init depth texture and FBO
	initializeOpenGLFunctions();
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

void myOpenGL::resizeGL(int width, int height)
{
	//未使用
	Q_UNUSED(width);
	Q_UNUSED(height);
}

void myOpenGL::paintGL()
{
	
	if (!conf)
		return;

	zoom = conf->getZoom();
	swing_angle = conf->getSwingAngle();
	elevate_angle = conf->getElevateAngle();
	center[0] = conf->getCenter()[0];
	center[1] = conf->getCenter()[1];
	center[2] = conf->getCenter()[2];

	int screenWidth = 621;
	int screenHeight = 611;
	GLuint uniloc;

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(6, (float)screenWidth / (float)screenHeight, 1, 200);
	glMatrixMode(GL_MODELVIEW);
	glShadeModel(GL_SMOOTH);
	
	glLoadIdentity();
	glClearColor(225.0 / 255, 220.0 / 255, 205.0 / 255.0,0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	gluLookAt(0, 0, zoom, 0, -0.5, 0, 0, 1, 0);     //三个坐标，依次设置脑袋的位置，眼睛看的位置和头顶的方向

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

	for (int k = 0; k < model->getTriangleModels().size(); k++)
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
		mesh = model->getTriangleModels()[k]->getParticleMesh();
		m_indices = mesh.getFaces();
		faceNum = m_indices.size();
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * faceNum, &m_indices[0], GL_STATIC_DRAW);                    //存储 面 数据 

		glBindBuffer(GL_ARRAY_BUFFER, vertex_handle);
		pd = model->getParticles();
		point = mesh.numVertices();
		unsigned int indexOffset = model->getTriangleModels()[k]->getIndexOffset();
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
	glLineWidth(0.1);
	for (int k = 0; k < model->getTriangleModels().size(); k++)
	{
		IndexedFaceMesh m = model->getTriangleModels()[k]->getParticleMesh();
		for (int i = 0; i < m.getBorderEdgesNum(); i++)
		{
			IndexedFaceMesh::Edge e = m.getEdges().at(m.getBorderEdges().at(i));
			unsigned int indexOffset = model->getTriangleModels()[k]->getIndexOffset();
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

	if (conf->getBodyPath() != "")
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
		body = model->getRigidBodies().at(0);
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
}


