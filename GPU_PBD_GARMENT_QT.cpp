#include "GPU_PBD_GARMENT_QT.h"


GPU_PBD_GARMENT_QT::GPU_PBD_GARMENT_QT(QWidget* parent)
	: QMainWindow(parent)
{
	this->resize(981, 628);
	setWindowFlags(windowFlags() & ~Qt::WindowMaximizeButtonHint);
	setFixedSize(this->width(), this->height());
	setWindowTitle(QString::fromLocal8Bit("三维人体服装模拟动画生成工具"));
	

	//导航栏
	QMenu* fileMenu;
	fileMenu = menuBar()->addMenu(QString::fromLocal8Bit("文件"));
	QAction* importFile = new QAction(QString::fromLocal8Bit("导入场景"), fileMenu);
	fileMenu->addAction(importFile);
	QAction* importGarment = new QAction(QString::fromLocal8Bit("导入服装模型"), fileMenu);
	fileMenu->addAction(importGarment);
	QAction* importAvatar = new QAction(QString::fromLocal8Bit("导入人体模型"), fileMenu);
	fileMenu->addAction(importAvatar);
	QAction* exportGarment=new QAction(QString::fromLocal8Bit("导出当前服装模型"), fileMenu);
	fileMenu->addAction(exportGarment);
	QAction* exitScene = new QAction(QString::fromLocal8Bit("退出"), fileMenu);
	fileMenu->addAction(exitScene);
	QMenu* aniMenu;
	aniMenu = menuBar()->addMenu(QString::fromLocal8Bit("动画"));
	startAni = new QAction(QString::fromLocal8Bit("开始播放"), aniMenu);
	startAni->setEnabled(false);
	aniMenu->addAction(startAni);
	stopAni = new QAction(QString::fromLocal8Bit("暂停播放"), aniMenu);
	stopAni->setEnabled(false);
	aniMenu->addAction(stopAni);
	resetAni = new QAction(QString::fromLocal8Bit("重置动画"), aniMenu);
	resetAni->setEnabled(false);
	aniMenu->addAction(resetAni);

	//工具栏
	toolBar = this->addToolBar(QString::fromLocal8Bit("动画"));
	startBtn = new QPushButton();
	startBtn->setFlat(true);
	QIcon icon;
	icon.addFile(".\\QtSource\\start.png");
	startBtn->setIcon(icon);
	startBtn->setEnabled(false);
	startBtn->setIconSize(QSize(15,20));

	dictBtn = new QPushButton();
	dictBtn->setFlat(true);
	QIcon icon2;
	icon2.addFile(".\\QtSource\\dictionary.png");
	dictBtn->setIcon(icon2);
	dictBtn->setIconSize(QSize(15, 20));

	impoGarBtn = new QPushButton();
	impoGarBtn->setFlat(true);
	QIcon icon4;
	icon4.addFile(".\\QtSource\\garment.png");
	impoGarBtn->setIcon(icon4);
	impoGarBtn->setIconSize(QSize(15, 20));

	impoAvaBtn = new QPushButton();
	impoAvaBtn->setFlat(true);
	QIcon icon5;
	icon5.addFile(".\\QtSource\\avatar.png");
	impoAvaBtn->setIcon(icon5);
	impoAvaBtn->setIconSize(QSize(15, 20));

	resetBtn = new QPushButton();
	resetBtn->setFlat(true);
	QIcon icon3;
	icon3.addFile(".\\QtSource\\reset.png");
	resetBtn->setIcon(icon3);
	resetBtn->setEnabled(false);
	resetBtn->setIconSize(QSize(15, 20));

	clearBtn = new QPushButton();
	clearBtn->setFlat(true);
	QIcon icon6;
	icon6.addFile(".\\QtSource\\clear.png");
	clearBtn->setIcon(icon6);
	clearBtn->setEnabled(false);
	clearBtn->setIconSize(QSize(15, 20));

	saveBtn = new QPushButton();
	saveBtn->setFlat(true);
	QIcon icon7;
	icon7.addFile(".\\QtSource\\save.png");
	saveBtn->setIcon(icon7);
	saveBtn->setEnabled(false);
	saveBtn->setIconSize(QSize(15, 20));
	
	toolBar->addWidget(dictBtn);
	toolBar->addWidget(impoGarBtn);
	toolBar->addWidget(impoAvaBtn);
	toolBar->addWidget(startBtn);
	toolBar->addWidget(resetBtn);
	toolBar->addWidget(clearBtn);
	toolBar->addWidget(saveBtn);
	
	//开始按钮、滚动条
	QLabel* patternInfo = new QLabel(QString::fromLocal8Bit("服装部件属性："));
	scrollArea = new QScrollArea(this);
	QWidget* widget = new QWidget();
	scrollArea->setStyleSheet(
		"QScrollBar:vertical"
		"{"
		"width:8px;"
		"background:rgba(0,0,0,0%);"
		"margin:0px,0px,0px,0px;"
		"padding-top:5px;"
		"padding-bottom:5px;"
		"}"
		"QScrollBar::handle:vertical"
		"{"
		"width:8px;"
		"background:rgba(0,0,0,25%);"
		" border-radius:4px;"
		"min-height:20;"
		"}"
		"QScrollBar::handle:vertical:hover"
		"{"
		"width:8px;"
		"background:rgba(0,0,0,50%);"
		" border-radius:4px;"
		"min-height:20;"
		"}"
		"QScrollBar::add-line:vertical"
		"{"
		"height:9px;width:8px;"
		"subcontrol-position:bottom;"
		"}"
		"QScrollBar::sub-line:vertical"
		"{"
		"height:9px;width:8px;"
		"subcontrol-position:top;"
		"}"
		"QScrollBar::add-line:vertical:hover"
		"{"
		"height:9px;width:8px;"
		"subcontrol-position:bottom;"
		"}"
		"QScrollBar::sub-line:vertical:hover"
		"{"
		"height:9px;width:8px;"
		"subcontrol-position:top;"
		"}"
		"QScrollBar::add-page:vertical,QScrollBar::sub-page:vertical"
		"{"
		"background:rgba(0,0,0,10%);"
		"border-radius:4px;"
		"}"
	);
	scrollArea->setFrameShape(QFrame::NoFrame);
	scrollArea2 = new QScrollArea(this);
	scrollArea2->setStyleSheet(
		"QScrollBar:vertical"
		"{"
		"width:8px;"
		"background:rgba(1,1,1,100%);"
		"margin:0px,0px,0px,0px;"
		"padding-top:5px;"
		"padding-bottom:5px;"
		"}"
		"QScrollBar::handle:vertical"
		"{"
		"width:8px;"
		"background:rgba(1,1,1,100%);"
		" border-radius:4px;"
		"min-height:20;"
		"}"
		"QScrollBar::handle:vertical:hover"
		"{"
		"width:8px;"
		"background:rgba(1,1,1,100%);"
		" border-radius:4px;"
		"min-height:20;"
		"}"
		"QScrollBar::add-line:vertical"
		"{"
		"height:9px;width:8px;"
		"subcontrol-position:bottom;"
		"}"
		"QScrollBar::sub-line:vertical"
		"{"
		"height:9px;width:8px;"
		"subcontrol-position:top;"
		"}"
		"QScrollBar::add-line:vertical:hover"
		"{"
		"height:9px;width:8px;"
		"subcontrol-position:bottom;"
		"}"
		"QScrollBar::sub-line:vertical:hover"
		"{"
		"height:9px;width:8px;"
		"subcontrol-position:top;"
		"}"
		"QScrollBar::add-page:vertical,QScrollBar::sub-page:vertical"
		"{"
		"background:rgba(1,1,1,100%);"
		"border-radius:4px;"
		"}"
	);
	scrollArea2->setFrameShape(QFrame::NoFrame);

	//场景显示区
	openGL = new myOpenGL();
	openGL->resize(621, 611);

	//布局管理器
	totalLayout = new QHBoxLayout();
	leftLayout = new QVBoxLayout();
	middleLayout = new QVBoxLayout();
	rightLayout = new QVBoxLayout();
	
	//控件加入布局管理器
	//topLayout->addWidget(toolBar);
	leftLayout->addWidget(patternInfo);
	leftLayout->addWidget(scrollArea);
	middleLayout->addWidget(openGL);
	rightLayout->addWidget(scrollArea2);
	
	//设置总布局管理器和比例
	totalLayout->addLayout(leftLayout);
	totalLayout->addLayout(middleLayout);
	totalLayout->addLayout(rightLayout);
	totalLayout->setStretchFactor(leftLayout, 195);
	totalLayout->setStretchFactor(middleLayout, 635);
	totalLayout->setStretchFactor(rightLayout, 160);

	widget->setLayout(totalLayout);
	setCentralWidget(widget);

	initDevice();
	
	//快捷键
	connect(dictBtn, SIGNAL(clicked(bool)), this, SLOT(readConfigFile()));
	connect(impoGarBtn, SIGNAL(clicked(bool)), this, SLOT(readGarmentFile()));
	connect(impoAvaBtn, SIGNAL(clicked(bool)), this, SLOT(readAvatarFile()));
	connect(startBtn, SIGNAL(clicked(bool)), this, SLOT(setPause()));
	connect(resetBtn, SIGNAL(clicked(bool)), this, SLOT(resetAnimation()));
	connect(clearBtn, SIGNAL(clicked(bool)), this,SLOT(clearScene()));
	connect(saveBtn, SIGNAL(clicked(bool)), this, SLOT(saveGarmentFile()));
	

	connect(exitScene, SIGNAL(triggered()), this, SLOT(exitApp()));
	connect(importFile, SIGNAL(triggered()), this, SLOT(readConfigFile()));
	connect(importGarment, SIGNAL(triggered()), this, SLOT(readGarmentFile()));
	connect(exportGarment, SIGNAL(triggered()), this, SLOT(saveGarmentFile()));
	connect(importAvatar, SIGNAL(triggered()), this, SLOT(readAvatarFile()));
	connect(startAni, SIGNAL(triggered()), this, SLOT(startAnimation()));
	connect(stopAni, SIGNAL(triggered()), this, SLOT(stopAnimation()));
	connect(resetAni, SIGNAL(triggered()), this, SLOT(resetAnimation()));
	
	
}

Configuration conf;
SimulationModel model;
DistanceFieldCollisionDetection cd;
TimeStepController sim;
std::vector<std::vector<OBJLoader::Vec3f>> globalPosition;
float seamTolerance = 0.015;
bool loadGarment = false;
bool loadAvatar = false;

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


//鼠标响应
bool mouseLeftDown;
bool mouseRightDown;
bool mouseMiddleDown;


//IndexedFaceMesh mesh;
//IndexedFaceMesh::Faces m_indices;
//int faceNum;
//RigidBody* body;
//IndexedFaceMesh mesh_body;
//IndexedFaceMesh::Faces body_indices;
//int body_faceNum;
//ParticleData pd;
//int point;
//VertexData body_VD;
//int pointNum;

int stepCount = -10;
bool firstLoad = true;
bool isReset = false;
bool isWait = false;


void GPU_PBD_GARMENT_QT::initDevice()
{
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
		printf("CUDA驱动版本:                                    %d.%d\n", driver_version / 1000, (driver_version % 1000) / 10);
		cudaRuntimeGetVersion(&runtime_version);
		printf("CUDA运行时版本:                                  %d.%d\n", runtime_version / 1000, (runtime_version % 1000) / 10);
		printf("设备计算能力:                                    %d.%d\n", deviceProp.major, deviceProp.minor);
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
}

void GPU_PBD_GARMENT_QT::showScrollArea()
{
	QVBoxLayout* scrollLayout = new QVBoxLayout();
	QWidget* scrollContent = new QWidget;
	for (int i = 0; i < conf.getClothCoeff().size(); i++)
	{
		QGroupBox* group = new QGroupBox();
		QLabel* label1 = new QLabel(tr("mass"));
		QLabel* label2 = new QLabel(tr("airDrag"));
		QLabel* label3 = new QLabel(tr("strecth"));
		QLabel* label4 = new QLabel(tr("bending"));
		QLabel* label5 = new QLabel(tr("collision"));
		QLabel* label6 = new QLabel(tr("friction"));


		QDoubleSpinBox* doubleSpinBox1 = new QDoubleSpinBox();
		doubleSpinBox1->setDecimals(5);
		doubleSpinBox1->setValue(conf.getClothCoeff().at(i)[3]);
		doubleSpinBox1->setRange(0.0, 2.0);
		doubleSpinBox1->setSingleStep(0.0005);
		QDoubleSpinBox* doubleSpinBox2 = new QDoubleSpinBox();
		doubleSpinBox2->setValue(conf.getClothCoeff().at(i)[4]);
		doubleSpinBox2->setRange(0.0, 10.5);
		doubleSpinBox2->setSingleStep(0.1);
		QDoubleSpinBox* doubleSpinBox3 = new QDoubleSpinBox();
		doubleSpinBox3->setValue(conf.getClothCoeff().at(i)[0]);
		doubleSpinBox3->setRange(0.1, 1.4);
		doubleSpinBox3->setSingleStep(0.1);
		QDoubleSpinBox* doubleSpinBox4 = new QDoubleSpinBox();
		doubleSpinBox4->setValue(conf.getClothCoeff().at(i)[2]);
		doubleSpinBox4->setRange(0.0, 1.3);
		doubleSpinBox4->setSingleStep(0.1);
		QDoubleSpinBox* doubleSpinBox5 = new QDoubleSpinBox();
		doubleSpinBox5->setDecimals(3);
		doubleSpinBox5->setValue(conf.getClothCoeff().at(i)[5]);
		doubleSpinBox5->setRange(0.0, 0.1);
		doubleSpinBox5->setSingleStep(0.001);
		QDoubleSpinBox* doubleSpinBox6 = new QDoubleSpinBox();
		doubleSpinBox6->setValue(conf.getClothCoeff().at(i)[6]);
		doubleSpinBox6->setRange(0.0, 1.5);
		doubleSpinBox6->setSingleStep(0.1);

		QHBoxLayout* h1 = new QHBoxLayout();
		h1->addWidget(label1);
		h1->addWidget(doubleSpinBox1);
		QHBoxLayout* h2 = new QHBoxLayout();
		h2->addWidget(label2);
		h2->addWidget(doubleSpinBox2);
		QHBoxLayout* h3 = new QHBoxLayout();
		h3->addWidget(label3);
		h3->addWidget(doubleSpinBox3);
		QHBoxLayout* h4 = new QHBoxLayout();
		h4->addWidget(label4);
		h4->addWidget(doubleSpinBox4);
		QHBoxLayout* h5 = new QHBoxLayout();
		h5->addWidget(label5);
		h5->addWidget(doubleSpinBox5);
		QHBoxLayout* h6 = new QHBoxLayout();
		h6->addWidget(label6);
		h6->addWidget(doubleSpinBox6);
		QVBoxLayout* v = new QVBoxLayout();
		v->addLayout(h1);
		v->addLayout(h2);
		v->addLayout(h3);
		v->addLayout(h4);
		v->addLayout(h5);
		v->addLayout(h6);

		group->setTitle(QString("pattern ") + QString::number(i + 1));
		group->setLayout(v);
		group->setMinimumSize(161, 131);
		group->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);

		scrollLayout->addWidget(group);
		groupBoxs.push_back(group);

		labels.push_back(label1);
		labels.push_back(label2);
		labels.push_back(label3);
		labels.push_back(label4);
		labels.push_back(label5);
		labels.push_back(label6);
		doubleSpinBoxs.push_back(doubleSpinBox1);
		doubleSpinBoxs.push_back(doubleSpinBox2);
		doubleSpinBoxs.push_back(doubleSpinBox3);
		doubleSpinBoxs.push_back(doubleSpinBox4);
		doubleSpinBoxs.push_back(doubleSpinBox5);
		doubleSpinBoxs.push_back(doubleSpinBox6);
		connect(doubleSpinBox1, SIGNAL(valueChanged(double)), this, SLOT(setAttrVal()));
		connect(doubleSpinBox2, SIGNAL(valueChanged(double)), this, SLOT(setAttrVal()));
		connect(doubleSpinBox3, SIGNAL(valueChanged(double)), this, SLOT(setAttrVal()));
		connect(doubleSpinBox4, SIGNAL(valueChanged(double)), this, SLOT(setAttrVal()));
		connect(doubleSpinBox5, SIGNAL(valueChanged(double)), this, SLOT(setAttrVal()));
		connect(doubleSpinBox6, SIGNAL(valueChanged(double)), this, SLOT(setAttrVal()));

	}
	scrollContent->setLayout(scrollLayout);
	scrollArea->widgetResizable();
	scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
	scrollArea->setWidget(scrollContent);
	scrollArea->setFrameShape(QFrame::NoFrame);
}

void GPU_PBD_GARMENT_QT::showScrollArea2()
{
	QVBoxLayout* scrollLayout = new QVBoxLayout();
	QWidget* scrollContent = new QWidget;
	int offset = 0;
	for (int i = 0; i < conf.getClothNum(); i++)
	{
		QGroupBox* group = new QGroupBox();
		int patternNum = conf.getPatternNum().at(i);
		int verticeNum = 0;
		int faceNum = 0;
		for (int j = 0; j < conf.getPatternNum().at(i); j++)
		{
			verticeNum=verticeNum	+model.getTriangleModels()[j+offset]->getParticleMesh().numVertices();
			faceNum = faceNum + model.getTriangleModels()[j + offset]->getParticleMesh().getFaceData().size();
		}
		QLabel* patternNumLabel = new QLabel();
		patternNumLabel->setText(QString::fromLocal8Bit("pattern num:     ") + QString::number(patternNum));
		QLabel* garmentVer = new QLabel();
		garmentVer->setText(QString::fromLocal8Bit("Vertice:      ") + QString::number(verticeNum));
		QLabel* garmentFace = new QLabel();
		garmentFace->setText(QString::fromLocal8Bit("Faces:        ") + QString::number(faceNum));

		QVBoxLayout* v = new QVBoxLayout();
		v->addWidget(patternNumLabel);
		v->addWidget(garmentVer);
		v->addWidget(garmentFace);

		group->setTitle(QString::fromLocal8Bit("服装 ") + QString::number(i + 1));
		group->setMinimumSize(161, 100);
		group->setLayout(v);
		group->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
		scrollLayout->addWidget(group);
	}

	avatar = new QGroupBox();
	avatar->setVisible(false);
	if (conf.getBodyPath() != "")
	{
		avatar->setTitle(QString::fromLocal8Bit("人体序列"));
		avatar->setVisible(true);
		//avatar->setMaximumHeight(70);
		curFrame = new QLabel();
		curFrame->setText(QString::fromLocal8Bit("current Frame:    ") + QString::number(conf.getCurrentFrame()));
		QLabel* totalFrame = new QLabel();
		totalFrame->setText(QString::fromLocal8Bit("total frame:    ") + QString::number(conf.getEndFrame()));
		QLabel* totAvaVer = new QLabel();
		totAvaVer->setText(QString::fromLocal8Bit("Vertice:       ") + QString::number(model.getRigidBodies()[0]->getVertexData().size()));
		QLabel* totAvaFaces = new QLabel();
		totAvaFaces->setText(QString::fromLocal8Bit("Faces:        ") + QString::number(model.getRigidBodies()[0]->getMesh().getFaceData().size()));
		QVBoxLayout* v2 = new QVBoxLayout();
		v2->addWidget(curFrame);
		v2->addWidget(totalFrame);
		v2->addWidget(totAvaVer);
		v2->addWidget(totAvaFaces);
		avatar->setMinimumSize(161, 100);
		avatar->setLayout(v2);
		scrollLayout->addWidget(avatar);
	}

	scrollContent->setLayout(scrollLayout);
	scrollArea2->widgetResizable();
	scrollArea2->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	scrollArea2->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
	scrollArea2->setWidget(scrollContent);
	scrollArea2->setFrameShape(QFrame::NoFrame);
}

void GPU_PBD_GARMENT_QT::readConfigFile()
{
	if (conf.getClothPath().size()!=0)
	{
		clearScene();
	}

	QString fileName = QFileDialog::getOpenFileName(this,
		tr("Open Spreadsheet"), "./experimentData/",
		tr("configure files (*.configFile)"));
	//cout << fileName.toStdString() << endl;
	if (fileName.toStdString() == "")
		return;
	conf=conf.getConfiguration(fileName.toStdString());
	if (!conf.isExtractValid())
	{
		QString title = "File Error";
		QString info = QString::fromLocal8Bit("配置文件格式有误");
		QMessageBox::information(this, title, info);
		conf.reset();
		return;
	}
	else
	{	//判读服装路径是否正确
		for (int i = 0; i < conf.getClothNum(); i++)
		{
			string clothPath = conf.getClothPath().at(i);
			ifstream filestream;
			filestream.open(clothPath);
			if (filestream.fail())
			{
				QString title = "File Error";
				cout << clothPath << endl;
				QString info = QString::fromLocal8Bit("服装模型路径有误");
				QMessageBox::information(this, title, info);
				conf.reset();
				return ;
			}
		}
		if (conf.getBvhName() != "")
		{
			string bodyPath = conf.getBodyPath();
			ifstream filestream;
			filestream.open(bodyPath);
			if (filestream.fail())
			{
				QString title = "File Error";
				cout << bodyPath << endl;
				QString info = QString::fromLocal8Bit("人体模型路径有误");
				QMessageBox::information(this, title, info);
				conf.reset();
				return;
			}
		}
	}

	if (!firstLoad)
	{
		openGL->mesh.release();
		openGL->faceNum = 0;
		openGL->body->release();
		openGL->mesh_body.release();
		openGL->body_faceNum = 0;
		openGL->point = 0;
		openGL->pointNum = 0;
		model.cleanup();
		stepCount = conf.getStepCount();
		globalPosition.clear();
		cd.cleanup();
		sim.reset();
	}
	firstLoad = false;
	openGL->conf = &conf;
	openGL->model = &model;
	buildModel();

	resetBtn->setEnabled(true);
	startBtn->setEnabled(true);
	startAni->setEnabled(true);
	stopAni->setEnabled(true);
	resetAni->setEnabled(true);
	clearBtn->setEnabled(true);
	saveBtn->setEnabled(true);

	if (conf.getBodyPath() != "" && conf.getEndFrame()>0)
	{
		progress = new QProgressDialog(QString::fromLocal8Bit("正在处理人体运动序列..."), QString::fromLocal8Bit("取消"), 0, conf.getEndFrame(), this);
		progress->setWindowModality(Qt::WindowModal);
		progress->setWindowTitle(QString::fromLocal8Bit("请稍等"));
		progress->show();


		int endFrame = conf.getEndFrame();
		globalPosition.reserve(endFrame);
		for (int i = 1; i <= endFrame; i++)
		{
			std::vector<OBJLoader::Vec3f> x;
			//string fileName = "D:\\Program\\GPU_PBD_Garment\\data\\walk\\sequence_vertex\\walk" + to_string(i) + ".obj";
			string fileName = conf.getBvhPath() + to_string(i) + ".obj";
			OBJLoader::loadObjVertex(fileName, &x);
			globalPosition.push_back(x);

			progress->setValue(i);
		}
	}

	//左滚动栏
	showScrollArea();
	//右滚动栏
	showScrollArea2();
}

void GPU_PBD_GARMENT_QT::importGarmentFile(string fileName)
{
	vector<float> translate = { 0.0,0.0,0.0 };
	vector<float> scale = { 1.0,1.0,1.0 };
	bool findSpecialSeam = false;
	createPatternMesh(fileName, translate, scale, findSpecialSeam);

	//为mesh粒子设置mass属性
	ParticleData& pd = model.getParticles();
	for (unsigned int i = 0; i < model.getTriangleModels().size(); i++)
	{
		float massCoeff = 1.0;
		unsigned int indexoffset = model.getTriangleModels().at(i)->getIndexOffset();
		unsigned int pdNum = model.getTriangleModels().at(i)->getParticleMesh().numVertices();
		for (unsigned int j = 0; j < pdNum; j++)
		{
			pd.setMass(indexoffset + j, massCoeff);
		}
	}
	//初始化服装缝合线等约束
	initClothConstraints();
	//设置各pattern的拉伸、弯曲等属性，初始化碰撞约束
	SimulationModel::TriangleModelVector& tm = model.getTriangleModels();
	for (unsigned int i = 0; i < tm.size(); i++)
	{
		TriangleModel* tri = model.getTriangleModels().at(i);
		tri->setFrictionCoeff(1.0);
		tri->setRestitutionCoeff(1.5);
		tri->setBendingCoeff(1.0);
		tri->setDampingCoeff(0.99);
		tri->setSlideFrictionCoeff(0.0);
		conf.setClothCoeff(vector<float>{1.0, 1.5, 1.0, 0.1, 0.99, 0.001,0});
		conf.setCollisionPairs(vector<unsigned int>{i, (unsigned int)tm.size()});

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
		cd.addCollisionSphereOnFaces(i, CollisionDetection::CollisionObject::TriangleModelCollisionObjectType, faces, faces.size(), &pd.getPosition(offset), nVert, 0.001);
	}

	resetBtn->setEnabled(true);
	startBtn->setEnabled(true);
	startAni->setEnabled(true);
	stopAni->setEnabled(true);
	resetAni->setEnabled(true);
	clearBtn->setEnabled(true);
	saveBtn->setEnabled(true);

	conf.setTranslate(translate);
	conf.setScale(scale);
	conf.setFindSpecialSeam(findSpecialSeam);
	conf.setClothNum(1);
	conf.setClothPath(fileName);
	conf.setZoom(50);
	conf.setCenter(vector<float>{0.0, 0.0, 0.0});

	openGL->conf = &conf;
	openGL->model = &model;
	showScrollArea();
}

void GPU_PBD_GARMENT_QT::readGarmentFile()
{
	QString fileName = QFileDialog::getOpenFileName(this,
		tr("Open Spreadsheet"), "./experimentData/",
		tr("Garment files (*.obj)"));
	//cout << fileName.toStdString() << endl;
	if (fileName.toStdString() == "")
		return;

	importGarmentFile(fileName.toStdString());
}

void GPU_PBD_GARMENT_QT::readAvatarFile()
{
	QString fileName = QFileDialog::getOpenFileName(this,
		tr("Open Configure File"), "./experimentData/",
		tr("Avatar files (*.obj)"));
	//cout << fileName.toStdString() << endl;
	if (fileName.toStdString() == "")
		return;
	SimulationModel::RigidBodyVector& rb = model.getRigidBodies();
	sim.setCollisionDetection(model, &cd);
	IndexedFaceMesh mesh;
	VertexData vd;
	loadObj(fileName.toStdString(), vd, mesh, Vector3r(1.0, 1.0, 1.0), false, false);
	RigidBody* rigidBody = new RigidBody();

	rigidBody->initBody(vd, mesh, Vector3r(1.0, 1.0, 1.0));
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

	conf.setBodyPath(fileName.toStdString());
	conf.setStartFrame(0);
	conf.setEndFrame(0);
	conf.setStepSize(1);
	conf.setCurrentFrame(0);
	conf.setTranslate(vector<float>{0.0, 0.0, 0.0});
	conf.setScale(vector<float>{1.0, 1.0, 1.0});
	conf.setFindSpecialSeam(false);
	conf.setZoom(62.0);
	conf.setCenter(vector<float>{0.0, 0.0, 0.0});

	openGL->conf = &conf;
	openGL->model = &model;
	clearBtn->setEnabled(true);
}

void saveSerGarmentFile(int curFrame)
{
	SimulationModel::TriangleModelVector triangleModel = model.getTriangleModels();

	using Vec3f = std::array<float, 3>;
	using Vec2f = std::array<float, 2>;
	std::vector<std::vector<Vec3f>> xVec;
	std::vector<std::vector<MeshFaceIndices>>facesVec;
	std::vector<std::vector<Vec3f>>normalsVec;
	std::vector<std::vector<Vec2f>>texcoordsVec;

	int patternOffset = 0;
	unsigned int preCloOffset = 0;
	unsigned int offset = 0;
	for (int t = 0; t < conf.getClothNum(); t++)
	{
		string fileName = "D://Program//GPU_PBD_GARMENT_QT//data//walk//new-garment//garment-" + to_string(curFrame) + "-" + to_string(t + 1) + ".obj";
		
		for (int i = patternOffset; i < patternOffset + conf.getPatternNum().at(t); i++)
		{
			std::vector<Vec3f> xData;
			std::vector<MeshFaceIndices> facesData;
			std::vector<Vec3f> normalsData;
			std::vector<Vec2f> texcoordsData;

			ParticleData pd = model.getParticles();
			IndexedFaceMesh faceMesh = triangleModel[i]->getParticleMesh();
			IndexedFaceMesh::Faces face = faceMesh.getFaces();
			IndexedFaceMesh::UVIndices uvIndice = faceMesh.getUVIndices();
			IndexedFaceMesh::NormalIndices normalIndice = faceMesh.getNormalIndices();
			IndexedFaceMesh::VertexNormals vertexNormal = faceMesh.getVertexNormals();
			IndexedFaceMesh::UVs uv = faceMesh.getUVs();
			offset = triangleModel[i]->getIndexOffset();
			unsigned int numPoint = faceMesh.numVertices();

			for (unsigned int j = 0; j < numPoint; j++)
			{
				Vector3r x = pd.getPosition(offset + j);
				Vec3f xx;
				//xx[0] = x[0] * 70 + 3.183; xx[1] = x[1] * 70 + 94.166; xx[2] = x[2] * 70 - 64.685;
				xx[0] = x[0] * 100 + 4.5 - 0.045; xx[1] = x[1] * 100 + 110.5 + 20 - 1.105; xx[2] = x[2] * 100 + 75.4 - 0.754 - 44;
				xData.push_back(xx);
			}
			for (unsigned int j = 0; j < vertexNormal.size(); j++)
			{
				Vector3r vernor = vertexNormal[j];
				Vec3f vernormal;
				vernormal[0] = vernor[0]; vernormal[1] = vernor[1]; vernormal[2] = vernor[2];
				normalsData.push_back(vernormal);
			}
			for (unsigned int j = 0; j < uv.size(); j++)
			{
				Vector2r texcor = uv[j];
				Vec2f UV;
				UV[0] = texcor[0]; UV[1] = texcor[1]; UV[2] = texcor[2];
				texcoordsData.push_back(UV);
			}
			for (int j = 0; j < face.size() / 3; j++)
			{
				MeshFaceIndices meshIndice;
				meshIndice.posIndices[0] = face[j * 3 + 0] + 1 + offset - preCloOffset;
				meshIndice.posIndices[1] = face[j * 3 + 1] + 1 + offset - preCloOffset;
				meshIndice.posIndices[2] = face[j * 3 + 2] + 1 + offset - preCloOffset;
				meshIndice.texIndices[0] = uvIndice[j * 3 + 0] + 1;
				meshIndice.texIndices[1] = uvIndice[j * 3 + 1] + 1;
				meshIndice.texIndices[2] = uvIndice[j * 3 + 2] + 1;
				meshIndice.normalIndices[0] = normalIndice[j * 3 + 0] + 1;
				meshIndice.normalIndices[1] = normalIndice[j * 3 + 1] + 1;
				meshIndice.normalIndices[2] = normalIndice[j * 3 + 2] + 1;
				facesData.push_back(meshIndice);
			}
			xVec.push_back(xData);
			facesVec.push_back(facesData);
			normalsVec.push_back(normalsData);
			texcoordsVec.push_back(texcoordsData);
		}
		patternOffset = patternOffset + conf.getPatternNum().at(t);
		for (int j = 0; j < xVec.size(); j++)
			preCloOffset += xVec[j].size();
		OBJLoader::savePatternObj(fileName, xVec, facesVec, normalsVec, texcoordsVec);
		xVec.resize(0);
		facesVec.resize(0);
		normalsVec.resize(0);
		texcoordsVec.resize(0);
	}
}

void GPU_PBD_GARMENT_QT::saveGarmentFile()
{
	QString fileName = QFileDialog::getSaveFileName(this,
		tr("Save Garment File"), "./data/",
		tr("Garment files(*obj)"));
	if (fileName.isNull())
	{
		return;
	}
	else
	{
		SimulationModel::TriangleModelVector triangleModel = model.getTriangleModels();

		using Vec3f = std::array<float, 3>;
		using Vec2f = std::array<float, 2>;
		std::vector<std::vector<Vec3f>> xVec;
		std::vector<std::vector<MeshFaceIndices>>facesVec;
		std::vector<std::vector<Vec3f>>normalsVec;
		std::vector<std::vector<Vec2f>>texcoordsVec;

		for (int i = 0; i < triangleModel.size(); i++)
		{
			std::vector<Vec3f> xData;
			std::vector<MeshFaceIndices> facesData;
			std::vector<Vec3f> normalsData;
			std::vector<Vec2f> texcoordsData;

			ParticleData pd = model.getParticles();
			IndexedFaceMesh faceMesh = triangleModel[i]->getParticleMesh();
			IndexedFaceMesh::Faces face = faceMesh.getFaces();
			IndexedFaceMesh::UVIndices uvIndice = faceMesh.getUVIndices();
			IndexedFaceMesh::NormalIndices normalIndece = faceMesh.getNormalIndices();
			IndexedFaceMesh::VertexNormals vertexNormal = faceMesh.getVertexNormals();
			
			IndexedFaceMesh::UVs uv = faceMesh.getUVs();
			unsigned int offset = triangleModel[i]->getIndexOffset();
			unsigned int numPoint = faceMesh.numVertices();

			for (unsigned int j = 0; j < numPoint; j++)
			{
				Vector3r x = pd.getPosition(offset + j);
				Vec3f xx;
				xx[0] = x[0]; xx[1] = x[1]; xx[2] = x[2];
				xData.push_back(xx);
			}
			for (unsigned int j = 0; j < vertexNormal.size(); j++)
			{
				Vector3r vernor = vertexNormal[j];
				Vec3f vernormal;
				vernormal[0] = vernor[0]; vernormal[1] = vernor[1]; vernormal[2] = vernor[2];
				normalsData.push_back(vernormal);
			}
			for (unsigned int j = 0; j < uv.size(); j++)
			{
				Vector2r texcor = uv[j];
				Vec2f UV;
				UV[0] = texcor[0]; UV[1] = texcor[1]; UV[2] = texcor[2];
				texcoordsData.push_back(UV);
			}
			for (int j = 0; j < face.size() / 3; j++)
			{
				MeshFaceIndices meshIndice;
				meshIndice.posIndices[0] = face[j * 3 + 0] + 1 + offset;
				meshIndice.posIndices[1] = face[j * 3 + 1] + 1 + offset;
				meshIndice.posIndices[2] = face[j * 3 + 2] + 1 + offset;
				meshIndice.texIndices[0] = uvIndice[j * 3 + 0] + 1;
				meshIndice.texIndices[1] = uvIndice[j * 3 + 1] + 1;
				meshIndice.texIndices[2] = uvIndice[j * 3 + 2] + 1;
				meshIndice.normalIndices[0] = normalIndece[j * 3 + 0] + 1;
				meshIndice.normalIndices[1] = normalIndece[j * 3 + 1] + 1;
				meshIndice.normalIndices[2] = normalIndece[j * 3 + 2] + 1;
				facesData.push_back(meshIndice);
			}
			xVec.push_back(xData);
			facesVec.push_back(facesData);
			normalsVec.push_back(normalsData);
			texcoordsVec.push_back(texcoordsData);
		}

		OBJLoader::savePatternObj(fileName.toStdString(), xVec, facesVec, normalsVec, texcoordsVec);
	}
}

void GPU_PBD_GARMENT_QT::setAttrVal()
{
	int i = 0;
	for (; i < doubleSpinBoxs.size(); i++)
	{
		if (doubleSpinBoxs[i]->hasFocus())
		{
			break;
		}
	}
	int groupIndex = i / 6;
	int attrIndex = i % 6;
	unsigned int indexOffset;
	unsigned int pdNum;
	switch (attrIndex)
	{
	case 0:
		indexOffset = model.getTriangleModels().at(groupIndex)->getIndexOffset();
		pdNum = model.getTriangleModels().at(groupIndex)->getParticleMesh().numVertices();
		for (unsigned int j = 0; j < pdNum; j++)
		{
			if(model.getParticles().getMass(indexOffset+j)!=0.0)
				model.getParticles().setMass(indexOffset + j, doubleSpinBoxs[i]->value());
		}
		break;
	case 1:
		model.getTriangleModels().at(groupIndex)->setAirDragCoeff(doubleSpinBoxs[i]->value());
		break;
	case 2:
		model.getTriangleModels().at(groupIndex)->setFrictionCoeff(doubleSpinBoxs[i]->value());
		break;
	case 3:
		model.getTriangleModels().at(groupIndex)->setBendingCoeff(doubleSpinBoxs[i]->value());
		break;
	case 4:
		model.getTriangleModels().at(groupIndex)->setCollisionCoeff(doubleSpinBoxs[i]->value());
		break;
	case 5:
		model.getTriangleModels().at(groupIndex)->setSlideFrictionCoeff(doubleSpinBoxs[i]->value());
		break;
	default:
		break;
	}
}

void GPU_PBD_GARMENT_QT::setPause()
{
	if (doPause)
	{
		doPause = false;
		QIcon icon;
		icon.addFile("D:\\Program\\GPU_PBD_GARMENT_QT\\QtSource\\pause.png");
		startBtn->setIcon(icon);
		timeStep();
	}
	else
	{
		QIcon icon;
		icon.addFile("D:\\Program\\GPU_PBD_GARMENT_QT\\QtSource\\start.png");
		startBtn->setIcon(icon);
		doPause = true;
		//timeStep();
	}
		
}

void GPU_PBD_GARMENT_QT::startAnimation()
{
	doPause = false;
	QIcon icon;
	icon.addFile("D:\\Program\\GPU_PBD_GARMENT_QT\\QtSource\\pause.png");
	startBtn->setIcon(icon);
	timeStep();
}

void GPU_PBD_GARMENT_QT::stopAnimation()
{
	QIcon icon;
	icon.addFile("D:\\Program\\GPU_PBD_GARMENT_QT\\QtSource\\start.png");
	startBtn->setIcon(icon);
	doPause = true;
}

void GPU_PBD_GARMENT_QT::resetAnimation()
{
	stepCount = -20;
	firstLoad = true;

	if (openGL->conf && openGL->conf->getBodyPath() != "")
	{
		curFrame->setText(QString::fromStdString("current Frame: 0"));
		conf.setCurrentFrame(0);
		stepCount = conf.getStepCount();
	}

	isReset = true;
	doPause = true;

	QIcon icon;
	icon.addFile("D:\\Program\\GPU_PBD_GARMENT_QT\\QtSource\\start.png");
	startBtn->setIcon(icon);

	for (int i = 0; i < model.getParticles().size(); i++)
	{
		ParticleData& pd = model.getParticles();
		pd.getPosition(i) = pd.getPosition0(i);
		pd.getLastPosition(i) = pd.getPosition0(i);
		pd.getOldPosition(i) = pd.getOldPosition(i);
		pd.getVelocity(i) = Vector3r(0.0, 0.0, 0.0);
	}
	if(conf.getBvhName()!="" && conf.getEndFrame()>0)
		updateAvater(conf.getBvhName(), 1);
	//sim.step(model, conf);
	
	for (unsigned int i = 0; i < model.getTriangleModels().size(); i++)
		model.getTriangleModels()[i]->updateMeshNormals(model.getParticles());
	openGL->update();
}

void GPU_PBD_GARMENT_QT::exitApp()
{
	this->close();
}

void GPU_PBD_GARMENT_QT::clearScene()
{
	delete garment;
	delete avatar;
	for (int i = 0; i < groupBoxs.size(); i++)
		delete groupBoxs[i];
	groupBoxs.resize(0);
	doubleSpinBoxs.resize(0);
	labels.resize(0);
	delete temp;
	startBtn->setEnabled(false);
	resetBtn->setEnabled(false);
	clearBtn->setEnabled(false);
	startAni->setEnabled(false);
	stopAni->setEnabled(false);
	resetAni->setEnabled(false);
	saveBtn->setEnabled(false);

	if (openGL->model)
	{
		openGL->mesh.release();
		openGL->faceNum = 0;
		
		openGL->point = 0;
		openGL->pointNum = 0;
		model.cleanup();
		stepCount = -20;
		if (openGL->conf->getBodyPath() != "")
		{
			openGL->body->release();
			openGL->mesh_body.release();
			openGL->body_faceNum = 0;
			stepCount = conf.getStepCount();
		}
		cd.cleanup();
		sim.reset();
		firstLoad = true;
	}

	QIcon icon;
	icon.addFile("D:\\Program\\GPU_PBD_GARMENT_QT\\QtSource\\start.png");
	doPause = true;
	startBtn->setIcon(icon);
	conf.setSwingAngle(0);
	conf.setElevateAngle(0);
	conf.reset();
	openGL->conf = nullptr;
	openGL->model = nullptr;
	openGL->update();
}

int preX = -1, preY = -1;
void GPU_PBD_GARMENT_QT::mousePressEvent(QMouseEvent* e)
{
	if (e->buttons() & (Qt::LeftButton | Qt::MiddleButton))
	{
		preX = e->globalX();
		preY = e->globalY();
	}
	else if (e->buttons() & Qt::RightButton)
	{
		openGL->polygonMode = !(openGL->polygonMode);
		openGL->update();
	}
}
void GPU_PBD_GARMENT_QT::wheelEvent(QWheelEvent* e)
{
	if (!openGL->conf)
		return;
	//鼠标在openGL控件上
	if(openGL->geometry().contains(this->mapFromGlobal(QCursor::pos())))
	{
		conf.setZoom(conf.getZoom() - e->delta() * 0.005);
		openGL->update();
		//cout << conf.getZoom() << endl;
	}
}
void GPU_PBD_GARMENT_QT::mouseMoveEvent(QMouseEvent* e)
{
	if (!openGL->conf)
		return;
	if (e->buttons() & Qt::LeftButton)
	{
		conf.setSwingAngle(conf.getSwingAngle() + (e->globalX() - preX)*0.5);
		conf.setElevateAngle(conf.getElevateAngle() + (e->globalY() - preY)*0.5);
		preX = e->globalX();
		preY = e->globalY();
		openGL->update();
	}
	else if (e->buttons() & Qt::MiddleButton)
	{
		vector<float> pre = conf.getCenter();
		vector<float> center = { pre[0] + (e->globalX() - preX) * (float)0.01 ,pre[1] - (e->globalY() - preY) * (float)0.01 ,0.0 };
		conf.setCenter(center);
		cout << center[0] << " " << center[1] << " " << center[2] << endl;
		cout << conf.getZoom() << " " << conf.getSwingAngle() << " " << conf.getElevateAngle() << endl;
		preX = e->globalX();
		preY = e->globalY();
		openGL->update();
	}
}




void GPU_PBD_GARMENT_QT::timeStep()
{
	while (!doPause)
	{
		if (conf.getBodyPath() != "")
		{
			unsigned int stepSize = conf.getStepSize();
			unsigned int startFrame = conf.getStartFrame();
			unsigned int endFrame = conf.getEndFrame();
			string bvhName = conf.getBvhName();
			if ((++stepCount)>0 &&  stepCount % stepSize == 0)
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
					//OBJLoader::saveScaledAvatar(currentFrame);
					updateAvater(bvhName, currentFrame);
					curFrame->setText(QString::number(currentFrame));
					//saveSerGarmentFile(currentFrame - 1);
				}
			}
			sim.step(model, conf);
			if ((stepCount + 1) % stepSize == 0)
				openGL->update();

			for (unsigned int i = 0; i < model.getTriangleModels().size(); i++)
				model.getTriangleModels()[i]->updateMeshNormals(model.getParticles());
		}
		else
		{
			sim.step(model, conf);
			openGL->update();
			for (unsigned int i = 0; i < model.getTriangleModels().size(); i++)
				model.getTriangleModels()[i]->updateMeshNormals(model.getParticles());
		}

		QEventLoop loop;
		QTimer::singleShot(10, &loop, SLOT(quit()));
		loop.exec();
	}
}

void GPU_PBD_GARMENT_QT::buildModel()
{
	unsigned int clothNum = conf.getClothNum();
	for (int i = 0; i < clothNum; i++)
	{
		string clothPath = conf.getClothPath().at(i);
		vector<float> translate = conf.getTranslate().at(0);
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
	stepCount = conf.getStepCount();
	SimulationModel::TriangleModelVector& tm = model.getTriangleModels();
	for (unsigned int i = 0; i < tm.size(); i++)
	{
		TriangleModel* tri = model.getTriangleModels().at(i);
		tri->setFrictionCoeff(conf.getClothCoeff().at(i)[0]);
		tri->setRestitutionCoeff(conf.getClothCoeff().at(i)[1]);
		tri->setBendingCoeff(conf.getClothCoeff().at(i)[2]);
		tri->setDampingCoeff(conf.getClothCoeff().at(i)[4]);
		tri->setSlideFrictionCoeff(conf.getClothCoeff().at(i)[6]);
		tri->setCollisionCoeff(conf.getClothCoeff().at(i)[5]);

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
		cd.addCollisionSphereOnFaces(i, CollisionDetection::CollisionObject::TriangleModelCollisionObjectType, faces, faces.size(), &pd.getPosition(offset), nVert, 0.001);
	}

	//为空气摩擦约束计算面积
	for (int i = 0; i < model.getTriangleModels().size(); i++)
	{
		IndexedFaceMesh::Faces faces= model.getTriangleModels().at(i)->getParticleMesh().getFaces();
		unsigned int indexOffset = model.getTriangleModels().at(i)->getIndexOffset();
		for (int j = 0; j < faces.size() / 3; j++)
		{
			unsigned int first = faces[j * 3] + indexOffset;
			unsigned int second = faces[j * 3 + 1] + indexOffset;
			unsigned int third = faces[j * 3 + 2] + indexOffset;
			Real area = (((pd.getPosition(first) - pd.getPosition(second)).cross(pd.getPosition(first) - pd.getPosition(third))).norm()) * 0.5;
			pd.setArea(first, pd.getArea(first) + area / 3.0);
			pd.setArea(second, pd.getArea(second) + area / 3.0);
			pd.setArea(third, pd.getArea(third) + area / 3.0);
		}
	}

	//创建人体模型
	if (conf.getBvhPath() != "")
	{
		string bvhName = conf.getBvhName();
		unsigned int startFrame = conf.getStartFrame();
		createAvaterOne(bvhName, startFrame);
	}
}

void GPU_PBD_GARMENT_QT::createPatternMesh(string str, const vector<float> translate, const vector<float> scale, bool findSpecialSeam)
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

	//更新vt的offset


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
			//20191107
			int normalIndices[3];
			for (int j = 0; j < 3; j++)
			{
				posIndices[j] = faces[i].posIndices[j] - 1;
				if (nTexCoords > 0)
				{
					texIndices[j] = faces[i].texIndices[j] - 1;
					mesh.addUVIndex(texIndices[j]);
					normalIndices[j] = faces[i].normalIndices[j] - 1;
					mesh.addNormalIndex(normalIndices[j]);
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
		IndexedFaceMesh::UVIndices normaIndices = mesh.getNormalIndices();
		Vector3r* points = new Vector3r[vd.size()];
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

		model.addTriangleModel(vd.size(), (m_indices.size()) / 3, &points[0], &indices[0], uvIndices, normaIndices, uvs,findSpecialSeam);
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

void GPU_PBD_GARMENT_QT::initClothConstraints()
{
	//最后调整缝合线位置
	initDistanceConstraints();
	initBendingConstraints();
	initSpecialSeamConstraints();
	initInnerPatterSeamConstraints();
	initBetweenPatternSeamConstraints();
}

void GPU_PBD_GARMENT_QT::initBetweenPatternSeamConstraints()
{
	ParticleData pd = model.getParticles();
	unsigned int patternOffset = 0;
	for (unsigned int n = 0; n < conf.getClothNum(); n++)
	{
		for (unsigned int i = patternOffset; i < conf.getPatternNum()[n] + patternOffset; i++)
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
				for (unsigned int k = i + 1; k < conf.getPatternNum()[n] + patternOffset; k++)
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
	}


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

void GPU_PBD_GARMENT_QT::initInnerPatterSeamConstraints()
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

void GPU_PBD_GARMENT_QT::initDistanceConstraints()
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

void GPU_PBD_GARMENT_QT::initBendingConstraints()
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
						if (model.getParticles().size() > 9800 && model.getParticles().size() < 10000)
						{
							model.addDistanceConstraint(vertex1, vertex2, cm);
							
						}
						else if (model.getParticles().size() == 11796)
						{

						}
						else if (model.getParticles().size() > 10000)
						{
							model.addDistanceConstraint(vertex1, vertex2, cm);
						}
						
					}
				}
			}
			mesh.release();
		}
	}
}

void GPU_PBD_GARMENT_QT::initSpecialSeamConstraints()
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

void GPU_PBD_GARMENT_QT::createAvaterOne(string filename, int currentFrame)
{
	SimulationModel::RigidBodyVector& rb = model.getRigidBodies();
	sim.setCollisionDetection(model, &cd);

	string bvhPath = conf.getBvhPath();

	//string fileName = bvhPath + filename + to_string(currentFrame) + ".obj";
	string fileName = conf.getBodyPath();

	IndexedFaceMesh mesh;
	VertexData vd;
	loadObj(fileName, vd, mesh, Vector3r(1.0, 1.0, 1.0), false, false);
	RigidBody* rigidBody = new RigidBody();

	rigidBody->initBody(vd, mesh, Vector3r(1.0, 1.0, 1.0));
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

void GPU_PBD_GARMENT_QT::loadObj(const std::string& filename, VertexData& vd, IndexedFaceMesh& mesh, const Vector3r& scale, bool isCloth, bool findSpecialSeam)
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

	cout << "Number of vertices: " << nPoints << endl;
}


void GPU_PBD_GARMENT_QT::updateAvater(string filename, int currentFrame)
{
	//START_TIMING("---1");
	std::vector<OBJLoader::Vec3f> x = globalPosition[currentFrame - 1];
	//STOP_TIMING_AVG_PRINT;
	//START_TIMING("---2");
	cout << "Loading motion........................walk" << currentFrame << ".obj" << endl;
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
