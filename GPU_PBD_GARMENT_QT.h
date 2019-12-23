#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_GPU_PBD_GARMENT_QT.h"
#include "ConfigurationLoader.h"
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
#include "MyVector.h"
#include "device_launch_parameters.h"
#include "myOpenGL.h"
#include <QHBoxLayout>
#include<QFileDialog>
#include<qtimer.h>
#include<QtGui>
#include<QWidget>
#include<QRadioButton>
#include<QScrollArea>
#include<QDoubleSpinBox>
#include<QPushButton>
#include<QScrollArea>
#include<QMenu>
#include<QMenuBar>
#include<QGroupBox>
#include<QFormLayout>
#include<QLabel>
#include<QMessageBox>
#include<QToolBar>
#include<QProgressDialog>
#include<QTimer>

using namespace std;
using namespace Utilities;
using namespace PBD;


class GPU_PBD_GARMENT_QT : public QMainWindow
{
	Q_OBJECT

public:
	GPU_PBD_GARMENT_QT(QWidget *parent = Q_NULLPTR);

private:
	//总布局
	QHBoxLayout* totalLayout;
	QVBoxLayout* leftLayout;
	QVBoxLayout* middleLayout;
	QVBoxLayout* rightLayout;
	//opengGL 显示区
	myOpenGL *openGL;
	//导航栏快捷键
	QToolBar* toolBar;
	QPushButton* startBtn;
	QPushButton* dictBtn;
	QPushButton* resetBtn;
	QPushButton* impoAvaBtn;
	QPushButton* impoGarBtn;
	QPushButton* clearBtn;
	QPushButton* saveBtn;
	//菜单
	QAction* startAni;
	QAction* stopAni;
	QAction* resetAni;
	QLabel* curFrame;
	
	QLabel* temp;
	QGroupBox* garment;
	QGroupBox* avatar;
	QScrollArea* scrollArea;
	QScrollArea* scrollArea2;
	vector<QDoubleSpinBox*> doubleSpinBoxs;
	vector<QGroupBox*> groupBoxs;
	vector<QLabel*> labels;
	Ui::GPU_PBD_GARMENT_QTClass ui;
	QProgressDialog* progress;
	QTimer* t;

private slots:
	void readConfigFile();
	void importGarmentFile(string fileName);
	void readGarmentFile();
	void saveGarmentFile();
	void readAvatarFile();
	void setAttrVal();
	void setPause();
	void startAnimation();
	void stopAnimation();
	void resetAnimation();
	void exitApp();
	void clearScene();
	//void loadBodySeq();
	
protected:
	void mouseMoveEvent(QMouseEvent* e);
	void mousePressEvent(QMouseEvent* e);
	void wheelEvent(QWheelEvent* e);

public:
	void initDevice();
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
	void timeStep();
	void showScrollArea();
	void showScrollArea2();
};
