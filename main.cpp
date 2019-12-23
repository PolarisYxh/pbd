#include "GPU_PBD_GARMENT_QT.h"
#include <QtWidgets/QApplication>


int main(int argc, char *argv[])
{
	QApplication a(argc, argv);

	GPU_PBD_GARMENT_QT w;
	w.show();

	a.setWindowIcon(QIcon("logo.ico"));
	_CrtDumpMemoryLeaks();
	return a.exec();
}
