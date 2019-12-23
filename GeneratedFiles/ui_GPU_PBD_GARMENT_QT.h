/********************************************************************************
** Form generated from reading UI file 'GPU_PBD_GARMENT_QT.ui'
**
** Created by: Qt User Interface Compiler version 5.12.3
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_GPU_PBD_GARMENT_QT_H
#define UI_GPU_PBD_GARMENT_QT_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_GPU_PBD_GARMENT_QTClass
{
public:
    QAction *actionImport;
    QWidget *centralWidget;

    void setupUi(QMainWindow *GPU_PBD_GARMENT_QTClass)
    {
        if (GPU_PBD_GARMENT_QTClass->objectName().isEmpty())
            GPU_PBD_GARMENT_QTClass->setObjectName(QString::fromUtf8("GPU_PBD_GARMENT_QTClass"));
        GPU_PBD_GARMENT_QTClass->resize(981, 628);
        actionImport = new QAction(GPU_PBD_GARMENT_QTClass);
        actionImport->setObjectName(QString::fromUtf8("actionImport"));
        centralWidget = new QWidget(GPU_PBD_GARMENT_QTClass);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        centralWidget->setEnabled(false);
        GPU_PBD_GARMENT_QTClass->setCentralWidget(centralWidget);

        retranslateUi(GPU_PBD_GARMENT_QTClass);

        QMetaObject::connectSlotsByName(GPU_PBD_GARMENT_QTClass);
    } // setupUi

    void retranslateUi(QMainWindow *GPU_PBD_GARMENT_QTClass)
    {
        GPU_PBD_GARMENT_QTClass->setWindowTitle(QApplication::translate("GPU_PBD_GARMENT_QTClass", "GPU_PBD_GARMENT_QT", nullptr));
        actionImport->setText(QApplication::translate("GPU_PBD_GARMENT_QTClass", "\345\257\274\345\205\245\346\250\241\345\236\213", nullptr));
    } // retranslateUi

};

namespace Ui {
    class GPU_PBD_GARMENT_QTClass: public Ui_GPU_PBD_GARMENT_QTClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_GPU_PBD_GARMENT_QT_H
