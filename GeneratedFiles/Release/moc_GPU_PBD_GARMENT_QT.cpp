/****************************************************************************
** Meta object code from reading C++ file 'GPU_PBD_GARMENT_QT.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.12.3)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../GPU_PBD_GARMENT_QT.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'GPU_PBD_GARMENT_QT.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.12.3. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_GPU_PBD_GARMENT_QT_t {
    QByteArrayData data[16];
    char stringdata0[199];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_GPU_PBD_GARMENT_QT_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_GPU_PBD_GARMENT_QT_t qt_meta_stringdata_GPU_PBD_GARMENT_QT = {
    {
QT_MOC_LITERAL(0, 0, 18), // "GPU_PBD_GARMENT_QT"
QT_MOC_LITERAL(1, 19, 14), // "readConfigFile"
QT_MOC_LITERAL(2, 34, 0), // ""
QT_MOC_LITERAL(3, 35, 17), // "importGarmentFile"
QT_MOC_LITERAL(4, 53, 6), // "string"
QT_MOC_LITERAL(5, 60, 8), // "fileName"
QT_MOC_LITERAL(6, 69, 15), // "readGarmentFile"
QT_MOC_LITERAL(7, 85, 15), // "saveGarmentFile"
QT_MOC_LITERAL(8, 101, 14), // "readAvatarFile"
QT_MOC_LITERAL(9, 116, 10), // "setAttrVal"
QT_MOC_LITERAL(10, 127, 8), // "setPause"
QT_MOC_LITERAL(11, 136, 14), // "startAnimation"
QT_MOC_LITERAL(12, 151, 13), // "stopAnimation"
QT_MOC_LITERAL(13, 165, 14), // "resetAnimation"
QT_MOC_LITERAL(14, 180, 7), // "exitApp"
QT_MOC_LITERAL(15, 188, 10) // "clearScene"

    },
    "GPU_PBD_GARMENT_QT\0readConfigFile\0\0"
    "importGarmentFile\0string\0fileName\0"
    "readGarmentFile\0saveGarmentFile\0"
    "readAvatarFile\0setAttrVal\0setPause\0"
    "startAnimation\0stopAnimation\0"
    "resetAnimation\0exitApp\0clearScene"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_GPU_PBD_GARMENT_QT[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
      12,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    0,   74,    2, 0x08 /* Private */,
       3,    1,   75,    2, 0x08 /* Private */,
       6,    0,   78,    2, 0x08 /* Private */,
       7,    0,   79,    2, 0x08 /* Private */,
       8,    0,   80,    2, 0x08 /* Private */,
       9,    0,   81,    2, 0x08 /* Private */,
      10,    0,   82,    2, 0x08 /* Private */,
      11,    0,   83,    2, 0x08 /* Private */,
      12,    0,   84,    2, 0x08 /* Private */,
      13,    0,   85,    2, 0x08 /* Private */,
      14,    0,   86,    2, 0x08 /* Private */,
      15,    0,   87,    2, 0x08 /* Private */,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 4,    5,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void GPU_PBD_GARMENT_QT::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<GPU_PBD_GARMENT_QT *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->readConfigFile(); break;
        case 1: _t->importGarmentFile((*reinterpret_cast< string(*)>(_a[1]))); break;
        case 2: _t->readGarmentFile(); break;
        case 3: _t->saveGarmentFile(); break;
        case 4: _t->readAvatarFile(); break;
        case 5: _t->setAttrVal(); break;
        case 6: _t->setPause(); break;
        case 7: _t->startAnimation(); break;
        case 8: _t->stopAnimation(); break;
        case 9: _t->resetAnimation(); break;
        case 10: _t->exitApp(); break;
        case 11: _t->clearScene(); break;
        default: ;
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject GPU_PBD_GARMENT_QT::staticMetaObject = { {
    &QMainWindow::staticMetaObject,
    qt_meta_stringdata_GPU_PBD_GARMENT_QT.data,
    qt_meta_data_GPU_PBD_GARMENT_QT,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *GPU_PBD_GARMENT_QT::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *GPU_PBD_GARMENT_QT::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_GPU_PBD_GARMENT_QT.stringdata0))
        return static_cast<void*>(this);
    return QMainWindow::qt_metacast(_clname);
}

int GPU_PBD_GARMENT_QT::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 12)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 12;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 12)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 12;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
