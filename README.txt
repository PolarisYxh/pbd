配置
win10+opengl+qt5.12.3+cuda10.2
vs2019解决方案配置：release
平台：x86
包含目录：
D:\Program\GPU_PBD_Garment\extern\glew\include\GL;D:\Program\GPU_PBD_Garment;D:\Program\GPU_PBD_Garment\extern\freeglut\include;D:\Program\GPU_PBD_Garment\extern;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include;C:\ProgramData\NVIDIA Corporation\CUDA Samples\v10.1\common\inc;$(IncludePath)

库目录
D:\Program\GPU_PBD_Garment\lib;C:\Program Files %28x86%29\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.20.27508\lib\x64;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib\x64;C:\ProgramData\NVIDIA Corporation\CUDA Samples\v10.1\common\lib\x64;$(LibraryPath)


C/C++ :-----------------

附件包含目录
D:\Program\GPU_PBD_Garment;%(AdditionalIncludeDirectories)
程序数据库（/zi)
等级1（/w1）
否(/sdl-)

优化： 最大优化(优选速度) (/O2)

预处理器定义
WIN32
_WINDOWS
NDEBUG
_CRT_SECURE_NO_DEPRECATE
PBD_DATA_PATH="../PBD_Garment_Simulation/data"
TW_NO_LIB_PRAGMA
TW_STATIC
FREEGLUT_LIB_PRAGMAS=0

代码生成
多线程DLL(/MD)
启用安全检查 (/GS)

链接器---------------------
附加库目录 
$(CUDA_PATH_V10_1)\lib\$(Platform);%(AdditionalLibraryDirectories)
附加依赖项
cudart.lib;lib\freeglut_rd.lib;opengl32.lib;glu32.lib;lib\AntTweakBar_rd.lib;lib\glew_rd.lib;glfw3.lib;%(AdditionalDependencies)

运行时点击任务栏的文件夹图标 用experimentData\walk\dress-311-Female Walk-9021.configFile模型，各个模型包含的模型路径需要更改





