####################################################################################
# This file is split into Three sections
# The first configures Qt and the source files for all platforms
# The second is the linux build
# The third the mac build
# (if your using windows you will need to add a fourth one!)
# first lets remove Qt core and gui not going to need it
####################################################################################
QT       -= core gui
####################################################################################
# This is the name of the plugin / final lib file
####################################################################################
TARGET=oceanFFT
####################################################################################
# for for mac we need a bundle so change the name
####################################################################################
macx:TARGET=oceanFFT.bundle
####################################################################################
# here we add the source files (and headers if required)
####################################################################################
SOURCES+=src/Plugin.cpp \
         src/Ocean.cpp \
         src/OceanNode.cpp \
         src/OceanCuda.cu

SOURCES-=src/OceanCuda.cu
HEADERS+=include/OceanNode.h \
         include/OceanCuda.h \
         include/Ocean.h \
         include/mathsUtils.h

OBJECTS_DIR = obj
INCLUDEPATH+=./include
INCLUDEPATH+=/usr/local/include
# these are defines required by Maya to re-define some C++
# stuff, we will add some more later to tell what platform
# we are on as well
DEFINES+=REQUIRE_IOSTREAM \
         _BOOL
####################################################################################
# These are the maya libs we need to link to, this will change depending
# upon which maya framework we use, just add them to the end of
# this list as required and they will be added to the build
####################################################################################
MAYALIBS=-lOpenMaya \
        -lFoundation \
        -lOpenMayaAnim
####################################################################################
# these are all the libs usually included by mayald in case you need
# them just add them to the list above and make sure you escape
####################################################################################
#-lOpenMayalib \
#-lOpenMayaAnim \
#-lOpenMaya \
#-lAnimSlice \
#-lDeformSlice \
#-lModifiers \
#-lDynSlice \
#-lKinSlice \
#-lModelSlice \
#-lNurbsSlice \
#-lPolySlice \
#-lProjectSlice \
#-lImage \
#-lShared \
#-lTranslators \
#-lDataModel \
#-lRenderModel \
#-lNurbsEngine \
#-lDependEngine \
#-lCommandEngine \
#-lFoundation \
#-lIMFbase \
#-lm -ldl
####################################################################################
# now tell linux we need to build a lib
####################################################################################
linux-g++*:TEMPLATE = lib
####################################################################################
# this tells qmake where maya is
####################################################################################
linux-g++*:MAYALOCATION=/opt/autodesk/maya
####################################################################################
# under linux we need to use the version of g++ used to build maya
# in this case g++412
####################################################################################
#linux-g++-64:QMAKE_CXX = g++412
####################################################################################
# set the include path for linux
####################################################################################
linux-g++*:INCLUDEPATH += $$MAYALOCATION/include \
                        /usr/X11R6/include
####################################################################################
# set which libs we need to include
####################################################################################
linux-g++*:LIBS += -L$$MAYALOCATION/lib \
                   $$MAYALIBS
####################################################################################
# tell maya we're building for linux
####################################################################################
linux:DEFINES+=linux
linux:DEFINES+=LINUX
####################################################################################
# linux flags
####################################################################################
linux*:QMAKE_CXX += -fPIC
####################################################################################
# tell maya we're building for Mac
####################################################################################
macx:DEFINES+=OSMac_
macx:MAYALOCATION=/Applications/Autodesk/maya2014
macx:CONFIG -= app_bundle
macx:INCLUDEPATH+=$$MAYALOCATION/devkit/include
####################################################################################
# under mac we need to build a bundle, to do this use
# the -bundle flag but we also need to not use -dynamic lib so
# remove this
####################################################################################
# path to cuda directory
macx:CUDA_DIR = /Developer/NVIDIA/CUDA-6.5
linux:CUDA_DIR = /opt/cuda-6.5

macx:LIBS +=-bundle
macx:LIBS += -L$$CUDA_DIR/lib -lcuda -lcudart -lcufft
linux:LIBS += -L$$CUDA_DIR/lib64 -lcuda -lcudart -lcufft

macx:LIBS -=-dynamiclib
####################################################################################

####################################################################################
macx:LIBS += -L$$MAYALOCATION/Maya.app/Contents/MacOS \
             $$MAYALIBS
####################################################################################

PROJECT_DIR = $$system(pwd)
# if we are on a mac define DARWIN
macx:DEFINES += DARWIN

# CUDA
NVCCFLAGS = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v -Xcompiler -fPIC

INCLUDEPATH += /usr/local/include
INCLUDEPATH +=./include /opt/local/include
INCLUDEPATH += $$CUDA_DIR/include
INCLUDEPATH += $$CUDA_DIR/samples/common/inc
INCLUDEPATH += $$CUDA_DIR/../shared/inc

DESTDIR=./

linux:CUDA_LIBS += -L$$CUDA_DIR/lib64
CUDA_LIBS += -L$$CUDA_DIR/lib
CUDA_LIBS += -L$$CUDA_DIR/samples/common/lib
CUDA_LIBS += -L/opt/local/lib
CUDA_LIBS += -lcudart  -lcufft

CUDA_SOURCES =src/OceanCuda.cu

CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')

# Prepare the extra compiler configuration
cuda.input = CUDA_SOURCES
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o

cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -g -G -gencode arch=compute_20,code=sm_20 -c $$NVCCFLAGS $$CUDA_INC $$CUDA_LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}

cuda.dependency_type = TYPE_C
cuda.depend_command = $$CUDA_DIR/bin/nvcc -g -G -M $$CUDA_INC $$NVCCFLAGS ${QMAKE_FILE_NAME}

QMAKE_EXTRA_UNIX_COMPILERS += cuda
