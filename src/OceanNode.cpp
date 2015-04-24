//----------------------------------------------------------------------------------------------------------------------
/// @class OceanNode class
/// @brief A class for generating a custom maya node for generating an ocean surface using fft and NVidia's Cuda
/// @author Toby Gilbert
/// @version 1.0
/// @date 03/04/15
//----------------------------------------------------------------------------------------------------------------------
#include "OceanNode.h"
#include <boost/lexical_cast.hpp>
//----------------------------------------------------------------------------------------------------------------------
/// @brief simple macro to check status and return if error
/// originally written by Sola Aina
//----------------------------------------------------------------------------------------------------------------------
#define CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL( stat , message )				\
    if( !status )										\
    {											\
        MString errorString = status.errorString() + " -- " + MString( message );	\
        MGlobal::displayError( errorString );						\
        return MStatus::kFailure;							\
    }											\
//----------------------------------------------------------------------------------------------------------------------
/// @brief simple macro to check status and return if error
/// originally written by Sola Aina
//----------------------------------------------------------------------------------------------------------------------

#define CHECK_STATUS_AND_RETURN_IF_FAIL( stat , message )					\
    if( !status )										\
    {											\
        MString errorString = status.errorString() + " -- " + MString( message );	\
        MGlobal::displayError( errorString );						\
    }											\

/// @brief macro to get rid of compiler warnings
#define UNUSED(arg) (void)arg;
//----------------------------------------------------------------------------------------------------------------------
MTypeId OceanNode::m_id( 0x70003 );		// numeric Id of node
const MString OceanNode::typeName( "oceanNode" );
MObject OceanNode::m_resolution;
MObject OceanNode::m_amplitude;
MObject OceanNode::m_output;
MObject OceanNode::m_windDirectionX;
MObject OceanNode::m_windDirectionZ;
MObject OceanNode::m_windSpeed;
MObject OceanNode::m_choppiness;
MObject OceanNode::m_time;
MObject OceanNode::m_frequency;
double OceanNode::m_wdx;
double OceanNode::m_wdz;
double OceanNode::m_ws;
double OceanNode::m_amp;
int OceanNode::m_res;
//----------------------------------------------------------------------------------------------------------------------
OceanNode::~OceanNode(){
    // delete the ocean node
    if(m_ocean !=0){
        delete m_ocean;
    }
}
//----------------------------------------------------------------------------------------------------------------------
// this method creates the attributes for our node and sets some default values etc
//----------------------------------------------------------------------------------------------------------------------
MStatus	OceanNode::initialize(){
    // Attributes to check whether the amplitude or wind vector have changed
    m_wdx = 0.0;
    m_wdz = 1.0;
    m_ws = 100.0;
    m_amp = 100.0;
    m_res = 0;

    MStatus status;

    // an emum attribute for use with the resolution
    MFnEnumAttribute enumAttr;

    // resolution
    m_resolution = enumAttr.create("resolution", "res", 0, &status);
    CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL(status, "Unable to create \"resolution\" attribute");
    enumAttr.addField("128x128", RES128);
    enumAttr.addField("256x256", RES256);
    enumAttr.addField("512x512", RES512);
    enumAttr.addField("1024x1024", RES1024);
    enumAttr.setKeyable(true);
    status = addAttribute(m_resolution);
    CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL(status, "Unable to add \"resolution\" attribute to OceanNode");

    // now we are going to add several number attributes
    MFnNumericAttribute	numAttr;

    // amplitde
    m_amplitude = numAttr.create( "amplitude", "amp", MFnNumericData::kDouble, 100.0, &status );
    CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL( status , "Unable to create \"amplitude\" attribute" );
    numAttr.setChannelBox( true );
    // add attribute
    status = addAttribute( m_amplitude );
    CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL( status , "Unable to add \"amplitude\" attribute to OceanNode" );

    // frequency
    m_frequency = numAttr.create("frequency", "frq", MFnNumericData::kDouble, 0.5, &status);
    CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL(status, "Unable to create \"frequency\" attribute");
    numAttr.setChannelBox(true);
    status = addAttribute(m_frequency);
    CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL(status, "Unable to add \"frequency\" attribute to OceanNodee");

    // the wind speed inputs
    MFnNumericAttribute	windDirectionAttr;
    m_windDirectionX = windDirectionAttr.create( "windDirectionX", "wdx", MFnNumericData::kDouble, 0.0, &status);
    CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL( status, "Unable to create \"wdx\" attribute");
    windDirectionAttr.setChannelBox(true);
    windDirectionAttr.setMin(0.0);
    windDirectionAttr.setMax(1.0);
    // add attribute
    status = addAttribute( m_windDirectionX );
    CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL(status, "Unable to add \"wdx\" attribute to OceanNode")

    m_windDirectionZ = windDirectionAttr.create( "windDirectionZ", "wdz", MFnNumericData::kDouble, 0.5, &status);
    CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL( status, "Unable to create \"wdy\" attribute");
    windDirectionAttr.setChannelBox(true);
    windDirectionAttr.setMin(0.0);
    windDirectionAttr.setMax(1.0);
    // add attribute
    status = addAttribute( m_windDirectionZ );
    CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL(status, "Unable to add \"wdz\" attribute to OceanNode");

    m_windSpeed = numAttr.create( "windSpeed", "ws", MFnNumericData::kDouble, 100.0, &status);
    CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL( status, "Unable to create \"ws\" attribute");
    numAttr.setChannelBox(true);
    // add attribute
    status = addAttribute( m_windSpeed );
    CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL(status, "Unable to add \"ws\" attribute to OceanNode");

    MFnNumericAttribute	chopAttr;
    m_choppiness = chopAttr.create("chopiness", "chp", MFnNumericData::kDouble, 0.0, &status);
    CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL(status, "Unable to create \"chopiness\" attribute");
    chopAttr.setChannelBox(true);
    chopAttr.setMax(2.0);
    chopAttr.setMin(0.0);
    // add attribute
    status = addAttribute(m_choppiness);
    CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL(status, "Unable to add \"choppiness\" attribute to OceanNode");

    // now the time inputs
    MFnNumericAttribute	timeAttr;
    m_time = timeAttr.create("time", "t", MFnNumericData::kDouble, 0.0, &status);
    CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL(status, "Unable to create \"t\" attribute");
    // Add the attribute
    status = addAttribute(m_time);
    timeAttr.setHidden(true);
    CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL(status, "Unable to add \"t\" attribute to OceanNode");

    // create the output attribute
    MFnTypedAttribute typeAttr;
    m_output = typeAttr.create("output", "out", MFnData::kMesh, &status);
    CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL(status, "Unable to create \"output\" attribute");
    typeAttr.setStorable(false);
    typeAttr.setHidden(true);
    status = addAttribute(m_output);
    CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL(status, "Unable to add \"output\" attribute to OceanNode");

    // this links the different elements together forcing a re-compute each time the values are changed
    attributeAffects(m_resolution, m_output);
    attributeAffects(m_amplitude,m_output);
    attributeAffects(m_windDirectionX, m_output);
    attributeAffects(m_windDirectionZ, m_output);
    attributeAffects(m_windSpeed, m_output);
    attributeAffects(m_choppiness, m_output);
    attributeAffects(m_time, m_output);

    // report all was good
    return MStatus::kSuccess;
}
//----------------------------------------------------------------------------------------------------------------------
// This method should be overridden in user defined nodes.
// Recompute the given output based on the nodes inputs.
// The plug represents the data value that needs to be recomputed, and the data block holds the storage
// for all of the node'_scale attributes.
//----------------------------------------------------------------------------------------------------------------------
MStatus OceanNode::compute( const MPlug &_plug , MDataBlock &_data ){
    MStatus status;
    // see if we get the output plug
    if( _plug == m_output){    

        MDataHandle dataHandle;

        dataHandle = _data.inputValue(m_resolution, &status);
        CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL(status, "Unable to get data handle for resolution plug");

        if (m_res != dataHandle.asInt()){
            switch(dataHandle.asInt()){
                case 0:
                    m_ocean->setResolution(128);
                    MGlobal::displayInfo("Resolution: 128");
                    break;
                case 1:
                    m_ocean->setResolution(256);
                    MGlobal::displayInfo("Resolution: 256");
                    break;
                case 2:
                    m_ocean->setResolution(512);
                    MGlobal::displayInfo("Resolution: 512");
                    break;
                case 3:
                    m_ocean->setResolution(1024);
                    MGlobal::displayInfo("Resolution: 1024");
                    break;
                default:
                    break;
            }
            m_res = dataHandle.asInt();
        }

        dataHandle = _data.inputValue( m_amplitude , &status );
        CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL( status , "Unable to get data handle for amplitude plug" );
        // now get the value for the data handle as a double
        double amp = dataHandle.asDouble();
        m_ocean->setAmplitude(amp);

        dataHandle = _data.inputValue(m_frequency, &status);
        CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL(status, "Unable to get handle for \"frequency\" plug");
        double freq = dataHandle.asDouble();
        m_ocean->setFrequency(freq);

        dataHandle = _data.inputValue(m_windDirectionX, &status);
        CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL(status, "Unable to get data handle for windDirectionX plug");
        // now get value for data handle
        double wdx = dataHandle.asDouble();
        dataHandle = _data.inputValue(m_windDirectionZ, &status);
        CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL(status, "Unable to get data handle for windDirectionY plug");
        // now get value for data handle
        double wdz = dataHandle.asDouble();
        m_ocean->setWindVector(make_float2(wdx, wdz));

        dataHandle = _data.inputValue(m_windSpeed, &status);
        CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL(status, "Unable to get data handle for windSpeed plug");
        // now get value for data handle
        double ws = dataHandle.asDouble();
        m_ocean->setWindSpeed(ws);

        // Only create a new frequency domain if either amplitude or the wind vecotr has changed
        if (m_amp != amp || m_wdx != wdx || m_wdz != wdz || m_ws != ws ){
            MGlobal::displayInfo("here");
            m_ocean->createH0();
            m_amp = amp;
            m_wdx = wdx;
            m_wdz = wdz;
            m_ws = ws;
        }

        dataHandle = _data.inputValue(m_choppiness, &status);
        CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL(status, "Unable to get data handle for the choppiness plug");
        double choppiness = dataHandle.asDouble();

        dataHandle = _data.inputValue(m_time, &status);
        CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL(status, "Unable to get data handle for time plug");
        MTime time = dataHandle.asTime();

        MDataHandle outputData = _data.outputValue(m_output, &status);
        CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL( status , "Unable to get data handle for output plug" );

        MFnMeshData mesh;
        MObject outputObject = mesh.create(&status);
        CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL(status, "Unable to create output mesh");

        // Find the current frame number we're on and create the grid based on this
        MAnimControl anim;
        anim.setMinTime(time);

        createGrid((int)pow(2.0, m_res+7), anim.currentTime().value()/24, choppiness, outputObject, status);

        CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL(status, "Unable to to create grid");

        outputData.set(outputObject);

        // clean the output plug, ie unset it from dirty so that maya does not re-evaluate it
        _data.setClean( _plug );

        return MStatus::kSuccess;
    }

    return MStatus::kUnknownParameter;
}
//----------------------------------------------------------------------------------------------------------------------
OceanNode::OceanNode(){
    MGlobal::displayInfo("Constructing new Ocean");
    m_ocean = new Ocean();
}
//----------------------------------------------------------------------------------------------------------------------
void* OceanNode::creator(){
    return new OceanNode();
}
//----------------------------------------------------------------------------------------------------------------------
void OceanNode::createGrid(int _resolution, double _time, double _choppiness, MObject& _outputData, MStatus &_status){
    int numTris = (_resolution-1)*(_resolution-1)*2;

    MFloatPointArray vertices;
    MIntArray numFaceVertices;
    MIntArray faceVertices;
    int tris[numTris*3];

    int width = 500;
    int depth = 500;

    // calculate the deltas for the x,z values of our point
    float wStep=(float)width/(float)_resolution;
    float dStep=(float)depth/(float)_resolution;
    // now we assume that the grid is centered at 0,0,0 so we make
    // it flow from -w/2 -d/2
    float xPos=-((float)width/2.0);
    float zPos=-((float)depth/2.0);

    // now loop from top left to bottom right and generate points

    m_ocean->update(_time);

    float2* heights = m_ocean->getHeights();
    float2* chopXArray = m_ocean->getChopX();
    float2* chopZArray = m_ocean->getChopZ();

    // Sourced form Jon Macey's NGL library
    for(int z=0; z<_resolution; z++){
        for(int x=0; x<_resolution; x++){
            // Divide the values we get out of the FFT by 50000 to get them in a suitable range
            float height = heights[z * _resolution + x].x/50000.0;
            float chopX = _choppiness * chopXArray[z * _resolution + x].x/50000.0;
            float chopZ= _choppiness * chopZArray[z * _resolution + x].x/50000.0;
            int sign = 1.0;
            if ((x+z) % 2 != 0){
                sign = -1.0;
            }
            vertices.append((xPos + (chopX * sign)), height * sign, (zPos + (chopZ * sign)));
            // calculate the new position
            zPos+=dStep;
        }
        // now increment to next z row
        xPos+=wStep;
        // we need to re-set the xpos for new row
        zPos=-((float)depth/2.0);
    }

    // Array for num vertices in each face
    for (int i=0; i<numTris; i++){
        numFaceVertices.append(3);
    }

    // Assign vertices to each face
    int fidx = 0;
    for (int i=0; i<(_resolution-1); i++){
        for (int j=0; j<(_resolution-1); j++){
            tris[fidx*3+0] = (i+1)*_resolution+j;
            tris[fidx*3+1] = i*_resolution+j+1;
            tris[fidx*3+2] = i*_resolution+j;
            fidx++;
            tris[fidx*3+0] = (i+1)*_resolution+j;
            tris[fidx*3+1] = (i+1)*_resolution+j+1;
            tris[fidx*3+2] = i*_resolution+j+1;
            fidx++;
        }
    }

    for (uint i=0; i<sizeof(tris)/sizeof(int); i++){
        faceVertices.append(tris[i]);
    }

   MFnMesh grid;
   grid.create(vertices.length(), numTris, vertices, numFaceVertices, faceVertices, _outputData, &_status);
}
//----------------------------------------------------------------------------------------------------------------------
