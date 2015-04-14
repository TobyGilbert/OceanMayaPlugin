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
MObject OceanNode::m_windDirectionY;
MObject OceanNode::m_windSpeed;
MObject OceanNode::m_choppiness;
MObject OceanNode::m_time;
double OceanNode::m_wdx;
double OceanNode::m_wdy;
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
    m_wdx = 1.0;
    m_wdy = 1.0;
    m_ws = 100.0;
    m_amp = 0.03;
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
    m_amplitude = numAttr.create( "amplitude", "amp", MFnNumericData::kDouble, 0.01, &status );
    CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL( status , "Unable to create \"amplitude\" attribute" );
    numAttr.setChannelBox( true );			// length attribute appears in channel box
    // add attribute
    status = addAttribute( m_amplitude );
    CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL( status , "Unable to add \"amplitude\" attribute to OceanNode" );

    // the wind speed inputs
    m_windDirectionX = numAttr.create( "windDirectionX", "wdx", MFnNumericData::kDouble, 0.5, &status);
    CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL( status, "Unable to create \"wdx\" attribute");
    numAttr.setChannelBox(true);
    // add attribute
    status = addAttribute( m_windDirectionX );
    CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL(status, "Unable to add \"wdx\" attribute to OceanNode")

    m_windDirectionY = numAttr.create( "windDirectionY", "wdy", MFnNumericData::kDouble, 0.5, &status);
    CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL( status, "Unable to create \"wdy\" attribute");
    numAttr.setChannelBox(true);
    // add attribute
    status = addAttribute( m_windDirectionY );
    CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL(status, "Unable to add \"wdy\" attribute to OceanNode");

    m_windSpeed = numAttr.create( "windSpeed", "ws", MFnNumericData::kDouble, 100.0, &status);
    CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL( status, "Unable to create \"ws\" attribute");
    numAttr.setChannelBox(true);
    // add attribute
    status = addAttribute( m_windSpeed );
    CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL(status, "Unable to add \"ws\" attribute to OceanNode");

    m_choppiness = numAttr.create("chopiness", "chp", MFnNumericData::kDouble, 0.0, &status);
    CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL(status, "Unable to create \"chopiness\" attribute");
    numAttr.setChannelBox(true);
    // add attribute
    status = addAttribute(m_choppiness);
    CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL(status, "Unable to add \"choppiness\" attribute to OceanNode");

    // now the time inputs
    m_time = numAttr.create("time", "t", MFnNumericData::kDouble, 0.0, &status);
    CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL(status, "Unable to create \"t\" attribute");
    // Add the attribute
    status = addAttribute(m_time);
    numAttr.setHidden(true);
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
    attributeAffects(m_windDirectionY, m_output);
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

        dataHandle = _data.inputValue(m_windDirectionX, &status);
        CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL(status, "Unable to get data handle for windDirectionX plug");
        // now get value for data handle
        double wdx = dataHandle.asDouble();
        dataHandle = _data.inputValue(m_windDirectionY, &status);
        CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL(status, "Unable to get data handle for windDirectionY plug");
        // now get value for data handle
        double wdy = dataHandle.asDouble();
        m_ocean->setWindVector(make_float2(wdx, wdy));

        dataHandle = _data.inputValue(m_windSpeed, &status);
        CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL(status, "Unable to get data handle for windSpeed plug");
        // now get value for data handle
        double ws = dataHandle.asDouble();
        m_ocean->setWindSpeed(ws);

        // Only create a new frequency domain if either amplitude or the wind vecotr has changed
        if (m_amp != amp || m_wdx != wdx || m_wdy != wdy || m_ws != ws ){
            MGlobal::displayInfo("here");
            m_ocean->createH0();
            m_amp = amp;
            m_wdx = wdx;
            m_wdy = wdy;
            m_ws = ws;
        }

        dataHandle = _data.inputValue(m_choppiness, &status);
        CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL(status, "Unable to get data handle for the choppiness plug");
        double choppiness = dataHandle.asDouble();

        dataHandle = _data.inputValue(m_time, &status);
        CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL(status, "Unable to get data handle for time plug");

        MDataHandle outputData = _data.outputValue(m_output, &status);
        CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL( status , "Unable to get data handle for output plug" );

        MFnMeshData mesh;
        MObject outputObject = mesh.create(&status);
        CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL(status, "Unable to create output mesh");

        // Find the current frame number we're on and create the grid based on this
        MTime frameNo;
        MAnimControl anim;
        // Set min and mix frames
        frameNo.setValue(0);
        anim.setMinTime(frameNo);

        createGrid((int)pow(2.0, m_res+7), anim.currentTime().value()/70.0, choppiness, outputObject, status);
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
    float2* chopYArray = m_ocean->getChopY();

    // Sourced form Jon Macey's NGL library
    for(int z=0; z<_resolution; z++){
        for(int x=0; x<_resolution; x++){
            float height = heights[z * _resolution + x].x/50000.0;
            float chopX = _choppiness * chopXArray[z * _resolution + x].x;
            float chopY = _choppiness * chopYArray[z * _resolution + x].x;
            int sign = 1.0;
            if ((x+z) % 2 != 0){
                sign = -1.0;
            }
            vertices.append((xPos + (chopX * sign)), height * sign, (zPos + (chopY * sign)));
            // calculate the new position
            xPos+=wStep;
        }
        // now increment to next z row
        zPos+=dStep;
        // we need to re-set the xpos for new row
        xPos=-((float)width/2.0);
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

    m_grid.create(vertices.length(), numTris, vertices, numFaceVertices, faceVertices, _outputData, &_status);
}
