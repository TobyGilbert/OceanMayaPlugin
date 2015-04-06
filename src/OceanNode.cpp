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
// as these are static class attributes we need to set them here so the static methods can see them
// in particular see  http://www.tutorialspoint.com/cplusplus/cpp_static_members.htm
// "By declaring a function member as static, you make it independent of any particular object of the class.
// A static member function can be called even if no objects of the class exist and the static functions are accessed using only
// the class name and the scope resolution operator ::.
// A static member function can only access static data member, other static member
// functions and any other functions from outside the class.
// Static member functions have a class scope and they do not have access
// to the this pointer of the class. You could use a static member function to determine whether some
// objects of the class have been created or not."
//----------------------------------------------------------------------------------------------------------------------
MTypeId OceanNode::m_id( 0x70003 );		// numeric Id of node
const MString OceanNode::typeName( "oceanNode" );
MObject OceanNode::m_amplitude;
MObject OceanNode::m_output;
MObject OceanNode::m_windSpeedX;
MObject OceanNode::m_windSpeedY;
MObject OceanNode::m_choppiness;
MObject OceanNode::m_time;
double OceanNode::m_wsx;
double OceanNode::m_wsy;
double OceanNode::m_amp;
//----------------------------------------------------------------------------------------------------------------------
OceanNode::~OceanNode(){
    // delete the noise node
    if(m_ocean !=0){
        delete m_ocean;
    }
}
//----------------------------------------------------------------------------------------------------------------------
// this method creates the attributes for our node and sets some default values etc
//----------------------------------------------------------------------------------------------------------------------
MStatus	OceanNode::initialize(){
    // Attributes to check whether the amplitude or wind vector have changed
    m_wsx = 1.0;
    m_wsy = 1.0;
    m_amp = 0.03;

    MStatus status;

    // now we are going to add several number attributes
    MFnNumericAttribute	numAttr;

    // amplitde
    m_amplitude = numAttr.create( "amplitude", "amp", MFnNumericData::kDouble, 0.03, &status );
    CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL( status , "Unable to create \"amplitude\" attribute" );
    numAttr.setChannelBox( true );			// length attribute appears in channel box
    // add attribute
    status = addAttribute( m_amplitude );
    CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL( status , "Unable to add \"amplitude\" attribute to OceanNode" );

    // the wind speed inputs
    m_windSpeedX = numAttr.create( "windspeedx", "wsx", MFnNumericData::kDouble, 1.0, &status);
    CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL( status, "Unable to create \"wsx\" attribute");
    numAttr.setChannelBox(true);
    // add attribute
    status = addAttribute( m_windSpeedX );
    CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL(status, "Unable to add \"wsx\" attribute to OceanNode")

    m_windSpeedY = numAttr.create( "windspeedy", "wsy", MFnNumericData::kDouble, 1.0, &status);
    CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL( status, "Unable to create \"windspeedy\" attribute");
    numAttr.setChannelBox(true);
    // add attribute
    status = addAttribute( m_windSpeedY );
    CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL(status, "Unable to add \"windspeedy\" attribute to OceanNode");

    m_choppiness = numAttr.create("chopiness", "chp", MFnNumericData::kDouble, 1.0, &status);
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
    attributeAffects(m_amplitude,m_output);
    attributeAffects(m_windSpeedX, m_output);
    attributeAffects(m_windSpeedY, m_output);
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

        dataHandle = _data.inputValue( m_amplitude , &status );
        CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL( status , "Unable to get data handle for amplitude plug" );
        // now get the value for the data handle as a double
        double amp = dataHandle.asDouble();
        m_ocean->setAmplitude(amp);

        dataHandle = _data.inputValue(m_windSpeedX, &status);
        CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL(status, "Unable to get data handle for windspeedx plug");
        // now get value for data handle
        double wsx = dataHandle.asDouble();
        dataHandle = _data.inputValue(m_windSpeedY, &status);
        CHECK_STATUS_AND_RETURN_MSTATUS_IF_FAIL(status, "Unable to get data handle for windspeedy plug");
        // now get value for data handle
        double wsy = dataHandle.asDouble();
        m_ocean->setWindVector(make_float2(wsx, wsy));

        // Only create a new frequency domain if either amplitude or the wind vecotr has changed
        if (m_amp != amp || m_wsx != wsx || m_wsy != wsy){
            MGlobal::displayInfo("here");
            m_ocean->createH0();
            m_amp = amp;
            m_wsx = wsx;
            m_wsy = wsy;
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
        frameNo.setValue(200);
        anim.setMaxTime(frameNo);

        createGrid(anim.currentTime().value()/100.0, choppiness, outputObject, status);
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
void OceanNode::createGrid(double _time, double _choppiness, MObject& _outputData, MStatus &_status){
    int res = 512;
    int numTris = (res-1)*(res-1)*2;

    MFloatPointArray vertices;
    MIntArray numFaceVertices;
    MIntArray faceVertices;
    int tris[numTris*3];

    int width = 500;
    int depth = 500;

    // calculate the deltas for the x,z values of our point
    float wStep=(float)width/(float)res;
    float dStep=(float)depth/(float)res;
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
    for(int z=0; z<res; z++){
        for(int x=0; x<res; x++){
            float height = heights[z * res + x].x/50000.0;
            float chopX = _choppiness * chopXArray[z * res + x].x;
            float chopY = _choppiness * chopYArray[z * res + x].x;
            int sign = 1.0;
            if ((x+z) % 2 != 0){
                sign = -1.0;
            }
            // grab the colour and use for the Y (height) only use the red channel
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
    for (int i=0; i<(res-1); i++){
        for (int j=0; j<(res-1); j++){
            tris[fidx*3+0] = (i+1)*res+j;
            tris[fidx*3+1] = i*res+j+1;
            tris[fidx*3+2] = i*res+j;
            fidx++;
            tris[fidx*3+0] = (i+1)*res+j;
            tris[fidx*3+1] = (i+1)*res+j+1;
            tris[fidx*3+2] = i*res+j+1;
            fidx++;
        }
    }

    for (uint i=0; i<sizeof(tris)/sizeof(int); i++){
        faceVertices.append(tris[i]);
    }

    m_grid.create(vertices.length(), numTris, vertices, numFaceVertices, faceVertices, _outputData, &_status);
}
