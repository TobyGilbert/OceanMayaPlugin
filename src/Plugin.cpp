#include "OceanNode.h"
#include <maya/MFnPlugin.h>

//----------------------------------------------------------------------------------------------------------------------

MStatus initializePlugin( MObject obj )
{
    MStatus   status;
    MFnPlugin plugin( obj, "", "Toby Gilbert" , "2015" );

  // register our nodes and the commands which will be called
    status = plugin.registerNode( "oceanNode", OceanNode::m_id, &OceanNode::creator, &OceanNode::initialize, MPxNode::kDependNode );
    if (!status)
    {
    status.perror("Unable to register OceanNode" );
        return status;
    }

    return status;
}

//----------------------------------------------------------------------------------------------------------------------

MStatus uninitializePlugin( MObject obj )
{
    MStatus   status;
    MFnPlugin plugin( obj );


  status = plugin.deregisterNode( OceanNode::m_id );
    if (!status)
    {
    status.perror( "unable to deregister OceanNode" );
        return status;
    }

    return status;
}
//----------------------------------------------------------------------------------------------------------------------


