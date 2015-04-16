/*! \mainpage Ocean FFT Maya Plugin using NVidia's Cuda
 *
 * \section intro_sec Introduction
 *
 * An ocean Maya plugin written by Toby Gilbert.  This plugin output an MPxMesh node which simulates the surface of an ocean
 *
 * \section install_sec Installation
 *
 * - The plugin uses Qt for compilation in order to create the makefile run qMake \n\n
 *      $:qmake \n\n
 * - The run make to build the .so \n\n
 *      $:make \n\n
 * - In Maya you now need to load the plugin.  Go to window > Settings/Preferences > Plugin Manager and browse for liboceanFFT.so or oceanFFT.bundle if on mac. \n
 * - Now in the script editor load the python script oceanTile.py from the local mayaScrips directory. \n
 * - Run the createScene function which takes a parameter, res, which is the number of tiles wide/long and creates a grid.  This function will create an oceanNode, meshNode, timeNode and a polySurface. \n
 * - The simulation should now be set up to play with its default parameters.  To edit any of the parameters go to the oceanNode.
 *
 * \section node The oceanNode
 * The oceanNode has seven parameters which the user can alter: \n\n
 * - Resolution \n
 *   The resolution of the ocean grid mesh. \n\n
 * - Frequency \n
 *   The frequency of the waves. \n\n
 * - Wind Direction X \n
 *   Parameter from 0-1 to control the wind direction. \n\n
 * - Wind Direction Z \n
 *   Parameter from 0-1 to control the wind direction. \n\n
 * - Wind Speed \n
 *   The velocity of the wind. \n\n
 * - Choppiness \n
 *   The How sharp the peaks of the waves are. \n\n
 *
 *
 *
 *
 */

#include "OceanNode.h"
#include <maya/MFnPlugin.h>

//----------------------------------------------------------------------------------------------------------------------

MStatus initializePlugin( MObject obj )
{
    MStatus   status;
    MFnPlugin plugin( obj, "", "Toby Gilbert" , "Any" );

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


