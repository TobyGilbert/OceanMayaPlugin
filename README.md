Ocean Maya Plugin
=================


Introduction:
 
An ocean Maya plugin written by Toby Gilbert.  This plugin output an MPxMesh node which simulates the surface of an ocean

Installation:

- The plugin uses Qt for compilation in order to create the makefile run qMake \n\n
      $:qmake \n\n
- The run make to build the .so \n\n
      $:make \n\n
- In Maya you now need to load the plugin.  Go to window > Settings/Preferences > Plugin Manager and browse for liboceanFFT.so or oceanFFT.bundle if on mac. \n
- In the script editor load the python script oceanTile.py from the local mayaScrips directory. \n
- Run the createScene function which takes a parameter, res, which is the number of tiles wide/long and creates a grid.  This function will create an oceanNode, meshNode, timeNode and a polySurface. \n
- The simulation should now be set up to play with its default parameters.  To edit any of the parameters go to the oceanNode.

The oceanNode

The oceanNode has seven parameters which the user can alter: \n
Resolution:
The resolution of the ocean grid mesh.
Amplitude
The maximum amplitude of any wave
Frequency
The frequency of the waves.
Wind Direction X
Parameter from 0-1 to control the wind direction.
Wind Direction Z
Parameter from 0-1 to control the wind direction.
Wind Speed
The velocity of the wind.
Choppiness
 How sharp the peaks of the waves are.

 OceanShader
 The OceanShader is a CGFX hardware shader. \n
 Installation
 - To load the shader first load the cgfxShader.so from the Plugin Manager.
 - Then assign a new CGFX shader to the oceanMesh.
 - Load the OceanShader.cgfx into the CGFX node. The mesh should now appear green.
 - To visulise the shader inside the viewpoint we need to enable hardware texturing.  To do this in the viewport go to Shading > Hardware Texturing.
 Usage:
 - Camera - This need to be assigned to the current camera (default persp).
 - Sun - This can be mapped to a locator and is used for moving the position of the sun.
 - Sea Base Colour - The main colour of the sea
 - Sea Top Colour -
 - Environment - The environment map used for reflections
