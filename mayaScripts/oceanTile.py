import maya.cmds as cmds

def deleteSun():
    cmds.select('Sun')
    cmds.delete()

def deleteGrid(res):
    for i in range(res*res):
        cmds.select('polySurface'+str(i+1))
        cmds.delete();
        cmds.select('oceanTransform'+str(i))
        cmds.delete();

def tileGrid(res):
    cmds.createNode('oceanNode')
    cmds.connectAttr('time1.outTime', 'oceanNode1.time')

    for i in range(res):
        for j in range(res):
            cmds.createNode('transform', n='oceanTransform'+str(i*res+j))
            cmds.createNode('mesh', n='oceanMesh'+str(i*res+j))
            cmds.connectAttr('oceanNode1.output', 'oceanMesh'+str(i*res+j)+'.inMesh')
            cmds.select('polySurface'+str(i*res+j+1))
            cmds.xform(ws=True, t=(i*497.0-(((res-1)*497.0)/2.0), 0.0, j*497.0-(((res-1)*497)/2.0)))
            
def createScene(res):
    #deleteSun()
    cmds.pointLight(n='Sun')
    cmds.select('Sun')
    cmds.xform(ws=True, t=(0.0, 20.0, -500.0))
           
    #deleteGrid(res)
    tileGrid(res)
    
createScene(3)