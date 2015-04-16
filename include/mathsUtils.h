#ifndef MATHSUTILS_H
#define MATHUTILS_H

#include <cuda_runtime.h>

float2 normalise(float2 _i) {
    float length = sqrt( (_i.x*_i.x) + (_i.y*_i.y) );
    return make_float2(_i.x / length, _i.y/length);
}
float3 normalise(float3 _i) {
    float length = sqrt( (_i.x*_i.x) + (_i.y*_i.y)  + (_i.z*_i.z) );
    return make_float3(_i.x/length, _i.y/length, _i.z/length);
}
float4 normalise(float4 _i) {
    float length = sqrt( (_i.x*_i.x) + (_i.y*_i.y) + (_i.z*_i.z) + (_i.w*_i.w) );
    return make_float4(_i.x/length, _i.y/length, _i.z/length, _i.w/length);
}

float length(float2 _i){
    return sqrt( (_i.x*_i.x) + (_i.y*_i.y) );
}

#endif
