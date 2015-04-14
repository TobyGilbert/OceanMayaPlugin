#include <shader.h>
#include <mi_shader_if.h>

struct zhang_illum_constant_t{
	miColor surfaceColour;
};


extern "C" DLLEXPORT int zhang_illum_constant_version(void){
	return(1);
}

extern "C" DLLEXPORT miBoolean zhang_illum_constant(miColor *result, miState *state, struct zhang_illum_constant_t *paras){
	// Check for illegal calls
	if (state->type == miRAY_SHADOW || state->type == miRAY_DISPLACE){
		return (miFALSE);
	}
	*result = *mi_eval_color(&paras->surfaceColour);
	result->a = 1;
	return(miTRUE);
}

