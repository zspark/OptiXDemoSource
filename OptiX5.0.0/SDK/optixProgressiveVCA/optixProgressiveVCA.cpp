/* 
 * Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//-----------------------------------------------------------------------------
//
//  optixProgressiveVCA.cpp -- Basic usage of progressive launch API and remote query of VCA.
//
//-----------------------------------------------------------------------------



#include <stdio.h>
#include <optix.h>
#include <sutil.h>
#include <math.h>
#include <stdlib.h>
#include <string>
#ifndef _WIN32
    #include <unistd.h>
    #include <cstring>
    #include <cctype>
#endif
#ifdef __APPLE__
    #include <GLUT/glut.h>
#elif defined( _WIN32 )
    #include <GL/freeglut.h>
#else
    #include <GL/glut.h>
#endif
#include <optixu/optixu_matrix_namespace.h>

const char* const SAMPLE_NAME = "optixProgressiveVCA";

void create_context( RTcontext* context, RTremotedevice rdev, RTbuffer* output_buffer, RTbuffer* stream_buffer );
void create_material( RTcontext context, RTmaterial* material );
void create_geometry( RTcontext context, RTmaterial material );

void cross( const float a[3], const float b[3], float result[3] );
void normalize( float a[3] );
float dot( const float a[3], const float b[3] );

// global variables
RTcontext  context = 0;
RTremotedevice rdev = 0;
RTbuffer output_buffer, stream_buffer;
RTmaterial material;

RTvariable var_eye;
RTvariable var_U;
RTvariable var_V;
RTvariable var_W;

std::string vca_url;
std::string vca_user;
std::string vca_password;
int         vca_num_nodes = 1;
int         vca_config_index = 0;

unsigned int width  = 800;
unsigned int height = 600;
int sqrt_occlusion_samples = 2;

int stream_bitrate;
int stream_fps;
float stream_gamma;
char stream_format[128];

float eye[3] = { 3.0f, 2.0f, -3.0f };
float eye_org[3] = { 3.0f, 2.0f, -3.0f };

// To track mouse movement
int xorg=0, yorg=0;
bool mouse_zoom = false;
float hfov_org = 75.0f, hfov = 75.0f;


void set_cam_vars( float eye[3], float fov )
{
    optix::float3 lookat, up, eyef3;
    optix::float3 camera_u, camera_v, camera_w;

    eyef3  = optix::make_float3( eye[0], eye[1], eye[2] );
    lookat = optix::make_float3( 0.0f, 0.3f, 0.0f );
    up     = optix::make_float3( 0.0f, 1.0f, 0.0f );

    const float aspect_ratio = (float)width/(float)height;

    sutil::calculateCameraVariables( eyef3, lookat, up, fov, aspect_ratio, camera_u, camera_v, camera_w );

    rtVariableSet3fv( var_eye, &eyef3.x );
    rtVariableSet3fv( var_U, &camera_u.x );
    rtVariableSet3fv( var_V, &camera_v.x );
    rtVariableSet3fv( var_W, &camera_w.x );
}

void create_remote_device( RTremotedevice *rdev )
{
    if( vca_url.empty() ) {
        printf( "No VCA cluster manager URL specified. Running locally.\n" );
        return;
    }

    printf( "Attempting to connect to: %s\n", vca_url.c_str() );

    RT_CHECK_ERROR( rtRemoteDeviceCreate( vca_url.c_str(), vca_user.c_str(), vca_password.c_str(), rdev ) );

    // Query compatible packages
    unsigned int num_configs;
    RT_CHECK_ERROR( rtRemoteDeviceGetAttribute( *rdev, RT_REMOTEDEVICE_ATTRIBUTE_NUM_CONFIGURATIONS, sizeof(unsigned int), &num_configs ) );

    if (num_configs == 0) {
        printf( "No compatible configurations found on this remote device\n" );
        exit(1);
    }

    // Display configurations
    printf( "Available configurations:\n" );
    for( unsigned int i=0; i<num_configs; ++i ) {
        char cfg_name[256];
        RT_CHECK_ERROR( rtRemoteDeviceGetAttribute( *rdev, (RTremotedeviceattribute)(RT_REMOTEDEVICE_ATTRIBUTE_CONFIGURATIONS+i), 256, cfg_name ) );
        printf(" [%d] %s\n", i, cfg_name );
    }

    printf( "Reserving %d nodes with configuration %d\n", vca_num_nodes, vca_config_index );
    RT_CHECK_ERROR( rtRemoteDeviceReserve( *rdev, vca_num_nodes, vca_config_index ) );

    printf( "Waiting for the cluster to be ready...." );
    while( true )
    {
        int ready;
        RT_CHECK_ERROR( rtRemoteDeviceGetAttribute( *rdev, RT_REMOTEDEVICE_ATTRIBUTE_STATUS, sizeof(int), &ready ) );
        if( ready == RT_REMOTEDEVICE_STATUS_READY )
            break;
        sutil::sleep( 10 );
    }
    printf( "done.\n" );

    return;
}

void create_context( RTcontext* context, RTremotedevice rdev, RTbuffer* output_buffer, RTbuffer* stream_buffer )
{
    // Context
    RT_CHECK_ERROR( rtContextCreate( context ) );

    // When using a remove device, it must be set right after creating the context.
    if( rdev ) {
        RT_CHECK_ERROR( rtContextSetRemoteDevice( *context, rdev ) );
    }

    RT_CHECK_ERROR( rtContextSetEntryPointCount( *context, 1 ) );
    RT_CHECK_ERROR( rtContextSetRayTypeCount( *context, 2 ) );
    RT_CHECK_ERROR( rtContextSetStackSize( *context, 320 ) );

    // Output buffer
    RT_CHECK_ERROR( rtBufferCreate( *context, RT_BUFFER_OUTPUT, output_buffer ) );
    RT_CHECK_ERROR( rtBufferSetFormat( *output_buffer, RT_FORMAT_FLOAT4 ) );
    RT_CHECK_ERROR( rtBufferSetSize2D( *output_buffer, width, height ) ); 
 
    RT_CHECK_ERROR( rtBufferCreate( *context, RT_BUFFER_PROGRESSIVE_STREAM, stream_buffer ) );
    RT_CHECK_ERROR( rtBufferSetFormat( *stream_buffer, RT_FORMAT_UNSIGNED_BYTE4 ) );
    RT_CHECK_ERROR( rtBufferSetSize2D( *stream_buffer, width, height ) );

    RT_CHECK_ERROR( rtBufferBindProgressiveStream( *stream_buffer, *output_buffer ) );

    // Ray generation program
    RTprogram raygen_program;
    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "camera.cu" );
    RT_CHECK_ERROR( rtProgramCreateFromPTXString( *context, ptx, "pinhole_camera", &raygen_program ) );
    RT_CHECK_ERROR( rtContextSetRayGenerationProgram( *context, 0, raygen_program ) );

    // Exception program
    RTprogram exception_program;
    RT_CHECK_ERROR( rtProgramCreateFromPTXString( *context, ptx, "exception", &exception_program ) );
    RT_CHECK_ERROR( rtContextSetExceptionProgram( *context, 0, exception_program ) );

    // Miss program
    RTprogram miss_program;
    ptx = sutil::getPtxString( SAMPLE_NAME, "constantbg.cu" );
    RT_CHECK_ERROR( rtProgramCreateFromPTXString( *context, ptx, "miss", &miss_program ) );
    RT_CHECK_ERROR( rtContextSetMissProgram( *context, 0, miss_program ) );
    
    // Variables
    RTvariable var_output_buffer;
    RTvariable var_scene_epsilon;
    RTvariable var_radiance_ray_type;
    RTvariable var_badcolor;
    RTvariable var_bgcolor;
    RTvariable var_frame;
    RT_CHECK_ERROR( rtContextDeclareVariable( *context, "output_buffer", &var_output_buffer ) );
    RT_CHECK_ERROR( rtContextDeclareVariable( *context, "scene_epsilon", &var_scene_epsilon ) );
    RT_CHECK_ERROR( rtContextDeclareVariable( *context, "radiance_ray_type", &var_radiance_ray_type ) );
    RT_CHECK_ERROR( rtContextDeclareVariable( *context, "bad_color", &var_badcolor ) );
    RT_CHECK_ERROR( rtContextDeclareVariable( *context, "bg_color", &var_bgcolor ) );
    RT_CHECK_ERROR( rtContextDeclareVariable( *context, "frame", &var_frame ) );
    RT_CHECK_ERROR( rtVariableSetObject( var_output_buffer, *output_buffer ) );
    RT_CHECK_ERROR( rtVariableSet1f( var_scene_epsilon, 1e-3f ) );
    RT_CHECK_ERROR( rtVariableSet1ui( var_radiance_ray_type, 0u ) );
    RT_CHECK_ERROR( rtVariableSet1ui( var_frame, 0u ) );
    RT_CHECK_ERROR( rtVariableSet3f( var_badcolor, 1.0f, 1.0f, 0.0f ) );
    RT_CHECK_ERROR( rtVariableSet3f( var_bgcolor, 0.462f, 0.725f, 0.0f ) );
    RT_CHECK_ERROR( rtContextDeclareVariable( *context, "eye",&var_eye ) );
    RT_CHECK_ERROR( rtContextDeclareVariable( *context, "U",&var_U ) );
    RT_CHECK_ERROR( rtContextDeclareVariable( *context, "V",&var_V ) );
    RT_CHECK_ERROR( rtContextDeclareVariable( *context, "W",&var_W ) );
}


void destroy_context()
{
    if( context ) {    
        RT_CHECK_ERROR( rtContextDestroy(context) );
        context = 0;
    }

    if( rdev ) {
        RT_CHECK_ERROR( rtRemoteDeviceRelease(rdev) );
        RT_CHECK_ERROR( rtRemoteDeviceDestroy(rdev) );
        rdev = 0;
    }
}


void create_material( RTcontext context, RTmaterial* material )
{
    RTvariable var_occlusion_distance;
    RTvariable var_sqrt_occlusion_samples;
    RTprogram  ch_radiance_program;
    RTprogram  ah_occlusion_program;

    /* Material. */
    RT_CHECK_ERROR( rtMaterialCreate(context,material) );
    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "ambocc.cu" );
    RT_CHECK_ERROR( rtProgramCreateFromPTXString( context, ptx, "closest_hit_radiance", &ch_radiance_program ) );
    RT_CHECK_ERROR( rtProgramCreateFromPTXString( context, ptx, "any_hit_occlusion", &ah_occlusion_program ) );
    RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( *material, 0, ch_radiance_program ) );
    RT_CHECK_ERROR( rtMaterialSetAnyHitProgram( *material, 1, ah_occlusion_program ) );

    /* Variables. */
    RT_CHECK_ERROR( rtMaterialDeclareVariable( *material, "occlusion_distance", &var_occlusion_distance ) );
    RT_CHECK_ERROR( rtVariableSet1f( var_occlusion_distance, 100.0f ) );
    RT_CHECK_ERROR( rtMaterialDeclareVariable( *material, "sqrt_occlusion_samples", &var_sqrt_occlusion_samples ) );
    RT_CHECK_ERROR( rtVariableSet1i( var_sqrt_occlusion_samples, sqrt_occlusion_samples ) );
}


void create_geometry( RTcontext context, RTmaterial material )
{
    RTgeometry sphere_geom;
    RTprogram  sphere_isect_program;
    RTprogram  sphere_bounds_program;
    RTvariable var_sphere;

    RTgeometry ground_geom;
    RTprogram  ground_isect_program;
    RTprogram  ground_bounds_program;
    RTvariable var_plane;
    RTvariable var_v1;
    RTvariable var_v2;
    RTvariable var_anchor;

    float v1[3], v2[3];
    float anchor[3];
    float normal[3];
    float inv_len2;
    float plane[4];

    RTgeometryinstance sphere_inst;
    RTgeometryinstance ground_inst;
    RTacceleration accel;
    RTgeometrygroup geometrygroup;
    RTvariable var_group;

    /* Sphere geometry. */
    RT_CHECK_ERROR( rtGeometryCreate( context,&sphere_geom ) );
    RT_CHECK_ERROR( rtGeometrySetPrimitiveCount( sphere_geom, 1 ) );
    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "sphere.cu" );
    RT_CHECK_ERROR( rtProgramCreateFromPTXString( context, ptx, "intersect", &sphere_isect_program ) );
    RT_CHECK_ERROR( rtProgramCreateFromPTXString( context, ptx, "bounds", &sphere_bounds_program) );
    RT_CHECK_ERROR( rtGeometrySetIntersectionProgram( sphere_geom, sphere_isect_program ) );
    RT_CHECK_ERROR( rtGeometrySetBoundingBoxProgram( sphere_geom, sphere_bounds_program ) );
    RT_CHECK_ERROR( rtGeometryDeclareVariable( sphere_geom, "sphere", &var_sphere ) );
    RT_CHECK_ERROR( rtVariableSet4f( var_sphere, 0.0f, 1.2f, 0.0f, 1.0f ) );

    /* Ground geometry. */
    RT_CHECK_ERROR( rtGeometryCreate( context, &ground_geom ) );
    RT_CHECK_ERROR( rtGeometrySetPrimitiveCount( ground_geom, 1 ) );
    ptx = sutil::getPtxString( SAMPLE_NAME, "parallelogram.cu" );
    RT_CHECK_ERROR( rtProgramCreateFromPTXString( context, ptx, "intersect", &ground_isect_program ) );
    RT_CHECK_ERROR( rtProgramCreateFromPTXString( context, ptx, "bounds", &ground_bounds_program ) );
    RT_CHECK_ERROR( rtGeometrySetIntersectionProgram( ground_geom,ground_isect_program ) );
    RT_CHECK_ERROR( rtGeometrySetBoundingBoxProgram( ground_geom,ground_bounds_program ) );
    RT_CHECK_ERROR( rtGeometryDeclareVariable( ground_geom, "plane", &var_plane ) );
    RT_CHECK_ERROR( rtGeometryDeclareVariable( ground_geom, "v1", &var_v1 ) );
    RT_CHECK_ERROR( rtGeometryDeclareVariable( ground_geom, "v2", &var_v2 ) );
    RT_CHECK_ERROR( rtGeometryDeclareVariable( ground_geom, "anchor", &var_anchor ) );

    anchor[0] = -30.0f;  anchor[1] =   0.0f;  anchor[2] =  20.0f;  
    v1[0]     =  40.0f;  v1[1]     =   0.0f;  v1[2]     =   0.0f;
    v2[0]     =   0.0f;  v2[1]     =   0.0f;  v2[2]     = -40.0f;
    cross( v1, v2, normal );
    normalize( normal );
    inv_len2 = 1.0f / dot( v1, v1 );
    v1[0] *= inv_len2; 
    v1[1] *= inv_len2; 
    v1[2] *= inv_len2; 

    inv_len2 = 1.0f / dot( v2, v2 );
    v2[0] *= inv_len2; 
    v2[1] *= inv_len2; 
    v2[2] *= inv_len2; 

    plane[0] = normal[0];
    plane[1] = normal[1];
    plane[2] = normal[2];
    plane[3] = dot( normal, anchor );

    RT_CHECK_ERROR( rtVariableSet4fv( var_plane, plane ) );
    RT_CHECK_ERROR( rtVariableSet3fv( var_v1, v1 ) );
    RT_CHECK_ERROR( rtVariableSet3fv( var_v2, v2 ) );
    RT_CHECK_ERROR( rtVariableSet3fv( var_anchor, anchor ) );

    /* Sphere instance. */
    RT_CHECK_ERROR( rtGeometryInstanceCreate( context, &sphere_inst ) );
    RT_CHECK_ERROR( rtGeometryInstanceSetGeometry( sphere_inst,sphere_geom ) );
    RT_CHECK_ERROR( rtGeometryInstanceSetMaterialCount( sphere_inst, 1 ) );
    RT_CHECK_ERROR( rtGeometryInstanceSetMaterial( sphere_inst, 0, material ) );

    /* Ground instance. */
    RT_CHECK_ERROR( rtGeometryInstanceCreate( context, &ground_inst ) );
    RT_CHECK_ERROR( rtGeometryInstanceSetGeometry( ground_inst, ground_geom ) );
    RT_CHECK_ERROR( rtGeometryInstanceSetMaterialCount( ground_inst, 1 ) );
    RT_CHECK_ERROR( rtGeometryInstanceSetMaterial( ground_inst, 0 ,material ) );

    /* Accelerations. */
    RT_CHECK_ERROR( rtAccelerationCreate( context, &accel ) );
    RT_CHECK_ERROR( rtAccelerationSetBuilder( accel, "NoAccel" ) );

    /* GeometryGroup. */
    RT_CHECK_ERROR( rtGeometryGroupCreate( context, &geometrygroup ) );
    RT_CHECK_ERROR( rtGeometryGroupSetChildCount( geometrygroup, 2 ) );
    RT_CHECK_ERROR( rtGeometryGroupSetChild( geometrygroup, 0, sphere_inst ) );
    RT_CHECK_ERROR( rtGeometryGroupSetChild( geometrygroup, 1, ground_inst ) );
    RT_CHECK_ERROR( rtGeometryGroupSetAcceleration( geometrygroup, accel ) );

    /* Attach to context */
    RT_CHECK_ERROR( rtContextDeclareVariable( context, "top_object", &var_group ) );
    RT_CHECK_ERROR( rtVariableSetObject( var_group, geometrygroup ) );
}


void cross( const float a[3], const float b[3], float result[3] )
{
    result[0] = a[1]*b[2] - a[2]*b[1];
    result[1] = a[2]*b[0] - a[0]*b[2];
    result[2] = a[0]*b[1] - a[1]*b[0];
}

float dot( const float a[3], const float b[3] )
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

void normalize( float a[3] )
{
    float inv_len = 1.0f / sqrtf( dot(a, a) );
    a[0] *= inv_len;
    a[1] *= inv_len;
    a[2] *= inv_len;
}

void display( RTbuffer buffer, RTformat buffer_format, RTsize element_size )
{

    GLenum gl_data_type;
    GLenum gl_format;

    switch (buffer_format)
    {
        case RT_FORMAT_UNSIGNED_BYTE4:
            gl_data_type = GL_UNSIGNED_BYTE;
            //gl_format    = 0x80E1; //GL_BGRA
            gl_format    = GL_RGBA;
            break;

        case RT_FORMAT_UNSIGNED_BYTE3:
            gl_data_type = GL_UNSIGNED_BYTE;
            //gl_format    = 0x80E0; //GL_BGR
            gl_format    = GL_RGB;
            break;

        case RT_FORMAT_FLOAT:
            gl_data_type = GL_FLOAT;
            gl_format    = GL_LUMINANCE;
            break;

        case RT_FORMAT_FLOAT3:
            gl_data_type = GL_FLOAT;
            gl_format    = GL_RGB;
            break;

        case RT_FORMAT_FLOAT4:
            gl_data_type = GL_FLOAT;
            gl_format    = GL_RGBA;
            break;

        default:
            fprintf(stderr, "Unrecognized buffer data type or format.\n");
            exit(2);
            break;
    }

    int align = 1;
    if      ((element_size % 8) == 0) align = 8; 
    else if ((element_size % 4) == 0) align = 4;
    else if ((element_size % 2) == 0) align = 2;
    glPixelStorei(GL_UNPACK_ALIGNMENT, align);

    GLvoid* imageData;
    RT_CHECK_ERROR( rtBufferMap(buffer,&imageData) );
    glDrawPixels( width, height, gl_format, gl_data_type, imageData );
    RT_CHECK_ERROR( rtBufferUnmap(buffer) );
}

//// GLUT callbacks

void glut_display(void)
{
    int ready;
    unsigned int subframe_count, max_subframes;
    RT_CHECK_ERROR( rtBufferGetProgressiveUpdateReady( stream_buffer, &ready, &subframe_count, &max_subframes ) );

    if( ready )
    {
        //printf("disp: subframe=%d max=%d\n", subframe_count, max_subframes );
        display( stream_buffer, RT_FORMAT_UNSIGNED_BYTE4, 4 );
        glutSwapBuffers();
    }

    // Calling rtContextLaunchProgressive while a launch is already in progress is a no-op,
    // so we can unconditionally call it here. In case the user interacted with the scene
    // and thus changed the state of the context (by calling rtVariableSet), this will restart
    // progressive rendering with the new context state.
    RT_CHECK_ERROR(rtContextLaunchProgressive2D(context, 0, width, height, 0));

}

void glut_mouse_button( int button, int state, int x, int y )
{
    if ((button == GLUT_LEFT_BUTTON) || (button == GLUT_RIGHT_BUTTON)) {
        if (state == GLUT_DOWN) {
            if (!mouse_zoom)
                yorg = y;
            mouse_zoom = true;
        } else {
            // GLUT_UP
            mouse_zoom = false;
            hfov_org = hfov;
        } 
    }
}

void glut_mouse_move( int x, int y)
{
    if (mouse_zoom) {
        hfov = (float)(hfov_org + (y-yorg));
        hfov = (hfov < 1.f) ? 1.f : hfov;
        hfov = (hfov > 150.f) ? 150.f : hfov; 
        set_cam_vars( eye, hfov ); 
    }
}

void glut_keyboard( unsigned char k, int x, int y )
{
    const char key = tolower(k);

    // Control gamma value with 'q' and 'a' keys
    if( key == 'a' || key == 'q' )
    {
        stream_gamma *= (key=='a' && stream_gamma > 0.5f) ? 0.8f : 1.0f;
        stream_gamma *= (key=='q' && stream_gamma < 5.0f) ? 1.0f/0.8f : 1.0f;
        RT_CHECK_ERROR( rtBufferSetAttribute(stream_buffer, RT_BUFFER_ATTRIBUTE_STREAM_GAMMA, sizeof(float), &stream_gamma) );
        printf("New stream gamma: %.2f\n", stream_gamma );
    }

    // Control bitrate with 'w' and 's' keys
    if( key == 's' || key == 'w' )
    {
        stream_bitrate -= (key=='s' && stream_bitrate >   200000) ? 200000 : 0;
        stream_bitrate += (key=='w' && stream_bitrate < 10000000) ? 200000 : 0;
        RT_CHECK_ERROR( rtBufferSetAttribute(stream_buffer, RT_BUFFER_ATTRIBUTE_STREAM_BITRATE, sizeof(int), &stream_bitrate) );
        printf("New stream bitrate: %d bps\n", stream_bitrate );
    }

    // Control fps with 'e' and 'd' keys
    if( key == 'd' || key == 'e' )
    {
        stream_fps -= (key=='d' && stream_fps > 5) ? 5 : 0;
        stream_fps += (key=='e' && stream_fps < 60) ? 5 : 0;
        RT_CHECK_ERROR( rtBufferSetAttribute(stream_buffer, RT_BUFFER_ATTRIBUTE_STREAM_FPS, sizeof(int), &stream_fps) );
        printf("New stream target fps: %d\n", stream_fps );
    }

    // Toggle format with 'f' key
    if( key == 'f' )
    {
        if( !strcmp(stream_format,"h264") )
            strcpy( stream_format, "png" );
        else
            strcpy( stream_format, "h264" );
        RT_CHECK_ERROR( rtBufferSetAttribute(stream_buffer, RT_BUFFER_ATTRIBUTE_STREAM_FORMAT, strlen(stream_format), stream_format) );
        printf("New stream format: %s\n", stream_format );
    }

    // Exit the application with ESC
    if ( k == 27 )
    {
        destroy_context();
        exit(0);
    }

}

// force it keep size.
void glut_resize( int w, int h )
{
    width = w;
    height = h;

    RT_CHECK_ERROR( rtBufferSetSize2D( output_buffer, width, height ) ); 
    RT_CHECK_ERROR( rtBufferSetSize2D( stream_buffer, width, height ) ); 

    set_cam_vars( eye, hfov );
}

void printUsage( int argc, char** argv )
{
    fprintf( stderr, "Usage: %s [options]\n", argv[0] );

    fprintf( stderr, "App Options:\n" \
                     "  -url <url>               Specify the VCA cluster manager WebSockets URL to connect to.\n" \
                     "                           URL format is wss://example.com:443, with wss:// and :443 required.\n" \

                     "                           If none is given, the application will run on the local machine.\n" \
                     "  -user <username>         The username for the account on the VCA cluster. Not needed when\n" \
                     "                           running locally.\n" \
                     "  -pass <password>         The password for the account on the VCA cluster. Not needed when\n" \
                     "                           running locally.\n" \
                     "  -nodes <n>               The number of nodes to reserve on the VCA cluster\n" \
                     "  -cfg <index>             The VCA configuration to use\n" \
                     "  -w <width>               Horizontal resolution in pixels\n" \
                     "  -h <height>              Vertical resolution in pixels\n" \
                     "  -s <samples>             Square root of the number of occlusion samples taken per pixel, per iteration\n" \
                     "\n" );

    fprintf( stderr, "App Keystrokes:\n" \
                     "  q/a  Increase/decrease gamma value\n" \
                     "  w/s  Increase/decrease stream bitrate (VCA only)\n" \
                     "  e/d  Increase/decrease stream max FPS (VCA only)\n" \
                     "  f    Toggle between different stream encoding formats (VCA only)\n" \
                     "\n" );
}

int main( int argc, char** argv )
{
    try
    {

        printUsage(argc,argv);

        for( int i=1; i<argc; ++i )
        {
            const std::string arg( argv[i] );

            if( arg == "-w" ) {
                if( i == argc-1 ) exit(1);
                width = atoi( argv[++i] );
            }
            if( arg == "-h" ) {
                if( i == argc-1 ) exit(1);
                height = atoi( argv[++i] );
            }
            if( arg == "-s" ) {
                if( i == argc-1 ) exit(1);
                sqrt_occlusion_samples = atoi( argv[++i] );
            }
            if( arg == "-url" ) {
                if( i == argc-1 ) exit(1);
                vca_url = argv[++i];
            }
            if( arg == "-user" ) {
                if( i == argc-1 ) exit(1);
                vca_user = argv[++i];
            }
            if( arg == "-pass" ) {
                if( i == argc-1 ) exit(1);
                vca_password = argv[++i];
            }
            if( arg == "-nodes" ) {
                if( i == argc-1 ) exit(1);
                vca_num_nodes = atoi( argv[++i] );
            }
            if( arg == "-cfg" ) {
                if( i == argc-1 ) exit(1);
                vca_config_index = atoi( argv[++i] );
            }
        }

        glutInit( &argc, argv );
        glutInitDisplayMode( GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE );
        glutInitWindowSize( width, height );
        glutCreateWindow( "optixProgressiveVCA" );

        create_remote_device( &rdev );
        create_context( &context, rdev, &output_buffer, &stream_buffer );
        create_material( context, &material );
        create_geometry( context, material );

        set_cam_vars( eye_org, hfov_org );

        RT_CHECK_ERROR( rtBufferGetAttribute(stream_buffer, RT_BUFFER_ATTRIBUTE_STREAM_BITRATE, sizeof(int), &stream_bitrate) );
        RT_CHECK_ERROR( rtBufferGetAttribute(stream_buffer, RT_BUFFER_ATTRIBUTE_STREAM_FPS, sizeof(int), &stream_fps) );
        RT_CHECK_ERROR( rtBufferGetAttribute(stream_buffer, RT_BUFFER_ATTRIBUTE_STREAM_GAMMA, sizeof(float), &stream_gamma) );
        RT_CHECK_ERROR( rtBufferGetAttribute(stream_buffer, RT_BUFFER_ATTRIBUTE_STREAM_FORMAT, sizeof(stream_format), stream_format) );
        printf( "Stream options: bitrate=%d, target fps=%d, gamma=%.2f, format=%s\n", stream_bitrate, stream_fps, stream_gamma, stream_format );

        RT_CHECK_ERROR( rtContextValidate(context) );
        RT_CHECK_ERROR( rtContextLaunchProgressive2D(context, 0, width, height, 0) );

        // register glut callbacks
        glutDisplayFunc( glut_display );
        glutIdleFunc( glut_display );
        glutReshapeFunc( glut_resize );
        glutKeyboardFunc( glut_keyboard );
        glutMouseFunc( glut_mouse_button );
        glutMotionFunc( glut_mouse_move );

        // register shutdown handler
#ifdef _WIN32
        glutCloseFunc( destroy_context );  // this function is freeglut-only
#else
        atexit( destroy_context );
#endif

        glutMainLoop();
        return 0;

    } SUTIL_CATCH( context )
}

