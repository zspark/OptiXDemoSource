
#include "Common.h"
#include "OptiXRaycastingContext.h"

#include <sutil.h>

#include <iostream>
#include <string.h>  // memcpy
#include <vector>


std::string ptxPath()
{
    return
        std::string(sutil::samplesPTXDir()) +
        "/optixRaycasting_generated_OptiXRaycastingContext.cu.ptx";
}

OptiXRaycastingContext::OptiXRaycastingContext()
{
  m_context = optix::Context::create();
  m_context->setRayTypeCount( 1 );
  m_context->setEntryPointCount( 1 );

  // Set small stack for a simple kernel without much shading.
  m_context->setStackSize( 200 );

  // Limit to single GPU for this sample, to simplify CUDA interop.
  const std::vector<int> enabled_devices = m_context->getEnabledDevices();
  m_context->setDevices( enabled_devices.begin(), enabled_devices.begin()+1 );
  m_optix_device_ordinal = enabled_devices[0];
  {
    m_cuda_device_ordinal = -1;
    m_context->getDeviceAttribute( m_optix_device_ordinal, RT_DEVICE_ATTRIBUTE_CUDA_DEVICE_ORDINAL, sizeof(int), &m_cuda_device_ordinal );
  }

  // Minimal OptiX scene hierarchy.

  m_geometry = m_context->createGeometry();
  const std::string ptx_path = ptxPath();
  m_geometry->setIntersectionProgram( m_context->createProgramFromPTXFile( ptx_path, "intersect" ) );

  m_material = m_context->createMaterial();
  optix::GeometryInstance geometry_instance = m_context->createGeometryInstance( m_geometry, &m_material, &m_material+1 );
  optix::GeometryGroup geometry_group = m_context->createGeometryGroup( &geometry_instance, &geometry_instance+1 );
  geometry_group->setAcceleration( m_context->createAcceleration( "Trbvh" ) );
  m_context[ "top_object" ]->set( geometry_group );

  // Closest hit program for returning geometry attributes.  No shading.
  optix::Program closest_hit = m_context->createProgramFromPTXFile( ptx_path, "closest_hit" );
  m_material->setClosestHitProgram( /*ray type*/ 0, closest_hit );

  // Raygen program that reads rays directly from an input buffer.
  optix::Program ray_gen = m_context->createProgramFromPTXFile( ptx_path, "ray_gen" );
  m_context->setRayGenerationProgram( /*entry point*/ 0, ray_gen );

  // Exception program for debugging
  /*
  optix::Program exception_program = m_context->createProgramFromPTXFile( ptx_path, "exception" );
  m_context->setExceptionProgram( 0, exception_program );
  m_context->setPrintEnabled( true );
  */

}

OptiXRaycastingContext::~OptiXRaycastingContext()
{
  if ( m_context ) m_context->destroy();
}

void OptiXRaycastingContext::setTriangles( int num_triangles, int32_t* indices, int num_vertices, float* positions, float* texcoords )
{
  if ( m_positions ) m_positions->destroy();
  if ( m_indices ) m_indices->destroy();
  if ( m_texcoords ) m_texcoords->destroy();

  m_positions = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, num_vertices );
  memcpy( m_positions->map(), positions, num_vertices*3*sizeof(float) );
  m_positions->unmap();

  m_indices  = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT3, num_triangles );
  memcpy( m_indices->map(), indices, num_triangles*3*sizeof(int32_t) );
  m_indices->unmap();

  // Connect required buffers to intersection program
  m_geometry->setPrimitiveCount( num_triangles );
  m_geometry[ "vertex_buffer" ]->set( m_positions );
  m_geometry[ "index_buffer"  ]->set( m_indices  );

  // Connect texcoord buffer, used for masking
  const int num_texcoords = texcoords ? num_vertices : 0;
  m_texcoords = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, num_texcoords );
  m_geometry[ "texcoord_buffer"       ]->set( m_texcoords );

  if ( texcoords ) {
    memcpy( m_texcoords->map(), texcoords, num_texcoords*2*sizeof(float) );
    m_texcoords->unmap();
  }

}

void OptiXRaycastingContext::setMask( const char* texture_filename )
{
  // Any-hit program for masking
  const std::string ptx_path = ptxPath();
  optix::Program any_hit = m_context->createProgramFromPTXFile( ptx_path, "any_hit" );
  m_material->setAnyHitProgram( /*ray type*/ 0, any_hit );

  optix::TextureSampler sampler = sutil::loadTexture( m_context, texture_filename, optix::make_float3(1.0f, 1.0f, 1.0f) );
  m_context[ "mask_sampler" ]->set( sampler );
}

void OptiXRaycastingContext::setRaysDevicePointer( const Ray* rays, size_t n )
{
  if ( m_rays ) m_rays->destroy();
  m_rays = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_USER, n );
  m_rays->setElementSize( sizeof(Ray) );
  m_rays->setDevicePointer( m_optix_device_ordinal, const_cast<Ray*>(rays) );
  m_context["rays"]->set( m_rays );
}

void OptiXRaycastingContext::setHitsDevicePointer( Hit* hits, size_t n )
{
  if ( m_hits ) m_hits->destroy();
  m_hits = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_USER, n );
  m_hits->setElementSize( sizeof(Hit) );
  m_hits->setDevicePointer( m_optix_device_ordinal, hits );
  m_context["hits"]->set( m_hits );
}

void OptiXRaycastingContext::execute()
{
  RTsize n;
  m_rays->getSize(n);
  m_context->launch( /*entry point*/ 0, n );
}

