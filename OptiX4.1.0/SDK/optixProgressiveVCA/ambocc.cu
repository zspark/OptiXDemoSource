/* 
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
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

#include <optix_world.h>
#include "random.h"
#include "commonStructs.h"

using namespace optix;

rtDeclareVariable(float,       scene_epsilon, , );
rtDeclareVariable(float,       occlusion_distance, , );
rtDeclareVariable(int,         sqrt_occlusion_samples, , );
rtDeclareVariable(rtObject,    top_object, , );
rtBuffer<unsigned int, 2>      rnd_seeds;

rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 

rtDeclareVariable(optix::Ray, ray,          rtCurrentRay, );
rtDeclareVariable(float,      t_hit,        rtIntersectionDistance, );
rtDeclareVariable(uint2,      launch_index, rtLaunchIndex, );
rtDeclareVariable(int,        frame, , );

rtDeclareVariable(unsigned int, subframe_idx, rtSubframeIndex, );

struct PerRayData_radiance
{
  float3 result;
  float importance;
  int depth;
};

struct PerRayData_occlusion
{
  float occlusion;
};

rtDeclareVariable(PerRayData_radiance,  prd_radiance,  rtPayload, );
rtDeclareVariable(PerRayData_occlusion, prd_occlusion, rtPayload, );

RT_PROGRAM void closest_hit_radiance()
{
  float3 phit    = ray.origin + t_hit * ray.direction;

  float3 world_shading_normal   = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
  float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
  float3 ffnormal = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);

  optix::Onb onb(ffnormal);

  unsigned int seed = rot_seed(rnd_seeds[launch_index], frame + subframe_idx);

  float       result           = 0.0f;
  const float inv_sqrt_samples = 1.0f / float(sqrt_occlusion_samples);
  for( int i=0; i<sqrt_occlusion_samples; ++i ) {
    for( int j=0; j<sqrt_occlusion_samples; ++j ) {

      PerRayData_occlusion prd_occ;
      prd_occ.occlusion = 0.0f;

      // Stratify samples via simple jitterring
      float u1 = (float(i) + rnd( seed ) )*inv_sqrt_samples;
      float u2 = (float(j) + rnd( seed ) )*inv_sqrt_samples;

      float3 dir;
      optix::cosine_sample_hemisphere( u1, u2, dir );
      onb.inverse_transform( dir );

      optix::Ray occlusion_ray = optix::make_Ray( phit, dir, 1, scene_epsilon,
                                                  occlusion_distance );
      rtTrace( top_object, occlusion_ray, prd_occ );

      result += 1.0f-prd_occ.occlusion;
    }
  }

  result /= (float)(sqrt_occlusion_samples*sqrt_occlusion_samples);


  prd_radiance.result = make_float3(result);
}

RT_PROGRAM void any_hit_occlusion()
{
  prd_occlusion.occlusion = 1.0f;

  rtTerminateRay();
}


// Phong and AO --------------------------------------------

// TODO: This was copied from phong.h.
// Phong.h can't be included directly from ambocc.cu and other samples because
// it declares the variables rtCurrentRay and rtIntersectionDistance.
// Possible solutions:
// remove those declarations from phong.h; or
// remove them from the samples; or
// put them in a common include.

rtDeclareVariable(float, Kd, ,);
rtDeclareVariable(float, Ka, ,);
rtDeclareVariable(float, Ks, ,);
rtDeclareVariable(float, Kr, ,);
rtDeclareVariable(float, phong_exp, ,);
rtDeclareVariable(int,               max_depth, , );
rtBuffer<BasicLight>                 lights;
rtDeclareVariable(float3,            ambient_light_color, , );
rtDeclareVariable(unsigned int,      radiance_ray_type, , );
rtDeclareVariable(unsigned int,      shadow_ray_type, , );
struct PerRayData_shadow
{
  float3 attenuation;
};
rtDeclareVariable(PerRayData_shadow,   prd_shadow, rtPayload, );
rtDeclareVariable(PerRayData_radiance, prd, rtPayload, );


static __device__ void phongShadowed()
{
  // this material is opaque, so it fully attenuates all shadow rays
  prd_shadow.attenuation = optix::make_float3(0.0f);
  rtTerminateRay();
}

static
__device__ void phongShade( float3 p_Kd,
                            float3 p_Ka,
                            float3 p_Ks,
                            float3 p_normal,
                            float  p_phong_exp,
                            float3 p_reflectivity )
{
  float3 hit_point = ray.origin + t_hit * ray.direction;

  // ambient contribution

  float3 result = p_Ka * ambient_light_color;

  // compute direct lighting
  unsigned int num_lights = lights.size();
  for(int i = 0; i < num_lights; ++i) {
    BasicLight light = lights[i];
    float Ldist = optix::length(light.pos - hit_point);
    float3 L = optix::normalize(light.pos - hit_point);
    float nDl = optix::dot( p_normal, L);

    // cast shadow ray
    float3 light_attenuation = make_float3(static_cast<float>( nDl > 0.0f ));
    if ( nDl > 0.0f && light.casts_shadow ) {
      PerRayData_shadow shadow_prd;
      shadow_prd.attenuation = make_float3(1.0f);
      optix::Ray shadow_ray = optix::make_Ray( hit_point, L, shadow_ray_type, scene_epsilon, Ldist );
      rtTrace(top_object, shadow_ray, shadow_prd);
      light_attenuation = shadow_prd.attenuation;
    }

    // If not completely shadowed, light the hit point
    if( fmaxf(light_attenuation) > 0.0f ) {
      float3 Lc = light.color * light_attenuation;

      result += p_Kd * nDl * Lc;

      float3 H = optix::normalize(L - ray.direction);
      float nDh = optix::dot( p_normal, H );
      if(nDh > 0) {
        float power = pow(nDh, p_phong_exp);
        result += p_Ks * power * Lc;
      }
    }
  }

  if( fmaxf( p_reflectivity ) > 0 ) {

    // ray tree attenuation
    PerRayData_radiance new_prd;
    new_prd.importance = prd.importance * optix::luminance( p_reflectivity );
    new_prd.depth = prd.depth + 1;

    // reflection ray
    if( new_prd.importance >= 0.01f && new_prd.depth <= max_depth) {
      float3 R = optix::reflect( ray.direction, p_normal );
      optix::Ray refl_ray = optix::make_Ray( hit_point, R, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX );
      rtTrace(top_object, refl_ray, new_prd);
      result += p_reflectivity * new_prd.result;
    }
  }

  // pass the color back up the tree
  prd.result = result;
}



RT_PROGRAM void closest_hit_radiance_phong_ao()
{
  float3 phit    = ray.origin + t_hit * ray.direction;

  float3 world_shading_normal   = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
  float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
  float3 ffnormal = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);

  optix::Onb onb(ffnormal);

  unsigned int seed = rot_seed( rnd_seeds[ launch_index ], frame );

  float       result           = 0.0f;
  const float inv_sqrt_samples = 1.0f / float(sqrt_occlusion_samples);
  for( int i=0; i<sqrt_occlusion_samples; ++i ) {
    for( int j=0; j<sqrt_occlusion_samples; ++j ) {

      PerRayData_occlusion prd_occ;
      prd_occ.occlusion = 0.0f;

      // Stratify samples via simple jitterring
      float u1 = (float(i) + rnd( seed ) )*inv_sqrt_samples;
      float u2 = (float(j) + rnd( seed ) )*inv_sqrt_samples;

      float3 dir;
      optix::cosine_sample_hemisphere( u1, u2, dir );
      onb.inverse_transform( dir );

      optix::Ray occlusion_ray = optix::make_Ray( phit, dir, 2, scene_epsilon,
                                                  occlusion_distance );
      rtTrace( top_object, occlusion_ray, prd_occ );

      result += 1.0f-prd_occ.occlusion;
    }
  }

  result /= (float)(sqrt_occlusion_samples*sqrt_occlusion_samples);

  // Phong
  phongShade( make_float3(Kd), make_float3(Ka), make_float3(Ks), ffnormal, phong_exp, make_float3(Kr) );

  // Phong result in prd.result

  prd_radiance.result = make_float3(result);

  //prd_radiance.result = make_float3(1.0f);

  //prd_radiance.result = 0.9f * prd_radiance.result + 0.1f * prd.result;

  prd_radiance.result.x *= prd.result.x;
  prd_radiance.result.y *= prd.result.y;
  prd_radiance.result.z *= prd.result.z;
}

RT_PROGRAM void any_hit_shadow()
{
  phongShadowed();
}

// ----------------------------------------------------------


