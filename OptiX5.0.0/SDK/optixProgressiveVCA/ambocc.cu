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

#include <optix_world.h>
#include "random.h"
#include "commonStructs.h"

using namespace optix;

rtDeclareVariable(float,       scene_epsilon, , );
rtDeclareVariable(float,       occlusion_distance, , );
rtDeclareVariable(int,         sqrt_occlusion_samples, , );
rtDeclareVariable(rtObject,    top_object, , );

rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 

rtDeclareVariable(optix::Ray,   ray,          rtCurrentRay, );
rtDeclareVariable(float,        t_hit,        rtIntersectionDistance, );
rtDeclareVariable(uint2,        launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2,        launch_dim,   rtLaunchDim, );
rtDeclareVariable(unsigned int, frame, , );

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
  float3 phit = ray.origin + t_hit * ray.direction;

  float3 world_shading_normal   = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
  float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
  float3 ffnormal = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);

  optix::Onb onb(ffnormal);

  unsigned int seed = tea<4>( launch_dim.x*launch_index.y+launch_index.x, frame + subframe_idx );

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



