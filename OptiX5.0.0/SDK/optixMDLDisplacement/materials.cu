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
#include "commonStructs.h"
#include "helpers.h"
#include "mdl_structs.h"

using namespace optix;


struct PerRayData_radiance
{
  float3 result;
  float importance;
  int depth;
};


rtBuffer<BasicLight> lights;

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_radiance, prd, rtPayload, );


rtDeclareVariable(float3, texcoord, attribute texcoord, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );

rtDeclareVariable(
  rtCallableProgramId<void(float3 *, MDL_SDK_State *, MDL_SDK_Res_data_pair *, void *)>,
  mdl_expr, , );
rtDeclareVariable(
  rtCallableProgramId<void(float3 *, MDL_Environment_state *, MDL_SDK_Res_data_pair *, void *)>,
  mdl_env_expr, , );


RT_PROGRAM void mdl_material_apply()
{
  float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
  float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));

  //
  // Initialize state for MDL
  //

  float world_to_object[16];
  float object_to_world[16];
  rtGetTransform(RT_WORLD_TO_OBJECT, world_to_object);
  rtGetTransform(RT_OBJECT_TO_WORLD, object_to_world);

  float3 hit_point = ray.origin + t_hit * ray.direction;

  float3 text_coords = texcoord;      // taking an address of an attribute is not supported
  float3 tangent_u = make_float3(0);
  float3 tangent_v = make_float3(0);

  MDL_SDK_State state = {
    world_shading_normal,
    world_geometric_normal,
    hit_point,
    0,                 // animation_time
    &text_coords,      // *text_coords;
    &tangent_u,        // *tangent_u;
    &tangent_v,        // *tangent_v;
    NULL,              // *text_results;
    NULL,              // *ro_data_segment;
    (float4 *) world_to_object,
    (float4 *) object_to_world,
    0                  // object_id
  };

  MDL_SDK_Res_data_pair res_data_pair = {
    NULL,
    NULL
  };

  //
  // Calculate tint
  //

  mdl_expr(&prd.result, &state, &res_data_pair, NULL);

  //
  // Calculate attenuation
  //

  float3 ffnormal = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);
  float3 attenuation = make_float3(0);

  unsigned int num_lights = lights.size();
  for ( int i = 0; i < num_lights; ++i ) {
    BasicLight light = lights[i];
    float3 L = optix::normalize(light.pos - hit_point);
    attenuation += max(0.f, optix::dot(ffnormal, L)) * light.color;
  }

  prd.result *= attenuation;
}

RT_PROGRAM void mdl_environment_apply()
{
  MDL_Environment_state state = {
    optix::normalize(ray.direction)
  };

  MDL_SDK_Res_data_pair res_data_pair = {
    NULL,
    NULL
  };

  mdl_env_expr(&prd.result, &state, &res_data_pair, NULL);
}
