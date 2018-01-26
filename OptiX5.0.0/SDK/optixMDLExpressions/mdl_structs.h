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

#pragma once

#include <optixu/optixu_vector_types.h>


/// The state of the MDL environment function.
struct MDL_Environment_state {
    float3                direction;               ///< state::direction() result
};

/// The MDL material state inside the MDL SDK.
struct MDL_SDK_State
{
    float3                normal;                  ///< state::normal() result
    float3                geom_normal;             ///< state::geom_normal() result
    float3                position;                ///< state::position() result
    float                 animation_time;          ///< state::animation_time() result
    const float3         *text_coords;             ///< state::texture_coordinate() table
    const float3         *tangent_u;               ///< state::texture_tangent_u() table
    const float3         *tangent_v;               ///< state::texture_tangent_v() table
    const float4         *text_results;            ///< texture results lookup table
    const unsigned char  *ro_data_segment;         ///< read only data segment

    // these fields are used only if the uniform state is included
    const float4         *world_to_object;         ///< world-to-object transform matrix
    const float4         *object_to_world;         ///< object-to-world transform matrix
    int                   object_id;               ///< state::object_id() result
};

/// The resource data structure required by the MDL SDK.
struct MDL_SDK_Res_data_pair
{
    void const *shared_data;  ///< currently unused, should be NULL
    void const *thread_data;  ///< will be provided as "self" parameter to texture functions
};
