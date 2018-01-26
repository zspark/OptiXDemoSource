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

#include <optixu/optixpp_namespace.h>

#ifndef MDLWRAPPERAPI
#  if defined( _WIN32 )
#    define MDLWRAPPERAPI extern "C" __declspec(dllimport)
#  else
#    define MDLWRAPPERAPI extern "C"
#  endif
#endif


/// Opaque type representing the MDL wrapper context.
typedef void *Mdl_wrapper_context;

/// Constructor of an MDL wrapper context.
MDLWRAPPERAPI Mdl_wrapper_context mdl_wrapper_create(
    RTcontext optix_ctx,
    const char *mdl_textures_ptx_path,
    const char *module_path,
    unsigned num_texture_spaces );

/// Destructor shutting down Neuray.
MDLWRAPPERAPI void mdl_wrapper_destroy( Mdl_wrapper_context mdl_ctx );

/// Adds a path to search for MDL modules and resources.
MDLWRAPPERAPI void mdl_wrapper_add_module_path(
    Mdl_wrapper_context mdl_ctx,
    const char *module_path );

/// Sets the path to the PTX file of mdl_textures.cu.
MDLWRAPPERAPI void mdl_wrapper_set_mdl_textures_ptx_path(
    Mdl_wrapper_context mdl_ctx,
    const char *mdl_textures_ptx_path );

/// Sets the content of the PTX code of mdl_textures.cu.
MDLWRAPPERAPI void mdl_wrapper_set_mdl_textures_ptx_string(
    Mdl_wrapper_context mdl_ctx,
    const char *mdl_textures_ptx_string );

/// Compiles the MDL expression given by the module and material name and the expression path
/// to an OptiX program.
MDLWRAPPERAPI RTprogram mdl_wrapper_compile_expression(
    Mdl_wrapper_context mdl_ctx,
    const char *module_name,
    const char *material_name,
    const char *expression_path );

/// Compiles the MDL expression given by the module and material name and the function name
/// as an environment to an OptiX program.
MDLWRAPPERAPI RTprogram mdl_wrapper_compile_environment(
    Mdl_wrapper_context mdl_ctx,
    const char *module_name,
    const char *function_name );

/// Clears the texture cache which holds references to the used OptiX buffers.
MDLWRAPPERAPI void mdl_wrapper_clear_texture_cache( Mdl_wrapper_context mdl_ctx );


/// A helper class for handling MDL in OptiX without requiring access to the MDL SDK header files.
class Mdl_wrapper
{
public:
    /// Constructs an Mdl_wrapper object.
    /// \param optix_ctx             the given OptiX context,
    /// \param mdl_textures_ptx_path the path to the PTX file of mdl_textures.cu.
    ///                              If you don't need texture support, this can be empty.
    ///                              It can also be set later or you can later set the PTX code
    ///                              directly (e.g. as obtained by the CUDA runtime compiler)
    /// \param module_path           an optional search path for MDL modules and resources.
    /// \param num_texture_spaces    the number of texture spaces provided in the MDL_SDK_State
    ///                              fields text_coords, tangent_t and tangent_v by the renderer.
    ///                              If invalid texture spaces are requested in the MDL materials,
    ///                              null values will be returned.
    Mdl_wrapper(
        optix::Context optix_ctx,
        const std::string &mdl_textures_ptx_path = std::string(),
        const std::string &module_path = std::string(),
        unsigned num_texture_spaces = 1 )
      : m_optix_ctx( optix_ctx )
    {
        m_wrapper_ctx = mdl_wrapper_create(
            optix_ctx->get(),
            mdl_textures_ptx_path.c_str(),
            module_path.c_str(),
            num_texture_spaces );
    }

    /// Destructor shutting down Neuray.
    ~Mdl_wrapper()
    {
        mdl_wrapper_destroy( m_wrapper_ctx );
    }

    /// Adds a path to search for MDL modules and resources.
    void add_module_path( const std::string &module_path )
    {
        mdl_wrapper_add_module_path( m_wrapper_ctx, module_path.c_str() );
    }

    /// Sets the path to the PTX file of mdl_textures.cu.
    void set_mdl_textures_ptx_path( const std::string &mdl_textures_ptx_path )
    {
        mdl_wrapper_set_mdl_textures_ptx_path( m_wrapper_ctx, mdl_textures_ptx_path.c_str() );
    }

    /// Sets the content of the PTX code of mdl_textures.cu.
    void set_mdl_textures_ptx_string( const std::string &mdl_textures_ptx_string )
    {
        mdl_wrapper_set_mdl_textures_ptx_string( m_wrapper_ctx, mdl_textures_ptx_string.c_str() );
    }

    /// Compiles the MDL expression given by the module and material name and the expression path
    /// to an OptiX program.
    optix::Program compile_expression(
        const std::string &module_name,
        const std::string &material_name,
        const std::string &expression_path )
    {
        return optix::Program::take(
            mdl_wrapper_compile_expression(
                m_wrapper_ctx,
                module_name.c_str(),
                material_name.c_str(),
                expression_path.c_str() ) );
    }

    /// Compiles the MDL expression given by the module and material name and the function name
    /// as an environment to an OptiX program.
    optix::Program compile_environment(
        const std::string &module_name,
        const std::string &function_name )
    {
        return optix::Program::take(
            mdl_wrapper_compile_environment(
                m_wrapper_ctx,
                module_name.c_str(),
                function_name.c_str() ) );
    }

    /// Clears the texture cache which holds references to the used OptiX buffers.
    void clear_texture_cache()
    {
        mdl_wrapper_clear_texture_cache( m_wrapper_ctx );
    }

private:
    /// OptiX context held to ensure proper reference counting.
    optix::Context m_optix_ctx;

    /// The MDL wrapper context.
    Mdl_wrapper_context m_wrapper_ctx;
};
