#
# Copyright (c) 2017 NVIDIA CORPORATION.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

# Building an own version of the mdl_wrapper requires the MDL SDK to be
# unpacked into "SDK/support/mdl-sdk" and reconfigure the project with CMake.
# With the MDL SDK you can also directly use the mdl_helper instead of
# the mdl_wrapper, making it possible to provide material parameters
# and to inspect the compiled materials.
#
# Access to the MDL SDK can be requested at:
# https://developer.nvidia.com/mdl-sdk


set(sources
  "${CMAKE_CURRENT_SOURCE_DIR}/mdl_helper.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/mdl_helper.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/mdl_wrapper.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/mdl_wrapper.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/neuray_loader.h"
  )

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  "${CMAKE_SOURCE_DIR}/support/mdl-sdk/include")


# Make the library.
set(mdl_wrapper_target "mdl_wrapper")
add_library(${mdl_wrapper_target} ${sources})

set(mdl_wrapper_FOUND TRUE)
set(mdl_wrapper_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Let the top level cmake know about that the mdl_wrapper is available
set(mdl_wrapper_FOUND ${mdl_wrapper_FOUND} PARENT_SCOPE)
set(mdl_wrapper_INCLUDE_DIR ${mdl_wrapper_INCLUDE_DIR} PARENT_SCOPE)

if(APPLE AND (RELEASE_PUBLIC OR RELEASE_PUBLIC_DEVELOPER))
  # If performing a release install, we want to use rpath for our install name.
  # The executables' rpaths will then be set to @executable_path so we can invoke
  # the samples from an arbitrary location and it will still find this library.
  set_target_properties(${mdl_wrapper_target} PROPERTIES
    INSTALL_NAME_DIR "@rpath"
    BUILD_WITH_INSTALL_RPATH ON
    )
endif()

# Use gcc rather than g++ to link if we are linking statically against libgcc_s and libstdc++
if(USING_GNU_C OR USING_GNU_CXX)
  if(GCC_LIBSTDCPP_HACK)
    set_target_properties(${mdl_wrapper_target} PROPERTIES LINKER_LANGUAGE "C")
    target_link_libraries(${mdl_wrapper_target} LINK_PRIVATE ${STATIC_LIBSTDCPP})
  endif()
endif()

target_link_libraries(${mdl_wrapper_target}
  optix
  optixu
  )

if(UNIX AND (NOT APPLE))
  target_link_libraries(${mdl_wrapper_target}
    dl

    # Force using RPATH instead of RUNPATH on Debian, as otherwise the RPATH of
    # the executable will not be used inside the wrapper library
    # (also see https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=859732)
    "-Wl,--disable-new-dtags"
    )
endif()

set_property(TARGET ${mdl_wrapper_target} PROPERTY FOLDER "${OPTIX_IDE_FOLDER}")


install(TARGETS ${mdl_wrapper_target} DESTINATION SDK/support/mdl_wrapper/lib)
install(FILES "mdl_wrapper.h" DESTINATION SDK/support/mdl_wrapper/include)

# Select MDL SDK files required by mdl_wrapper.
# Note: nv_freeimage.dll/so will already be copied from dice
if( WIN32 )
  set(MDL_DIR ${CMAKE_SOURCE_DIR}/support/mdl-sdk/nt-x86-64/lib)
  list(APPEND MDL_FILES ${MDL_DIR}/dds.dll)
  list(APPEND MDL_FILES ${MDL_DIR}/libmdl_sdk.dll)
elseif( APPLE )
  set(MDL_DIR ${CMAKE_SOURCE_DIR}/support/mdl-sdk/macosx-x86-64/lib)
  list(APPEND MDL_FILES ${MDL_DIR}/dds.so)
  list(APPEND MDL_FILES ${MDL_DIR}/libmdl_sdk.so)
elseif( UNIX )
  set(MDL_DIR ${CMAKE_SOURCE_DIR}/support/mdl-sdk/linux-x86-64/lib)
  list(APPEND MDL_FILES ${MDL_DIR}/dds.so)
  list(APPEND MDL_FILES ${MDL_DIR}/libmdl_sdk.so)
else()
  message( FATAL_ERROR "Unknown platform, no MDL SDK libraries available" )
endif()

# Copy the MDL SDK to runtime/library directories depending on platform,
# so we can run SDK samples from the build dir
foreach(config ${CMAKE_CONFIGURATION_TYPES} ${CMAKE_BUILD_TYPE})
  foreach(file ${MDL_FILES})
    get_filename_component(file_without_path "${file}" NAME)
    if( WIN32 )
      configure_file("${file}" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${config}/${file_without_path}" COPYONLY)
    else()
      configure_file("${file}" "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${file_without_path}" COPYONLY)
    endif()
  endforeach()
endforeach()

if(RELEASE_INSTALL_BINARY_SAMPLES AND NOT RELEASE_STATIC_BUILD)
  install(TARGETS ${mdl_wrapper_target}
    RUNTIME DESTINATION ${SDK_BINARY_INSTALL_DIR}
    LIBRARY DESTINATION ${SDK_BINARY_INSTALL_DIR}
    )

  install(FILES ${MDL_FILES}
    DESTINATION ${SDK_BINARY_INSTALL_DIR}
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    )
endif()
