# ----------------------------------------------------------------------------
# CMakeLists.txt
#
#  Created on: 22 Jan 2021
#      Author: Kiwon Um
#        Mail: kiwon.um@telecom-paris.fr
#
# Description: CMake configuration
#
# Copyright 2021-2024 Kiwon Um
#
# The copyright to the computer program(s) herein is the property of Kiwon Um,
# Telecom Paris, France. The program(s) may be used and/or copied only with
# the written permission of Kiwon Um or in accordance with the terms and
# conditions stipulated in the agreement/contract under which the program(s)
# have been supplied.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.0)

SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

project(viscoElasticFluidSimulation)
# Ajouter les sources d'ImGui
file(GLOB IMGUI_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/dep/imgui/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dep/imgui/backends/imgui_impl_glfw.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dep/imgui/backends/imgui_impl_opengl3.cpp
)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dep/imgui)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dep/imgui/backends)

add_executable(
  ${PROJECT_NAME}
  src/main.cpp
    src/Vector.hpp
 ${IMGUI_SOURCES} # Ajouter les sources d'ImGui ici
)

target_sources(${PROJECT_NAME} PRIVATE dep/glad/src/glad.c)
target_include_directories(${PROJECT_NAME} PRIVATE dep/glad/include/)

add_subdirectory(dep/glfw)
target_link_libraries(${PROJECT_NAME} PRIVATE glfw)

# Liens OpenGL
find_package(OpenGL REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenGL::GL)


 #Ajouter OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE ${CMAKE_DL_LIBS})

add_custom_command(TARGET ${PROJECT_NAME}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${PROJECT_NAME}> ${CMAKE_CURRENT_SOURCE_DIR})

