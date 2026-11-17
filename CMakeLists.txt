cmake_minimum_required(VERSION 3.18)
project(TensorConsole VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- options (CI / release friendly) ---------------------------------------
option(TC_STATIC_CRT "Statically link the MSVC runtime (portable release exe)" ON)
option(TC_INSTALL "Generate install rules and CPack release archives" ON)
set(SERVIC_ROOT "" CACHE PATH "Local clone of yauntyour/servic.cpp (empty = FetchContent)")
set(SERVIC_GIT_TAG "main" CACHE STRING "servic.cpp git ref used by FetchContent")

if(MSVC AND TC_STATIC_CRT)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# ---- servic.cpp (header-only, Boost.Asio) ----------------------------------
# Resolution order: -DSERVIC_ROOT=... > $SERVIC_ROOT > sibling clone > FetchContent.
# CI pins the ref with -DSERVIC_GIT_TAG=<sha|tag> or pre-seeds the source via
# -DFETCHCONTENT_SOURCE_DIR_SERVIC=<dir> for fully offline builds.
if(NOT SERVIC_ROOT AND DEFINED ENV{SERVIC_ROOT})
  set(SERVIC_ROOT "$ENV{SERVIC_ROOT}")
endif()
if(NOT SERVIC_ROOT AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../servic.cpp/servic.hpp")
  set(SERVIC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../servic.cpp")
endif()
if(SERVIC_ROOT AND EXISTS "${SERVIC_ROOT}/servic.hpp")
  set(SERVIC_INCLUDE_DIR "${SERVIC_ROOT}")
  message(STATUS "servic.cpp: local clone at ${SERVIC_ROOT}")
else()
  include(FetchContent)
  FetchContent_Declare(servic
    GIT_REPOSITORY https://github.com/yauntyour/servic.cpp.git
    GIT_TAG ${SERVIC_GIT_TAG}
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE)
  FetchContent_MakeAvailable(servic)
  set(SERVIC_INCLUDE_DIR "${servic_SOURCE_DIR}")
  message(STATUS "servic.cpp: FetchContent ref ${SERVIC_GIT_TAG}")
endif()

# ---- Boost (headers only: Asio) --------------------------------------------
# ubuntu: apt libboost-dev | macos: brew boost | windows: vcpkg boost-asio
# or point -DBOOST_ROOT=<dir> at any header tree.
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()
find_package(Boost QUIET)
if(NOT Boost_FOUND)
  message(FATAL_ERROR
    "Boost headers (Asio) not found. Install libboost-dev / brew boost / "
    "vcpkg boost-asio, or configure with -DBOOST_ROOT=<dir>.")
endif()
find_package(Threads REQUIRED)

# ---- shared dependency interface -------------------------------------------
add_library(tc_deps INTERFACE)
target_include_directories(tc_deps INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party
  ${SERVIC_INCLUDE_DIR})
if(TARGET Boost::headers)
  target_link_libraries(tc_deps INTERFACE Boost::headers)
else()
  target_include_directories(tc_deps INTERFACE ${Boost_INCLUDE_DIRS})
endif()
target_link_libraries(tc_deps INTERFACE Threads::Threads)
if(WIN32)
  target_link_libraries(tc_deps INTERFACE ws2_32 mswsock)
  target_compile_definitions(tc_deps INTERFACE _WIN32_WINNT=0x0A00 NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
if(MSVC)
  target_compile_options(tc_deps INTERFACE /utf-8 /bigobj)
else()
  target_compile_options(tc_deps INTERFACE -Wall -Wno-unused-parameter)
endif()

add_library(tc_core STATIC
  src/common.cpp
  src/events.cpp
  src/store.cpp
  src/process.cpp
  src/adapter.cpp
  src/profiler.cpp
  src/monitor.cpp
  src/jobs.cpp
  src/api.cpp
)
target_link_libraries(tc_core PUBLIC tc_deps)

add_executable(tcd src/main.cpp)
target_link_libraries(tcd PRIVATE tc_core)

add_executable(tc cli/tc.cpp)
target_link_libraries(tc PRIVATE tc_deps)

# ---- install / release packaging -------------------------------------------
# Archive layout is runnable as-is: tcd/tc at the root next to web/, so the
# default --data ./data --web ./web resolve without extra flags.
if(TC_INSTALL)
  install(TARGETS tcd tc RUNTIME DESTINATION .)
  install(DIRECTORY web/ DESTINATION web)
  install(DIRECTORY sdk/ DESTINATION sdk)
  install(FILES README.md DESTINATION .)
  install(FILES docs/DESIGN.md DESTINATION docs)

  set(CPACK_PACKAGE_NAME "tensorconsole")
  set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
  set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
  set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
  set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Train/Infer visualization console for TensorN C++ backends")
  set(CPACK_PACKAGE_VENDOR "yauntyour")
  set(CPACK_VERBATIM_VARIABLES YES)
  if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(_tc_arch "x86_64")
    set(CPACK_GENERATOR "ZIP")
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(_tc_arch "${CMAKE_SYSTEM_PROCESSOR}")
    set(CPACK_GENERATOR "TGZ")
  else()
    set(_tc_arch "${CMAKE_SYSTEM_PROCESSOR}")
    set(CPACK_GENERATOR "TGZ")
  endif()
  set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${PROJECT_VERSION}-${CMAKE_SYSTEM_NAME}-${_tc_arch}")
  set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY TRUE)
  set(CPACK_STRIP_FILES TRUE)
  include(CPack)
endif()

message(STATUS "TensorConsole ${PROJECT_VERSION} | ${CMAKE_BUILD_TYPE} | ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Boost ${Boost_VERSION} | servic: ${SERVIC_INCLUDE_DIR}")