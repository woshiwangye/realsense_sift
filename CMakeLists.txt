cmake_minimum_required(VERSION 3.16)
project(SIFT)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})
IF("${CMAKE_BUILD_TYPE}" STREQUAL "Release" OR "${CMAKE_BUILD_TYPE}" STREQUAL "RELEASE")
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -march=native")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -fopenmp")
ELSE()
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fsanitize=leak -g")
ENDIF()

find_package(OpenCV 4.5 REQUIRED)
if(NOT OpenCV_FOUND)
  message(FATAL_ERROR "OpenCV > 4.5 not found.")
endif()

find_package(realsense2 REQUIRED)
MESSAGE("realsense include dirs: " ${realsense2_INCLUDE_DIRS})
MESSAGE("realsense include dirs: " ${realsense2_LIBRARY})

MESSAGE("opencv include dirs: " ${OpenCV_INCLUDE_DIRS})

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${realsense2_INCLUDE_DIRS}
    # ${OpenGL_INCLUDE_DIR}
)

add_executable(imshow apps/rs-imshow.cpp
)

link_directories(/usr/lib/x86_64-linux-gnu)
target_link_libraries(imshow
    ${OpenCV_LIBS}
    ${realsense2_LIBRARY}
    # ${OPENGL_LIBRARY}
    # glut
    realsense2
)
