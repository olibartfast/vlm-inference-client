cmake_minimum_required(VERSION 3.10)
project(vlm-inference-client)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find required packages
find_package(CURL REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(OpenCV REQUIRED)  # Add OpenCV

# Add nlohmann_json
include(FetchContent)
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.2  # Specify a version/tag here
)
FetchContent_MakeAvailable(nlohmann_json)
message(STATUS "nlohmann_json_SOURCE_DIR: ${nlohmann_json_SOURCE_DIR}")

# Add cpp-base64
FetchContent_Declare(
  cpp_base64
  GIT_REPOSITORY https://github.com/ReneNyffenegger/cpp-base64.git
  GIT_TAG master  # You might want to specify a specific commit or tag for stability
)
FetchContent_MakeAvailable(cpp_base64)
message(STATUS "cpp_base64_SOURCE_DIR: ${cpp_base64_SOURCE_DIR}")

# Add cxxopts
FetchContent_Declare(
  cxxopts
  GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
  GIT_TAG v3.1.1
)
FetchContent_MakeAvailable(cxxopts)
message(STATUS "cxxopts_SOURCE_DIR: ${cxxopts_SOURCE_DIR}")

# Add the executable
add_executable(${PROJECT_NAME} main.cpp ${cpp_base64_SOURCE_DIR}/base64.cpp)

# Link libraries
target_link_libraries(${PROJECT_NAME} 
  PRIVATE 
    CURL::libcurl
    OpenSSL::SSL
    OpenSSL::Crypto
    ${OpenCV_LIBS}  # Link OpenCV
)

# Include directories
target_include_directories(${PROJECT_NAME} 
  PRIVATE 
    ${cpp_base64_SOURCE_DIR}
    ${nlohmann_json_SOURCE_DIR}
    ${cxxopts_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}  # Include OpenCV headers
)
