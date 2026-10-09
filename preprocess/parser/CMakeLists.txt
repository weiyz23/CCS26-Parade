CMAKE_MINIMUM_REQUIRED(VERSION 3.10 FATAL_ERROR)
project(data_processor LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -flto")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


find_library(BGPSTREAM_LIB bgpstream)
if(NOT BGPSTREAM_LIB)
    message(FATAL_ERROR "BGPStream library not found")
else()
    message(STATUS "BGPStream library found: ${BGPSTREAM_LIB}")
endif()


include_directories(
    include
)


add_library(path_extractor INTERFACE)
target_include_directories(path_extractor
    INTERFACE
    include
)
target_link_libraries(path_extractor
    INTERFACE
    bgpstream
    pthread
    rt
)

add_executable(data_processor
    src/data_parser.cpp
)
target_include_directories(data_processor
    PRIVATE
    include
)
target_link_libraries(data_processor
    path_extractor
    bgpstream
    rt
)

add_executable(processor
    src/processor.cpp
)
target_include_directories(processor
    PRIVATE
    include
)
target_link_libraries(processor
    bgpstream
    rt
)

# Summarize compact reference_paths.bin without scanning it in Python.
add_executable(reference_summary
    src/reference_summary.cpp
)

# Build graph context from older parser caches without reparsing MRT files.
add_executable(cache_graph_context
    src/cache_graph_context.cpp
)
target_include_directories(cache_graph_context
    PRIVATE
    include
)

add_executable(monitor
    src/monitor.cpp
)
target_include_directories(monitor
    PRIVATE
    include
)
target_link_libraries(monitor
    pthread
    rt
)

message(STATUS "Executable output directory: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "Data processor target: data_processor")
message(STATUS "Processor target: processor")
message(STATUS "Reference summary target: reference_summary")
message(STATUS "Cache graph context target: cache_graph_context")
message(STATUS "Monitor target: monitor")
