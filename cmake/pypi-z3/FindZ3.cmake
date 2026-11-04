if(Z3_FOUND)
    return()
endif()
find_package(Python3 COMPONENTS Interpreter REQUIRED)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import z3; print(z3.__path__[0])"
    OUTPUT_VARIABLE Z3_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE Z3_PYTHON_RESULT
)
if(NOT Z3_PYTHON_RESULT EQUAL 0 OR Z3_PATH STREQUAL "")
    message(FATAL_ERROR "Failed to locate z3 Python package. Ensure z3-solver>=4.13.0 is installed.")
endif()
message("-- Find Z3 in path: ${Z3_PATH}")
find_path(Z3_INCLUDE_DIR NO_DEFAULT_PATH NAMES z3++.h PATHS ${Z3_PATH}/include)
if(WIN32)
    find_file(Z3_RUNTIME_LIBRARY NO_DEFAULT_PATH NAMES libz3.dll z3.dll PATHS ${Z3_PATH}/bin ${Z3_PATH}/lib ${Z3_PATH}/lib64)
    find_library(Z3_IMPORT_LIBRARY NO_DEFAULT_PATH NAMES libz3 z3 PATHS ${Z3_PATH}/bin ${Z3_PATH}/lib ${Z3_PATH}/lib64)
    set(Z3_DLL "${Z3_RUNTIME_LIBRARY}")
    if(Z3_DLL AND NOT Z3_IMPORT_LIBRARY)
        set(_z3_import_dir "${CMAKE_BINARY_DIR}/z3-import")
        file(MAKE_DIRECTORY "${_z3_import_dir}")
        set(Z3_IMPORT_LIBRARY "${_z3_import_dir}/libz3.lib")
        if(NOT EXISTS "${Z3_IMPORT_LIBRARY}")
            if(COMMAND tilelang_generate_windows_import_library)
                tilelang_generate_windows_import_library("${Z3_DLL}" Z3_IMPORT_LIBRARY "libz3")
            else()
                message(FATAL_ERROR "tilelang_generate_windows_import_library is unavailable for Windows Z3 import-lib generation")
            endif()
        endif()
    endif()
    set(Z3_LIBRARY "${Z3_IMPORT_LIBRARY}")
else()
    find_library(Z3_LIBRARY NO_DEFAULT_PATH NAMES z3 libz3 PATHS ${Z3_PATH}/bin ${Z3_PATH}/lib ${Z3_PATH}/lib64)
    if(APPLE AND Z3_LIBRARY)
        set(Z3_ORIGINAL_LIBRARY "${Z3_LIBRARY}")
        set(Z3_LIBRARY "${CMAKE_BINARY_DIR}/lib/libz3${CMAKE_SHARED_LIBRARY_SUFFIX}")
        file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
        configure_file("${Z3_ORIGINAL_LIBRARY}" "${Z3_LIBRARY}" COPYONLY)
        execute_process(
            COMMAND /usr/bin/install_name_tool -id "@rpath/libz3${CMAKE_SHARED_LIBRARY_SUFFIX}" "${Z3_LIBRARY}"
            RESULT_VARIABLE Z3_INSTALL_NAME_RESULT
        )
        if(NOT Z3_INSTALL_NAME_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to normalize Z3 install name for ${Z3_LIBRARY}")
        endif()
    endif()
    # PyPI z3-solver >= 4.13.1 ships Linux libz3 with a versioned SONAME
    # (e.g. libz3.so.4.15); only 4.13.0 ships an unversioned libz3.so SONAME.
    # DT_NEEDED is recorded from the linked library's SONAME, so linking
    # against a versioned SONAME pins the built wheel to the exact
    # z3-solver version used at build time: any other runtime z3-solver
    # fails at import with "OSError: libz3.so.<ver>: cannot open shared
    # object file" because only the build version ships that file name.
    # Every z3-solver release keeps an unversioned libz3.so alongside the
    # versioned copy, so normalize the SONAME of a build-tree copy (mirrors
    # the macOS install_name_tool handling above) to make the produced
    # binaries resolvable against any z3-solver in the supported range.
    if(UNIX AND NOT APPLE AND Z3_LIBRARY)
        find_program(Z3_PATCHELF_EXECUTABLE patchelf)
        if(NOT Z3_PATCHELF_EXECUTABLE)
            message(FATAL_ERROR "patchelf is required to normalize the PyPI z3 SONAME (pip install patchelf)")
        endif()
        set(Z3_ORIGINAL_LIBRARY "${Z3_LIBRARY}")
        set(Z3_LIBRARY "${CMAKE_BINARY_DIR}/lib/libz3${CMAKE_SHARED_LIBRARY_SUFFIX}")
        file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
        configure_file("${Z3_ORIGINAL_LIBRARY}" "${Z3_LIBRARY}" COPYONLY)
        execute_process(
            COMMAND "${Z3_PATCHELF_EXECUTABLE}" --set-soname libz3${CMAKE_SHARED_LIBRARY_SUFFIX} "${Z3_LIBRARY}"
            RESULT_VARIABLE Z3_SONAME_RESULT
        )
        if(NOT Z3_SONAME_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to normalize Z3 SONAME for ${Z3_LIBRARY}")
        endif()
    endif()
endif()
message("-- Found Z3 include dir: ${Z3_INCLUDE_DIR}")
message("-- Found Z3 library: ${Z3_LIBRARY}")
if(APPLE AND Z3_ORIGINAL_LIBRARY)
    message("-- Found Z3 original library: ${Z3_ORIGINAL_LIBRARY}")
endif()
if(WIN32)
    message("-- Found Z3 runtime library: ${Z3_RUNTIME_LIBRARY}")
endif()
if(NOT Z3_INCLUDE_DIR OR NOT Z3_LIBRARY)
    message(FATAL_ERROR "Could not find Z3 library or include directory")
endif()
if(WIN32 AND NOT Z3_RUNTIME_LIBRARY)
    message(FATAL_ERROR "Could not find Z3 runtime library")
endif()
if(WIN32)
    add_library(z3::libz3 SHARED IMPORTED GLOBAL)
else()
    add_library(z3::libz3 SHARED IMPORTED GLOBAL)
endif()
set_target_properties(z3::libz3
    PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${Z3_INCLUDE_DIR}
)
if(WIN32)
    set_target_properties(z3::libz3 PROPERTIES IMPORTED_LOCATION "${Z3_RUNTIME_LIBRARY}")
else()
    set_target_properties(z3::libz3 PROPERTIES IMPORTED_LOCATION "${Z3_LIBRARY}")
endif()
if(WIN32 AND DEFINED Z3_IMPORT_LIBRARY AND EXISTS "${Z3_IMPORT_LIBRARY}")
    set_target_properties(z3::libz3 PROPERTIES IMPORTED_IMPLIB "${Z3_IMPORT_LIBRARY}")
endif()
set(Z3_CXX_INCLUDE_DIRS ${Z3_INCLUDE_DIR})
set(Z3_C_INCLUDE_DIRS ${Z3_INCLUDE_DIR})
set(Z3_FOUND TRUE)
