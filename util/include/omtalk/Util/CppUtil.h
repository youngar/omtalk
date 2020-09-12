#ifndef OMTALK_CPPUTIL_H
#define OMTALK_CPPUTIL_H

/// C preprocessor utilities
#define OMTALK_TOKEN_PASTE(a, b) a##b

/// Basic stringification using the preprocessor. x will not be expanded.
/// Whatever is passed in as x will be turned to a literal char array.
#define OMTALK_STRINGIFY_RAW(x) #x

/// Stringify x. x will be expanded by the preprocessor.
#define OMTALK_STRINGIFY(x) OMTALK_STRINGIFY_RAW(x)

/// The name of the current C++ source file. A char[] literal.
#define OMTALK_FILE_STR() __FILE__

/// The current line number in the C++ source file. A char[] literal.
#define OMTALK_LINE_STR() OMTALK_STRINGIFY(__LINE__)

/// The string literal file:line, describing the current line in the source
/// code. A char[] literal.
#define OMTALK_LOCATION_STR() OMTALK_FILE_STR() ":" OMTALK_LINE_STR()
#if defined(__clang__) || defined(__GNUC__)
#define OMTALK_FUNCTION_STR() __PRETTY_FUNCTION__
#else
#define OMTALK_FUNCTION_STR() "<unknown>"
#endif

#endif