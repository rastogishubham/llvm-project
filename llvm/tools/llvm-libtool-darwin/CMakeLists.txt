set(LLVM_LINK_COMPONENTS
  BinaryFormat
  CAS
  CASObjectFormats
  Core
  Object
  Support
  TargetParser
  TextAPI
  CASUtil
  ${LLVM_TARGETS_TO_BUILD}
  )

add_llvm_tool(llvm-libtool-darwin
  llvm-libtool-darwin.cpp
)

if(LLVM_INSTALL_CCTOOLS_SYMLINKS)
  add_llvm_tool_symlink(libtool llvm-libtool-darwin)
endif()
