set(LLVM_LINK_COMPONENTS
  DebugInfoDWARF
  Support
  CAS
  CASObjectFormats
  MCCAS
  CASUtil
  BinaryFormat
  TargetParser
  Object
  )

add_llvm_tool(llvm-cas-dump
  llvm-cas-dump.cpp
  CASDWARFObject.cpp
  MCCASPrinter.cpp
)
