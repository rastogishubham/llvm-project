set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  BitReader
  BitWriter
  CodeGen
  Core
  IRReader
  IPO
  LTO
  MC
  Object
  Support
  Target
  TargetParser
  CAS
  RemoteCachingService
  RemoteNullService
  )

add_llvm_tool(llvm-lto
  llvm-lto.cpp

  DEPENDS intrinsics_gen
  )
