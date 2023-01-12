set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  DebugInfoDWARF
  DebugInfoGSYM
  MC
  MCCAS
  Object
  Support
  Target
  TargetParser
  )

add_llvm_tool(llvm-gsymutil
  llvm-gsymutil.cpp

  )
