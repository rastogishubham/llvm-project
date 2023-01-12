set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  CodeGen
  Core
  MC
  MCCAS
  MIRParser
  Support
  Target
  TargetParser
  )

add_llvm_unittest(MITests
  LiveIntervalTest.cpp
  )
