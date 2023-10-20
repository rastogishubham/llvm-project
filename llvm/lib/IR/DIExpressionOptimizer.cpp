//===- DIExpressionOptimizer.cpp - Constant folding of DIExpressions ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements functions to constant fold DIExpressions. Which were
// declared in DIExpressionOptimizer.h
//
//===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/DebugInfoMetadata.h"

using namespace llvm;

static std::optional<uint64_t> isConstantVal(DIExpression::ExprOperand Op) {
  if (Op.getOp() == dwarf::DW_OP_constu)
    return Op.getArg(0);
  return std::nullopt;
}

static bool isNeutralElement(uint64_t Op, uint64_t Val) {
  switch (Op) {
  case dwarf::DW_OP_plus:
  case dwarf::DW_OP_minus:
  case dwarf::DW_OP_shl:
  case dwarf::DW_OP_shr:
    return Val == 0;
  case dwarf::DW_OP_mul:
  case dwarf::DW_OP_div:
    return Val == 1;
  default:
    return false;
  }
}

static std::optional<uint64_t> foldOperationIfPossible(
    DIExpression::ExprOperand Op1, DIExpression::ExprOperand Op2,
    DIExpression::ExprOperand Op3, bool &ConstantValCheckFailed) {

  auto Operand1 = isConstantVal(Op1);
  auto Operand2 = isConstantVal(Op2);

  if (!Operand1 || !Operand2) {
    ConstantValCheckFailed = true;
    return std::nullopt;
  }

  auto Oper1 = *Operand1;
  auto Oper2 = *Operand2;

  bool ResultOverflowed;
  switch (Op3.getOp()) {
  case dwarf::DW_OP_plus: {
    auto Result = SaturatingAdd(Oper1, Oper2, &ResultOverflowed);
    if (ResultOverflowed)
      return std::nullopt;
    return Result;
  }
  case dwarf::DW_OP_minus: {
    if (Oper1 < Oper2)
      return std::nullopt;
    return Oper1 - Oper2;
  }
  case dwarf::DW_OP_shl: {
    if ((uint64_t)countl_zero(Oper1) < Oper2)
      return std::nullopt;
    return Oper1 << Oper2;
  }
  case dwarf::DW_OP_shr: {
    if ((uint64_t)countr_zero(Oper1) < Oper2)
      return std::nullopt;
    return Oper1 >> Oper2;
  }
  case dwarf::DW_OP_mul: {
    auto Result = SaturatingMultiply(Oper1, Oper2, &ResultOverflowed);
    if (ResultOverflowed)
      return std::nullopt;
    return Result;
  }
  case dwarf::DW_OP_div: {
    if (Oper2)
      return Oper1 / Oper2;
    return std::nullopt;
  }
  default:
    return std::nullopt;
  }
}

static bool operationsAreFoldableAndCommutative(uint64_t Op1, uint64_t Op2) {
  return Op1 == Op2 && (Op1 == dwarf::DW_OP_plus || Op1 == dwarf::DW_OP_mul);
}

static void consumeOneOperator(DIExpressionCursor &Cursor, uint64_t &Loc,
                               const DIExpression::ExprOperand &Op) {
  Cursor.consume(1);
  Loc = Loc + Op.getSize();
}

void startFromBeginning(uint64_t &Loc, DIExpressionCursor &Cursor,
                        ArrayRef<uint64_t> WorkingOps) {
  Cursor.assignNewExpr(WorkingOps);
  Loc = 0;
}

static SmallVector<uint64_t>
canonicalizeDwarfOperations(ArrayRef<uint64_t> WorkingOps) {
  DIExpressionCursor Cursor(WorkingOps);
  uint64_t Loc = 0;
  SmallVector<uint64_t> ResultOps;
  while (Loc < WorkingOps.size()) {
    auto Op = Cursor.peek();
    /// Expression has no operations, break.
    if (!Op)
      break;
    auto OpRaw = Op->getOp();
    auto OpArg = Op->getArg(0);

    if (OpRaw >= dwarf::DW_OP_lit0 && OpRaw <= dwarf::DW_OP_lit31) {
      ResultOps.push_back(dwarf::DW_OP_constu);
      ResultOps.push_back(OpRaw - dwarf::DW_OP_lit0);
      consumeOneOperator(Cursor, Loc, *Cursor.peek());
      continue;
    }
    if (OpRaw == dwarf::DW_OP_plus_uconst) {
      ResultOps.push_back(dwarf::DW_OP_constu);
      ResultOps.push_back(OpArg);
      ResultOps.push_back(dwarf::DW_OP_plus);
      consumeOneOperator(Cursor, Loc, *Cursor.peek());
      continue;
    }
    uint64_t PrevLoc = Loc;
    consumeOneOperator(Cursor, Loc, *Cursor.peek());
    ResultOps.append(WorkingOps.begin() + PrevLoc, WorkingOps.begin() + Loc);
  }
  return ResultOps;
}

static SmallVector<uint64_t>
optimizeDwarfOperations(ArrayRef<uint64_t> WorkingOps) {
  DIExpressionCursor Cursor(WorkingOps);
  uint64_t Loc = 0;
  SmallVector<uint64_t> ResultOps;
  while (Loc < WorkingOps.size()) {
    auto Op1 = Cursor.peek();
    /// Expression has no operations, exit.
    if (!Op1)
      break;
    auto Op1Raw = Op1->getOp();
    auto Op1Arg = Op1->getArg(0);

    if (Op1Raw == dwarf::DW_OP_constu && Op1Arg == 0) {
      ResultOps.push_back(dwarf::DW_OP_lit0);
      consumeOneOperator(Cursor, Loc, *Cursor.peek());
      continue;
    }

    auto Op2 = Cursor.peekNext();
    /// Expression has no more operations, copy into ResultOps and exit.
    if (!Op2) {
      uint64_t PrevLoc = Loc;
      consumeOneOperator(Cursor, Loc, *Cursor.peek());
      ResultOps.append(WorkingOps.begin() + PrevLoc, WorkingOps.begin() + Loc);
      break;
    }
    auto Op2Raw = Op2->getOp();

    if (Op1Raw == dwarf::DW_OP_constu && Op2Raw == dwarf::DW_OP_plus) {
      ResultOps.push_back(dwarf::DW_OP_plus_uconst);
      ResultOps.push_back(Op1Arg);
      consumeOneOperator(Cursor, Loc, *Cursor.peek());
      consumeOneOperator(Cursor, Loc, *Cursor.peek());
      continue;
    }
    uint64_t PrevLoc = Loc;
    consumeOneOperator(Cursor, Loc, *Cursor.peek());
    ResultOps.append(WorkingOps.begin() + PrevLoc, WorkingOps.begin() + Loc);
  }
  return ResultOps;
}

static bool tryFoldNoOpMath(ArrayRef<DIExpression::ExprOperand> Ops,
                            uint64_t &Loc, DIExpressionCursor &Cursor,
                            SmallVectorImpl<uint64_t> &WorkingOps) {

  if (isConstantVal(Ops[0]) &&
      isNeutralElement(Ops[1].getOp(), Ops[0].getArg(0))) {
    WorkingOps.erase(WorkingOps.begin() + Loc, WorkingOps.begin() + Loc + 3);
    startFromBeginning(Loc, Cursor, WorkingOps);
    return true;
  }
  return false;
}

static bool tryFoldConstants(ArrayRef<DIExpression::ExprOperand> Ops,
                             uint64_t &Loc, DIExpressionCursor &Cursor,
                             SmallVectorImpl<uint64_t> &WorkingOps) {

  bool ConstantValCheckFailed = false;
  auto Result =
      foldOperationIfPossible(Ops[0], Ops[1], Ops[2], ConstantValCheckFailed);
  if (ConstantValCheckFailed)
    return false;
  if (!Result) {
    consumeOneOperator(Cursor, Loc, Ops[0]);
    return true;
    }
    WorkingOps.erase(WorkingOps.begin() + Loc + 2,
                     WorkingOps.begin() + Loc + 5);
    WorkingOps[Loc] = dwarf::DW_OP_constu;
    WorkingOps[Loc + 1] = *Result;
    startFromBeginning(Loc, Cursor, WorkingOps);
    return true;
}

static bool tryFoldCommutativeMath(ArrayRef<DIExpression::ExprOperand> Ops,
                                   uint64_t &Loc, DIExpressionCursor &Cursor,
                                   SmallVectorImpl<uint64_t> &WorkingOps) {

  bool ConstantValCheckFailed = false;
  if (operationsAreFoldableAndCommutative(Ops[1].getOp(), Ops[3].getOp())) {
    auto Result =
        foldOperationIfPossible(Ops[0], Ops[2], Ops[1], ConstantValCheckFailed);
    if (ConstantValCheckFailed)
      return false;
    if (!Result) {
      consumeOneOperator(Cursor, Loc, Ops[0]);
      return true;
    }
    WorkingOps.erase(WorkingOps.begin() + Loc + 3,
                     WorkingOps.begin() + Loc + 6);
    WorkingOps[Loc] = dwarf::DW_OP_constu;
    WorkingOps[Loc + 1] = *Result;
    startFromBeginning(Loc, Cursor, WorkingOps);
    return true;
  }
  return false;
}

static bool tryFoldCommutativeMathWithArgInBetween(
    ArrayRef<DIExpression::ExprOperand> Ops, uint64_t &Loc,
    DIExpressionCursor &Cursor, SmallVectorImpl<uint64_t> &WorkingOps) {

  bool ConstantValCheckFailed = false;
  if (Ops[2].getOp() == dwarf::DW_OP_LLVM_arg &&
      operationsAreFoldableAndCommutative(Ops[1].getOp(), Ops[3].getOp()) &&
      operationsAreFoldableAndCommutative(Ops[3].getOp(), Ops[5].getOp())) {
    auto Result =
        foldOperationIfPossible(Ops[0], Ops[4], Ops[1], ConstantValCheckFailed);
    if (ConstantValCheckFailed)
      return false;
    if (!Result) {
      consumeOneOperator(Cursor, Loc, Ops[0]);
      return true;
    }
    WorkingOps.erase(WorkingOps.begin() + Loc + 6,
                     WorkingOps.begin() + Loc + 9);
    WorkingOps[Loc] = dwarf::DW_OP_constu;
    WorkingOps[Loc + 1] = *Result;
    startFromBeginning(Loc, Cursor, WorkingOps);
    return true;
  }
  return false;
}

DIExpression *DIExpression::foldConstantMath() {

  SmallVector<uint64_t, 8> WorkingOps(Elements.begin(), Elements.end());
  uint64_t Loc = 0;
  SmallVector<uint64_t> ResultOps = canonicalizeDwarfOperations(WorkingOps);
  DIExpressionCursor Cursor(ResultOps);
  SmallVector<DIExpression::ExprOperand, 8> Ops;

  while (Loc < ResultOps.size()) {
    Ops.clear();

    auto Op = Cursor.peek();
    // Expression has no operations, exit.
    if (!Op)
      break;

    Ops.push_back(*Op);

    if (!isConstantVal(Ops[0])) {
      // Early exit, all of the following patterns start with a constant value.
      consumeOneOperator(Cursor, Loc, *Op);
      continue;
    }

    Op = Cursor.peekNext();
    // All following patterns require at least 2 Operations, exit.
    if (!Op)
      break;

    Ops.push_back(*Op);

    if (tryFoldNoOpMath(Ops, Loc, Cursor, ResultOps))
      continue;

    Op = Cursor.peekNextN(2);
    // Op[1] could still match a pattern, skip iteration.
    if (!Op) {
      consumeOneOperator(Cursor, Loc, Ops[0]);
      continue;
    }

    Ops.push_back(*Op);
    if (tryFoldConstants(Ops, Loc, Cursor, ResultOps))
      continue;

    Op = Cursor.peekNextN(3);
    // Op[1] and Op[2] could still match a pattern, skip iteration.
    if (!Op) {
      consumeOneOperator(Cursor, Loc, Ops[0]);
      continue;
    }

    Ops.push_back(*Op);
    if (tryFoldCommutativeMath(Ops, Loc, Cursor, ResultOps))
      continue;

    Op = Cursor.peekNextN(4);
    if (!Op) {
      consumeOneOperator(Cursor, Loc, Ops[0]);
      continue;
    }

    Ops.push_back(*Op);
    Op = Cursor.peekNextN(5);
    if (!Op) {
      consumeOneOperator(Cursor, Loc, Ops[0]);
      continue;
    }

    Ops.push_back(*Op);
    if (tryFoldCommutativeMathWithArgInBetween(Ops, Loc, Cursor, ResultOps))
      continue;

    consumeOneOperator(Cursor, Loc, Ops[0]);
  }
  ResultOps = optimizeDwarfOperations(ResultOps);
  auto *Result = DIExpression::get(getContext(), ResultOps);
  assert(Result->isValid() && "concatenated expression is not valid");
  return Result;
}
