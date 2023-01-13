//===- CASUtil/Utils.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CASUtil/Utils.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/CAS/CASID.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/CASObjectFormats/CASObjectReader.h"
#include "llvm/CASObjectFormats/Encoding.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;
using namespace llvm::cas;
using namespace llvm::casobjectformats;
using namespace llvm::casobjectformats::reader;

Expected<CASID> cas::readCASIDBuffer(cas::ObjectStore &CAS,
                                     MemoryBufferRef Buffer) {
  if (identify_magic(Buffer.getBuffer()) != file_magic::cas_id)
    return createStringError(std::errc::invalid_argument,
                             "buffer does not contain a CASID");

  StringRef Remaining =
      Buffer.getBuffer().substr(StringRef(casidObjectMagicPrefix).size());
  uint32_t Size;
  if (auto E = encoding::consumeVBR8(Remaining, Size))
    return std::move(E);

  StringRef CASIDStr = Remaining.substr(0, Size);
  return CAS.parseID(CASIDStr);
}

void cas::writeCASIDBuffer(const CASID &ID, llvm::raw_ostream &OS) {
  OS << casidObjectMagicPrefix;
  SmallString<256> CASIDStr;
  raw_svector_ostream(CASIDStr) << ID;

  // Write out the size of the CASID so that we can read it back properly even
  // if the buffer has additional padding (e.g. after getting added in an
  // archive).
  SmallString<2> SizeBuf;
  encoding::writeVBR8(CASIDStr.size(), SizeBuf);
  OS << SizeBuf << CASIDStr;
}

static std::string getDescription(const CASSection &Info) {
  std::string Description;
  {
    raw_string_ostream(Description) << Info.Name << " | MemProt: " << Info.Prot;
  }
  return Description;
}

static std::string getDescription(const CASBlock &Info) {
  std::string Description;
  {
    raw_string_ostream OS(Description);
    OS << "size = " << formatv("{0:x4}", Info.Size)
       << ", align = " << Info.Alignment
       << ", alignment-offset = " << Info.AlignmentOffset;
    if (Info.isZeroFill())
      OS << ", zero-fill";
  }
  return Description;
}

static std::string
getDescription(const CASBlockFixup &Fixup, StringRef TargetName,
               std::function<const char *(uint8_t)> GetEdgeName) {
  std::string Description;
  {
    raw_string_ostream OS(Description);
    OS << formatv("{0:x4}", Fixup.Offset) << ", addend = ";
    if (Fixup.Addend >= 0)
      OS << formatv("+{0:x8}", Fixup.Addend);
    else
      OS << formatv("-{0:x8}", -Fixup.Addend);
    OS << ", kind = " << GetEdgeName(Fixup.Kind) << ", target = " << TargetName;
  }
  return Description;
}

static std::string
getDescription(StringRef Name, const CASSymbol &Info,
               std::function<const char *(uint8_t)> GetScopeName,
               std::function<const char *(uint8_t)> GetLinkageName) {
  bool IsDefined = Info.isDefined();
  std::string Description;
  {
    raw_string_ostream OS(Description);
    OS << Name << " | ";
    if (IsDefined) {
      OS << "offset: " << formatv("{0:x4}", Info.Offset) << ", ";
    }
    OS << "linkage: "
       << formatv("{0:6}", GetLinkageName((uint8_t)Info.Linkage));
    if (IsDefined) {
      OS << ", scope: " << formatv("{0:8}", GetLinkageName((uint8_t)Info.Scope))
         << ", " << (Info.IsLive ? "live" : "dead");
      if (Info.IsCallable)
        OS << ", callable";
      if (Info.IsAutoHide)
        OS << ", autohide";
    }
  }
  return Description;
  return Info.Name.str();
}

static cl::opt<bool> SortSections("print-cas-object-sort-sections",
                                  cl::desc("Sort sections before printing"),
                                  cl::init(false));

Error casobjectformats::printCASObject(
    const reader::CASObjectReader &Reader, raw_ostream &OS, bool omitCASID,
    std::function<const char *(uint8_t)> GetEdgeName,
    std::function<const char *(uint8_t)> GetScopeName,
    std::function<const char *(uint8_t)> GetLinkageName) {
  Triple TT = Reader.getTargetTriple();
  OS << "Triple: " << TT.str() << "\n\n";

  struct PrintBlock;
  struct PrintSymbol;

  struct PrintSection {
    std::string Name;
    std::string Description;
    SmallVector<PrintBlock *> Blocks;
  };
  struct PrintBlock {
    struct Fixup {
      std::string Description;
    };

    CASBlockRef Ref;
    std::string Description;
    std::string Content;
    PrintSection *Section = nullptr;
    SmallVector<PrintSymbol *> Symbols;
    SmallVector<Fixup> Fixups;
    llvm::cas::CASID ID;
    PrintBlock(cas::CASID _ID) : ID(_ID) {}
  };
  struct PrintSymbol {
    unsigned Offset;
    std::string Name;
    std::string Description;
    PrintBlock *Block = nullptr;
  };

  SmallVector<std::unique_ptr<PrintSection>> Sections;
  auto createSection = [&]() -> PrintSection * {
    Sections.push_back(std::make_unique<PrintSection>());
    return Sections.back().get();
  };

  SmallVector<std::unique_ptr<PrintBlock>> Blocks;
  auto createBlock = [&](cas::CASID ID) -> PrintBlock * {
    Blocks.push_back(std::make_unique<PrintBlock>(ID));
    return Blocks.back().get();
  };

  SmallVector<std::unique_ptr<PrintSymbol>> Symbols;
  auto createSymbol = [&]() -> PrintSymbol * {
    Symbols.push_back(std::make_unique<PrintSymbol>());
    return Symbols.back().get();
  };

  SmallDenseMap<CASSectionRef, PrintSection *> SectionMap;
  DenseMap<CASBlockRef, PrintBlock *> BlockMap;
  DenseMap<CASSymbolRef, PrintSymbol *> SymbolMap;
  unsigned AnonNameIdx = 0;

  auto recordSection =
      [&](CASSectionRef SectionRef) -> Expected<PrintSection *> {
    PrintSection *&MappedSection = SectionMap[SectionRef];
    if (MappedSection)
      return MappedSection;
    Expected<CASSection> Info = Reader.materialize(SectionRef);
    if (!Info)
      return Info.takeError();
    auto *PrintSection = createSection();
    PrintSection->Name = Info->Name.str();
    PrintSection->Description = getDescription(*Info);
    MappedSection = PrintSection;
    return PrintSection;
  };

  auto recordBlock = [&](CASBlockRef BlockRef) -> Expected<PrintBlock *> {
    PrintBlock *&MappedBlock = BlockMap[BlockRef];
    if (MappedBlock)
      return MappedBlock;
    Expected<CASBlock> Info = Reader.materialize(BlockRef);
    if (!Info)
      return Info.takeError();
    auto *PrintBlock = createBlock(Info->BlockContentID);
    auto PrintSection = recordSection(Info->SectionRef);
    if (!PrintSection)
      return PrintSection.takeError();
    PrintBlock->Ref = BlockRef;
    PrintBlock->Description = getDescription(*Info);
    PrintBlock->Content = Info->Content.value_or(StringRef()).str();
    PrintBlock->Section = *PrintSection;
    PrintBlock->Section->Blocks.push_back(PrintBlock);
    MappedBlock = PrintBlock;
    return PrintBlock;
  };

  auto recordSymbol = [&](CASSymbolRef SymbolRef,
                          CASSymbol Info) -> Expected<PrintSymbol *> {
    PrintSymbol *&MappedSymbol = SymbolMap[SymbolRef];
    if (MappedSymbol)
      return MappedSymbol;
    auto *PrintSymbol = createSymbol();
    PrintSymbol->Offset = Info.Offset;
    if (!Info.Name.empty()) {
      PrintSymbol->Name = Info.Name.str();
    } else {
      raw_string_ostream(PrintSymbol->Name)
          << "<anon-" << (AnonNameIdx++) << '>';
    }
    PrintSymbol->Description =
        getDescription(PrintSymbol->Name, Info, GetScopeName, GetLinkageName);
    if (Info.BlockRef) {
      auto PrintBlock = recordBlock(*Info.BlockRef);
      if (!PrintBlock)
        return PrintBlock.takeError();
      PrintSymbol->Block = *PrintBlock;
    }
    if (PrintSymbol->Block)
      PrintSymbol->Block->Symbols.push_back(PrintSymbol);
    MappedSymbol = PrintSymbol;
    return PrintSymbol;
  };

  Error E = Reader.forEachSymbol(
      [&](CASSymbolRef SymbolRef, CASSymbol Info) -> Error {
        Expected<PrintSymbol *> Symbol = recordSymbol(SymbolRef, Info);
        if (!Symbol)
          return Symbol.takeError();
        return Error::success();
      });
  if (E)
    return E;

  // Check for \p size() at each iteration because new blocks can be added
  // while reading the fixups, due to new symbols getting discovered when using
  // nestedv1 format.
  // FIXME: Get nestedv1 to provide all the symbols that can be referenced from
  // the translation unit, via \p forEachSymbol().
  for (unsigned I = 0; I < Blocks.size(); ++I) {
    PrintBlock &PBlock = *Blocks[I];
    Error E = Reader.materializeFixups(
        PBlock.Ref, [&](const CASBlockFixup &Fixup) -> Error {
          PrintSymbol *Target = SymbolMap[Fixup.TargetRef];
          if (!Target) {
            Expected<CASSymbol> Info = Reader.materialize(Fixup.TargetRef);
            if (!Info)
              return Info.takeError();
            Expected<PrintSymbol *> PrintSym =
                recordSymbol(Fixup.TargetRef, *Info);
            if (!PrintSym)
              return PrintSym.takeError();
            Target = *PrintSym;
          }
          std::string FixupDesc =
              getDescription(Fixup, Target->Name, GetEdgeName);
          PBlock.Fixups.push_back(PrintBlock::Fixup{std::move(FixupDesc)});
          return Error::success();
        });
    if (E)
      return E;
  }

  if (SortSections) {
    llvm::sort(Sections, [](const auto &LHS, const auto &RHS) {
      return LHS->Description < RHS->Description;
    });
  }
  for (const auto &Section : Sections) {
    OS << "SECTION: " << Section->Description << '\n';
    OS << "{\n";
    for (const auto *Block : Section->Blocks) {
      OS.indent(2) << "BLOCK: " << Block->Description << '\n';
      if (!omitCASID)
        OS.indent(2) << "Block-Cas-ID: " << Block->ID << '\n';
      OS.indent(2) << "{\n";
      for (const auto *Symbol : Block->Symbols) {
        OS.indent(4) << "SYMBOL: " << Symbol->Description;
        if (Section->Name == "__TEXT,__cstring") {
          OS << " | \"";
          OS.write_escaped(&Block->Content[Symbol->Offset]);
          OS << '"';
        }
        OS << '\n';
      }
      for (const auto &Fixup : Block->Fixups) {
        OS.indent(4) << "FIXUP: " << Fixup.Description << '\n';
      }
      OS.indent(2) << "}\n";
    }
    OS << "}\n\n";
  }

  OS << "UNDEFINED SYMBOLS\n{\n";
  for (const auto &Symbol : Symbols) {
    if (Symbol->Block)
      continue;
    OS.indent(2) << Symbol->Description << '\n';
  }
  OS << "}\n";
  return Error::success();
}
