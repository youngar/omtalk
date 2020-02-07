#ifndef OMRTALK_PARSER_HPP
#define OMRTALK_PARSER_HPP

#include <llvm/Support/SMLoc.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Identifier.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Module.h>

#include <cassert>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace omtalk {

mlir::OwningModuleRef parseSourceFile(const llvm::SourceMgr &sourceMgr,
                                      mlir::MLIRContext *context);

mlir::OwningModuleRef parseSourceFile(llvm::StringRef filename,
                                      mlir::MLIRContext *context);

mlir::OwningModuleRef parseSourceFile(llvm::StringRef filename,
                                      llvm::SourceMgr &sourceMgr,
                                      mlir::MLIRContext *context);

mlir::OwningModuleRef parseSourceString(llvm::StringRef moduleStr,
                                        mlir::MLIRContext *context);

class Parser {
 public:
  Parser(const llvm::SourceMgr &source_mgr, mlir::MLIRContext *context);

  mlir::MLIRContext *context() { return context_; }
  const llvm::SourceMgr &source_mgr() { return source_mgr_; }

 private:
  mlir::MLIRContext *const context_;
  const llvm::SourceMgr &source_mgr_;
  llvm::StringRef buffer_;
  const char *loc_;
};

class Module {
 public:
  Module(mlir::MLIRContext &context);

  mlir::ModuleOp module();

 private:
  mlir::ModuleOp module_;
  mlir::OpBuilder builder_;
};

Module::Module(mlir::MLIRContext &context) : builder_(&context) {
  module_ = mlir::ModuleOp::create(builder_.getUnknownLoc());
}

mlir::ModuleOp Module::module() { return module_; }

inline Parser::Parser(const llvm::SourceMgr &source_mgr,
                      mlir::MLIRContext *context)
    : context_(context), source_mgr_(source_mgr) {
  auto buffer_id = source_mgr.getMainFileID();
  buffer_ = source_mgr_.getMemoryBuffer(buffer_id)->getBuffer();
  loc_ = buffer_.begin();
}

mlir::OwningModuleRef parse_source_file(const llvm::SourceMgr &source_mgr,
                                        mlir::MLIRContext *context) {
  auto source_buff = source_mgr.getMemoryBuffer(source_mgr.getMainFileID());
  mlir::OwningModuleRef module(mlir::ModuleOp::create(mlir::FileLineColLoc::get(
      source_buff->getBufferIdentifier(), 0, 0, context)));

  // TODO Parse the source file

  // if (failed(verify(*module))) {
  //   return nullptr;
  // }
  return nullptr;
}

mlir::OwningModuleRef parse_source_file(llvm::StringRef filename,
                                        mlir::MLIRContext *context);

mlir::OwningModuleRef parse_source_file(llvm::StringRef filename,
                                        llvm::SourceMgr &sourceMgr,
                                        mlir::MLIRContext *context);

mlir::OwningModuleRef parse_source_string(llvm::StringRef module_str,
                                          mlir::MLIRContext *context) {
  auto mem_buffer = llvm::MemoryBuffer::getMemBuffer(module_str);
  if (!mem_buffer) {
    return nullptr;
  }
  llvm::SourceMgr source_mgr;
  source_mgr.AddNewSourceBuffer(std::move(mem_buffer), llvm::SMLoc());
  return parse_source_file(source_mgr, context);
}

}  // namespace omtalk

#endif