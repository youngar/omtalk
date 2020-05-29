#include <gtest/gtest.h>
#include <stdlib.h>

#include <fstream>
#include <iostream>

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Analysis/Verifier.h"
#include "mlir/Dialect/Omtalk/Omtalk.h"
#include "mlir/Dialect/Omtalk/IR/OmtalkDialect.h"
#include "mlir/Dialect/Omtalk/Parser.h"
#include "mlir/Dialect/Omtalk/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"

// TEST(Omtalk, DISABLED_Module) {
//   mlir::MLIRContext context;
//   mlir::OpBuilder builder(&context);
//   mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());

//   // Function Prototype
//   auto location = builder.getUnknownLoc();
//   mlir::Type i64_type = builder.getIntegerType(64);

//   auto func_type = builder.getFunctionType({}, {i64_type});
//   mlir::FuncOp function =
//       mlir::FuncOp::create(location, "test_function", func_type);

//   // Function Body
//   auto &entryBlock = *function.addEntryBlock();
//   builder.setInsertionPointToStart(&entryBlock);

//   auto value1 = builder.create<omtalk::ConstantOp>(
//       builder.getUnknownLoc(), i64_type, builder.getI64IntegerAttr(69));
//   auto value2 = builder.create<omtalk::ConstantOp>(
//       builder.getUnknownLoc(), i64_type, builder.getI64IntegerAttr(100));

//   auto value3 = builder.create<omtalk::IAddOp>(builder.getUnknownLoc(),
//                                                i64_type, value1, value2);

//   // Return
//   omtalk::ReturnOp returnOp =
//       builder.create<omtalk::ReturnOp>(builder.getUnknownLoc(), value3);

//   module.push_back(function);
//   module.dump();
// }

// TEST(Omtalk, DISABLED_Function_Args) {
//   omtalk::Process process;
//   omtalk::Thread thread(process);
//   omtalk::VirtualMachine vm(thread);

//   mlir::MLIRContext context;
//   mlir::OpBuilder builder(&context);
//   mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());

//   mlir::Type i64_type = builder.getIntegerType(64);
//   mlir::Type box_type = omtalk::BoxType::get(&context);
//   auto location = builder.getUnknownLoc();

//   // Add 10
//   {
//     auto func_type = builder.getFunctionType({box_type}, {box_type});
//     mlir::FuncOp add_ten = mlir::FuncOp::create(location, "add_ten",
//     func_type);

//     auto &entryBlock = *add_ten.addEntryBlock();
//     auto args = entryBlock.getArguments();
//     builder.setInsertionPointToStart(&entryBlock);

//     auto value1 = builder.create<omtalk::ConstantOp>(
//         builder.getUnknownLoc(), box_type, builder.getI64IntegerAttr(10));

//     auto value3 = builder.create<omtalk::IAddOp>(builder.getUnknownLoc(),
//                                                  box_type, value1, args[0]);

//     omtalk::ReturnOp returnOp =
//         builder.create<omtalk::ReturnOp>(builder.getUnknownLoc(), value3);

//     module.push_back(add_ten);
//   }

//   module.dump();

//   // run
//   {
//     auto func_type = builder.getFunctionType({}, {box_type});
//     mlir::FuncOp run = mlir::FuncOp::create(location, "run", func_type);
//     auto &entryBlock = *run.addEntryBlock();
//     auto block_args = entryBlock.getArguments();
//     builder.setInsertionPointToStart(&entryBlock);

//     auto value1 = builder.create<omtalk::ConstantOp>(
//         builder.getUnknownLoc(), box_type, builder.getI64IntegerAttr(5));

//     llvm::ArrayRef<mlir::Value> args = {value1};
//     auto value2 = builder.create<omtalk::SendOp>(
//         builder.getUnknownLoc(), box_type, llvm::StringRef("add_ten"), args);

//     omtalk::ReturnOp returnOp =
//         builder.create<omtalk::ReturnOp>(builder.getUnknownLoc(), value2);

//     module.push_back(run);
//   }

//   module.dump();

//   // Optimize
//   {
//     mlir::PassManager pm(&context);

//     pm.addPass(mlir::createInlinerPass());
//     pm.addPass(omtalk::createDeadFunctionEliminationPass());
//     if (mlir::failed(pm.run(module))) {
//       // TODO handle error
//     }
//     module.dump();
//   }

//   // Lower to Standard
//   {
//     mlir::PassManager pm(&context);
//     pm.addPass(omtalk::createLowerPass());
//     if (mlir::failed(pm.run(module))) {
//       // TODO handle error
//     }
//     module.dump();
//   }

//   // Lower to LLVM
//   {
//     mlir::PassManager pm(&context);
//     pm.addPass(omtalk::createToLlvmLoweringPass());
//     if (mlir::failed(pm.run(module))) {
//       // TODO handle error
//     }
//     module.dump();
//   }

//   // JIT
//   {
//     auto llvmModule = mlir::translateModuleToLLVMIR(module);
//     llvm::InitializeNativeTarget();
//     llvm::InitializeNativeTargetAsmPrinter();
//     mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

//     auto optPipeline = mlir::makeOptimizingTransformer(
//         /*optLevel=*/3, /*sizeLevel=*/0,
//         /*targetMachine=*/nullptr);
//     if (auto err = optPipeline(llvmModule.get())) {
//       llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
//     }

//     llvm::errs() << *llvmModule << "\n";

//     auto maybeEngine = mlir::ExecutionEngine::create(module, optPipeline);
//     assert(maybeEngine && "failed to construct an execution engine");
//     auto &engine = maybeEngine.get();

//     auto invocationResult = engine->invoke("run");
//     if (invocationResult) {
//       llvm::errs() << "JIT invocation failed\n";
//       // return -1;
//     }
//   }
// }

namespace omtalk {

class ObjectBuilder {
 public:
 private:
};

}  // namespace omtalk

//===----------------------------------------------------------------------===//
// Integer Primitives
//===----------------------------------------------------------------------===//

namespace mlir {

template <typename LHS, typename RHS>
void emitIntegerAdd(Location location, OpBuilder builder, LHS lhs, RHS rhs) {
  // mlir::Type box_type = ::omtalk::BoxType::get(location->getContext());
  // mlir::Type box_type = omtalk::BoxType::get(&context);

  // auto func_type = builder.getFunctionType({box_type}, {box_type});
  // mlir::FuncOp func = mlir::FuncOp::create(location, "Integer_+", func_type);
  // auto &entryBlock = *func.addEntryBlock();
  // auto args = entryBlock.getArguments();
  // builder.setInsertionPointToStart(&entryBlock);

  // auto rhs = args[0];
  // auto lhs = args[1];

  // auto val = builder.create<SubIOp>(location, rhs,
  //                              builder.create<ConstantOp>(location, 1));
  // val = builder.create<AddIOp>(location, val, lhs);

  // // builder.create<mlir::ReturnOp>(location, val);

  // omtalk::ReturnOp returnOp = builder.create<omtalk::ReturnOp>(location,
  // val);

  // module.push_back(func);
}



}  // namespace mlir

//===----------------------------------------------------------------------===//
// Integer Tests
//===----------------------------------------------------------------------===//

TEST(Omtalk, Send) {
  omtalk::Process process;
  omtalk::Thread thread(process);
  omtalk::VirtualMachine vm(thread);

  mlir::MLIRContext context;
  mlir::OpBuilder builder(&context);
  mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());

  mlir::Type i64_type = builder.getIntegerType(64);
  mlir::Type box_type = omtalk::BoxType::get(&context);
  mlir::Type bint_type = omtalk::BoxIntType::get(&context);
  mlir::Type bref_type = omtalk::BoxRefType::get(&context);
  auto location = builder.getUnknownLoc();

  // run
  {
    auto func_type = builder.getFunctionType({}, {box_type});
    mlir::FuncOp run = mlir::FuncOp::create(location, "run", func_type);
    auto &entryBlock = *run.addEntryBlock();
    auto block_args = entryBlock.getArguments();
    builder.setInsertionPointToStart(&entryBlock);

    auto value1 = builder.create<omtalk::ConstantIntOp>(
        location, bint_type, builder.getI64IntegerAttr(4));

    auto value2 = builder.create<omtalk::ConstantIntOp>(
        location, bint_type, builder.getI64IntegerAttr(3));

    llvm::ArrayRef<mlir::Value> args = {value2};
    auto value3 = builder.create<omtalk::SendOp>(location, box_type, value1,
                                                 llvm::StringRef("+"), args);

    omtalk::ReturnOp returnOp =
        builder.create<omtalk::ReturnOp>(location, value3);

    module.push_back(run);
  }

  module.dump();

  // Canonicalize
  {
    std::cout << "\n!!! Canonicalization\n";
    mlir::PassManager pm(&context);
    pm.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
    if (mlir::failed(pm.run(module))) {
      // TODO handle error
    }
    module.dump();
  }

  // Optimize
  {
     std::cout << "\n!!! Optimize\n";
    mlir::PassManager pm(&context);

    pm.addPass(mlir::createInlinerPass());
    pm.addPass(omtalk::createDeadFunctionEliminationPass());
    if (mlir::failed(pm.run(module))) {
      // TODO handle error
    }
    module.dump();
  }

  // Lower to Standard
  {
    std::cout << "\n!!! Lower to Standard\n";
    mlir::PassManager pm(&context);
    pm.addPass(omtalk::createLowerPass());
    if (mlir::failed(pm.run(module))) {
      // TODO handle error
    }
    module.dump();
  }

  // Lower to LLVM
  {
    std::cout << "\n!!! Lower to LLVM\n";

    mlir::PassManager pm(&context);
    pm.addPass(omtalk::createToLlvmLoweringPass());
    if (mlir::failed(pm.run(module))) {
      // TODO handle error
    }
    module.dump();
  }

  // JIT
  {

    auto llvmModule = mlir::translateModuleToLLVMIR(module);
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

    auto optPipeline = mlir::makeOptimizingTransformer(
        /*optLevel=*/3, /*sizeLevel=*/0,
        /*targetMachine=*/nullptr);
    if (auto err = optPipeline(llvmModule.get())) {
      llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    }

    llvm::errs() << *llvmModule << "\n";

    auto maybeEngine = mlir::ExecutionEngine::create(module, optPipeline);
    assert(maybeEngine && "failed to construct an execution engine");
    auto &engine = maybeEngine.get();

    auto invocationResult = engine->invoke("run");
    if (invocationResult) {
      llvm::errs() << "JIT invocation failed\n";
      // return -1;
    }
  }
}