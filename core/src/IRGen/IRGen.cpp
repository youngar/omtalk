#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/ScopedHashTable.h>
#include <mlir/Dialect/Omtalk/IR/OmtalkDialect.h>
#include <mlir/Dialect/Omtalk/IR/OmtalkOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Module.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/IR/Verifier.h>
#include <omtalk/IRGen/IRGen.h>
#include <omtalk/Parser/Location.h>

using namespace omtalk;
using namespace omtalk::irgen;
using namespace omtalk::parser;

namespace {

class IRGen {
public:
  IRGen(mlir::MLIRContext &context) : builder(&context) {}

  mlir::Location loc(Location loc) {
    return builder.getFileLineColLoc(builder.getIdentifier(loc.filename),
                                     loc.start.line, loc.start.line);
  }

  /// convert an identifier list into a single composite selector name.
  /// eg val: val: val: => val:val:val:
  std::string getSelectorString(IdentifierList selector) {
    std::string name = "";
    for (const auto &id : selector) {
      name += id.value;
    }
    return name;
  }

  //===--------------------------------------------------------------------===//
  // Literals
  //===--------------------------------------------------------------------===//

  mlir::omtalk::ConstantIntOp irGen(const IntegerExpr &expr) {
    auto location = loc(expr.location);
    auto value = builder.getIntegerAttr(builder.getIntegerType(64), expr.value);
    return builder.create<mlir::omtalk::ConstantIntOp>(
        location, mlir::omtalk::BoxIntType::get(builder.getContext()), value);
  }

  mlir::omtalk::ConstantFloatOp irGen(const FloatExpr &expr) {
    auto location = loc(expr.location);
    auto value = builder.getFloatAttr(builder.getF64Type(), expr.value);
    return builder.create<mlir::omtalk::ConstantFloatOp>(
        location, mlir::omtalk::BoxRefType::get(builder.getContext()), value);
  }

  mlir::omtalk::ConstantSymbolOp irGen(const SymbolExpr &expr) {
    auto location = loc(expr.location);
    auto value = builder.getStringAttr(expr.value);
    return builder.create<mlir::omtalk::ConstantSymbolOp>(
        location, mlir::omtalk::BoxRefType::get(builder.getContext()), value);
  }

  //===--------------------------------------------------------------------===//
  // Expressions
  //===--------------------------------------------------------------------===//

  mlir::Value irGenNil(const IdentifierExpr &expr) {
    auto location = loc(expr.location);
    auto value = builder.getIntegerAttr(builder.getIntegerType(64), 0);
    return builder.create<mlir::omtalk::ConstantRefOp>(
        location, mlir::omtalk::BoxRefType::get(builder.getContext()), value);
  }

  mlir::Value irGen(const IdentifierExpr &expr) {
    if (expr.name == "nil") {
      return irGenNil(expr);
    }
    return symbolTable.lookup(expr.name);
  }

  mlir::omtalk::SendOp irGen(const SendExpr &expr) {
    auto ty = mlir::omtalk::BoxUnkType::get(builder.getContext());
    auto recv = irGenExpr(*(expr.parameters[0]));
    auto message = getSelectorString(expr.selector);
    auto location = loc(expr.location);

    llvm::SmallVector<mlir::Value, 4> operands;
    for (int i = 1; i < expr.parameters.size(); i++) {
      operands.push_back(irGenExpr(*(expr.parameters[i])));
    }

    auto sendOp = builder.create<mlir::omtalk::SendOp>(location, ty, recv,
                                                       message, operands);

    return sendOp;
  }

  mlir::omtalk::ReturnOp irGen(const ReturnExpr &expr) {
    auto value = irGenExpr(*expr.value);
    auto location = loc(expr.location);
    return builder.create<mlir::omtalk::ReturnOp>(location, value);
  }

  mlir::omtalk::NonlocalReturnOp irGen(const NonlocalReturnExpr &expr) {
    auto value = irGenExpr(*expr.value);
    auto location = loc(expr.location);
    return builder.create<mlir::omtalk::NonlocalReturnOp>(location, value);
  }

  mlir::Value irGenExpr(const Expr &expr) {
    switch (expr.kind) {
    case ExprKind::Integer:
      return irGen(expr.cast<IntegerExpr>());
    case ExprKind::Float:
      return irGen(expr.cast<FloatExpr>());
    case ExprKind::String:
      assert(false && "StringExpr");
    case ExprKind::Symbol:
      assert(false && "SymbolExpr");
    case ExprKind::Array:
      assert(false && "ArrayExpr");
    case ExprKind::Identifier:
      return irGen(expr.cast<IdentifierExpr>());
    case ExprKind::Send:
      return irGen(expr.cast<SendExpr>());
    case ExprKind::Block:
      return irGen(expr.cast<BlockExpr>());
    default:
      llvm::outs() << static_cast<int>(expr.kind) << "\n";
      assert(false && "Unhandled expression type");
      break;
    }
  }

  std::optional<mlir::Value> irGenStatement(const Expr &expr) {
    switch (expr.kind) {
    case ExprKind::Return:
      irGen(expr.cast<ReturnExpr>());
      break;
    case ExprKind::NonlocalReturn:
      irGen(expr.cast<NonlocalReturnExpr>());
      break;
    default:
      return irGenExpr(expr);
    }
    return std::nullopt;
  }

  mlir::omtalk::BlockOp irGen(const BlockExpr &expr) {
    auto ty = mlir::omtalk::BoxUnkType::get(builder.getContext());
    auto location = loc(expr.location);
    llvm::SmallVector<mlir::Type, 6> inputs(expr.parameters.size(), ty);
    llvm::SmallVector<mlir::Type, 1> results(1, ty);
    auto resultType = ty;
    auto funcType = builder.getFunctionType(inputs, results);

    auto blockOp = builder.create<mlir::omtalk::BlockOp>(
        location, resultType, mlir::TypeAttr::get(funcType));

    mlir::OpBuilder::InsertionGuard guard(builder);

    auto region = &blockOp.body();
    auto block = builder.createBlock(region);

    llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> var_scope(
        symbolTable);
    for (const auto &param : expr.parameters) {
      auto arg = block->addArgument(ty);
      symbolTable.insert(param.value, arg);
    }

    builder.setInsertionPointToStart(block);

    for (const auto &expr : expr.body) {
      irGenStatement(*expr);
    }

    return blockOp;
  }

  mlir::omtalk::MethodOp irGen(const Method &method) {
    auto location = loc(method.location);
    auto sym_name = getSelectorString(method.selector);
    auto methodOp = builder.create<mlir::omtalk::MethodOp>(
        location, sym_name, 1 + method.parameters.size());

    mlir::OpBuilder::InsertionGuard guard(builder);

    auto region = &methodOp.body();
    auto block = builder.createBlock(region);

    llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> var_scope(
        symbolTable);
    auto ty = mlir::omtalk::BoxUnkType::get(builder.getContext());
    symbolTable.insert("self", block->addArgument(ty));
    for (const auto &param : method.parameters) {
      symbolTable.insert(param.value, block->addArgument(ty));
    }

    builder.setInsertionPointToStart(block);

    for (const auto &expr : method.body) {
      irGenStatement(*expr);
    }
  }

  mlir::omtalk::KlassOp irGen(const Klass &klass) {

    auto location = loc(klass.location);
    std::string name = klass.name.value;
    std::string super = "Object";
    if (klass.super) {
      super = klass.super->value;
    }

    auto klassOp = builder.create<mlir::omtalk::KlassOp>(location, name, super);

    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      auto region = &klassOp.body();
      auto block = builder.createBlock(region);
      builder.setInsertionPointToStart(block);

      if (klass.fields) {
        for (const auto &field : (*klass.fields).elements) {
          builder.create<mlir::omtalk::FieldOp>(loc(field.location),
                                                field.value);
        }
      }

      for (const auto &method : klass.methods) {
        auto methodOp = irGen(*method);
      }

      builder.create<mlir::omtalk::KlassEndOp>(loc(klass.location));
    }

    return klassOp;
  }

  mlir::ModuleOp irGen(Module &module) {

    moduleOp = mlir::ModuleOp::create(builder.getUnknownLoc());

    for (const auto &klass : module.klasses) {
      moduleOp.push_back(irGen(*klass));
    }

    if (mlir::failed(mlir::verify(moduleOp))) {
      moduleOp.emitError("module verification error");
      return nullptr;
    }

    return moduleOp;
  }

private:
  mlir::ModuleOp moduleOp;
  mlir::OpBuilder builder;
  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;
};

} // namespace

mlir::OwningModuleRef omtalk::irgen::irGen(mlir::MLIRContext &context,
                                           parser::Module &module) {

  IRGen irGen(context);
  return irGen.irGen(module);
}
