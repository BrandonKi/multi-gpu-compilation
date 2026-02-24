// RUN: polygeist-opt --convert-polygeist-to-llvm %s | FileCheck %s

module {
  func.func @f() {
    "polygeist.alternatives"() ({
      "polygeist.polygeist_yield"() : () -> ()
    }) : () -> ()
    return
  }
}

// CHECK-NOT: polygeist.alternatives
