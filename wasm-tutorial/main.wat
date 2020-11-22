(module
 (type $FUNCSIG$vii (func (param i32 i32)))
 (type $FUNCSIG$vi (func (param i32)))
 (import "env" "numLog" (func $numLog (param i32)))
 (import "env" "strLog" (func $strLog (param i32 i32)))
 (import "env" "memory" (memory $0 2))
 (table 0 anyfunc)
 (data (i32.const 16) "Hello from C!\00")
 (export "memory" (memory $0))
 (export "main" (func $main))
 (export "greet" (func $greet))
 (export "getDoubleNumber" (func $getDoubleNumber))
 (func $main (; 2 ;) (result i32)
  (i32.const 42)
 )
 (func $greet (; 3 ;)
  (call $strLog
   (i32.const 16)
   (i32.const 13)
  )
 )
 (func $getDoubleNumber (; 4 ;) (param $0 i32)
  (call $numLog
   (i32.shl
    (get_local $0)
    (i32.const 1)
   )
  )
 )


  (func $add (param $lhs i32) (param $rhs i32) (result i32)
    get_local $lhs
    get_local $rhs
    i32.add)
  (export "add" (func $add))
)