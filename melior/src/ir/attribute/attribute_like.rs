use crate::{
    ir::{r#type::TypeId, Type},
    ContextRef,
};
use melior_macro::attribute_check_functions;
use mlir_sys::{
    mlirAttributeDump, mlirAttributeGetContext, mlirAttributeGetType, mlirAttributeGetTypeID,
    MlirAttribute,
};

/// Trait for attribute-like types.
pub trait AttributeLike<'c> {
    /// Converts a attribute into a raw object.
    fn to_raw(&self) -> MlirAttribute;

    /// Returns a context.
    fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirAttributeGetContext(self.to_raw())) }
    }

    /// Returns a type.
    fn r#type(&self) -> Type {
        unsafe { Type::from_raw(mlirAttributeGetType(self.to_raw())) }
    }

    /// Returns a type ID.
    fn type_id(&self) -> TypeId<'c> {
        unsafe { TypeId::from_raw(mlirAttributeGetTypeID(self.to_raw())) }
    }

    /// Dumps a attribute.
    fn dump(&self) {
        unsafe { mlirAttributeDump(self.to_raw()) }
    }

    attribute_check_functions!(
        mlirAttributeIsALocation,
        mlirAttributeIsAAffineMap,
        mlirAttributeIsAArray,
        mlirAttributeIsADictionary,
        mlirAttributeIsAFloat,
        mlirAttributeIsAInteger,
        mlirAttributeIsABool,
        mlirAttributeIsAIntegerSet,
        mlirAttributeIsAOpaque,
        mlirAttributeIsAString,
        mlirAttributeIsASymbolRef,
        mlirAttributeIsAFlatSymbolRef,
        mlirAttributeIsAType,
        mlirAttributeIsAUnit,
        mlirAttributeIsAElements,
        mlirAttributeIsADenseBoolArray,
        mlirAttributeIsADenseI8Array,
        mlirAttributeIsADenseI16Array,
        mlirAttributeIsADenseI32Array,
        mlirAttributeIsADenseI64Array,
        mlirAttributeIsADenseF32Array,
        mlirAttributeIsADenseF64Array,
        mlirAttributeIsADenseElements,
        mlirAttributeIsADenseIntElements,
        mlirAttributeIsADenseFPElements,
        mlirAttributeIsADenseResourceElements,
        mlirAttributeIsASparseElements,
        mlirAttributeIsAStridedLayout,
        mlirAttributeIsAGPUObjectAttr,
        mlirAttributeIsASparseTensorEncodingAttr,
    );
}
