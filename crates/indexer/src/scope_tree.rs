//! Scope-tree bookkeeping used by the iterative parser traversal.
//!
//! A [`ScopeFrame`] is pushed when the traversal enters a naming scope
//! (impl/struct/class/trait/enum/module/function) and popped by byte-range
//! once the traversal moves past it. The frame stack yields both the
//! `parent_id` edge and the `::`-joined qualified name for nested entities.

use crate::entity::EntityKind;

pub(crate) struct ScopeFrame {
    pub entity_id: u64,
    pub end_byte: usize,
    pub name: String,
    pub kind: EntityKind,
}

/// Join enclosing scope names with the entity's own name (`Foo::bar`).
pub(crate) fn build_qualified_name(scopes: &[ScopeFrame], own: &str) -> String {
    if scopes.is_empty() {
        return own.to_string();
    }
    let mut out = String::new();
    for s in scopes {
        out.push_str(&s.name);
        out.push_str("::");
    }
    out.push_str(own);
    out
}

/// Whether a scope kind is a *type* scope — a function directly inside one is a
/// method rather than a free function.
pub(crate) fn is_type_scope(kind: EntityKind) -> bool {
    matches!(
        kind,
        EntityKind::Struct
            | EntityKind::Class
            | EntityKind::Trait
            | EntityKind::Enum
            | EntityKind::Block // impl blocks
    )
}
