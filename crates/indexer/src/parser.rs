//! Tree-sitter parsing + entity extraction.
//!
//! Traversal is ITERATIVE via a [`TreeCursor`] (`goto_first_child` /
//! `goto_next_sibling` / `goto_parent`) — recursion would overflow the stack
//! on minified/generated files (TASK-2.1 guard). Byte ranges are byte offsets,
//! never char offsets, so every slice goes through `source.get(a..b)` rather
//! than `&source[a..b]` (which panics on non-UTF-8 boundaries). The `Tree` is
//! kept alive for the whole traversal so borrowed `Node`s stay valid.

use std::path::Path;

use agent_types::{AgentError, Result};
use tree_sitter::Node;

use crate::entity::{stable_id, CodeEntity, EntityKind};
use crate::scope_tree::{build_qualified_name, is_type_scope, ScopeFrame};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Language {
    Rust,
    Python,
    TypeScript,
}

impl Language {
    /// Detect a supported language by file extension.
    pub fn from_path(path: &Path) -> Option<Self> {
        match path.extension().and_then(|e| e.to_str()) {
            Some("rs") => Some(Language::Rust),
            Some("py" | "pyi") => Some(Language::Python),
            Some("ts" | "tsx" | "mts" | "cts") => Some(Language::TypeScript),
            _ => None,
        }
    }

    fn ts_language(self) -> tree_sitter::Language {
        match self {
            Language::Rust => tree_sitter_rust::language(),
            Language::Python => tree_sitter_python::language(),
            Language::TypeScript => tree_sitter_typescript::language_typescript(),
        }
    }
}

/// Parse `source` (whose kind is inferred from `path`) into a flat list of
/// [`CodeEntity`]s carrying scope-tree parent edges. Unsupported file types
/// yield an empty list rather than an error.
pub fn parse(path: &Path, source: &str) -> Result<Vec<CodeEntity>> {
    let lang = match Language::from_path(path) {
        Some(l) => l,
        None => return Ok(Vec::new()),
    };

    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&lang.ts_language())
        .map_err(|e| AgentError::Index(format!("set_language: {e}")))?;
    let tree = parser
        .parse(source, None)
        .ok_or_else(|| AgentError::Index("tree-sitter returned no tree".to_string()))?;

    let mut entities: Vec<CodeEntity> = Vec::new();
    let mut scopes: Vec<ScopeFrame> = Vec::new();

    // Iterative pre-order walk.
    let mut cursor = tree.walk();
    loop {
        let node = cursor.node();
        let start = node.start_byte();

        // Leaving any scope whose extent ends at or before this node.
        while scopes.last().map(|s| s.end_byte <= start).unwrap_or(false) {
            scopes.pop();
        }

        if let Some((kind, is_scope)) = classify(lang, node.kind()) {
            if let Some(name) = entity_name(lang, node, source) {
                let effective_kind = refine_kind(kind, &scopes);
                let qualified_name = build_qualified_name(&scopes, &name);
                let id = stable_id(path, effective_kind, &qualified_name);
                let (sb, eb) = (node.start_byte(), node.end_byte());
                let entity = CodeEntity {
                    id,
                    kind: effective_kind,
                    qualified_name,
                    signature: signature_of(node, source),
                    docstring: docstring_of(node, source),
                    path: path.to_path_buf(),
                    byte_range: (sb, eb),
                    line_range: (
                        node.start_position().row as u32 + 1,
                        node.end_position().row as u32 + 1,
                    ),
                    parent_id: scopes.last().map(|s| s.entity_id),
                };
                if is_scope {
                    scopes.push(ScopeFrame {
                        entity_id: entity.id,
                        end_byte: eb,
                        name,
                        kind: effective_kind,
                    });
                }
                entities.push(entity);
            }
        }

        // Advance the cursor in pre-order.
        if cursor.goto_first_child() {
            continue;
        }
        loop {
            if cursor.goto_next_sibling() {
                break;
            }
            if !cursor.goto_parent() {
                return Ok(entities);
            }
        }
    }
}

/// Map a node kind to (entity kind, introduces-scope?) for the language.
fn classify(lang: Language, kind: &str) -> Option<(EntityKind, bool)> {
    match lang {
        Language::Rust => match kind {
            "function_item" => Some((EntityKind::Function, true)),
            "struct_item" => Some((EntityKind::Struct, true)),
            "enum_item" => Some((EntityKind::Enum, true)),
            "trait_item" => Some((EntityKind::Trait, true)),
            "impl_item" => Some((EntityKind::Struct, true)),
            "mod_item" => Some((EntityKind::Module, true)),
            _ => None,
        },
        Language::Python => match kind {
            "function_definition" => Some((EntityKind::Function, true)),
            "class_definition" => Some((EntityKind::Class, true)),
            _ => None,
        },
        Language::TypeScript => match kind {
            "function_declaration" => Some((EntityKind::Function, true)),
            "class_declaration" | "abstract_class_declaration" => Some((EntityKind::Class, true)),
            "method_definition" => Some((EntityKind::Method, true)),
            "interface_declaration" => Some((EntityKind::Trait, true)),
            "enum_declaration" => Some((EntityKind::Enum, true)),
            _ => None,
        },
    }
}

/// A free function directly inside a type scope is really a method.
fn refine_kind(kind: EntityKind, scopes: &[ScopeFrame]) -> EntityKind {
    if kind == EntityKind::Function {
        if let Some(top) = scopes.last() {
            if is_type_scope(top.kind) {
                return EntityKind::Method;
            }
        }
    }
    kind
}

/// Extract the declared name of a definition node.
fn entity_name(lang: Language, node: Node, source: &str) -> Option<String> {
    // Rust `impl Foo` uses the `type` field, not `name`.
    if lang == Language::Rust && node.kind() == "impl_item" {
        let ty = node.child_by_field_name("type")?;
        return slice(source, ty.start_byte(), ty.end_byte())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
    }
    let name_node = node.child_by_field_name("name")?;
    slice(source, name_node.start_byte(), name_node.end_byte())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Signature = source from the node start up to the body-opening brace/colon.
fn signature_of(node: Node, source: &str) -> String {
    let start = node.start_byte();
    let end = match node.child_by_field_name("body") {
        Some(body) => body.start_byte(),
        None => node.end_byte(),
    };
    let end = end.max(start);
    slice(source, start, end)
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

/// Docstring = the run of comment nodes immediately preceding the definition.
fn docstring_of(node: Node, source: &str) -> Option<String> {
    let mut comments: Vec<String> = Vec::new();
    let mut prev = node.prev_sibling();
    while let Some(p) = prev {
        if p.kind().contains("comment") {
            if let Some(text) = slice(source, p.start_byte(), p.end_byte()) {
                comments.push(text.trim().to_string());
            }
            prev = p.prev_sibling();
        } else {
            break;
        }
    }
    if comments.is_empty() {
        return None;
    }
    comments.reverse();
    Some(comments.join("\n"))
}

/// Safe byte-range slice — returns `None` on a non-UTF-8 boundary instead of
/// panicking.
fn slice(source: &str, a: usize, b: usize) -> Option<&str> {
    source.get(a..b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn rust_impl_method_has_qualified_name_and_parent() {
        let src = "impl Foo { fn bar() {} }";
        let path = PathBuf::from("lib.rs");
        let entities = parse(&path, src).unwrap();

        let foo = entities
            .iter()
            .find(|e| e.qualified_name == "Foo")
            .expect("Foo entity present");
        let bar = entities
            .iter()
            .find(|e| e.qualified_name == "Foo::bar")
            .expect("Foo::bar entity present");

        assert_eq!(bar.kind, EntityKind::Method);
        assert_eq!(bar.parent_id, Some(foo.id));
    }

    #[test]
    fn rust_free_function_is_function_not_method() {
        let src = "fn top_level() {}";
        let entities = parse(&PathBuf::from("m.rs"), src).unwrap();
        let f = entities
            .iter()
            .find(|e| e.qualified_name == "top_level")
            .unwrap();
        assert_eq!(f.kind, EntityKind::Function);
        assert_eq!(f.parent_id, None);
        assert!(f.signature.starts_with("fn top_level"));
    }

    #[test]
    fn rust_docstring_from_preceding_comments() {
        let src = "// hello doc\nfn documented() {}";
        let entities = parse(&PathBuf::from("d.rs"), src).unwrap();
        let f = entities
            .iter()
            .find(|e| e.qualified_name == "documented")
            .unwrap();
        assert_eq!(f.docstring.as_deref(), Some("// hello doc"));
    }

    #[test]
    fn python_method_inside_class() {
        let src = "class Animal:\n    def speak(self):\n        pass\n";
        let entities = parse(&PathBuf::from("a.py"), src).unwrap();
        let m = entities
            .iter()
            .find(|e| e.qualified_name == "Animal::speak")
            .expect("Animal::speak present");
        assert_eq!(m.kind, EntityKind::Method);
    }

    #[test]
    fn unsupported_extension_is_empty() {
        let entities = parse(&PathBuf::from("notes.txt"), "hello").unwrap();
        assert!(entities.is_empty());
    }
}
