//! Skills registry: load skills from `.agent/skills/*.toml`, activate per-turn.
//!
//! Built-in skills: `code-review` (trigger-driven), `hinglish-mode` (config-driven,
//! activates every turn when `lang == LanguageMode::Hinglish`).
//! Compile regexes once at load (TASK-7.2 context guard).

use std::path::{Path, PathBuf};

use agent_types::{AgentError, LanguageMode, Result};
use regex::Regex;

/// A skill provides a turn-scoped system prompt fragment.
pub trait Skill: Send + Sync {
    fn name(&self) -> &str;
    fn system_prompt_fragment(&self) -> &str;
    /// Check if this skill should activate for the given message + language mode.
    fn should_activate(&self, user_msg: &str, lang: LanguageMode) -> bool;
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SkillConfig {
    name: String,
    triggers: Vec<String>,
    prompt: String,
    /// Explicit opt-in to replace an already-loaded skill of the same name.
    ///
    /// Without this, a duplicate name is an error rather than a silent shadow.
    #[serde(default)]
    override_existing: bool,
}

/// One workspace skill that could not be loaded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillLoadError {
    /// File the problem was found in.
    pub path: PathBuf,
    /// Actionable reason, including line, column, and field where available.
    pub reason: String,
}

impl std::fmt::Display for SkillLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.path.display(), self.reason)
    }
}

/// A user-defined skill loaded from TOML.
struct TomlSkill {
    config: SkillConfig,
    trigger_regexes: Vec<Regex>,
    trigger_substrings: Vec<String>,
}

impl Skill for TomlSkill {
    fn name(&self) -> &str {
        &self.config.name
    }
    fn system_prompt_fragment(&self) -> &str {
        &self.config.prompt
    }
    fn should_activate(&self, user_msg: &str, _lang: LanguageMode) -> bool {
        let msg_lower = user_msg.to_lowercase();
        self.trigger_substrings
            .iter()
            .any(|s| msg_lower.contains(s))
            || self.trigger_regexes.iter().any(|r| r.is_match(user_msg))
    }
}

/// Built-in code-review skill.
struct CodeReviewSkill;

impl Skill for CodeReviewSkill {
    fn name(&self) -> &str {
        "code-review"
    }
    fn system_prompt_fragment(&self) -> &str {
        "You are performing a code review. Evaluate: correctness, performance, \
         security, readability, test coverage. Provide specific line references."
    }
    fn should_activate(&self, user_msg: &str, _lang: LanguageMode) -> bool {
        let lower = user_msg.to_lowercase();
        lower.contains("review") || lower.contains("code review")
    }
}

/// Built-in hinglish-mode skill: config-driven, activates every turn in Hinglish mode.
struct HinglishSkill;

impl Skill for HinglishSkill {
    fn name(&self) -> &str {
        "hinglish-mode"
    }
    fn system_prompt_fragment(&self) -> &str {
        "LANGUAGE MODE: Hinglish (Hindi written in the English/Latin alphabet — NEVER Devanagari script).\n\
         Reason, plan, and explain concepts EXCLUSIVELY in Hinglish.\n\n\
         LANG-GUARD RULE (non-negotiable):\n\
         All code blocks, variable/function/type names, tool calls, JSON keys and schemas,\n\
         file paths, and shell commands MUST remain strictly English/ASCII.\n\
         Hinglish is ONLY for prose: explanations, reasoning, summaries.\n\
         A single Devanagari codepoint in machine surfaces = immediate rejection."
    }
    fn should_activate(&self, _user_msg: &str, lang: LanguageMode) -> bool {
        lang == LanguageMode::Hinglish
    }
}

pub struct SkillRegistry {
    skills: Vec<Box<dyn Skill>>,
}

impl SkillRegistry {
    /// Load built-ins plus workspace skills, failing on any invalid input.
    ///
    /// Use [`load_with_diagnostics`](Self::load_with_diagnostics) in interactive
    /// paths, where a malformed workspace file must not remove the built-ins.
    pub fn load(skills_dir: Option<&Path>) -> Result<Self> {
        let (registry, errors) = Self::load_with_diagnostics(skills_dir);
        if let Some(first) = errors.first() {
            return Err(AgentError::Tool {
                name: "skills".into(),
                reason: first.to_string(),
            });
        }
        Ok(registry)
    }

    /// Load built-ins first, then workspace skills, reporting per-file problems.
    ///
    /// Built-ins are always present, so malformed workspace input degrades one
    /// file rather than disabling skills entirely. A duplicate name is reported
    /// instead of silently shadowing, unless the file sets
    /// `override_existing = true`.
    pub fn load_with_diagnostics(skills_dir: Option<&Path>) -> (Self, Vec<SkillLoadError>) {
        let mut skills: Vec<Box<dyn Skill>> =
            vec![Box::new(CodeReviewSkill), Box::new(HinglishSkill)];
        let mut errors: Vec<SkillLoadError> = Vec::new();

        let Some(dir) = skills_dir else {
            return (Self { skills }, errors);
        };
        if !dir.is_dir() {
            return (Self { skills }, errors);
        }

        let mut paths = match std::fs::read_dir(dir) {
            Ok(entries) => entries
                .flatten()
                .map(|entry| entry.path())
                .filter(|path| {
                    path.is_file() && path.extension().and_then(|ext| ext.to_str()) == Some("toml")
                })
                .collect::<Vec<_>>(),
            Err(error) => {
                errors.push(SkillLoadError {
                    path: dir.to_path_buf(),
                    reason: format!("cannot read skills directory: {error}"),
                });
                return (Self { skills }, errors);
            }
        };
        // Deterministic order so duplicate resolution does not depend on the
        // filesystem's directory ordering.
        paths.sort();

        for path in paths {
            let content = match std::fs::read_to_string(&path) {
                Ok(content) => content,
                Err(error) => {
                    errors.push(SkillLoadError {
                        path,
                        reason: format!("cannot read file: {error}"),
                    });
                    continue;
                }
            };

            // `toml`'s error already names the line, column, and offending field.
            let config: SkillConfig = match toml::from_str(&content) {
                Ok(config) => config,
                Err(error) => {
                    errors.push(SkillLoadError {
                        path,
                        reason: error.to_string(),
                    });
                    continue;
                }
            };

            if config.name.trim().is_empty() {
                errors.push(SkillLoadError {
                    path,
                    reason: "field 'name' must not be empty".into(),
                });
                continue;
            }

            let existing = skills
                .iter()
                .position(|skill| skill.name() == config.name.as_str());
            if let Some(index) = existing {
                if !config.override_existing {
                    errors.push(SkillLoadError {
                        path,
                        reason: format!(
                            "duplicate skill name '{}' would shadow an already loaded skill; \
                             set 'override_existing = true' to replace it intentionally",
                            config.name
                        ),
                    });
                    continue;
                }
                match compile_toml_skill(config) {
                    Ok(skill) => skills[index] = Box::new(skill),
                    Err(error) => errors.push(SkillLoadError {
                        path,
                        reason: error.to_string(),
                    }),
                }
                continue;
            }

            match compile_toml_skill(config) {
                Ok(skill) => skills.push(Box::new(skill)),
                Err(error) => errors.push(SkillLoadError {
                    path,
                    reason: error.to_string(),
                }),
            }
        }

        (Self { skills }, errors)
    }

    /// Activate skills for the current turn. Returns system prompt fragments to append.
    pub fn activate(&self, user_msg: &str, lang: LanguageMode) -> Vec<&str> {
        self.skills
            .iter()
            .filter(|s| s.should_activate(user_msg, lang))
            .map(|s| s.system_prompt_fragment())
            .collect()
    }
}

fn compile_toml_skill(cfg: SkillConfig) -> Result<TomlSkill> {
    let mut regexes = Vec::new();
    let mut substrings = Vec::new();

    for trigger in &cfg.triggers {
        if let Some(pattern) = trigger.strip_prefix("regex:") {
            let re = Regex::new(pattern).map_err(|e| AgentError::Tool {
                name: "skills".into(),
                reason: format!("bad trigger regex '{}': {e}", pattern),
            })?;
            regexes.push(re);
        } else {
            substrings.push(trigger.to_lowercase());
        }
    }

    Ok(TomlSkill {
        config: cfg,
        trigger_regexes: regexes,
        trigger_substrings: substrings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_review_activates_on_review_keyword() {
        let reg = SkillRegistry::load(None).unwrap();
        let fragments = reg.activate("please do a code review of src/", LanguageMode::En);
        assert!(fragments.iter().any(|f| f.contains("code review")));
    }

    #[test]
    fn hinglish_activates_only_in_hinglish_mode() {
        let reg = SkillRegistry::load(None).unwrap();
        let frags_en = reg.activate("hello", LanguageMode::En);
        assert!(!frags_en.iter().any(|f| f.contains("Hinglish")));

        let frags_hi = reg.activate("hello", LanguageMode::Hinglish);
        assert!(frags_hi.iter().any(|f| f.contains("Hinglish")));
    }

    #[test]
    fn hinglish_contains_lang_guard_rule() {
        let reg = SkillRegistry::load(None).unwrap();
        let frags = reg.activate("anything", LanguageMode::Hinglish);
        let hinglish_frag = frags.iter().find(|f| f.contains("Hinglish")).unwrap();
        assert!(hinglish_frag.contains("LANG-GUARD RULE"));
        assert!(hinglish_frag.contains("Devanagari"));
    }

    fn write_skill(dir: &Path, file: &str, body: &str) {
        std::fs::write(dir.join(file), body).unwrap();
    }

    #[test]
    fn malformed_workspace_skill_preserves_builtins_and_reports_location() {
        // **Validates: Requirements 2.16, 3.6**
        let dir = tempfile::tempdir().unwrap();
        write_skill(dir.path(), "broken.toml", "name = [not valid TOML");
        write_skill(
            dir.path(),
            "good.toml",
            "name = \"valid-skill\"\ntriggers = [\"activate-me\"]\nprompt = \"do the thing\"\n",
        );

        let (registry, errors) = SkillRegistry::load_with_diagnostics(Some(dir.path()));

        // Built-ins survive a malformed file, and the valid skill still loads.
        assert!(registry
            .activate("anything", LanguageMode::Hinglish)
            .iter()
            .any(|fragment| fragment.contains("Hinglish")));
        assert!(registry
            .activate("activate-me", LanguageMode::En)
            .iter()
            .any(|fragment| fragment.contains("do the thing")));

        assert_eq!(errors.len(), 1);
        assert!(errors[0].path.ends_with("broken.toml"));
        // The `toml` error names the line and column of the failure.
        assert!(errors[0].reason.contains("line"));

        // The strict entry point still fails for callers that require it.
        assert!(SkillRegistry::load(Some(dir.path())).is_err());
    }

    #[test]
    fn duplicate_skill_name_is_reported_instead_of_silently_shadowing() {
        // **Validates: Requirements 2.16**
        let dir = tempfile::tempdir().unwrap();
        write_skill(
            dir.path(),
            "shadow.toml",
            "name = \"code-review\"\ntriggers = [\"review\"]\nprompt = \"HIJACKED\"\n",
        );

        let (registry, errors) = SkillRegistry::load_with_diagnostics(Some(dir.path()));
        assert_eq!(errors.len(), 1);
        assert!(errors[0].reason.contains("duplicate skill name"));
        assert!(errors[0].reason.contains("override_existing"));

        // The built-in is intact, not replaced.
        let fragments = registry.activate("please review this", LanguageMode::En);
        assert!(fragments.iter().any(|f| f.contains("code review")));
        assert!(!fragments.iter().any(|f| f.contains("HIJACKED")));
    }

    #[test]
    fn explicit_override_replaces_the_existing_skill() {
        // **Validates: Requirements 2.16**
        let dir = tempfile::tempdir().unwrap();
        write_skill(
            dir.path(),
            "override.toml",
            "name = \"code-review\"\ntriggers = [\"review\"]\nprompt = \"CUSTOM REVIEW\"\n\
             override_existing = true\n",
        );

        let (registry, errors) = SkillRegistry::load_with_diagnostics(Some(dir.path()));
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");

        let fragments = registry.activate("please review this", LanguageMode::En);
        assert!(fragments.iter().any(|f| f.contains("CUSTOM REVIEW")));
        // Replacement, not duplication.
        assert_eq!(fragments.len(), 1);
    }

    #[test]
    fn unknown_field_and_bad_regex_are_actionable_not_panics() {
        // **Validates: Requirements 2.16**
        let dir = tempfile::tempdir().unwrap();
        write_skill(
            dir.path(),
            "a-unknown.toml",
            "name = \"x\"\ntriggers = []\nprompt = \"p\"\nunexpected_field = 1\n",
        );
        write_skill(
            dir.path(),
            "b-regex.toml",
            "name = \"y\"\ntriggers = [\"regex:([unclosed\"]\nprompt = \"p\"\n",
        );

        let (registry, errors) = SkillRegistry::load_with_diagnostics(Some(dir.path()));
        assert_eq!(errors.len(), 2);
        assert!(errors[0].reason.contains("unexpected_field"));
        assert!(errors[1].reason.contains("trigger regex"));
        // Built-ins remain usable.
        assert!(!registry
            .activate("anything", LanguageMode::Hinglish)
            .is_empty());
    }

    #[test]
    fn missing_directory_is_not_an_error() {
        // **Validates: Requirements 3.6**
        let dir = tempfile::tempdir().unwrap();
        let (registry, errors) =
            SkillRegistry::load_with_diagnostics(Some(&dir.path().join("absent")));
        assert!(errors.is_empty());
        assert!(!registry
            .activate("anything", LanguageMode::Hinglish)
            .is_empty());
    }

    #[test]
    fn no_skills_activate_for_unrelated_message() {
        let reg = SkillRegistry::load(None).unwrap();
        let frags = reg.activate("build the project", LanguageMode::En);
        assert!(frags.is_empty());
    }
}
