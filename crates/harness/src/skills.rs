//! Skills registry: load skills from `.agent/skills/*.toml`, activate per-turn.
//!
//! Built-in skills: `code-review` (trigger-driven), `hinglish-mode` (config-driven,
//! activates every turn when `lang == LanguageMode::Hinglish`).
//! Compile regexes once at load (TASK-7.2 context guard).

use std::path::Path;

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
    /// Load skills from `.agent/skills/*.toml` + built-ins.
    pub fn load(skills_dir: Option<&Path>) -> Result<Self> {
        let mut skills: Vec<Box<dyn Skill>> = Vec::new();

        // Built-ins.
        skills.push(Box::new(CodeReviewSkill));
        skills.push(Box::new(HinglishSkill));

        // User-defined from TOML.
        if let Some(dir) = skills_dir {
            if dir.is_dir() {
                let entries = std::fs::read_dir(dir)
                    .map_err(|e| AgentError::Tool {
                        name: "skills".into(),
                        reason: format!("read skills dir: {e}"),
                    })?;
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().and_then(|e| e.to_str()) == Some("toml") {
                        let content = std::fs::read_to_string(&path).map_err(|e| {
                            AgentError::Tool {
                                name: "skills".into(),
                                reason: format!("read {}: {e}", path.display()),
                            }
                        })?;
                        let cfg: SkillConfig = toml::from_str(&content).map_err(|e| {
                            AgentError::Tool {
                                name: "skills".into(),
                                reason: format!("parse {}: {e}", path.display()),
                            }
                        })?;
                        let skill = compile_toml_skill(cfg)?;
                        skills.push(Box::new(skill));
                    }
                }
            }
        }

        Ok(Self { skills })
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

    #[test]
    fn no_skills_activate_for_unrelated_message() {
        let reg = SkillRegistry::load(None).unwrap();
        let frags = reg.activate("build the project", LanguageMode::En);
        assert!(frags.is_empty());
    }
}
