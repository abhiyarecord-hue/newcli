//! Localization mode. Hinglish = Hindi written in the Latin/English alphabet.
//! Pure Devanagari is intentionally unsupported (plan.md section 5).
//! Signature verbatim from plan.md section 3.

#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LanguageMode {
    #[default]
    En,
    Hinglish,
}
