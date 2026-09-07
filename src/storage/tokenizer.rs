//! Configurable text analysis for BM25.
//!
//! The pipeline is: split -> lowercase -> drop stopwords -> stem -> hash (FNV-1a).
//! Every stage after the split is optional, and the whole configuration travels
//! with the store: an index built with English stemming can only be queried with
//! English stemming, so a mismatch forces a rebuild rather than silently wrong
//! scores (see `AnalyzerConfig` handling in the engine).

use rust_stemmers::{Algorithm, Stemmer};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::storage::stopwords;

/// Hash a token's bytes via FNV-1a (64-bit). Same constants as `id_pool::fnv1a64`,
/// kept inline here to avoid a cross-module visibility change.
#[inline]
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// How documents and queries are turned into token hashes.
///
/// Defaults to English: stemming and the bundled English stopword list are on, so
/// "running shoes" and "run shoe" match out of the box. Set `language` to `"none"`
/// for the pre-0.9 behaviour (split + lowercase only).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalyzerConfig {
    /// Snowball language for stemming, or `"none"` to disable stemming.
    #[serde(default = "default_language")]
    pub language: String,
    /// Stopwords to drop. `None` means "use the bundled list for `language`";
    /// an explicit empty list means "keep every token".
    #[serde(default)]
    pub stopwords: Option<Vec<String>>,
    /// When true (default) tokens split on any non-alphanumeric character.
    /// When false only Unicode whitespace splits, so `foo-bar` stays one token.
    #[serde(default = "default_split_on_punctuation")]
    pub split_on_punctuation: bool,
}

fn default_language() -> String {
    "english".to_string()
}

fn default_split_on_punctuation() -> bool {
    true
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        Self {
            language: default_language(),
            stopwords: None,
            split_on_punctuation: true,
        }
    }
}

impl AnalyzerConfig {
    /// The pre-0.9 analyzer: split and lowercase, nothing else.
    pub fn plain() -> Self {
        Self {
            language: "none".to_string(),
            stopwords: Some(Vec::new()),
            split_on_punctuation: true,
        }
    }

    /// Reject a configuration naming a language we cannot stem, so the error
    /// arrives at `Database.open` instead of as silently unstemmed tokens.
    pub fn validate(&self) -> Result<(), String> {
        if algorithm_for(&self.language).is_none() && !is_no_stemming(&self.language) {
            return Err(format!(
                "unknown text language {:?}; supported: {}, or \"none\" to disable stemming",
                self.language,
                SUPPORTED_LANGUAGES.join(", ")
            ));
        }
        Ok(())
    }
}

/// Languages with a Snowball stemmer. The first five also ship a stopword list.
pub const SUPPORTED_LANGUAGES: &[&str] = &[
    "english",
    "french",
    "german",
    "spanish",
    "portuguese",
    "italian",
    "dutch",
    "swedish",
    "norwegian",
    "danish",
    "russian",
];

fn is_no_stemming(language: &str) -> bool {
    matches!(language, "none" | "" | "off")
}

fn algorithm_for(language: &str) -> Option<Algorithm> {
    Some(match language {
        "english" => Algorithm::English,
        "french" => Algorithm::French,
        "german" => Algorithm::German,
        "spanish" => Algorithm::Spanish,
        "portuguese" => Algorithm::Portuguese,
        "italian" => Algorithm::Italian,
        "dutch" => Algorithm::Dutch,
        "swedish" => Algorithm::Swedish,
        "norwegian" => Algorithm::Norwegian,
        "danish" => Algorithm::Danish,
        "russian" => Algorithm::Russian,
        _ => return None,
    })
}

/// Turns text into token hashes according to an [`AnalyzerConfig`].
///
/// Build once per index; `analyze` takes `&self` and allocates only what the
/// lowercase and stemming stages require.
pub struct TextAnalyzer {
    config: AnalyzerConfig,
    stemmer: Option<Stemmer>,
    stopwords: HashSet<String>,
}

impl TextAnalyzer {
    pub fn new(config: AnalyzerConfig) -> Self {
        let stemmer = algorithm_for(&config.language).map(Stemmer::create);
        // An explicit list always wins, including an explicit empty one; `None`
        // falls back to the bundled list for the language (empty if there is none).
        let raw: Vec<String> = match &config.stopwords {
            Some(words) => words.clone(),
            None => stopwords::for_language(&config.language)
                .map(|list| list.iter().map(|w| (*w).to_string()).collect())
                .unwrap_or_default(),
        };
        // Stopwords are matched against produced tokens, so they must be split the
        // same way documents are. Without this an entry like "aren't" could never
        // match: the tokenizer yields "aren" and "t", and both would end up indexed
        // as ordinary terms. Splitting here also lets a caller pass contractions or
        // phrases and get what they meant.
        let split_on_punctuation = config.split_on_punctuation;
        let stopwords: HashSet<String> = raw
            .iter()
            .flat_map(|word| {
                word.split(move |c: char| {
                    if split_on_punctuation {
                        !c.is_alphanumeric()
                    } else {
                        c.is_whitespace()
                    }
                })
                .filter(|piece| !piece.is_empty())
                .map(|piece| piece.to_lowercase())
                .collect::<Vec<_>>()
            })
            .collect();
        Self {
            config,
            stemmer,
            stopwords,
        }
    }

    pub fn config(&self) -> &AnalyzerConfig {
        &self.config
    }

    #[inline]
    fn is_separator(&self, c: char) -> bool {
        if self.config.split_on_punctuation {
            !c.is_alphanumeric()
        } else {
            c.is_whitespace()
        }
    }

    /// Normalise one raw token: lowercase, drop if a stopword, then stem.
    /// Returns `None` when the token carries no signal.
    fn normalize(&self, raw: &str) -> Option<String> {
        if raw.is_empty() {
            return None;
        }
        let lowered = if raw.bytes().all(|b| !b.is_ascii_uppercase() && b < 0x80) {
            raw.to_string()
        } else {
            raw.to_lowercase()
        };
        if self.stopwords.contains(&lowered) {
            return None;
        }
        match &self.stemmer {
            Some(stemmer) => Some(stemmer.stem(&lowered).into_owned()),
            None => Some(lowered),
        }
    }

    /// Tokenize `text` into FNV-1a hashes of the analysed tokens.
    ///
    /// Determinism: a pure function of the input bytes and the config, so two opens
    /// with the same documents and config rebuild identical posting lists.
    pub fn analyze(&self, text: &str) -> Vec<u64> {
        let mut out = Vec::with_capacity(text.len() / 6); // approx. avg word length
        for raw in text.split(|c: char| self.is_separator(c)) {
            if let Some(token) = self.normalize(raw) {
                out.push(fnv1a64(token.as_bytes()));
            }
        }
        out
    }

    /// Number of tokens `analyze` would produce, without hashing them.
    pub fn token_count(&self, text: &str) -> u32 {
        let mut n: u32 = 0;
        for raw in text.split(|c: char| self.is_separator(c)) {
            if self.normalize(raw).is_some() {
                n = n.saturating_add(1);
            }
        }
        n
    }
}

impl Default for TextAnalyzer {
    fn default() -> Self {
        Self::new(AnalyzerConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain() -> TextAnalyzer {
        TextAnalyzer::new(AnalyzerConfig::plain())
    }

    fn english() -> TextAnalyzer {
        TextAnalyzer::default()
    }

    // -- tokenization behaviour shared by every config ------------------------

    #[test]
    fn lowercases_ascii() {
        let a = plain();
        assert_eq!(a.analyze("Foo BAR baz"), a.analyze("foo bar baz"));
    }

    #[test]
    fn splits_on_non_alphanumeric() {
        assert_eq!(plain().analyze("hello, world! 123-abc").len(), 4);
    }

    #[test]
    fn drops_empty_tokens() {
        assert_eq!(plain().analyze("   ,,,a,,,b...").len(), 2);
    }

    #[test]
    fn unicode_words_kept() {
        // Non-ASCII alphanumerics are part of tokens, not separators, and they
        // exercise the lowercase path that pure-ASCII input skips.
        let a = plain();
        assert_eq!(a.analyze("café résumé naïve").len(), 3);
        assert_eq!(
            a.analyze("CAFÉ"),
            a.analyze("café"),
            "non-ASCII uppercase must fold to the same token"
        );
        assert_ne!(a.analyze("café"), a.analyze("cafe"));
    }

    #[test]
    fn token_count_matches_analyze_len() {
        for a in [plain(), english()] {
            for s in [
                "",
                "one",
                "one two",
                "one,, two!! three",
                "the running shoes",
            ] {
                assert_eq!(a.token_count(s) as usize, a.analyze(s).len(), "input={s:?}");
            }
        }
    }

    // -- stemming --------------------------------------------------------------

    #[test]
    fn stemming_matches_inflections() {
        // The roadmap card's acceptance criterion.
        let a = english();
        assert_eq!(a.analyze("running shoes"), a.analyze("run shoe"));
    }

    #[test]
    fn stemming_off_keeps_inflections_distinct() {
        let a = plain();
        assert_ne!(a.analyze("running shoes"), a.analyze("run shoe"));
    }

    #[test]
    fn other_languages_stem() {
        // Pairs verified against the Snowball stemmers themselves, not assumed.
        for (language, inflected, base) in [
            ("german", "kinder", "kind"),
            ("german", "häuser", "haus"),
            ("french", "chevaux", "cheval"),
            ("spanish", "corriendo", "correr"),
            ("portuguese", "livros", "livro"),
        ] {
            let a = TextAnalyzer::new(AnalyzerConfig {
                language: language.into(),
                stopwords: Some(Vec::new()),
                split_on_punctuation: true,
            });
            assert_eq!(
                a.analyze(inflected),
                a.analyze(base),
                "{language}: {inflected} should stem to {base}"
            );
        }
    }

    // -- stopwords -------------------------------------------------------------

    #[test]
    fn bundled_stopwords_are_dropped() {
        let a = english();
        assert_eq!(
            a.analyze("the quick brown fox"),
            a.analyze("quick brown fox")
        );
    }

    #[test]
    fn explicit_empty_list_keeps_stopwords() {
        let cfg = AnalyzerConfig {
            stopwords: Some(Vec::new()),
            ..Default::default()
        };
        let a = TextAnalyzer::new(cfg);
        assert_ne!(
            a.analyze("the quick brown fox"),
            a.analyze("quick brown fox")
        );
    }

    #[test]
    fn custom_stopwords_replace_the_bundled_list() {
        let cfg = AnalyzerConfig {
            stopwords: Some(vec!["quick".into()]),
            ..Default::default()
        };
        let a = TextAnalyzer::new(cfg);
        // "quick" is a stopword now, and "the" - bundled but not listed - is not.
        assert_eq!(a.analyze("the quick fox"), a.analyze("the fox"));
    }

    #[test]
    fn contraction_stopwords_match_the_tokens_they_produce() {
        // "aren't" tokenizes to "aren" + "t"; both must be dropped, or the list
        // entry is dead weight and the fragments get indexed as real terms.
        let a = english();
        assert!(a.analyze("aren't").is_empty());
        assert!(a.analyze("AREN'T").is_empty());
        assert_eq!(a.analyze("they aren't going"), a.analyze("going"));
    }

    #[test]
    fn user_supplied_contractions_are_split_too() {
        let cfg = AnalyzerConfig {
            language: "none".into(),
            stopwords: Some(vec!["don't".into()]),
            split_on_punctuation: true,
        };
        let a = TextAnalyzer::new(cfg);
        assert!(a.analyze("don't").is_empty());
        assert!(a.analyze("don").is_empty());
    }

    #[test]
    fn custom_stopwords_are_case_insensitive() {
        let cfg = AnalyzerConfig {
            stopwords: Some(vec!["Quick".into()]),
            ..Default::default()
        };
        assert!(TextAnalyzer::new(cfg).analyze("QUICK").is_empty());
    }

    // -- tokenizer split -------------------------------------------------------

    #[test]
    fn whitespace_only_split_keeps_hyphenates_together() {
        let cfg = AnalyzerConfig {
            language: "none".into(),
            stopwords: Some(Vec::new()),
            split_on_punctuation: false,
        };
        let a = TextAnalyzer::new(cfg);
        assert_eq!(a.analyze("foo-bar baz").len(), 2);
        assert_ne!(a.analyze("foo-bar"), a.analyze("foo bar"));
    }

    // -- config validation -----------------------------------------------------

    #[test]
    fn unknown_language_is_rejected() {
        let cfg = AnalyzerConfig {
            language: "klingon".into(),
            ..Default::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("klingon"), "{err}");
    }

    #[test]
    fn supported_languages_are_accepted() {
        assert!(AnalyzerConfig::plain().validate().is_ok());
        for lang in SUPPORTED_LANGUAGES {
            let cfg = AnalyzerConfig {
                language: (*lang).to_string(),
                ..Default::default()
            };
            assert!(cfg.validate().is_ok(), "{lang}");
        }
    }
}
