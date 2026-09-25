use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::model::Skill;
use crate::error::{Error, Result};

/// Bundled defaults, embedded in the binary.
pub const BUNDLED: &[(&str, &str)] = &[
    ("anti-sycophancy-terse.yaml", include_str!("../../assets/skills/anti-sycophancy-terse.yaml")),
    ("explain-like-principal.yaml", include_str!("../../assets/skills/explain-like-principal.yaml")),
    ("test-driven-loop.yaml", include_str!("../../assets/skills/test-driven-loop.yaml")),
    ("markdown-memory-sync.yaml", include_str!("../../assets/skills/markdown-memory-sync.yaml")),
];

#[derive(Debug, Default)]
pub struct LoadReport {
    pub skills: BTreeMap<String, Skill>,
    /// Bad files are reported, never fatal: one broken custom skill must not
    /// take down hot-reload for the rest.
    pub errors: Vec<(PathBuf, String)>,
}

/// Write bundled skills that are not on disk yet (never overwrites user edits).
pub fn install_bundled(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    for (file, body) in BUNDLED {
        let p = dir.join(file);
        if !p.exists() {
            std::fs::write(p, body)?;
        }
    }
    Ok(())
}

pub fn load_dir(dir: &Path) -> Result<LoadReport> {
    let mut report = LoadReport::default();
    let mut files: Vec<PathBuf> =
        std::fs::read_dir(dir)?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_file()).collect();
    files.sort();
    for path in files {
        if !matches!(path.extension().and_then(|e| e.to_str()), Some("yaml" | "yml" | "md")) {
            continue;
        }
        let parsed = std::fs::read_to_string(&path).map_err(Error::from).and_then(|text| parse_skill(&path, &text));
        match parsed {
            Ok(skill) => {
                if report.skills.contains_key(&skill.name) {
                    report.errors.push((path, format!("duplicate skill name {:?}", skill.name)));
                } else {
                    report.skills.insert(skill.name.clone(), skill);
                }
            }
            Err(e) => report.errors.push((path, e.to_string())),
        }
    }
    Ok(report)
}

/// `.yaml`/`.yml`: the skill itself. `.md`: YAML front matter, with the
/// Markdown body used as `prompt_injection` when that field is empty.
pub fn parse_skill(path: &Path, text: &str) -> Result<Skill> {
    let fail = |msg: String| Error::SkillFile { path: path.to_path_buf(), msg };
    let mut skill: Skill = match path.extension().and_then(|e| e.to_str()) {
        Some("yaml" | "yml") => serde_yaml::from_str(text).map_err(|e| fail(e.to_string()))?,
        Some("md") => {
            let (front, body) = split_front_matter(text).ok_or_else(|| fail("missing YAML front matter".into()))?;
            let mut s: Skill = serde_yaml::from_str(&front).map_err(|e| fail(e.to_string()))?;
            if s.prompt_injection.trim().is_empty() {
                s.prompt_injection = body.trim().to_string();
            }
            s
        }
        _ => return Err(fail("unsupported extension".into())),
    };
    skill.prompt_injection = skill.prompt_injection.trim_end().to_string();
    skill.validate().map_err(|e| fail(e.to_string()))?;
    Ok(skill)
}

fn split_front_matter(text: &str) -> Option<(String, String)> {
    let text = text.replace("\r\n", "\n");
    let rest = text.strip_prefix("---\n")?;
    let end = rest.find("\n---")?;
    let front = rest[..end].to_string();
    let body = rest[end + 4..].trim_start_matches(['\n', '-']).to_string();
    Some((front, body))
}
