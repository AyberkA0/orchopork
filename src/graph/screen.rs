//! Whole-screen capture for agents that need to see a desktop app (or
//! anything else on screen). Off unless `limits.allow_screen_capture` is on:
//! the screen shows whatever the user has open.

use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use tokio::process::Command;

/// Captures every monitor into a PNG at `path`, scaled down to at most
/// 1920 px wide so vision models accept it.
pub async fn capture(path: &Path) -> Result<(), String> {
    let out = path.to_string_lossy().to_string();
    // Success is judged by the file appearing: an older shot of the same
    // name must not make a failed capture look like a good one.
    let _ = std::fs::remove_file(path);
    let attempts: Vec<(String, Vec<String>)> = if cfg!(windows) {
        let script = format!(
            r#"$ErrorActionPreference = 'Stop'
Add-Type -Name Dpi -Namespace W -MemberDefinition '[DllImport("user32.dll")] public static extern bool SetProcessDPIAware();'
[W.Dpi]::SetProcessDPIAware() | Out-Null
Add-Type -AssemblyName System.Windows.Forms, System.Drawing
$b = [System.Windows.Forms.SystemInformation]::VirtualScreen
$bmp = New-Object System.Drawing.Bitmap $b.Width, $b.Height
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($b.Left, $b.Top, 0, 0, $bmp.Size)
$w = [Math]::Min(1920, $b.Width); $h = [int]($b.Height * $w / $b.Width)
$small = New-Object System.Drawing.Bitmap $bmp, $w, $h
$small.Save('{}', [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose(); $bmp.Dispose(); $small.Dispose()"#,
            out.replace('\'', "''")
        );
        vec![("powershell".into(), vec!["-NoProfile".into(), "-NonInteractive".into(), "-Command".into(), script])]
    } else if cfg!(target_os = "macos") {
        vec![("screencapture".into(), vec!["-x".into(), out.clone()])]
    } else {
        vec![
            ("grim".into(), vec![out.clone()]),
            ("import".into(), vec!["-window".into(), "root".into(), out.clone()]),
            ("gnome-screenshot".into(), vec!["-f".into(), out.clone()]),
            ("scrot".into(), vec!["-o".into(), out.clone()]),
        ]
    };
    let mut errors = Vec::new();
    for (exe, args) in attempts {
        let run = Command::new(&exe).args(&args).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::piped()).kill_on_drop(true).output();
        match tokio::time::timeout(Duration::from_secs(30), run).await {
            Ok(Ok(o)) if o.status.success() && path.is_file() => {
                if cfg!(target_os = "macos") {
                    let _ = Command::new("sips").args(["-Z", "1920", &out]).stdout(Stdio::null()).stderr(Stdio::null()).status().await;
                }
                return Ok(());
            }
            Ok(Ok(o)) => errors.push(format!("{exe}: {}", String::from_utf8_lossy(&o.stderr).trim())),
            Ok(Err(e)) => errors.push(format!("{exe}: {e}")),
            Err(_) => errors.push(format!("{exe}: timed out")),
        }
    }
    Err(format!("could not capture the screen ({})", errors.join("; ")))
}

#[cfg(test)]
mod tests {
    /// Captures the real screen: `cargo test screen_capture -- --ignored`.
    /// Only checks that a PNG came out; the file is deleted right away.
    #[tokio::test]
    #[ignore]
    async fn screen_capture_writes_a_png() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("screen.png");
        super::capture(&p).await.unwrap();
        let bytes = std::fs::read(&p).unwrap();
        assert_eq!(&bytes[1..4], b"PNG");
        let w = u32::from_be_bytes(bytes[16..20].try_into().unwrap());
        let h = u32::from_be_bytes(bytes[20..24].try_into().unwrap());
        assert!(w > 0 && w <= 1920 && h > 0, "{w}x{h}");
        eprintln!("captured {w}x{h}, {} KB", bytes.len() / 1024);
    }
}
