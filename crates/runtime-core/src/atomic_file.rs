//! Cross-platform atomic file replacement.
//!
//! The replacement is prepared in an exclusively-created temporary file in
//! the destination directory. A successful return means the complete byte
//! slice is visible at the destination. Cancellation is checked immediately
//! before the namespace replacement; cancellation observed after that point
//! cannot turn a committed write into an ambiguous error.
//!
//! These guarantees assume a local filesystem with normal atomic rename or
//! replace behavior. Network shares and removable/FAT-like filesystems may
//! provide weaker replacement, flush, or crash-durability semantics. Unix
//! parent-directory flush is best effort because not every filesystem allows
//! directories to be opened or flushed.

use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use agent_types::{AgentError, Result};
use tokio_util::sync::CancellationToken;

/// Default number of exclusive temporary-name creation attempts.
pub const DEFAULT_TEMP_CREATE_ATTEMPTS: usize = 64;
/// Hard ceiling preventing accidentally unbounded temporary-name retries.
pub const MAX_TEMP_CREATE_ATTEMPTS: usize = 1_024;
/// Default number of transient namespace replacement attempts.
pub const DEFAULT_REPLACE_ATTEMPTS: usize = 16;
/// Hard ceiling preventing accidentally unbounded replacement retries.
pub const MAX_REPLACE_ATTEMPTS: usize = 128;

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);
// Windows ReplaceFileW can reject simultaneous replacement calls even when
// every writer owns a distinct source. Keep the commit point process-local and
// short; exclusive temporaries still provide cross-process safety.
static COMMIT_LOCK: Mutex<()> = Mutex::new(());

/// Explicit bounds and durability controls for [`atomic_replace`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AtomicWriteOptions {
    /// Number of unique names tried after `AlreadyExists` collisions.
    pub temp_create_attempts: usize,
    /// Number of attempts for transient concurrent namespace races.
    pub replace_attempts: usize,
    /// Request a best-effort parent-directory flush after commit on Unix.
    pub sync_parent: bool,
}

impl Default for AtomicWriteOptions {
    fn default() -> Self {
        Self {
            temp_create_attempts: DEFAULT_TEMP_CREATE_ATTEMPTS,
            replace_attempts: DEFAULT_REPLACE_ATTEMPTS,
            sync_parent: true,
        }
    }
}
/// Owns an exclusively-created temporary and removes it unless committed.
///
/// Cleanup runs during normal error returns and unwinding. Cleanup failure is
/// intentionally ignored in `Drop`; the operation's original error remains
/// authoritative.
#[derive(Debug)]
pub struct AtomicWriteGuard {
    path: PathBuf,
    file: Option<File>,
    committed: bool,
}

impl AtomicWriteGuard {
    fn create(destination: &Path, attempts: usize) -> io::Result<Self> {
        validate_attempts(attempts)?;
        let parent = normalized_parent(destination)?;
        let file_name = destination.file_name().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "destination has no file name")
        })?;
        let seed = unique_seed();

        for attempt in 0..attempts {
            let path = parent.join(temporary_name(file_name, seed, attempt));
            let mut options = OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            match options.open(&path) {
                Ok(file) => {
                    return Ok(Self {
                        path,
                        file: Some(file),
                        committed: false,
                    });
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }

        Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!("failed to create a unique temporary after {attempts} attempts"),
        ))
    }

    /// Path of the owned same-directory temporary file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    fn file_mut(&mut self) -> io::Result<&mut File> {
        self.file.as_mut().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::BrokenPipe,
                "temporary file is already closed",
            )
        })
    }

    fn close(&mut self) {
        drop(self.file.take());
    }

    fn mark_committed(&mut self) {
        self.committed = true;
    }
}

impl Drop for AtomicWriteGuard {
    fn drop(&mut self) {
        self.close();
        if !self.committed {
            let _ = fs::remove_file(&self.path);
        }
    }
}
/// Atomically replaces `path` with `bytes` on the destination filesystem.
///
/// The destination's parent must already exist. This function does not delete
/// destinations or create backups; callers retain their existing explicit
/// deletion, confinement, and backup policies. If preparation, cancellation,
/// or replacement fails, an existing destination is left untouched and the
/// owned temporary is removed.
pub async fn atomic_replace(
    path: impl AsRef<Path>,
    bytes: &[u8],
    options: AtomicWriteOptions,
    cancel: &CancellationToken,
) -> Result<()> {
    atomic_replace_inner(path.as_ref(), bytes, options, cancel, NoHooks)
}

/// Synchronous [`atomic_replace`] for callers outside an async context.
///
/// The replacement itself performs no async I/O, so a synchronous consumer gets
/// identical guarantees without needing a runtime or a `block_on` shim of its
/// own. Every consumer therefore shares one implementation instead of
/// re-deriving temp naming and commit ordering.
pub fn atomic_replace_blocking(
    path: impl AsRef<Path>,
    bytes: &[u8],
    options: AtomicWriteOptions,
    cancel: &CancellationToken,
) -> Result<()> {
    atomic_replace_inner(path.as_ref(), bytes, options, cancel, NoHooks)
}

trait AtomicWriteHooks {
    fn temporary_created(&mut self, _path: &Path) -> io::Result<()> {
        Ok(())
    }

    fn before_commit(&mut self, _path: &Path) -> io::Result<()> {
        Ok(())
    }

    fn after_commit(&mut self, _path: &Path) {}
}

struct NoHooks;
impl AtomicWriteHooks for NoHooks {}

fn atomic_replace_inner<H: AtomicWriteHooks>(
    destination: &Path,
    bytes: &[u8],
    options: AtomicWriteOptions,
    cancel: &CancellationToken,
    mut hooks: H,
) -> Result<()> {
    validate_options(options)?;
    if cancel.is_cancelled() {
        return Err(AgentError::Cancelled);
    }

    let mut guard = AtomicWriteGuard::create(destination, options.temp_create_attempts)?;
    hooks.temporary_created(guard.path())?;
    guard.file_mut()?.write_all(bytes)?;
    guard.file_mut()?.sync_all()?;
    guard.close();

    // The temporary is created 0600 so no reader can observe partial content.
    // Committing by rename would make that the destination's mode, silently
    // stripping bits an existing file had — an executable script would stop
    // being executable. Apply the final mode after the content is written and
    // before the commit, so the restrictive window is preserved.
    inherit_destination_mode(guard.path(), destination)?;

    hooks.before_commit(guard.path())?;
    let commit_lock = COMMIT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if cancel.is_cancelled() {
        return Err(AgentError::Cancelled);
    }

    platform_replace(guard.path(), destination, options.replace_attempts)?;
    guard.mark_committed();
    drop(commit_lock);
    hooks.after_commit(destination);

    if options.sync_parent {
        sync_parent_best_effort(destination);
    }
    Ok(())
}

fn validate_options(options: AtomicWriteOptions) -> io::Result<()> {
    validate_bound(
        "temp_create_attempts",
        options.temp_create_attempts,
        MAX_TEMP_CREATE_ATTEMPTS,
    )?;
    validate_bound(
        "replace_attempts",
        options.replace_attempts,
        MAX_REPLACE_ATTEMPTS,
    )
}

fn validate_attempts(attempts: usize) -> io::Result<()> {
    validate_bound("temp_create_attempts", attempts, MAX_TEMP_CREATE_ATTEMPTS)
}

fn validate_bound(name: &str, value: usize, maximum: usize) -> io::Result<()> {
    if (1..=maximum).contains(&value) {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{name} must be between 1 and {maximum}"),
        ))
    }
}
/// Mode applied to a newly created destination.
///
/// A create-and-write would land on `0666 & !umask`, commonly `0644`. The umask
/// has no read-only query, so this uses `0644` directly rather than taking a
/// `libc` dependency to read it. A caller needing something stricter sets it
/// after the replacement.
#[cfg(unix)]
const NEW_DESTINATION_MODE: u32 = 0o644;

/// Give the temporary the mode the destination should end up with.
#[cfg(unix)]
fn inherit_destination_mode(temporary: &Path, destination: &Path) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;

    match fs::metadata(destination) {
        // Preserve an existing file's permissions exactly.
        Ok(metadata) => fs::set_permissions(temporary, metadata.permissions()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            fs::set_permissions(temporary, fs::Permissions::from_mode(NEW_DESTINATION_MODE))
        }
        Err(error) => Err(error),
    }
}

#[cfg(not(unix))]
fn inherit_destination_mode(_temporary: &Path, _destination: &Path) -> io::Result<()> {
    // Windows has no mode bits to carry across the replacement.
    Ok(())
}

fn normalized_parent(destination: &Path) -> io::Result<&Path> {
    let parent = destination
        .parent()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "destination has no parent"))?;
    if parent.as_os_str().is_empty() {
        Ok(Path::new("."))
    } else {
        Ok(parent)
    }
}

fn unique_seed() -> u128 {
    let time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed) as u128;
    time ^ ((std::process::id() as u128) << 64) ^ sequence
}

fn temporary_name(file_name: &std::ffi::OsStr, seed: u128, attempt: usize) -> OsString {
    let mut name = OsString::from(".");
    name.push(file_name);
    name.push(format!(".atomic-{seed:032x}-{attempt:04x}.tmp"));
    name
}

#[cfg(unix)]
fn platform_replace(source: &Path, destination: &Path, _attempts: usize) -> io::Result<()> {
    fs::rename(source, destination)
}

#[cfg(windows)]
fn platform_replace(source: &Path, destination: &Path, attempts: usize) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use std::ptr;
    use std::thread;
    use std::time::Duration;
    use windows_sys::Win32::Storage::FileSystem::{
        MoveFileExW, ReplaceFileW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH,
    };

    fn wide(path: &Path) -> Vec<u16> {
        path.as_os_str().encode_wide().chain(Some(0)).collect()
    }

    fn transient(error: &io::Error) -> bool {
        // ERROR_SHARING_VIOLATION, ERROR_UNABLE_TO_REMOVE_REPLACED, and
        // ERROR_UNABLE_TO_MOVE_REPLACEMENT. All leave the source owned by the
        // cleanup guard, so a bounded retry remains unambiguous.
        matches!(error.raw_os_error(), Some(32 | 1175 | 1176))
    }

    let source = wide(source);
    let destination = wide(destination);
    let mut last_error = None;

    for attempt in 0..attempts {
        // SAFETY: Both paths are valid, NUL-terminated UTF-16 buffers that
        // remain alive for the call. Optional pointer parameters are null.
        let replaced = unsafe {
            ReplaceFileW(
                destination.as_ptr(),
                source.as_ptr(),
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
            )
        };
        if replaced != 0 {
            return Ok(());
        }

        let replace_error = io::Error::last_os_error();
        let error = if matches!(replace_error.raw_os_error(), Some(2 | 3)) {
            // ReplaceFileW requires an existing destination. MoveFileExW
            // supplies atomic creation and remains replace-capable if another
            // process wins the existence race before this call.
            // SAFETY: The same stable, NUL-terminated buffers are used here.
            let moved = unsafe {
                MoveFileExW(
                    source.as_ptr(),
                    destination.as_ptr(),
                    MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
                )
            };
            if moved != 0 {
                return Ok(());
            }
            io::Error::last_os_error()
        } else {
            replace_error
        };

        if !transient(&error) || attempt + 1 == attempts {
            return Err(error);
        }
        last_error = Some(error);
        thread::sleep(Duration::from_millis(1));
    }

    Err(last_error.unwrap_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "replacement attempts must be nonzero",
        )
    }))
}

#[cfg(not(any(unix, windows)))]
compile_error!("atomic replacement is implemented only for supported Unix and Windows targets");

#[cfg(unix)]
fn sync_parent_best_effort(destination: &Path) {
    if let Ok(parent) = normalized_parent(destination) {
        if let Ok(directory) = File::open(parent) {
            let _ = directory.sync_all();
        }
    }
}

#[cfg(windows)]
fn sync_parent_best_effort(_destination: &Path) {
    // FlushFileBuffers is performed by File::sync_all before replacement and
    // MOVEFILE_WRITE_THROUGH is used for creation. Windows does not expose the
    // Unix parent-directory fsync model through ordinary directory handles.
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::{Arc, Barrier};

    fn temp_files(directory: &Path) -> Vec<PathBuf> {
        fs::read_dir(directory)
            .expect("read test directory")
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.contains(".atomic-") && name.ends_with(".tmp"))
            })
            .collect()
    }

    struct CancelBeforeCommit(CancellationToken);
    impl AtomicWriteHooks for CancelBeforeCommit {
        fn before_commit(&mut self, _path: &Path) -> io::Result<()> {
            self.0.cancel();
            Ok(())
        }
    }

    struct CancelAfterCommit(CancellationToken);
    impl AtomicWriteHooks for CancelAfterCommit {
        fn after_commit(&mut self, _path: &Path) {
            self.0.cancel();
        }
    }

    struct FailBeforeCommit;
    impl AtomicWriteHooks for FailBeforeCommit {
        fn before_commit(&mut self, _path: &Path) -> io::Result<()> {
            Err(io::Error::other("injected commit fault"))
        }
    }

    struct CommitBarrier(Arc<Barrier>);
    impl AtomicWriteHooks for CommitBarrier {
        fn before_commit(&mut self, _path: &Path) -> io::Result<()> {
            self.0.wait();
            Ok(())
        }
    }

    // **Validates: Requirements 2.10, 3.8**
    #[tokio::test]
    async fn creates_and_replaces_only_with_complete_bytes() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let destination = directory.path().join("state.json");
        let cancel = CancellationToken::new();

        atomic_replace(
            &destination,
            b"first",
            AtomicWriteOptions::default(),
            &cancel,
        )
        .await
        .expect("create destination");
        atomic_replace(
            &destination,
            b"second",
            AtomicWriteOptions::default(),
            &cancel,
        )
        .await
        .expect("replace destination");

        assert_eq!(fs::read(&destination).expect("read destination"), b"second");
        assert!(temp_files(directory.path()).is_empty());
    }

    // **Validates: Requirements 2.10**
    #[test]
    fn cancellation_at_commit_preserves_destination_and_cleans_temporary() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let destination = directory.path().join("history.jsonl");
        fs::write(&destination, b"old").expect("write old destination");
        let cancel = CancellationToken::new();

        let result = atomic_replace_inner(
            &destination,
            b"new",
            AtomicWriteOptions::default(),
            &cancel,
            CancelBeforeCommit(cancel.clone()),
        );

        assert!(matches!(result, Err(AgentError::Cancelled)));
        assert_eq!(fs::read(&destination).expect("read destination"), b"old");
        assert!(temp_files(directory.path()).is_empty());
    }

    // **Validates: Requirements 2.10**
    #[test]
    fn cancellation_after_commit_reports_success() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let destination = directory.path().join("snapshot.json");
        fs::write(&destination, b"old").expect("write old destination");
        let cancel = CancellationToken::new();

        let result = atomic_replace_inner(
            &destination,
            b"committed",
            AtomicWriteOptions::default(),
            &cancel,
            CancelAfterCommit(cancel.clone()),
        );

        assert!(result.is_ok());
        assert!(cancel.is_cancelled());
        assert_eq!(
            fs::read(&destination).expect("read destination"),
            b"committed"
        );
    }
    // **Validates: Requirements 2.10, 3.13**
    #[test]
    fn faults_and_unwinding_remove_owned_temporaries() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let destination = directory.path().join("policy.md");
        fs::write(&destination, b"old").expect("write old destination");
        let cancel = CancellationToken::new();

        let fault = atomic_replace_inner(
            &destination,
            b"new",
            AtomicWriteOptions::default(),
            &cancel,
            FailBeforeCommit,
        );
        assert!(fault.is_err());
        assert_eq!(fs::read(&destination).expect("read destination"), b"old");
        assert!(temp_files(directory.path()).is_empty());

        let unwound = catch_unwind(AssertUnwindSafe(|| {
            let guard = AtomicWriteGuard::create(&destination, 1).expect("create guard");
            assert!(guard.path().exists());
            panic!("injected unwind");
        }));
        assert!(unwound.is_err());
        assert!(temp_files(directory.path()).is_empty());
    }

    // **Validates: Requirements 2.10**
    #[test]
    fn temporaries_are_exclusive_unique_and_same_directory() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let destination = directory.path().join("state.txt");
        let first = AtomicWriteGuard::create(&destination, 1).expect("first guard");
        let second = AtomicWriteGuard::create(&destination, 1).expect("second guard");

        assert_ne!(first.path(), second.path());
        assert_eq!(first.path().parent(), Some(directory.path()));
        assert_eq!(second.path().parent(), Some(directory.path()));
        assert_eq!(temp_files(directory.path()).len(), 2);
    }

    // **Validates: Requirements 2.10**
    #[test]
    fn concurrent_commits_never_expose_or_leave_partial_files() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let destination = directory.path().join("shared.bin");
        fs::write(&destination, b"old").expect("write old destination");
        let payloads: Vec<Vec<u8>> = (1_u8..=8).map(|byte| vec![byte; 32 * 1024]).collect();
        let barrier = Arc::new(Barrier::new(payloads.len()));

        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for payload in &payloads {
                let destination = destination.clone();
                let barrier = barrier.clone();
                handles.push(scope.spawn(move || {
                    atomic_replace_inner(
                        &destination,
                        payload,
                        AtomicWriteOptions::default(),
                        &CancellationToken::new(),
                        CommitBarrier(barrier),
                    )
                }));
            }
            for handle in handles {
                handle
                    .join()
                    .expect("writer thread")
                    .expect("atomic writer");
            }
        });

        let final_bytes = fs::read(&destination).expect("read final destination");
        assert!(payloads.contains(&final_bytes));
        assert!(temp_files(directory.path()).is_empty());
    }

    // **Validates: Requirements 2.10**
    #[tokio::test]
    async fn invalid_retry_bounds_do_not_touch_destination() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let destination = directory.path().join("bounded.txt");
        fs::write(&destination, b"old").expect("write old destination");
        let options = AtomicWriteOptions {
            temp_create_attempts: 0,
            ..AtomicWriteOptions::default()
        };

        assert!(
            atomic_replace(&destination, b"new", options, &CancellationToken::new())
                .await
                .is_err()
        );
        assert_eq!(fs::read(&destination).expect("read destination"), b"old");
        assert!(temp_files(directory.path()).is_empty());
    }
    #[cfg(unix)]
    // **Validates: Requirements 2.10**
    #[test]
    fn unix_temporary_is_mode_0600() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().expect("temporary directory");
        let destination = directory.path().join("secret.txt");
        let guard = AtomicWriteGuard::create(&destination, 1).expect("create guard");
        let mode = fs::metadata(guard.path())
            .expect("temporary metadata")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o600);
    }

    #[cfg(windows)]
    // **Validates: Requirements 2.10**
    #[test]
    fn windows_sharing_violation_preserves_destination_and_cleans_temporary() {
        use std::os::windows::ffi::OsStrExt;
        use std::ptr;
        use windows_sys::Win32::Foundation::{CloseHandle, GENERIC_READ, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::Storage::FileSystem::{
            CreateFileW, FILE_ATTRIBUTE_NORMAL, FILE_SHARE_READ, OPEN_EXISTING,
        };

        struct HandleGuard(windows_sys::Win32::Foundation::HANDLE);
        impl Drop for HandleGuard {
            fn drop(&mut self) {
                // SAFETY: The handle was returned by CreateFileW and is closed once here.
                unsafe { CloseHandle(self.0) };
            }
        }

        let directory = tempfile::tempdir().expect("temporary directory");
        let destination = directory.path().join("locked.txt");
        fs::write(&destination, b"old").expect("write old destination");
        let wide: Vec<u16> = destination
            .as_os_str()
            .encode_wide()
            .chain(Some(0))
            .collect();
        // SAFETY: `wide` is a stable, NUL-terminated UTF-16 path. Null security
        // and template parameters are permitted. FILE_SHARE_DELETE is omitted
        // intentionally to force replacement to fail.
        let handle = unsafe {
            CreateFileW(
                wide.as_ptr(),
                GENERIC_READ,
                FILE_SHARE_READ,
                ptr::null(),
                OPEN_EXISTING,
                FILE_ATTRIBUTE_NORMAL,
                ptr::null_mut(),
            )
        };
        assert_ne!(handle, INVALID_HANDLE_VALUE, "open destination lock");
        let _handle = HandleGuard(handle);

        let result = atomic_replace_inner(
            &destination,
            b"new",
            AtomicWriteOptions::default(),
            &CancellationToken::new(),
            NoHooks,
        );
        assert!(result.is_err(), "sharing violation must fail replacement");
        assert_eq!(fs::read(&destination).expect("read destination"), b"old");
        assert!(temp_files(directory.path()).is_empty());
    }
}
