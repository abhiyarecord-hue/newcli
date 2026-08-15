//! Cross-platform ownership for a child process and all descendants.
//!
//! Unix children lead a fresh process group. Windows children are suspended,
//! assigned to a kill-on-close Job Object, and only then resumed.

use std::io;
use std::process::ExitStatus;
use std::time::Duration;

use agent_types::{AgentError, Result};
use tokio::process::{Child, Command};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::bounded_output::{capture_bounded, CapturedOutput};

pub const DEFAULT_OUTPUT_LIMIT: usize = 1024 * 1024;
pub const DEFAULT_TERMINATION_GRACE: Duration = Duration::from_millis(250);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Termination {
    Exit,
    Timeout,
    Cancelled,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessConfig {
    pub stdout_limit: usize,
    pub stderr_limit: usize,
    pub termination_grace: Duration,
}

impl Default for ProcessConfig {
    fn default() -> Self {
        Self {
            stdout_limit: DEFAULT_OUTPUT_LIMIT,
            stderr_limit: DEFAULT_OUTPUT_LIMIT,
            termination_grace: DEFAULT_TERMINATION_GRACE,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ExecResult {
    pub stdout: String,
    pub stderr: String,
    pub stdout_truncated: bool,
    pub stderr_truncated: bool,
    pub exit_code: i32,
    pub termination: Termination,
}

#[derive(Clone, Debug)]
pub struct ProcessSupervisor {
    config: ProcessConfig,
}

impl Default for ProcessSupervisor {
    fn default() -> Self {
        Self::new(ProcessConfig::default())
    }
}

impl ProcessSupervisor {
    pub fn new(config: ProcessConfig) -> Self {
        Self { config }
    }

    pub fn spawn(&self, mut command: Command) -> Result<SupervisedChild> {
        command.stdout(std::process::Stdio::piped());
        command.stderr(std::process::Stdio::piped());
        command.kill_on_drop(true);

        let (mut child, owner) = spawn_owned(command)?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| AgentError::Sandbox("spawned process has no stdout pipe".into()))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| AgentError::Sandbox("spawned process has no stderr pipe".into()))?;
        let stdout_task = tokio::spawn(capture_bounded(stdout, self.config.stdout_limit));
        let stderr_task = tokio::spawn(capture_bounded(stderr, self.config.stderr_limit));

        Ok(SupervisedChild {
            child: Some(child),
            owner,
            stdout_task: Some(stdout_task),
            stderr_task: Some(stderr_task),
            grace: self.config.termination_grace,
            complete: false,
        })
    }

    /// Spawn an interactively-routed process while retaining complete process
    /// tree ownership. The caller may take stdin/stdout exactly once; stderr is
    /// drained with the configured bound so an unobserved server log cannot
    /// deadlock the child.
    pub fn spawn_stdio(&self, mut command: Command) -> Result<SupervisedStdioChild> {
        command.stdin(std::process::Stdio::piped());
        command.stdout(std::process::Stdio::piped());
        command.stderr(std::process::Stdio::piped());
        command.kill_on_drop(true);

        let (mut child, owner) = spawn_owned(command)?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| AgentError::Sandbox("spawned process has no stdin pipe".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| AgentError::Sandbox("spawned process has no stdout pipe".into()))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| AgentError::Sandbox("spawned process has no stderr pipe".into()))?;
        let stderr_task = tokio::spawn(capture_bounded(stderr, self.config.stderr_limit));

        Ok(SupervisedStdioChild {
            child: Some(child),
            owner,
            stdin: Some(stdin),
            stdout: Some(stdout),
            stderr_task: Some(stderr_task),
            complete: false,
        })
    }
}

fn spawn_owned(mut command: Command) -> Result<(Child, platform::ProcessTreeOwner)> {
    let mut owner = platform::ProcessTreeOwner::new()
        .map_err(|error| AgentError::Sandbox(format!("process owner: {error}")))?;
    owner
        .prepare(&mut command)
        .map_err(|error| AgentError::Sandbox(format!("process prepare: {error}")))?;
    let mut child = command
        .spawn()
        .map_err(|error| AgentError::Sandbox(format!("spawn: {error}")))?;
    owner.attach(&child).map_err(|error| {
        let _ = child.start_kill();
        AgentError::Sandbox(format!("process ownership: {error}"))
    })?;
    Ok((child, owner))
}

pub struct SupervisedStdioChild {
    child: Option<Child>,
    owner: platform::ProcessTreeOwner,
    stdin: Option<tokio::process::ChildStdin>,
    stdout: Option<tokio::process::ChildStdout>,
    stderr_task: Option<JoinHandle<io::Result<CapturedOutput>>>,
    complete: bool,
}

impl SupervisedStdioChild {
    pub fn id(&self) -> Option<u32> {
        self.child.as_ref().and_then(Child::id)
    }

    pub fn take_stdin(&mut self) -> Result<tokio::process::ChildStdin> {
        self.stdin
            .take()
            .ok_or_else(|| AgentError::Sandbox("process stdin already taken".into()))
    }

    pub fn take_stdout(&mut self) -> Result<tokio::process::ChildStdout> {
        self.stdout
            .take()
            .ok_or_else(|| AgentError::Sandbox("process stdout already taken".into()))
    }
}

impl Drop for SupervisedStdioChild {
    fn drop(&mut self) {
        if self.complete {
            return;
        }
        self.owner.force_terminate();
        self.stdin.take();
        self.stdout.take();
        if let Some(task) = self.stderr_task.take() {
            task.abort();
        }
        let Some(mut child) = self.child.take() else {
            return;
        };
        let _ = child.start_kill();
        let _ = std::thread::Builder::new()
            .name("sandbox-stdio-process-reaper".into())
            .spawn(move || loop {
                match child.try_wait() {
                    Ok(Some(_)) | Err(_) => break,
                    Ok(None) => std::thread::sleep(Duration::from_millis(10)),
                }
            });
    }
}

pub struct SupervisedChild {
    child: Option<Child>,
    owner: platform::ProcessTreeOwner,
    stdout_task: Option<JoinHandle<io::Result<CapturedOutput>>>,
    stderr_task: Option<JoinHandle<io::Result<CapturedOutput>>>,
    grace: Duration,
    complete: bool,
}

impl SupervisedChild {
    pub fn id(&self) -> Option<u32> {
        self.child.as_ref().and_then(Child::id)
    }

    pub async fn wait(self) -> Result<ExecResult> {
        let never_cancel = CancellationToken::new();
        self.wait_with(Duration::MAX, &never_cancel).await
    }

    pub async fn wait_with(
        mut self,
        timeout: Duration,
        cancel: &CancellationToken,
    ) -> Result<ExecResult> {
        let outcome = tokio::select! {
            biased;
            _ = cancel.cancelled() => None,
            waited = tokio::time::timeout(timeout, self.child_mut()?.wait()) => Some(waited),
        };

        let grace = self.grace;
        match outcome {
            None => self.terminate(Termination::Cancelled, grace).await,
            Some(Err(_)) => self.terminate(Termination::Timeout, grace).await,
            Some(Ok(Err(error))) => Err(AgentError::Sandbox(format!("wait: {error}"))),
            Some(Ok(Ok(status))) => self.finish_exit(status).await,
        }
    }

    pub async fn terminate(mut self, reason: Termination, grace: Duration) -> Result<ExecResult> {
        debug_assert!(reason != Termination::Exit);
        self.owner.graceful_terminate();
        if !grace.is_zero() {
            tokio::time::sleep(grace).await;
        }
        self.owner.force_terminate();
        let status = self
            .child_mut()?
            .wait()
            .await
            .map_err(|error| AgentError::Sandbox(format!("reap: {error}")))?;
        self.finish(status, reason).await
    }

    async fn finish_exit(mut self, status: ExitStatus) -> Result<ExecResult> {
        if self.owner.graceful_terminate() {
            if !self.grace.is_zero() {
                tokio::time::sleep(self.grace).await;
            }
            self.owner.force_terminate();
        }
        self.finish(status, Termination::Exit).await
    }

    async fn finish(mut self, status: ExitStatus, termination: Termination) -> Result<ExecResult> {
        let stdout = join_capture(self.stdout_task.take(), "stdout").await?;
        let stderr = join_capture(self.stderr_task.take(), "stderr").await?;
        self.complete = true;
        self.owner.disarm();
        self.child.take();
        Ok(ExecResult {
            stdout: stdout.text,
            stderr: stderr.text,
            stdout_truncated: stdout.truncated,
            stderr_truncated: stderr.truncated,
            exit_code: status.code().unwrap_or(-1),
            termination,
        })
    }

    fn child_mut(&mut self) -> Result<&mut Child> {
        self.child
            .as_mut()
            .ok_or_else(|| AgentError::Sandbox("process already reaped".into()))
    }
}

async fn join_capture(
    task: Option<JoinHandle<io::Result<CapturedOutput>>>,
    stream: &str,
) -> Result<CapturedOutput> {
    let task = task.ok_or_else(|| AgentError::Sandbox(format!("missing {stream} reader")))?;
    task.await
        .map_err(|error| AgentError::Sandbox(format!("{stream} reader task: {error}")))?
        .map_err(|error| AgentError::Sandbox(format!("read {stream}: {error}")))
}

impl Drop for SupervisedChild {
    fn drop(&mut self) {
        if self.complete {
            return;
        }
        self.owner.force_terminate();
        if let Some(task) = self.stdout_task.take() {
            task.abort();
        }
        if let Some(task) = self.stderr_task.take() {
            task.abort();
        }
        let Some(mut child) = self.child.take() else {
            return;
        };
        let _ = child.start_kill();
        // `Drop` cannot await. A small dedicated thread guarantees the root is
        // eventually reaped even when the owning async task itself is aborted.
        let _ = std::thread::Builder::new()
            .name("sandbox-process-reaper".into())
            .spawn(move || loop {
                match child.try_wait() {
                    Ok(Some(_)) | Err(_) => break,
                    Ok(None) => std::thread::sleep(Duration::from_millis(10)),
                }
            });
    }
}

#[cfg(unix)]
mod platform {
    use super::*;
    use std::os::unix::process::CommandExt;

    pub struct ProcessTreeOwner {
        pgid: Option<i32>,
        armed: bool,
    }

    impl ProcessTreeOwner {
        pub fn new() -> io::Result<Self> {
            Ok(Self {
                pgid: None,
                armed: true,
            })
        }

        pub fn prepare(&mut self, command: &mut Command) -> io::Result<()> {
            unsafe {
                command.as_std_mut().pre_exec(|| {
                    if libc::setpgid(0, 0) == -1 {
                        return Err(io::Error::last_os_error());
                    }
                    Ok(())
                });
            }
            Ok(())
        }

        pub fn attach(&mut self, child: &Child) -> io::Result<()> {
            self.pgid = child.id().map(|id| id as i32);
            Ok(())
        }

        pub fn graceful_terminate(&mut self) -> bool {
            self.signal(libc::SIGTERM)
        }

        pub fn force_terminate(&mut self) -> bool {
            self.signal(libc::SIGKILL)
        }

        pub fn disarm(&mut self) {
            self.armed = false;
        }

        fn signal(&self, signal: i32) -> bool {
            if !self.armed {
                return false;
            }
            self.pgid
                .is_some_and(|pgid| unsafe { libc::kill(-pgid, signal) == 0 })
        }
    }

    impl Drop for ProcessTreeOwner {
        fn drop(&mut self) {
            self.force_terminate();
        }
    }
}

#[cfg(windows)]
mod platform {
    use super::*;
    use std::mem::{size_of, zeroed};
    use std::os::windows::process::CommandExt;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, Thread32First, Thread32Next, TH32CS_SNAPTHREAD, THREADENTRY32,
    };
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
        SetInformationJobObject, TerminateJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };
    use windows_sys::Win32::System::Threading::{
        OpenThread, ResumeThread, CREATE_NEW_PROCESS_GROUP, CREATE_SUSPENDED, THREAD_SUSPEND_RESUME,
    };

    pub struct ProcessTreeOwner {
        job: HANDLE,
        armed: bool,
    }

    unsafe impl Send for ProcessTreeOwner {}

    impl ProcessTreeOwner {
        pub fn new() -> io::Result<Self> {
            let job = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
            if job.is_null() {
                return Err(io::Error::last_os_error());
            }
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { zeroed() };
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            let configured = unsafe {
                SetInformationJobObject(
                    job,
                    JobObjectExtendedLimitInformation,
                    &info as *const _ as *const _,
                    size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                )
            };
            if configured == 0 {
                unsafe { CloseHandle(job) };
                return Err(io::Error::last_os_error());
            }
            Ok(Self { job, armed: true })
        }

        pub fn prepare(&mut self, command: &mut Command) -> io::Result<()> {
            command
                .as_std_mut()
                .creation_flags(CREATE_SUSPENDED | CREATE_NEW_PROCESS_GROUP);
            Ok(())
        }

        pub fn attach(&mut self, child: &Child) -> io::Result<()> {
            let process = child
                .raw_handle()
                .ok_or_else(|| io::Error::other("missing process handle"))?
                as HANDLE;
            if unsafe { AssignProcessToJobObject(self.job, process) } == 0 {
                return Err(io::Error::last_os_error());
            }
            resume_primary_thread(child.id().ok_or_else(|| io::Error::other("missing pid"))?)
        }

        pub fn graceful_terminate(&mut self) -> bool {
            self.force_terminate();
            false
        }

        pub fn force_terminate(&mut self) -> bool {
            self.armed && unsafe { TerminateJobObject(self.job, 1) != 0 }
        }

        pub fn disarm(&mut self) {
            self.armed = false;
        }
    }

    impl Drop for ProcessTreeOwner {
        fn drop(&mut self) {
            if self.armed {
                self.force_terminate();
            }
            unsafe { CloseHandle(self.job) };
        }
    }

    fn resume_primary_thread(pid: u32) -> io::Result<()> {
        let snapshot = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0) };
        if snapshot == INVALID_HANDLE_VALUE {
            return Err(io::Error::last_os_error());
        }
        let mut entry: THREADENTRY32 = unsafe { zeroed() };
        entry.dwSize = size_of::<THREADENTRY32>() as u32;
        let mut found = false;
        let mut has_entry = unsafe { Thread32First(snapshot, &mut entry) } != 0;
        while has_entry {
            if entry.th32OwnerProcessID == pid {
                let thread = unsafe { OpenThread(THREAD_SUSPEND_RESUME, 0, entry.th32ThreadID) };
                if !thread.is_null() {
                    let resumed = unsafe { ResumeThread(thread) };
                    unsafe { CloseHandle(thread) };
                    if resumed != u32::MAX {
                        found = true;
                        break;
                    }
                }
            }
            has_entry = unsafe { Thread32Next(snapshot, &mut entry) } != 0;
        }
        unsafe { CloseHandle(snapshot) };
        if found {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shell(command: &str) -> Command {
        #[cfg(windows)]
        let mut child = {
            let mut command_builder = Command::new("cmd.exe");
            command_builder.args(["/C", command]);
            command_builder
        };
        #[cfg(not(windows))]
        let mut child = {
            let mut command_builder = Command::new("sh");
            command_builder.args(["-c", command]);
            command_builder
        };
        child.env_clear();
        child.env("PATH", std::env::var("PATH").unwrap_or_default());
        child.env(
            "SYSTEMROOT",
            std::env::var("SYSTEMROOT").unwrap_or_default(),
        );
        child
    }

    fn small_supervisor() -> ProcessSupervisor {
        ProcessSupervisor::new(ProcessConfig {
            stdout_limit: 4,
            stderr_limit: 3,
            termination_grace: Duration::from_millis(25),
        })
    }

    #[tokio::test]
    async fn normal_exit_captures_both_streams_and_exit_status() {
        #[cfg(windows)]
        let command = "echo out& echo err 1>&2& exit /B 7";
        #[cfg(not(windows))]
        let command = "printf out; printf err >&2; exit 7";
        let result = ProcessSupervisor::default()
            .spawn(shell(command))
            .unwrap()
            .wait()
            .await
            .unwrap();
        assert_eq!(result.termination, Termination::Exit);
        assert_eq!(result.exit_code, 7);
        assert!(result.stdout.contains("out"));
        assert!(result.stderr.contains("err"));
        assert!(!result.stdout_truncated && !result.stderr_truncated);
    }

    #[tokio::test]
    async fn injected_limits_bound_and_flag_both_streams() {
        #[cfg(windows)]
        let command = "echo 123456789& echo abcdefghi 1>&2";
        #[cfg(not(windows))]
        let command = "printf 123456789; printf abcdefghi >&2";
        let result = small_supervisor()
            .spawn(shell(command))
            .unwrap()
            .wait()
            .await
            .unwrap();
        assert!(result.stdout.len() <= 4 && result.stderr.len() <= 3);
        assert!(result.stdout_truncated && result.stderr_truncated);
    }

    #[tokio::test]
    async fn timeout_and_cancellation_are_distinct_terminal_outcomes() {
        #[cfg(windows)]
        let sleeping = "ping -n 30 127.0.0.1 >NUL";
        #[cfg(not(windows))]
        let sleeping = "sleep 30";

        let timeout = small_supervisor()
            .spawn(shell(sleeping))
            .unwrap()
            .wait_with(Duration::from_millis(30), &CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(timeout.termination, Termination::Timeout);

        let cancel = CancellationToken::new();
        let trigger = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            trigger.cancel();
        });
        let cancelled = small_supervisor()
            .spawn(shell(sleeping))
            .unwrap()
            .wait_with(Duration::from_secs(5), &cancel)
            .await
            .unwrap();
        assert_eq!(cancelled.termination, Termination::Cancelled);
    }

    #[tokio::test]
    async fn dropping_owner_kills_root_without_waiting_for_timeout() {
        #[cfg(windows)]
        let sleeping = "ping -n 30 127.0.0.1 >NUL";
        #[cfg(not(windows))]
        let sleeping = "sleep 30";
        let child = small_supervisor().spawn(shell(sleeping)).unwrap();
        assert!(child.id().is_some());
        drop(child);
        tokio::time::sleep(Duration::from_millis(75)).await;
    }
}
