//! Narrow Unix system calls for pinned observation and bounded process supervision.
#![allow(unsafe_code)]
use super::{ProcessLimits, ToolExecutionContext, ToolExecutionError, commitment, error};
use abi_capability::Digest;
use serde::Serialize;
use std::{
    ffi::CString,
    fs::{File, Metadata, OpenOptions},
    io::{self, Read},
    os::{
        fd::AsRawFd,
        unix::{
            ffi::OsStrExt,
            fs::{MetadataExt, OpenOptionsExt},
            process::CommandExt,
        },
    },
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

#[derive(Serialize, PartialEq, Eq)]
struct Identity {
    device: u64,
    inode: u64,
    owner: u32,
    mode: u32,
}
impl Identity {
    fn of(metadata: &Metadata) -> Self {
        Self {
            device: metadata.dev(),
            inode: metadata.ino(),
            owner: metadata.uid(),
            mode: metadata.mode(),
        }
    }
}
pub(super) struct Pinned {
    pub(super) cwd: String,
    directory: File,
    directory_id: Identity,
    executable: PathBuf,
    executable_id: Identity,
    executable_digest: Digest,
    #[cfg(test)]
    pub(super) after_verify: Option<std::sync::Arc<dyn Fn() + Send + Sync>>,
}
impl Pinned {
    pub(super) fn new(path: &Path) -> Result<Self, ToolExecutionError> {
        if !std::fs::symlink_metadata(path).is_ok_and(|m| m.is_dir() && !m.file_type().is_symlink())
        {
            return Err(error("process_directory_refused"));
        }
        let canonical = path
            .canonicalize()
            .map_err(|_| error("process_directory_refused"))?;
        let cwd = canonical
            .to_str()
            .filter(|s| s.len() <= 4096)
            .ok_or_else(|| error("process_directory_refused"))?
            .to_owned();
        let directory = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK)
            .open(&canonical)
            .map_err(|_| error("process_directory_refused"))?;
        let directory_id = Identity::of(
            &directory
                .metadata()
                .map_err(|_| error("process_directory_refused"))?,
        );
        let executable = Path::new("/bin/pwd")
            .canonicalize()
            .map_err(|_| error("process_executable_refused"))?;
        let (executable_id, executable_digest) = executable_state(&executable)?;
        let pinned = Self {
            cwd,
            directory,
            directory_id,
            executable,
            executable_id,
            executable_digest,
            #[cfg(test)]
            after_verify: None,
        };
        pinned.verify()?;
        Ok(pinned)
    }
    pub(super) fn identity_digest(&self) -> Digest {
        commitment(&(
            "abi.process.pins.v1",
            &self.cwd,
            &self.directory_id,
            &self.executable,
            &self.executable_id,
            self.executable_digest,
        ))
    }
    pub(super) fn verify(&self) -> Result<(), ToolExecutionError> {
        let current =
            std::fs::symlink_metadata(&self.cwd).map_err(|_| error("process_directory_changed"))?;
        let fd = self
            .directory
            .metadata()
            .map_err(|_| error("process_directory_changed"))?;
        if !current.is_dir()
            || current.file_type().is_symlink()
            || Identity::of(&current) != self.directory_id
            || Identity::of(&fd) != self.directory_id
        {
            return Err(error("process_directory_changed"));
        }
        let (identity, digest) = executable_state(&self.executable)?;
        if identity != self.executable_id || digest != self.executable_digest {
            return Err(error("process_executable_changed"));
        }
        let executable = CString::new(self.executable.as_os_str().as_bytes())
            .map_err(|_| error("process_executable_refused"))?;
        // SAFETY: valid live FD and NUL-terminated paths. Effective credentials
        // enforce the kernel ACL policy; unsupported AT_EACCESS fails closed.
        if unsafe {
            libc::faccessat(
                libc::AT_FDCWD,
                executable.as_ptr(),
                libc::X_OK,
                libc::AT_EACCESS,
            )
        } != 0
            || unsafe {
                libc::faccessat(
                    self.directory.as_raw_fd(),
                    c".".as_ptr(),
                    libc::X_OK,
                    libc::AT_EACCESS,
                )
            } != 0
        {
            return Err(error("process_permission_refused"));
        }
        #[cfg(test)]
        if let Some(hook) = &self.after_verify {
            hook();
        }
        Ok(())
    }
    fn command(&self) -> Command {
        let mut command = Command::new(&self.executable);
        command
            .env_clear()
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .process_group(0);
        let fd = self.directory.as_raw_fd();
        // SAFETY: directory remains alive through spawn. Only async-signal-safe
        // fchdir occurs after fork; no allocation, locks or environment access.
        unsafe {
            command.pre_exec(move || {
                if libc::fchdir(fd) == 0 {
                    Ok(())
                } else {
                    Err(io::Error::last_os_error())
                }
            });
        }
        command
    }
}
fn executable_state(path: &Path) -> Result<(Identity, Digest), ToolExecutionError> {
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK)
        .open(path)
        .map_err(|_| error("process_executable_refused"))?;
    let metadata = file
        .metadata()
        .map_err(|_| error("process_executable_refused"))?;
    if !metadata.is_file()
        || metadata.uid() != 0
        || metadata.mode() & 0o022 != 0
        || metadata.len() > 16 * 1024 * 1024
    {
        return Err(error("process_executable_refused"));
    }
    let mut bytes = Vec::new();
    // A bounded read also refuses a concurrently growing/replaced executable.
    (&mut file)
        .take(16 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| error("process_executable_refused"))?;
    let after = file
        .metadata()
        .map_err(|_| error("process_executable_refused"))?;
    if bytes.len() as u64 != metadata.len()
        || Identity::of(&metadata) != Identity::of(&after)
        || metadata.mtime() != after.mtime()
        || metadata.mtime_nsec() != after.mtime_nsec()
        || metadata.ctime() != after.ctime()
        || metadata.ctime_nsec() != after.ctime_nsec()
    {
        return Err(error("process_executable_changed"));
    }
    Ok((
        Identity::of(&metadata),
        commitment(&("abi.process.executable.v1", bytes)),
    ))
}

struct Pipes {
    stdout: std::process::ChildStdout,
    stderr: std::process::ChildStderr,
    out: Vec<u8>,
    err: Vec<u8>,
    out_eof: bool,
    err_eof: bool,
    failed: bool,
}
impl Pipes {
    fn new(child: &mut Child) -> Result<Self, ToolExecutionError> {
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| error("process_pipe_failed"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| error("process_pipe_failed"))?;
        if nonblocking(stdout.as_raw_fd()).is_err() || nonblocking(stderr.as_raw_fd()).is_err() {
            return Err(error("process_pipe_failed"));
        }
        Ok(Self {
            stdout,
            stderr,
            out: Vec::new(),
            err: Vec::new(),
            out_eof: false,
            err_eof: false,
            failed: false,
        })
    }
    fn read(&mut self, limit: usize) -> bool {
        if !self.failed {
            self.failed = read_pipe(
                &mut self.stdout,
                &mut self.out,
                &mut self.out_eof,
                self.err.len(),
                limit,
            )
            .is_err()
                || read_pipe(
                    &mut self.stderr,
                    &mut self.err,
                    &mut self.err_eof,
                    self.out.len(),
                    limit,
                )
                .is_err();
        }
        !self.failed
    }
}

#[derive(Default)]
pub(super) struct Runner {
    child: Option<Child>,
    #[cfg(test)]
    fail_kill_once: bool,
    #[cfg(test)]
    fail_wait_once: bool,
    #[cfg(test)]
    hold_reap: bool,
    #[cfg(test)]
    last_pid: Option<u32>,
}
impl Runner {
    pub(super) fn run(
        &mut self,
        pinned: &Pinned,
        limits: ProcessLimits,
        context: ToolExecutionContext<'_>,
    ) -> Result<Vec<u8>, ToolExecutionError> {
        self.run_command(pinned.command(), limits, context)
    }
    fn run_command(
        &mut self,
        mut command: Command,
        limits: ProcessLimits,
        context: ToolExecutionContext<'_>,
    ) -> Result<Vec<u8>, ToolExecutionError> {
        if self.child.is_some() && !self.cleanup(limits.cleanup_grace_ms) {
            return Err(error("process_cleanup_uncertain"));
        }
        if context.should_stop() {
            return Err(error("process_stopped"));
        }
        let deadline = Instant::now()
            .checked_add(Duration::from_millis(limits.timeout_ms))
            .ok_or_else(|| error("process_invalid_limits"))?
            .min(context.deadline);
        let child = command.spawn().map_err(|_| error("process_spawn_failed"))?;
        #[cfg(test)]
        {
            self.last_pid = Some(child.id());
        }
        self.child = Some(child);
        let mut pipes = match Pipes::new(self.child.as_mut().expect("owned child")) {
            Ok(pipes) => pipes,
            Err(err) => {
                self.cleanup(limits.cleanup_grace_ms);
                return Err(err);
            }
        };
        let execution = loop {
            if context.should_stop() || Instant::now() >= deadline {
                break false;
            }
            if !pipes.read(limits.max_output_bytes) {
                break false;
            }
            match exited(self.child.as_ref().expect("owned child").id()) {
                Ok(true) => break true,
                Ok(false) => thread::sleep(Duration::from_millis(1)),
                Err(_) => break false,
            }
        };
        // Kill the group while its unreaped leader still pins the PGID.
        let killed = self.kill_current().is_ok();
        let cleanup_deadline = Instant::now() + Duration::from_millis(limits.cleanup_grace_ms);
        let mut status = None;
        let mut pipes_ok = true;
        loop {
            pipes_ok &= pipes.read(limits.max_output_bytes);
            if killed && status.is_none() {
                match self.reap_current() {
                    Ok(Some(observed)) => status = Some(observed),
                    Ok(None) => {}
                    Err(_) => break,
                }
            }
            if status.is_some() && ((pipes.out_eof && pipes.err_eof) || !pipes_ok) {
                break;
            }
            if Instant::now() >= cleanup_deadline {
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
        if status.is_some() {
            self.child = None;
        }
        if !execution
            || !killed
            || !pipes_ok
            || !status.is_some_and(|s| s.success())
            || !pipes.out_eof
            || !pipes.err_eof
            || !pipes.err.is_empty()
            || context.should_stop()
            || Instant::now() >= deadline
        {
            return Err(error("process_execution_unverified"));
        }
        Ok(pipes.out)
    }
    fn cleanup(&mut self, grace: u64) -> bool {
        if self.child.is_none() {
            return true;
        }
        if self.kill_current().is_err() {
            return false;
        }
        let deadline = Instant::now() + Duration::from_millis(grace);
        loop {
            match self.reap_current() {
                Ok(Some(_)) => {
                    self.child = None;
                    return true;
                }
                Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(1)),
                _ => return false,
            }
        }
    }

    fn kill_current(&mut self) -> io::Result<()> {
        #[cfg(test)]
        if std::mem::take(&mut self.fail_kill_once) {
            return Err(io::Error::other("injected kill failure"));
        }
        kill_group(self.child.as_ref().expect("owned child").id())
    }

    fn reap_current(&mut self) -> io::Result<Option<std::process::ExitStatus>> {
        #[cfg(test)]
        {
            if std::mem::take(&mut self.fail_wait_once) {
                return Err(io::Error::other("injected wait failure"));
            }
            if self.hold_reap {
                return Ok(None);
            }
        }
        self.child.as_mut().expect("owned child").try_wait()
    }
}
impl Drop for Runner {
    fn drop(&mut self) {
        #[cfg(test)]
        {
            self.fail_kill_once = false;
            self.fail_wait_once = false;
            self.hold_reap = false;
        }
        self.cleanup(250);
    }
}
fn nonblocking(fd: i32) -> io::Result<()> {
    // SAFETY: fcntl operates on a live owned pipe descriptor.
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags < 0 || unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}
fn read_pipe(
    pipe: &mut impl Read,
    bytes: &mut Vec<u8>,
    eof: &mut bool,
    other: usize,
    limit: usize,
) -> io::Result<()> {
    if *eof {
        return Ok(());
    }
    let mut buffer = [0_u8; 4096];
    match pipe.read(&mut buffer) {
        Ok(0) => {
            *eof = true;
            Ok(())
        }
        Ok(count) if count <= limit.saturating_sub(bytes.len().saturating_add(other)) => {
            bytes.extend_from_slice(&buffer[..count]);
            Ok(())
        }
        Err(err)
            if matches!(
                err.kind(),
                io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
            ) =>
        {
            Ok(())
        }
        _ => Err(io::Error::other("process_pipe_limit")),
    }
}
fn exited(pid: u32) -> io::Result<bool> {
    // SAFETY: zero is valid initialization; waitid fills this bounded object.
    let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
    // WNOWAIT observes status without releasing the PGID identity reservation.
    if unsafe {
        libc::waitid(
            libc::P_PID,
            pid,
            &raw mut info,
            libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
        )
    } != 0
    {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: si_pid is valid after successful waitid (zero means no change).
    Ok(unsafe { info.si_pid() } != 0)
}
fn kill_group(pid: u32) -> io::Result<()> {
    let pid = i32::try_from(pid).map_err(|_| io::Error::other("process_id_refused"))?;
    // SAFETY: caller owns the unreaped leader; negative PID selects its group.
    if unsafe { libc::kill(-pid, libc::SIGKILL) } == 0 {
        return Ok(());
    }
    let err = io::Error::last_os_error();
    if err.raw_os_error() == Some(libc::ESRCH) {
        Ok(())
    } else {
        // Darwin can return EPERM for a group containing only a zombie leader.
        // Do not ignore permission errors: independently prove that the owned
        // leader exited and is the group's sole member before reaping it.
        #[cfg(target_os = "macos")]
        if err.raw_os_error() == Some(libc::EPERM)
            && exited(pid.cast_unsigned())?
            && sole_group_leader(pid)
        {
            return Ok(());
        }
        Err(err)
    }
}

#[cfg(target_os = "macos")]
fn sole_group_leader(pid: i32) -> bool {
    let mut members = [0_i32; 2];
    // SAFETY: valid bounded output buffer; the unreaped leader still pins pid.
    // A full buffer, missing group or any additional member refuses the proof.
    let count = unsafe { libc::proc_listpgrppids(pid, members.as_mut_ptr().cast(), 8) };
    count == 1 && members[0] == pid
}

#[cfg(test)]
#[path = "supervisor_tests.rs"]
mod tests;
