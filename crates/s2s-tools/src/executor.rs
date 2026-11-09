//! Async tool executor (tokio): concurrent dispatch, per-call timeouts,
//! cancellation by `call_id`, and the client-tool bridge.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard, RwLock};
use std::time::Duration;

use serde_json::Value;
use tokio::sync::{mpsc, oneshot};
use tokio::task::AbortHandle;
use tokio::time::Instant;

use crate::parser::{CallId, ToolCall};
use crate::spec::{ToolLocation, ToolRegistry};

/// The future a [`ToolHandler`] returns: the tool's JSON result or an error
/// message for the model.
pub type HandlerFuture = Pin<Box<dyn Future<Output = Result<Value, String>> + Send + 'static>>;

/// A server-side tool implementation.
///
/// Implemented for any `Fn(Value) -> impl Future<Output = Result<Value, String>>`,
/// so a closure works:
///
/// ```
/// # use serde_json::{json, Value};
/// # fn check(_: impl s2s_tools::ToolHandler) {}
/// check(|args: Value| async move { Ok::<_, String>(json!({"echo": args})) });
/// ```
pub trait ToolHandler: Send + Sync + 'static {
    /// Run the tool with validated `arguments`.
    fn call(&self, arguments: Value) -> HandlerFuture;
}

impl<F, Fut> ToolHandler for F
where
    F: Fn(Value) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<Value, String>> + Send + 'static,
{
    fn call(&self, arguments: Value) -> HandlerFuture {
        Box::pin(self(arguments))
    }
}

/// How a call ended.
#[derive(Clone, Debug, PartialEq)]
pub enum ToolStatus {
    /// The tool returned this JSON.
    Ok(Value),
    /// The tool failed (or could not be run), with a message for the model.
    Error(String),
    /// No result within the timeout.
    Timeout,
    /// Cancelled by `call_id` before it finished.
    Cancelled,
}

impl ToolStatus {
    /// The protocol name of the status: `ok`, `error`, `timeout` or `cancelled`.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Ok(_) => "ok",
            Self::Error(_) => "error",
            Self::Timeout => "timeout",
            Self::Cancelled => "cancelled",
        }
    }
}

/// The final result of one call. Exactly one is emitted per dispatched call.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolOutcome {
    /// The call.
    pub call_id: CallId,
    /// Tool name.
    pub name: String,
    /// How it ended.
    pub status: ToolStatus,
    /// Time from dispatch to completion.
    pub elapsed: Duration,
}

/// Events from the [`Executor`], in the order they happen.
#[derive(Clone, Debug, PartialEq)]
pub enum ExecutorEvent {
    /// A call started running (`tool.status {state: "running"}`).
    Dispatched {
        /// The call.
        call_id: CallId,
        /// Tool name.
        name: String,
        /// Where it runs.
        location: ToolLocation,
    },
    /// A client-side call: send `tool.call {call_id, name, arguments}` to the
    /// client and pass its `tool.result` to
    /// [`Executor::submit_client_result`].
    ClientCall(ToolCall),
    /// A call finished; format it with
    /// [`InjectionFormatter`](crate::InjectionFormatter) and inject it.
    Finished(ToolOutcome),
}

/// [`Executor`] settings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutorConfig {
    /// Timeout for server tools without their own.
    pub server_timeout: Duration,
    /// How long to wait for a client's `tool.result`.
    pub client_timeout: Duration,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            server_timeout: Duration::from_secs(10),
            client_timeout: Duration::from_secs(30),
        }
    }
}

/// Errors registering a handler.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RegisterError {
    /// The tool is not in the registry.
    UnknownTool(String),
    /// The tool is declared client-side.
    ClientTool(String),
}

impl std::fmt::Display for RegisterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownTool(n) => write!(f, "unknown tool {n:?}"),
            Self::ClientTool(n) => write!(f, "tool {n:?} is client-side"),
        }
    }
}

impl std::error::Error for RegisterError {}

/// Errors dispatching a call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DispatchError {
    /// A call with this id is still in flight.
    DuplicateCallId(CallId),
    /// Not called from within a tokio runtime.
    NoRuntime,
}

impl std::fmt::Display for DispatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateCallId(id) => write!(f, "call {id} is already in flight"),
            Self::NoRuntime => f.write_str("dispatch needs a tokio runtime"),
        }
    }
}

impl std::error::Error for DispatchError {}

/// Errors delivering a client result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SubmitError {
    /// No such call in flight (never dispatched, already finished, timed out
    /// or cancelled). Late results land here and are dropped.
    UnknownCall(CallId),
    /// The call is a server-side call.
    NotClientCall(CallId),
}

impl std::fmt::Display for SubmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownCall(id) => write!(f, "no client call {id} in flight"),
            Self::NotClientCall(id) => write!(f, "call {id} is not a client call"),
        }
    }
}

impl std::error::Error for SubmitError {}

struct Registered {
    handler: Arc<dyn ToolHandler>,
    timeout: Option<Duration>,
}

struct InFlight {
    name: String,
    started: Instant,
    abort: Option<AbortHandle>,
    /// Client calls: where the client's result goes.
    client: Option<oneshot::Sender<Result<Value, String>>>,
}

struct Inner {
    registry: Arc<ToolRegistry>,
    config: ExecutorConfig,
    handlers: RwLock<HashMap<String, Registered>>,
    inflight: Mutex<HashMap<CallId, InFlight>>,
    events: mpsc::UnboundedSender<ExecutorEvent>,
}

impl Inner {
    fn inflight(&self) -> MutexGuard<'_, HashMap<CallId, InFlight>> {
        // A panic while holding the lock cannot leave the map inconsistent
        // (every update is a single insert/remove), so poisoning is ignored.
        self.inflight.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn emit(&self, ev: ExecutorEvent) {
        // The receiver may be gone (session ending); nothing to do then.
        let _ = self.events.send(ev);
    }

    /// Finish a call if it is still in flight. Whoever removes the entry
    /// emits the outcome, so each call finishes exactly once.
    fn finish(&self, call_id: &CallId, status: ToolStatus) -> bool {
        let Some(entry) = self.inflight().remove(call_id) else {
            return false;
        };
        if let Some(a) = entry.abort {
            a.abort();
        }
        self.emit(ExecutorEvent::Finished(ToolOutcome {
            call_id: call_id.clone(),
            name: entry.name,
            status,
            elapsed: entry.started.elapsed(),
        }));
        true
    }
}

/// Finishes the call with an error if the task ends without finishing it
/// (the handler panicked). On normal completion or abort it is a no-op,
/// because the entry is already gone.
struct FinishGuard {
    inner: Arc<Inner>,
    call_id: CallId,
}

impl Drop for FinishGuard {
    fn drop(&mut self) {
        let status = if std::thread::panicking() {
            ToolStatus::Error("tool handler panicked".into())
        } else {
            ToolStatus::Error("tool task ended without a result".into())
        };
        self.inner.finish(&self.call_id, status);
    }
}

/// Runs tool calls concurrently and reports their outcomes as
/// [`ExecutorEvent`]s.
///
/// Cheap to clone (all clones share state). [`dispatch`](Self::dispatch)
/// returns immediately; the frame loop keeps running while tools execute
/// (plan §4.2). Every dispatched call yields exactly one
/// [`ExecutorEvent::Finished`].
#[derive(Clone)]
pub struct Executor {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for Executor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Executor")
            .field("registry", &self.inner.registry)
            .field("config", &self.inner.config)
            .field("in_flight", &self.in_flight())
            .finish_non_exhaustive()
    }
}

impl Executor {
    /// A new executor and the receiver for its events.
    pub fn new(
        registry: Arc<ToolRegistry>,
        config: ExecutorConfig,
    ) -> (Self, mpsc::UnboundedReceiver<ExecutorEvent>) {
        let (tx, rx) = mpsc::unbounded_channel();
        let inner = Inner {
            registry,
            config,
            handlers: RwLock::new(HashMap::new()),
            inflight: Mutex::new(HashMap::new()),
            events: tx,
        };
        (
            Self {
                inner: Arc::new(inner),
            },
            rx,
        )
    }

    /// The tool registry.
    pub fn registry(&self) -> &Arc<ToolRegistry> {
        &self.inner.registry
    }

    /// Register the handler for server tool `name` (replacing any previous
    /// one), using the default server timeout.
    pub fn register_handler(
        &self,
        name: &str,
        handler: impl ToolHandler,
    ) -> Result<(), RegisterError> {
        self.register(name, Arc::new(handler), None)
    }

    /// Register a handler with its own timeout.
    pub fn register_handler_with_timeout(
        &self,
        name: &str,
        handler: impl ToolHandler,
        timeout: Duration,
    ) -> Result<(), RegisterError> {
        self.register(name, Arc::new(handler), Some(timeout))
    }

    fn register(
        &self,
        name: &str,
        handler: Arc<dyn ToolHandler>,
        timeout: Option<Duration>,
    ) -> Result<(), RegisterError> {
        let spec = self
            .inner
            .registry
            .get(name)
            .ok_or_else(|| RegisterError::UnknownTool(name.to_string()))?;
        if spec.location == ToolLocation::Client {
            return Err(RegisterError::ClientTool(name.to_string()));
        }
        self.inner
            .handlers
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(name.to_string(), Registered { handler, timeout });
        Ok(())
    }

    /// Ids of calls in flight, sorted.
    pub fn in_flight(&self) -> Vec<CallId> {
        let mut ids: Vec<CallId> = self.inner.inflight().keys().cloned().collect();
        ids.sort();
        ids
    }

    /// Whether `call_id` is in flight.
    pub fn is_in_flight(&self, call_id: &CallId) -> bool {
        self.inner.inflight().contains_key(call_id)
    }

    /// Start a call. Must be called within a tokio runtime.
    ///
    /// - Server tool: its handler runs on a new task with the tool's timeout.
    /// - Client tool: emits [`ExecutorEvent::ClientCall`] and waits (up to the
    ///   client timeout) for [`submit_client_result`](Self::submit_client_result).
    /// - Unknown tool or missing handler: finishes at once with
    ///   [`ToolStatus::Error`], so the model hears about it.
    ///
    /// Arguments are assumed validated (the parser does that).
    pub fn dispatch(&self, call: ToolCall) -> Result<(), DispatchError> {
        let rt = tokio::runtime::Handle::try_current().map_err(|_| DispatchError::NoRuntime)?;
        let inner = &self.inner;
        let ToolCall {
            call_id,
            name,
            arguments,
        } = call;
        let mut map = inner.inflight();
        if map.contains_key(&call_id) {
            return Err(DispatchError::DuplicateCallId(call_id));
        }
        let location = inner.registry.get(&name).map(|s| s.location);
        let entry = InFlight {
            name: name.clone(),
            started: Instant::now(),
            abort: None,
            client: None,
        };
        map.insert(call_id.clone(), entry);
        let dispatched = ExecutorEvent::Dispatched {
            call_id: call_id.clone(),
            name: name.clone(),
            location: location.unwrap_or_default(),
        };
        match location {
            None => {
                drop(map);
                inner.emit(dispatched);
                inner.finish(
                    &call_id,
                    ToolStatus::Error(format!("unknown tool {name:?}")),
                );
            }
            Some(ToolLocation::Server) => {
                let registered = inner
                    .handlers
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .get(&name)
                    .map(|r| (r.handler.clone(), r.timeout));
                let Some((handler, timeout)) = registered else {
                    drop(map);
                    inner.emit(dispatched);
                    inner.finish(
                        &call_id,
                        ToolStatus::Error(format!("no handler for tool {name:?}")),
                    );
                    return Ok(());
                };
                inner.emit(dispatched);
                let timeout = timeout.unwrap_or(inner.config.server_timeout);
                let guard = FinishGuard {
                    inner: inner.clone(),
                    call_id: call_id.clone(),
                };
                // The lock is held until the abort handle is stored, so the
                // task cannot finish (and remove the entry) before that.
                let task = rt.spawn(async move {
                    let status = match tokio::time::timeout(timeout, handler.call(arguments)).await
                    {
                        Ok(Ok(v)) => ToolStatus::Ok(v),
                        Ok(Err(e)) => ToolStatus::Error(e),
                        Err(_) => ToolStatus::Timeout,
                    };
                    guard.inner.finish(&guard.call_id, status);
                });
                if let Some(e) = map.get_mut(&call_id) {
                    e.abort = Some(task.abort_handle());
                }
            }
            Some(ToolLocation::Client) => {
                let (tx, rx) = oneshot::channel();
                inner.emit(dispatched);
                inner.emit(ExecutorEvent::ClientCall(ToolCall {
                    call_id: call_id.clone(),
                    name,
                    arguments,
                }));
                let timeout = inner.config.client_timeout;
                let guard = FinishGuard {
                    inner: inner.clone(),
                    call_id: call_id.clone(),
                };
                let task = rt.spawn(async move {
                    let status = match tokio::time::timeout(timeout, rx).await {
                        Ok(Ok(Ok(v))) => ToolStatus::Ok(v),
                        Ok(Ok(Err(e))) => ToolStatus::Error(e),
                        Ok(Err(_)) => ToolStatus::Error("client result channel closed".into()),
                        Err(_) => ToolStatus::Timeout,
                    };
                    guard.inner.finish(&guard.call_id, status);
                });
                if let Some(e) = map.get_mut(&call_id) {
                    e.abort = Some(task.abort_handle());
                    e.client = Some(tx);
                }
            }
        }
        Ok(())
    }

    /// Deliver a client's `tool.result {call_id, json}`.
    pub fn submit_client_result(&self, call_id: &CallId, json: Value) -> Result<(), SubmitError> {
        self.submit(call_id, Ok(json))
    }

    /// Deliver a client-side failure for a client call.
    pub fn submit_client_error(
        &self,
        call_id: &CallId,
        message: impl Into<String>,
    ) -> Result<(), SubmitError> {
        self.submit(call_id, Err(message.into()))
    }

    fn submit(&self, call_id: &CallId, result: Result<Value, String>) -> Result<(), SubmitError> {
        let mut map = self.inner.inflight();
        let entry = map
            .get_mut(call_id)
            .ok_or_else(|| SubmitError::UnknownCall(call_id.clone()))?;
        let tx = entry
            .client
            .take()
            .ok_or_else(|| SubmitError::NotClientCall(call_id.clone()))?;
        // The waiting task finishes the call; if it is already gone (it
        // timed out this instant), the result is dropped like any late one.
        tx.send(result)
            .map_err(|_| SubmitError::UnknownCall(call_id.clone()))
    }

    /// Cancel a call (`session.cancel_tool`, or the model's `<tool_cancel>`).
    /// Emits [`ToolStatus::Cancelled`] and returns `true` if it was in
    /// flight. A server handler is aborted at its next `.await`; a late
    /// client result is rejected with [`SubmitError::UnknownCall`].
    pub fn cancel(&self, call_id: &CallId) -> bool {
        self.inner.finish(call_id, ToolStatus::Cancelled)
    }

    /// Cancel every call in flight; returns the ids that were cancelled.
    pub fn cancel_all(&self) -> Vec<CallId> {
        self.in_flight()
            .into_iter()
            .filter(|id| self.cancel(id))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::ToolSpec;
    use serde_json::json;

    fn registry() -> Arc<ToolRegistry> {
        let obj = json!({"type": "object"});
        Arc::new(
            ToolRegistry::from_specs([
                ToolSpec::new("echo", "", obj.clone()),
                ToolSpec::new("sleep", "", obj.clone()),
                ToolSpec::new("fail", "", obj.clone()),
                ToolSpec::new("panic", "", obj.clone()),
                ToolSpec::new("unhandled", "", obj.clone()),
                ToolSpec::client("ask_client", "", obj),
            ])
            .unwrap(),
        )
    }

    fn call(id: &str, name: &str, args: Value) -> ToolCall {
        ToolCall {
            call_id: id.into(),
            name: name.into(),
            arguments: args,
        }
    }

    /// Executor with handlers; `sleep` sleeps `args.ms` milliseconds.
    fn executor() -> (Executor, mpsc::UnboundedReceiver<ExecutorEvent>) {
        let (ex, rx) = Executor::new(registry(), ExecutorConfig::default());
        ex.register_handler("echo", |a: Value| async move { Ok(a) })
            .unwrap();
        ex.register_handler("sleep", |a: Value| async move {
            let ms = a["ms"].as_u64().unwrap_or(0);
            tokio::time::sleep(Duration::from_millis(ms)).await;
            Ok(json!({"slept": ms}))
        })
        .unwrap();
        ex.register_handler("fail", |_a: Value| async move { Err("boom".to_string()) })
            .unwrap();
        ex.register_handler("panic", |_a: Value| async move {
            tokio::task::yield_now().await;
            panic!("handler bug")
        })
        .unwrap();
        (ex, rx)
    }

    async fn next_finished(rx: &mut mpsc::UnboundedReceiver<ExecutorEvent>) -> ToolOutcome {
        loop {
            match rx.recv().await.expect("event stream ended") {
                ExecutorEvent::Finished(o) => return o,
                _ => continue,
            }
        }
    }

    #[test]
    fn registration_rules() {
        let (ex, _rx) = Executor::new(registry(), ExecutorConfig::default());
        let h = |_a: Value| async move { Ok(Value::Null) };
        assert_eq!(
            ex.register_handler("nope", h),
            Err(RegisterError::UnknownTool("nope".into()))
        );
        assert_eq!(
            ex.register_handler("ask_client", h),
            Err(RegisterError::ClientTool("ask_client".into()))
        );
        assert!(ex
            .register_handler_with_timeout("echo", h, Duration::from_secs(1))
            .is_ok());
        assert_eq!(
            ex.dispatch(call("c_1", "echo", json!({}))),
            Err(DispatchError::NoRuntime)
        );
        assert!(format!("{ex:?}").contains("Executor"));
    }

    #[tokio::test(start_paused = true)]
    async fn ok_error_and_events_in_order() {
        let (ex, mut rx) = executor();
        ex.dispatch(call("c_1", "echo", json!({"x": 1}))).unwrap();
        assert_eq!(
            rx.recv().await.unwrap(),
            ExecutorEvent::Dispatched {
                call_id: "c_1".into(),
                name: "echo".into(),
                location: ToolLocation::Server
            }
        );
        let o = next_finished(&mut rx).await;
        assert_eq!(o.status, ToolStatus::Ok(json!({"x": 1})));
        assert_eq!((o.call_id.as_str(), o.name.as_str()), ("c_1", "echo"));

        ex.dispatch(call("c_2", "fail", json!({}))).unwrap();
        assert_eq!(
            next_finished(&mut rx).await.status,
            ToolStatus::Error("boom".into())
        );
        assert!(ex.in_flight().is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn unknown_tool_and_missing_handler_finish_with_error() {
        let (ex, mut rx) = executor();
        ex.dispatch(call("c_1", "ghost", json!({}))).unwrap();
        let o = next_finished(&mut rx).await;
        assert!(matches!(o.status, ToolStatus::Error(ref m) if m.contains("unknown tool")));
        ex.dispatch(call("c_2", "unhandled", json!({}))).unwrap();
        let o = next_finished(&mut rx).await;
        assert!(matches!(o.status, ToolStatus::Error(ref m) if m.contains("no handler")));
    }

    #[tokio::test(start_paused = true)]
    async fn parallel_calls_complete_by_duration_not_order() {
        let (ex, mut rx) = executor();
        let t0 = Instant::now();
        ex.dispatch(call("c_1", "sleep", json!({"ms": 300})))
            .unwrap();
        ex.dispatch(call("c_2", "sleep", json!({"ms": 100})))
            .unwrap();
        ex.dispatch(call("c_3", "sleep", json!({"ms": 200})))
            .unwrap();
        assert_eq!(ex.in_flight(), ["c_1".into(), "c_2".into(), "c_3".into()]);
        assert_eq!(
            ex.dispatch(call("c_2", "echo", json!({}))),
            Err(DispatchError::DuplicateCallId("c_2".into()))
        );
        let mut order = Vec::new();
        for _ in 0..3 {
            let o = next_finished(&mut rx).await;
            assert!(matches!(o.status, ToolStatus::Ok(_)));
            order.push(o.call_id.0);
        }
        assert_eq!(order, ["c_2", "c_3", "c_1"]);
        // Concurrent: total time is the longest call, not the sum.
        assert_eq!(t0.elapsed(), Duration::from_millis(300));
    }

    #[tokio::test(start_paused = true)]
    async fn timeouts_default_and_per_tool() {
        let (ex, mut rx) = executor();
        ex.dispatch(call("c_1", "sleep", json!({"ms": 60_000})))
            .unwrap();
        let o = next_finished(&mut rx).await;
        assert_eq!(o.status, ToolStatus::Timeout);
        assert_eq!(o.elapsed, ExecutorConfig::default().server_timeout);

        ex.register_handler_with_timeout(
            "sleep",
            |_a: Value| async move {
                tokio::time::sleep(Duration::from_secs(5)).await;
                Ok(Value::Null)
            },
            Duration::from_millis(250),
        )
        .unwrap();
        ex.dispatch(call("c_2", "sleep", json!({}))).unwrap();
        let o = next_finished(&mut rx).await;
        assert_eq!(
            (o.status, o.elapsed),
            (ToolStatus::Timeout, Duration::from_millis(250))
        );
    }

    #[tokio::test(start_paused = true)]
    async fn cancel_by_id_and_all() {
        let (ex, mut rx) = executor();
        ex.dispatch(call("c_1", "sleep", json!({"ms": 1000})))
            .unwrap();
        ex.dispatch(call("c_2", "sleep", json!({"ms": 1000})))
            .unwrap();
        ex.dispatch(call("c_3", "sleep", json!({"ms": 1000})))
            .unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(ex.cancel(&"c_2".into()));
        assert!(!ex.cancel(&"c_2".into()), "only once");
        assert!(!ex.cancel(&"c_99".into()));
        let o = next_finished(&mut rx).await;
        assert_eq!(
            (o.call_id.as_str(), &o.status),
            ("c_2", &ToolStatus::Cancelled)
        );
        assert_eq!(ex.cancel_all(), ["c_1".into(), "c_3".into()]);
        for _ in 0..2 {
            assert_eq!(next_finished(&mut rx).await.status, ToolStatus::Cancelled);
        }
        // Aborted handlers never report: no further events after their deadline.
        tokio::time::sleep(Duration::from_secs(5)).await;
        assert!(rx.try_recv().is_err());
        assert!(ex.in_flight().is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn panicking_handler_reports_error() {
        let (ex, mut rx) = executor();
        ex.dispatch(call("c_1", "panic", json!({}))).unwrap();
        let o = next_finished(&mut rx).await;
        assert_eq!(o.status, ToolStatus::Error("tool handler panicked".into()));
        assert!(!ex.is_in_flight(&"c_1".into()));
    }

    #[tokio::test(start_paused = true)]
    async fn client_bridge_round_trip() {
        let (ex, mut rx) = executor();
        let args = json!({"q": "battery level"});
        ex.dispatch(call("c_7", "ask_client", args.clone()))
            .unwrap();
        assert!(matches!(
            rx.recv().await.unwrap(),
            ExecutorEvent::Dispatched {
                location: ToolLocation::Client,
                ..
            }
        ));
        assert_eq!(
            rx.recv().await.unwrap(),
            ExecutorEvent::ClientCall(call("c_7", "ask_client", args))
        );
        assert!(ex.is_in_flight(&"c_7".into()));
        tokio::time::sleep(Duration::from_millis(500)).await;
        ex.submit_client_result(&"c_7".into(), json!({"level": 0.8}))
            .unwrap();
        let o = next_finished(&mut rx).await;
        assert_eq!(o.status, ToolStatus::Ok(json!({"level": 0.8})));
        assert_eq!(o.elapsed, Duration::from_millis(500));
        // Late / duplicate results are rejected.
        assert_eq!(
            ex.submit_client_result(&"c_7".into(), json!(1)),
            Err(SubmitError::UnknownCall("c_7".into()))
        );
    }

    #[tokio::test(start_paused = true)]
    async fn client_bridge_error_timeout_cancel() {
        let (ex, mut rx) = executor();
        ex.dispatch(call("c_1", "ask_client", json!({}))).unwrap();
        ex.submit_client_error(&"c_1".into(), "permission denied")
            .unwrap();
        assert_eq!(
            next_finished(&mut rx).await.status,
            ToolStatus::Error("permission denied".into())
        );

        ex.dispatch(call("c_2", "ask_client", json!({}))).unwrap();
        let o = next_finished(&mut rx).await;
        assert_eq!(
            (o.status, o.elapsed),
            (ToolStatus::Timeout, Duration::from_secs(30))
        );
        assert_eq!(
            ex.submit_client_result(&"c_2".into(), json!(1)),
            Err(SubmitError::UnknownCall("c_2".into()))
        );

        ex.dispatch(call("c_3", "ask_client", json!({}))).unwrap();
        assert!(ex.cancel(&"c_3".into()));
        assert_eq!(next_finished(&mut rx).await.status, ToolStatus::Cancelled);
        assert_eq!(
            ex.submit_client_result(&"c_3".into(), json!(1)),
            Err(SubmitError::UnknownCall("c_3".into()))
        );

        // Server calls do not accept client results.
        ex.dispatch(call("c_4", "sleep", json!({"ms": 50})))
            .unwrap();
        assert_eq!(
            ex.submit_client_result(&"c_4".into(), json!(1)),
            Err(SubmitError::NotClientCall("c_4".into()))
        );
        assert!(matches!(
            next_finished(&mut rx).await.status,
            ToolStatus::Ok(_)
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn many_concurrent_calls_finish_exactly_once() {
        let (ex, mut rx) = executor();
        let n = 200;
        for i in 0..n {
            let (name, args) = if i % 2 == 0 {
                ("echo", json!({"i": i}))
            } else {
                ("sleep", json!({"ms": i % 7}))
            };
            ex.dispatch(call(&format!("c_{i}"), name, args)).unwrap();
        }
        // Race cancellations against completions.
        for i in (0..n).step_by(3) {
            ex.cancel(&format!("c_{i}").into());
        }
        let mut seen = std::collections::HashSet::new();
        while seen.len() < n {
            let o = next_finished(&mut rx).await;
            assert!(
                seen.insert(o.call_id.0.clone()),
                "duplicate outcome {}",
                o.call_id
            );
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
        while let Ok(ev) = rx.try_recv() {
            assert!(
                !matches!(ev, ExecutorEvent::Finished(_)),
                "extra outcome {ev:?}"
            );
        }
        assert!(ex.in_flight().is_empty());
    }
}
