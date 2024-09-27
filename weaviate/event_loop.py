import asyncio
import threading
import time
from concurrent.futures import Future
from typing import Any, Callable, Coroutine, Dict, Generic, Optional, TypeVar, cast
from typing_extensions import ParamSpec


class EventLoopClosedError(RuntimeError):
    """Raise an exception if the event loop is closed."""

    def __init__(self) -> None:
        super().__init__(
            "The event loop is closed. This may have been caused by trying to use an async function "
            "after the event loop was terminated. Ensure that the event loop is running before making async calls."
        )


# Define type variables for use in function signatures
P = ParamSpec("P")
T = TypeVar("T")


class _Future(Future, Generic[T]):
    def result(self, timeout: Optional[float] = None) -> T:
        return cast(T, super().result(timeout))


class _EventLoop:
    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        self.loop: Optional[asyncio.AbstractEventLoop] = loop

    def start(self) -> None:
        if self.loop is not None:
            return
        self.loop = self.__start_new_event_loop()

    def _ensure_event_loop_is_running(self) -> asyncio.AbstractEventLoop:
        """Helper function to ensure the event loop is not None and running."""
        if self.loop is None or self.loop.is_closed():
            raise EventLoopClosedError()
        return self.loop

    def run_until_complete(
        self, f: Callable[P, Coroutine[Any, Any, T]], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        """Runs the provided coroutine in a blocking manner."""
        loop = self._ensure_event_loop_is_running()  # Ensure loop is not None and running

        try:
            fut = asyncio.run_coroutine_threadsafe(f(*args, **kwargs), loop)
            return fut.result()
        except RuntimeError as e:
            # Re-raise with a more descriptive message if the event loop is closed
            if "Event loop is closed" in str(e):
                raise EventLoopClosedError() from e
            raise RuntimeError(
                f"Failed to run coroutine '{f.__name__}' due to an unexpected error: {str(e)}"
            ) from e  # Provide context on other runtime errors

    def schedule(
        self, f: Callable[P, Coroutine[Any, Any, T]], *args: P.args, **kwargs: P.kwargs
    ) -> _Future[T]:
        """Schedules the provided coroutine for asynchronous execution."""
        loop = self._ensure_event_loop_is_running()  # Ensure loop is not None and running

        try:
            fut = asyncio.run_coroutine_threadsafe(f(*args, **kwargs), loop)
            return cast(_Future[T], fut)  # Ensure we return a valid _Future instance
        except RuntimeError as e:
            # Re-raise with a more descriptive message if the event loop is closed
            if "Event loop is closed" in str(e):
                raise EventLoopClosedError() from e
            raise RuntimeError(
                f"Failed to schedule coroutine '{f.__name__}' due to an unexpected error: {str(e)}"
            ) from e  # Provide context on other runtime errors

    def shutdown(self) -> None:
        if self.loop is None:
            return
        self.loop.call_soon_threadsafe(self.loop.stop)

    @staticmethod
    def __run_event_loop(loop: asyncio.AbstractEventLoop) -> None:
        try:
            loop.run_forever()
        finally:
            # This is entered when loop.stop is scheduled from the main thread
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    @staticmethod
    def __start_new_event_loop() -> asyncio.AbstractEventLoop:
        loop = asyncio.new_event_loop()

        event_loop = threading.Thread(
            target=_EventLoop.__run_event_loop,
            daemon=True,
            args=(loop,),
            name="eventLoop",
        )
        event_loop.start()

        while not loop.is_running():
            time.sleep(0.01)

        return loop

    @staticmethod
    def patch_exception_handler(loop: asyncio.AbstractEventLoop) -> None:
        """
        This patches the asyncio exception handler to ignore the `BlockingIOError: [Errno 35] Resource temporarily unavailable` error
        that is emitted by `aio.grpc` when multiple event loops are used in separate threads. This error is not actually an implementation/call error,
        it's just a problem with grpc's cython implementation of `aio.Channel.__init__` whereby a `socket.recv(1)` call only works on the first call with
        all subsequent calls to `aio.Channel.__init__` throwing the above error.

        This call within the `aio.Channel.__init__` method does not affect the functionality of the library and can be safely ignored.

        Context:
            - https://github.com/grpc/grpc/issues/25364
            - https://github.com/grpc/grpc/pull/36096
        """

        def exception_handler(loop: asyncio.AbstractEventLoop, context: Dict[str, Any]) -> None:
            if "exception" in context:
                err = f"{type(context['exception']).__name__}: {context['exception']}"
                if "BlockingIOError: [Errno 35] Resource temporarily unavailable" == err:
                    return
            loop.default_exception_handler(context)

        loop.set_exception_handler(exception_handler)

    def __del__(self) -> None:
        self.shutdown()


class _EventLoopSingleton:
    _instance: Optional[_EventLoop] = None

    @classmethod
    def get_instance(cls) -> _EventLoop:
        if cls._instance is not None:
            return cls._instance
        cls._instance = _EventLoop()
        cls._instance.start()
        return cls._instance

    def __del__(self) -> None:
        if self._instance is not None:
            self._instance.shutdown()
            self._instance = None
