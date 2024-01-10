from __future__ import annotations

import datetime
import time

from copy import copy
from threading import Thread, Event
from typing import Any, Dict, Optional, Tuple, Union, cast

from grpc import _channel  # type: ignore
from grpc_health.v1 import health_pb2  # type: ignore
from httpx import (
    AsyncClient,
    Client,
    ConnectError,
    HTTPError,
    Limits,
    ReadError,
    ReadTimeout,
    RemoteProtocolError,
    RequestError,
    Response,
    Timeout,
    get,
)
from authlib.integrations.httpx_client import AsyncOAuth2Client, OAuth2Client  # type: ignore

from weaviate import __version__ as client_version
from weaviate.auth import (
    AuthCredentials,
    AuthApiKey,
    AuthClientCredentials,
)
from weaviate.config import ConnectionConfig
from weaviate.connect.authentication import _Auth
from weaviate.connect.base import (
    _ConnectionBase,
    ConnectionParams,
    JSONPayload,
    _get_proxies,
    PYPI_TIMEOUT,
    TIMEOUT_TYPE_RETURN,
)
from weaviate.embedded import EmbeddedDB
from weaviate.exceptions import (
    AuthenticationFailedException,
    WeaviateGrpcUnavailable,
    WeaviateStartUpError,
)
from weaviate.util import (
    is_weaviate_domain,
    is_weaviate_too_old,
    is_weaviate_client_too_old,
    PYPI_PACKAGE_URL,
    _decode_json_response_dict,
    _ServerVersion,
)
from weaviate.warnings import _Warnings

from weaviate.proto.v1 import weaviate_pb2_grpc

Session = Union[Client, OAuth2Client]
AsyncSession = Union[AsyncClient, AsyncOAuth2Client]


class _Connection(_ConnectionBase):
    """
    Connection class used to communicate to a weaviate instance.
    """

    def __init__(
        self,
        connection_params: ConnectionParams,
        auth_client_secret: Optional[AuthCredentials],
        timeout_config: Tuple[float, float],
        proxies: Union[dict, str, None],
        trust_env: bool,
        additional_headers: Optional[Dict[str, Any]],
        connection_config: ConnectionConfig,
        embedded_db: Optional[EmbeddedDB] = None,
    ):
        self.url = connection_params._http_url
        self.embedded_db = embedded_db
        self._api_version_path = "/v1"
        self._aclient: Optional[AsyncSession] = None
        self._client: Session
        self.__additional_headers = {}
        self.__auth = auth_client_secret
        self._connection_params = connection_params
        self._grpc_available = False
        self._grpc_stub: Optional[weaviate_pb2_grpc.WeaviateStub] = None
        self._grpc_stub_async: Optional[weaviate_pb2_grpc.WeaviateStub] = None
        self.timeout_config = timeout_config
        self.__connection_config = connection_config
        self._weaviate_version: _ServerVersion

        self._headers = {"content-type": "application/json"}
        if additional_headers is not None:
            if not isinstance(additional_headers, dict):
                raise TypeError(
                    f"'additional_headers' must be of type dict or None. Given type: {type(additional_headers)}."
                )
            self.__additional_headers = additional_headers
            for key, value in additional_headers.items():
                self._headers[key.lower()] = value

        self._proxies = _get_proxies(proxies, trust_env)

        # auth secrets can contain more information than a header (refresh tokens and lifetime) and therefore take
        # precedent over headers
        if "authorization" in self._headers and auth_client_secret is not None:
            _Warnings.auth_header_and_auth_secret()
            self._headers.pop("authorization")

        # if there are API keys included add them right away to headers
        if auth_client_secret is not None and isinstance(auth_client_secret, AuthApiKey):
            self._headers["authorization"] = "Bearer " + auth_client_secret.api_key

    def connect(self, skip_init_checks: bool) -> None:
        self._create_clients(self.__auth)
        if not skip_init_checks:
            # first connection attempt
            try:
                self._server_version = self.get_meta()["version"]
            except (ConnectError, ReadError, RemoteProtocolError) as e:
                raise WeaviateStartUpError(f"Could not connect to Weaviate:{e}.") from e

            if self._server_version < "1.14":
                _Warnings.weaviate_server_older_than_1_14(self._server_version)
            if is_weaviate_too_old(self._server_version):
                _Warnings.weaviate_too_old_vs_latest(self._server_version)

            try:
                pkg_info = get(PYPI_PACKAGE_URL, timeout=PYPI_TIMEOUT).json()
                pkg_info = pkg_info.get("info", {})
                latest_version = pkg_info.get("version", "unknown version")
                if is_weaviate_client_too_old(client_version, latest_version):
                    _Warnings.weaviate_client_too_old_vs_latest(client_version, latest_version)
            except RequestError:
                pass  # ignore any errors related to requests, it is a best-effort warning
        else:
            self._server_version = ""
        self._weaviate_version = _ServerVersion.from_string(self._server_version)

    def __make_sync_client(self) -> Client:
        return Client(
            headers=self._get_request_header(),
            timeout=Timeout(None, connect=self.timeout_config[0], read=self.timeout_config[1]),
            proxies=self._proxies,
            limits=Limits(
                max_connections=self.__connection_config.session_pool_maxsize,
                max_keepalive_connections=self.__connection_config.session_pool_connections,
            ),
        )

    def __make_async_client(self) -> AsyncClient:
        return AsyncClient(
            headers=self._get_request_header(),
            timeout=Timeout(None, connect=self.timeout_config[0], read=self.timeout_config[1]),
            proxies=self._proxies,
            limits=Limits(
                max_connections=self.__connection_config.session_pool_maxsize,
                max_keepalive_connections=self.__connection_config.session_pool_connections,
            ),
        )

    def __make_clients(self) -> None:
        self._client = self.__make_sync_client()

    def _create_clients(self, auth_client_secret: Optional[AuthCredentials]) -> None:
        """Creates sync and async httpx clients.

        Either through authlib.oauth2 if authentication is enabled or a normal httpx sync client otherwise.

        Raises
        ------
        ValueError
            If no authentication credentials provided but the Weaviate server has OpenID configured.
        """
        # API keys are separate from OIDC and do not need any config from weaviate
        if auth_client_secret is not None and isinstance(auth_client_secret, AuthApiKey):
            self.__make_clients()
            return

        if "authorization" in self._headers and auth_client_secret is None:
            self.__make_clients()
            return

        oidc_url = self.url + self._api_version_path + "/.well-known/openid-configuration"
        with self.__make_sync_client() as client:
            response = client.get(
                oidc_url,
            )
        if response.status_code == 200:
            # Some setups are behind proxies that return some default page - for example a login - for all requests.
            # If the response is not json, we assume that this is the case and try unauthenticated access. Any auth
            # header provided by the user is unaffected.
            try:
                resp = response.json()
            except Exception:
                _Warnings.auth_cannot_parse_oidc_config(oidc_url)
                self.__make_clients()
                return

            if auth_client_secret is not None and not isinstance(auth_client_secret, AuthApiKey):
                _auth = _Auth(resp, auth_client_secret, self)
                self._client = _auth.get_auth_session()

                if isinstance(auth_client_secret, AuthClientCredentials):
                    # credentials should only be saved for client credentials, otherwise use refresh token
                    self._create_background_token_refresh(_auth)
                else:
                    self._create_background_token_refresh()

            else:
                msg = f""""No login credentials provided. The weaviate instance at {self.url} requires login credentials.

                    Please check our documentation at https://weaviate.io/developers/weaviate/client-libraries/python#authentication
                    for more information about how to use authentication."""

                if is_weaviate_domain(self.url):
                    msg += """

                    You can instantiate the client with login credentials for WCS using

                    client = weaviate.Client(
                      url=YOUR_WEAVIATE_URL,
                      auth_client_secret=weaviate.AuthClientPassword(
                        username = YOUR_WCS_USER,
                        password = YOUR_WCS_PW,
                      ))
                    """
                raise AuthenticationFailedException(msg)
        elif response.status_code == 404 and auth_client_secret is not None:
            _Warnings.auth_with_anon_weaviate()
            self.__make_clients()
        else:
            self.__make_clients()

    def get_current_bearer_token(self) -> str:
        if "authorization" in self._headers:
            return self._headers["authorization"]
        elif isinstance(self._client, OAuth2Client):
            return f"Bearer {self._client.token['access_token']}"
        return ""

    def _create_background_token_refresh(self, _auth: Optional[_Auth] = None) -> None:
        """Create a background thread that periodically refreshes access and refresh tokens.

        While the underlying library refreshes tokens, it does not have an internal cronjob that checks every
        X-seconds if a token has expired. If there is no activity for longer than the refresh tokens lifetime, it will
        expire. Therefore, refresh manually shortly before expiration time is up."""
        assert isinstance(self._client, OAuth2Client)
        if "refresh_token" not in self._client.token and _auth is None:
            return

        expires_in: int = self._client.token.get(
            "expires_in", 60
        )  # use 1minute as token lifetime if not supplied
        self._shutdown_background_event = Event()

        def periodic_refresh_token(refresh_time: int, _auth: Optional[_Auth]) -> None:
            time.sleep(max(refresh_time - 30, 1))
            while (
                self._shutdown_background_event is not None
                and not self._shutdown_background_event.is_set()
            ):
                # use refresh token when available
                try:
                    if "refresh_token" in cast(OAuth2Client, self._client).token:
                        assert isinstance(self._client, OAuth2Client)
                        self._client.token = self._client.refresh_token(
                            self._client.metadata["token_endpoint"]
                        )
                        refresh_time = self._client.token.get("expires_in") - 30
                    else:
                        # client credentials usually does not contain a refresh token => get a new token using the
                        # saved credentials
                        assert _auth is not None
                        assert isinstance(self._client, OAuth2Client)
                        new_session = _auth.get_auth_session()
                        self._client.token = new_session.fetch_token()
                except (HTTPError, ReadTimeout) as exc:
                    # retry again after one second, might be an unstable connection
                    refresh_time = 1
                    _Warnings.token_refresh_failed(exc)

                time.sleep(max(refresh_time, 1))

        demon = Thread(
            target=periodic_refresh_token,
            args=(expires_in, _auth),
            daemon=True,
            name="TokenRefresh",
        )
        demon.start()

    def open_async(self) -> None:
        # Careful, this method is not async but it must be called from an async context where
        # there is a running event loop since grpc.aio.Channel makes a call to cygrpc.get_working_loop()
        if self._aclient is None:
            self._aclient = self.__make_async_client()
        if self._grpc_stub_async is None:
            self._grpc_channel_async = self._connection_params._grpc_channel(async_channel=True)
            assert self._grpc_channel_async is not None
            self._grpc_stub_async = weaviate_pb2_grpc.WeaviateStub(self._grpc_channel_async)

    async def aclose(self) -> None:
        if self._aclient is not None:
            await self._aclient.aclose()
            self._aclient = None
        if self._grpc_stub_async is not None:
            assert self._grpc_channel_async is not None
            await self._grpc_channel_async.close()
            self._grpc_stub_async = None

    def close(self) -> None:
        """Shutdown connection class gracefully."""
        # in case an exception happens before definition of these members
        if (
            hasattr(self, "_shutdown_background_event")
            and self._shutdown_background_event is not None
        ):
            self._shutdown_background_event.set()
        if hasattr(self, "_client"):
            self._client.close()

    def _get_request_header(self) -> dict:
        """
        Returns the correct headers for a request.

        Returns
        -------
        dict
            Request header as a dict.
        """
        return self._headers

    def __get_headers_for_async(self) -> Dict[str, str]:
        if "authorization" in self._headers:
            return self._headers

        auth_token = self.get_current_bearer_token()
        if auth_token == "":
            return self._headers

        # bearer token can change over time (OIDC) so we need to get the current one for each request
        copied_headers = copy(self._headers)
        copied_headers.update({"authorization": self.get_current_bearer_token()})
        return copied_headers

    def delete(
        self,
        path: str,
        weaviate_object: Optional[JSONPayload] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Response:
        """
        Make a DELETE request to the Weaviate server instance.

        Parameters
        ----------
        path : str
            Sub-path to the Weaviate resources. Must be a valid Weaviate sub-path.
            e.g. '/meta' or '/objects', without version.
        weaviate_object : dict, optional
            Object is used as payload for DELETE request. By default None.
        params : dict, optional
            Additional request parameters, by default None

        Returns
        -------
        httpx.Response
            The response, if request was successful.

        Raises
        ------
        httpx.ConnectionError
            If the DELETE request could not be made.
        """
        if self.embedded_db is not None:
            self.embedded_db.ensure_running()
        request_url = self.url + self._api_version_path + path

        # Must build manually because httpx is opinionated about sending JSON in DELETE requests
        # From httpx docs:
        # Note that the data, files, json and content parameters are not available on this function, as DELETE requests should not include a request body.
        request = self._client.build_request(
            "DELETE",
            url=request_url,
            json=weaviate_object,
            params=params,
        )
        res = self._client.send(request)
        return cast(Response, res)

    def patch(
        self,
        path: str,
        weaviate_object: JSONPayload,
        params: Optional[Dict[str, Any]] = None,
    ) -> Response:
        """
        Make a PATCH request to the Weaviate server instance.

        Parameters
        ----------
        path : str
            Sub-path to the Weaviate resources. Must be a valid Weaviate sub-path.
            e.g. '/meta' or '/objects', without version.
        weaviate_object : dict
            Object is used as payload for PATCH request.
        params : dict, optional
            Additional request parameters, by default None
        Returns
        -------
        httpx.Response
            The response, if request was successful.

        Raises
        ------
        httpx.ConnectionError
            If the PATCH request could not be made.
        """
        if self.embedded_db is not None:
            self.embedded_db.ensure_running()
        request_url = self.url + self._api_version_path + path

        res = self._client.patch(
            url=request_url,
            json=weaviate_object,
            params=params,
        )
        return cast(Response, res)

    def post(
        self,
        path: str,
        weaviate_object: JSONPayload,
        params: Optional[Dict[str, Any]] = None,
    ) -> Response:
        """
        Make a POST request to the Weaviate server instance.

        Parameters
        ----------
        path : str
            Sub-path to the Weaviate resources. Must be a valid Weaviate sub-path.
            e.g. '/meta' or '/objects', without version.
        weaviate_object : dict
            Object is used as payload for POST request.
        params : dict, optional
            Additional request parameters, by default None
        external_url: Is an external (non-weaviate) url called

        Returns
        -------
        httpx.Response
            The response, if request was successful.

        Raises
        ------
        httpx.ConnectionError
            If the POST request could not be made.
        """
        if self.embedded_db is not None:
            self.embedded_db.ensure_running()
        request_url = self.url + self._api_version_path + path

        res = self._client.post(
            url=request_url,
            json=weaviate_object,
            params=params,
            headers=self.__get_headers_for_async(),
        )
        return cast(Response, res)

    async def apost(
        self,
        path: str,
        weaviate_object: JSONPayload,
        params: Optional[Dict[str, Any]] = None,
    ) -> Response:
        """
        Make a async POST request to the Weaviate server instance.

        Parameters
        ----------
        path : str
            Sub-path to the Weaviate resources. Must be a valid Weaviate sub-path.
            e.g. '/meta' or '/objects', without version.
        weaviate_object : dict
            Object is used as payload for POST request.
        params : dict, optional
            Additional request parameters, by default None
        external_url: Is an external (non-weaviate) url called

        Returns
        -------
        httpx.Response
            The response, if request was successful.

        Raises
        ------
        httpx.ConnectionError
            If the POST request could not be made.
        """
        assert self._aclient is not None

        if self.embedded_db is not None:
            self.embedded_db.ensure_running()
        request_url = self.url + self._api_version_path + path

        return await self._aclient.post(
            url=request_url,
            json=weaviate_object,
            params=params,
        )

    def put(
        self,
        path: str,
        weaviate_object: JSONPayload,
        params: Optional[Dict[str, Any]] = None,
    ) -> Response:
        """
        Make a PUT request to the Weaviate server instance.

        Parameters
        ----------
        path : str
            Sub-path to the Weaviate resources. Must be a valid Weaviate sub-path.
            e.g. '/meta' or '/objects', without version.
        weaviate_object : dict
            Object is used as payload for PUT request.
        params : dict, optional
            Additional request parameters, by default None
        Returns
        -------
        httpx.Response
            The response, if request was successful.

        Raises
        ------
        httpx.ConnectionError
            If the PUT request could not be made.
        """
        if self.embedded_db is not None:
            self.embedded_db.ensure_running()
        request_url = self.url + self._api_version_path + path

        res = self._client.put(
            url=request_url,
            json=weaviate_object,
            params=params,
        )
        return cast(Response, res)

    def get(
        self, path: str, params: Optional[Dict[str, Any]] = None, external_url: bool = False
    ) -> Response:
        """Make a GET request.

        Parameters
        ----------
        path : str
            Sub-path to the Weaviate resources. Must be a valid Weaviate sub-path.
            e.g. '/meta' or '/objects', without version.
        params : dict, optional
            Additional request parameters, by default None
        external_url: Is an external (non-weaviate) url called

        Returns
        -------
        httpx.Response
            The response if request was successful.

        Raises
        ------
        httpx.ConnectionError
            If the GET request could not be made.
        """
        if self.embedded_db is not None:
            self.embedded_db.ensure_running()
        if params is None:
            params = {}

        if external_url:
            request_url = path
        else:
            request_url = self.url + self._api_version_path + path

        res = self._client.get(
            url=request_url,
            params=params,
        )
        return cast(Response, res)

    def head(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Response:
        """
        Make a HEAD request to the server.

        Parameters
        ----------
        path : str
            Sub-path to the resources. Must be a valid sub-path.
            e.g. '/meta' or '/objects', without version.
        params : dict, optional
            Additional request parameters, by default None

        Returns
        -------
        httpx.Response
            The response to the request.

        Raises
        ------
        httpx.ConnectionError
            If the HEAD request could not be made.
        """
        if self.embedded_db is not None:
            self.embedded_db.ensure_running()
        request_url = self.url + self._api_version_path + path

        res = self._client.head(
            url=request_url,
            params=params,
        )
        return cast(Response, res)

    @property
    def proxies(self) -> dict:
        return self._proxies

    def wait_for_weaviate(self, startup_period: int) -> None:
        """
        Waits until weaviate is ready or the timelimit given in 'startup_period' has passed.

        Parameters
        ----------
        startup_period : int
            Describes how long the client will wait for weaviate to start in seconds.

        Raises
        ------
        WeaviateStartUpError
            If weaviate takes longer than the timelimit to respond.
        """

        ready_url = self.url + self._api_version_path + "/.well-known/ready"
        with Client(headers=self._get_request_header()) as client:
            for _i in range(startup_period):
                try:
                    res: Response = client.get(ready_url)
                    res.raise_for_status()
                    return
                except (ConnectError, HTTPError):
                    time.sleep(1)

            try:
                res = client.get(ready_url)
                res.raise_for_status()
                return
            except (ConnectError, HTTPError) as error:
                raise WeaviateStartUpError(
                    f"Weaviate did not start up in {startup_period} seconds. Either the Weaviate URL {self.url} is wrong or Weaviate did not start up in the interval given in 'startup_period'."
                ) from error

    @property
    def grpc_stub(self) -> Optional[weaviate_pb2_grpc.WeaviateStub]:
        return self._grpc_stub

    @property
    def server_version(self) -> str:
        """
        Version of the weaviate instance.
        """
        return self._server_version

    @property
    def grpc_available(self) -> bool:
        return self._grpc_available

    def get_proxies(self) -> dict:
        return self._proxies

    @property
    def additional_headers(self) -> Dict[str, str]:
        return self.__additional_headers

    def get_meta(self) -> Dict[str, str]:
        """
        Returns the meta endpoint.
        """
        response = self.get(path="/meta")
        res = _decode_json_response_dict(response, "Meta endpoint")
        assert res is not None
        return res


class ConnectionV4(_Connection):
    def __init__(
        self,
        connection_params: ConnectionParams,
        auth_client_secret: Optional[AuthCredentials],
        timeout_config: TIMEOUT_TYPE_RETURN,
        proxies: Union[dict, str, None],
        trust_env: bool,
        additional_headers: Optional[Dict[str, Any]],
        connection_config: ConnectionConfig,
        embedded_db: Optional[EmbeddedDB] = None,
    ):
        super().__init__(
            connection_params,
            auth_client_secret,
            timeout_config,
            proxies,
            trust_env,
            additional_headers,
            connection_config,
            embedded_db,
        )

    def connect(self, skip_init_checks: bool) -> None:
        super().connect(skip_init_checks)
        # create GRPC channel. If Weaviate does not support GRPC then error now.
        if self._connection_params._has_grpc:
            grpc_channel = self._connection_params._grpc_channel(async_channel=False)
            assert grpc_channel is not None
            self._grpc_stub = weaviate_pb2_grpc.WeaviateStub(grpc_channel)

            self._grpc_available = True
            if not skip_init_checks:
                try:
                    res: health_pb2.HealthCheckResponse = grpc_channel.unary_unary(
                        "/grpc.health.v1.Health/Check",
                        request_serializer=health_pb2.HealthCheckRequest.SerializeToString,
                        response_deserializer=health_pb2.HealthCheckResponse.FromString,
                    )(health_pb2.HealthCheckRequest(), timeout=1)
                    if res.status != health_pb2.HealthCheckResponse.SERVING:
                        raise WeaviateGrpcUnavailable(f"Weaviate v{self.server_version}")
                except _channel._InactiveRpcError as e:
                    raise WeaviateGrpcUnavailable(f"Weaviate v{self.server_version}") from e
        else:
            raise WeaviateGrpcUnavailable(
                "You must provide the gRPC port in `connection_params` to use gRPC."
            )

    @property
    def grpc_stub(self) -> Optional[weaviate_pb2_grpc.WeaviateStub]:
        if not self._grpc_available:
            raise WeaviateGrpcUnavailable(
                "Did you forget to call client.connect() before using the client?"
            )
        return self._grpc_stub

    @property
    def agrpc_stub(self) -> Optional[weaviate_pb2_grpc.WeaviateStub]:
        if not self._grpc_available:
            raise WeaviateGrpcUnavailable(
                "Did you forget to call client.connect() before using the client?"
            )
        return self._grpc_stub_async


def _get_epoch_time() -> int:
    """
    Get the current epoch time as an integer.

    Returns
    -------
    int
        Current epoch time.
    """

    dts = datetime.datetime.utcnow()
    return round(time.mktime(dts.timetuple()) + dts.microsecond / 1e6)
